"""
Phase 2 Memory Pool Management System

Enhanced memory management for Mark Six AI system targeting 60-80% memory efficiency improvement
and 30% training speedup through intelligent caching and tensor pooling.

Expert Panel Approved: Memory Management Engineer + Performance Profiling Analyst
Expected Impact: Optimize 15.6/31.9GB RAM usage (50% underutilized)
"""

import torch
import numpy as np
import psutil
import threading
import time
import gc
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
import weakref
import warnings

@dataclass
class MemoryPoolConfig:
    """Configuration for memory pool management."""
    # Pool sizes (in GB)
    batch_cache_gb: float = 8.0
    feature_cache_gb: float = 6.0
    tensor_pool_gb: float = 4.0
    working_memory_gb: float = 8.0
    
    # Memory pressure thresholds
    warning_threshold: float = 0.80
    emergency_threshold: float = 0.90
    cleanup_threshold: float = 0.85
    
    # Cache policies
    lru_cleanup_ratio: float = 0.3  # Clean 30% when threshold hit
    cache_ttl_seconds: int = 1800   # 30 minutes
    enable_compression: bool = True
    
    # Performance tuning
    prealloc_tensor_sizes: List[Tuple[int, ...]] = None
    memory_check_interval: float = 10.0  # seconds


class TensorPool:
    """
    High-performance tensor memory pool for reducing allocation overhead.
    Reuses tensors to minimize memory fragmentation and allocation time.
    """
    
    def __init__(self, max_size_gb: float = 4.0, device: str = 'cuda'):
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.device = torch.device(device)
        self.current_size = 0
        
        # Pools organized by size and dtype
        self.pools = defaultdict(lambda: defaultdict(list))  # size -> dtype -> [tensors]
        self.in_use = set()  # Track tensor ids to avoid WeakSet comparison issues
        self.tensor_refs = {}  # Map tensor id to weak reference
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'allocations': 0,
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_allocated_bytes': 0
        }
        
        # Pre-allocate common tensor sizes
        self._preallocate_common_sizes()
    
    def _preallocate_common_sizes(self):
        """Pre-allocate tensors for common Mark Six operations."""
        common_sizes = [
            (8, 6),      # Batch size 8, 6 numbers (base batch)
            (16, 6),     # Batch size 16, 6 numbers 
            (28, 6),     # Batch size 28, 6 numbers (Phase 1 optimized)
            (32, 6),     # Batch size 32, 6 numbers
            (8, 17),     # Features: batch_size 8, 17 features
            (16, 17),    # Features: batch_size 16, 17 features  
            (28, 17),    # Features: batch_size 28, 17 features
            (32, 17),    # Features: batch_size 32, 17 features
            (10, 6),     # Temporal sequence length 10, 6 numbers
            (20, 6),     # Temporal sequence length 20, 6 numbers
        ]
        
        with self.lock:
            for size in common_sizes:
                # Pre-allocate 2-3 tensors of each common size
                for _ in range(3):
                    for dtype in [torch.long, torch.float32]:
                        if self.current_size < self.max_size_bytes // 2:  # Use max 50% for prealloc
                            tensor = torch.empty(size, dtype=dtype, device=self.device)
                            tensor_size = tensor.numel() * tensor.element_size()
                            
                            self.pools[size][dtype].append(tensor)
                            self.current_size += tensor_size
                            
        print(f"✅ TensorPool pre-allocated common sizes: {len(common_sizes)} patterns")
    
    def get_tensor(self, size: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get a tensor from the pool or allocate new one."""
        with self.lock:
            self.stats['allocations'] += 1
            
            # Try to get from pool
            if size in self.pools and dtype in self.pools[size] and self.pools[size][dtype]:
                tensor = self.pools[size][dtype].pop()
                
                # Track tensor using id
                tensor_id = id(tensor)
                self.in_use.add(tensor_id)
                self.tensor_refs[tensor_id] = weakref.ref(tensor)
                self.stats['hits'] += 1
                
                # Clear the tensor data (security/correctness)
                tensor.zero_()
                return tensor
            
            # Pool miss - allocate new tensor
            self.stats['misses'] += 1
            
            try:
                tensor = torch.empty(size, dtype=dtype, device=self.device)
                tensor_size = tensor.numel() * tensor.element_size()
                
                # Check if we need to free space
                if self.current_size + tensor_size > self.max_size_bytes:
                    self._evict_unused_tensors(tensor_size)
                
                self.current_size += tensor_size
                self.stats['total_allocated_bytes'] += tensor_size
                
                # Track tensor using id
                tensor_id = id(tensor)
                self.in_use.add(tensor_id)
                self.tensor_refs[tensor_id] = weakref.ref(tensor)
                
                return tensor
                
            except torch.cuda.OutOfMemoryError:
                # Emergency cleanup and retry
                self._emergency_cleanup()
                tensor = torch.empty(size, dtype=dtype, device=self.device)
                
                # Track tensor using id
                tensor_id = id(tensor)
                self.in_use.add(tensor_id)
                self.tensor_refs[tensor_id] = weakref.ref(tensor)
                
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return a tensor to the pool for reuse."""
        tensor_id = id(tensor)
        
        if tensor_id not in self.in_use:
            return  # Not from our pool
        
        with self.lock:
            size = tuple(tensor.shape)
            dtype = tensor.dtype
            
            # Return to appropriate pool
            self.pools[size][dtype].append(tensor)
            
            # Remove from tracking
            self.in_use.discard(tensor_id)
            if tensor_id in self.tensor_refs:
                del self.tensor_refs[tensor_id]
    
    def _evict_unused_tensors(self, needed_bytes: int):
        """Evict unused tensors to free memory."""
        freed_bytes = 0
        
        # Iterate through pools and free least recently used tensors
        for size_dict in self.pools.values():
            for dtype_list in size_dict.values():
                while dtype_list and freed_bytes < needed_bytes:
                    tensor = dtype_list.pop(0)  # Remove oldest
                    tensor_size = tensor.numel() * tensor.element_size()
                    freed_bytes += tensor_size
                    self.current_size -= tensor_size
                    self.stats['evictions'] += 1
                    
                    if freed_bytes >= needed_bytes:
                        return
    
    def _emergency_cleanup(self):
        """Emergency tensor pool cleanup."""
        print("⚠️ TensorPool emergency cleanup triggered")
        
        with self.lock:
            # Clear all unused tensors
            total_freed = 0
            for size_dict in self.pools.values():
                for dtype_list in size_dict.values():
                    while dtype_list:
                        tensor = dtype_list.pop()
                        total_freed += tensor.numel() * tensor.element_size()
            
            self.current_size -= total_freed
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print(f"🧹 Freed {total_freed / (1024**2):.1f}MB from tensor pool")
    
    def get_stats(self) -> Dict:
        """Get tensor pool performance statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / max(1, total_requests)) * 100
            
            return {
                'hit_rate_pct': hit_rate,
                'total_requests': total_requests,
                'current_size_mb': self.current_size / (1024**2),
                'max_size_mb': self.max_size_bytes / (1024**2),
                'utilization_pct': (self.current_size / self.max_size_bytes) * 100,
                'evictions': self.stats['evictions'],
                'active_tensors': len(self.in_use)
            }


class IntelligentBatchCache:
    """
    Intelligent caching system for processed batches with compression and LRU eviction.
    Targets repeated data processing patterns in Mark Six training.
    """
    
    def __init__(self, max_size_gb: float = 8.0, enable_compression: bool = True):
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.enable_compression = enable_compression
        self.current_size = 0
        
        # Cache storage
        self.cache = OrderedDict()  # LRU ordering
        self.access_counts = defaultdict(int)
        self.timestamps = {}
        self.lock = threading.RLock()
        
        # Compression setup (optional dependency)
        if enable_compression:
            try:
                import lz4.frame
                self.compressor = lz4.frame
                self.compression_available = True
            except ImportError:
                print("⚠️ lz4 not available, disabling compression")
                self.compressor = None
                self.compression_available = False
        else:
            self.compressor = None
            self.compression_available = False
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'compressions': 0,
            'compression_ratio': 0.0,
            'evictions': 0
        }
    
    def _generate_key(self, data_description: str, parameters: Dict = None) -> str:
        """Generate cache key from data description and parameters."""
        import hashlib
        param_str = str(sorted(parameters.items())) if parameters else ""
        key_string = f"{data_description}_{param_str}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve item from cache with LRU update."""
        with self.lock:
            if key not in self.cache:
                self.stats['misses'] += 1
                return None
            
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            
            self.access_counts[key] += 1
            self.timestamps[key] = time.time()
            self.stats['hits'] += 1
            
            # Decompress if needed
            if self.compression_available and isinstance(value, bytes):
                try:
                    import pickle
                    decompressed = self.compressor.decompress(value)
                    return pickle.loads(decompressed)
                except Exception:
                    # Fallback to raw value
                    return value
            
            return value
    
    def put(self, key: str, value: Any, data_size_hint: int = None):
        """Store item in cache with compression and size management."""
        with self.lock:
            # Serialize and optionally compress
            serialized_value = value
            if self.compression_available:
                try:
                    import pickle
                    pickled = pickle.dumps(value)
                    compressed = self.compressor.compress(pickled)
                    
                    compression_ratio = len(compressed) / max(1, len(pickled))
                    self.stats['compression_ratio'] = (
                        self.stats['compression_ratio'] * 0.9 + compression_ratio * 0.1
                    )
                    self.stats['compressions'] += 1
                    
                    serialized_value = compressed
                except Exception:
                    # Compression failed, store raw
                    pass
            
            # Estimate size
            if data_size_hint:
                item_size = data_size_hint
            else:
                try:
                    if isinstance(serialized_value, (bytes, bytearray)):
                        item_size = len(serialized_value)
                    elif hasattr(serialized_value, 'nbytes'):
                        item_size = serialized_value.nbytes
                    else:
                        item_size = len(str(serialized_value)) * 2  # Rough estimate
                except:
                    item_size = 1024  # Default fallback
            
            # Evict if necessary
            if self.current_size + item_size > self.max_size_bytes:
                self._evict_lru_entries(item_size)
            
            # Store in cache
            self.cache[key] = serialized_value
            self.access_counts[key] = 1
            self.timestamps[key] = time.time()
            self.current_size += item_size
    
    def _evict_lru_entries(self, needed_space: int):
        """Evict least recently used entries."""
        freed_space = 0
        
        # Remove oldest entries first (start of OrderedDict)
        keys_to_remove = []
        for key in self.cache:
            if freed_space >= needed_space:
                break
            keys_to_remove.append(key)
            
            # Estimate freed space
            value = self.cache[key]
            if isinstance(value, (bytes, bytearray)):
                freed_space += len(value)
            elif hasattr(value, 'nbytes'):
                freed_space += value.nbytes
            else:
                freed_space += 1024  # Estimate
        
        # Actually remove the entries
        for key in keys_to_remove:
            del self.cache[key]
            if key in self.access_counts:
                del self.access_counts[key]
            if key in self.timestamps:
                del self.timestamps[key]
            self.stats['evictions'] += 1
        
        self.current_size -= freed_space
    
    def get_stats(self) -> Dict:
        """Get cache performance statistics."""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / max(1, total_requests)) * 100
            
            return {
                'hit_rate_pct': hit_rate,
                'total_requests': total_requests,
                'cache_entries': len(self.cache),
                'current_size_mb': self.current_size / (1024**2),
                'max_size_mb': self.max_size_bytes / (1024**2),
                'compression_ratio': self.stats['compression_ratio'],
                'evictions': self.stats['evictions']
            }


class MemoryPoolManager:
    """
    Unified memory pool management system for Mark Six AI.
    Coordinates tensor pools, batch caches, and system memory monitoring.
    """
    
    def __init__(self, config: MemoryPoolConfig = None):
        self.config = config or MemoryPoolConfig()
        
        # Initialize memory pools
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.tensor_pool = TensorPool(
            max_size_gb=self.config.tensor_pool_gb,
            device=device
        )
        
        self.batch_cache = IntelligentBatchCache(
            max_size_gb=self.config.batch_cache_gb,
            enable_compression=self.config.enable_compression
        )
        
        # System monitoring
        self.memory_monitor = threading.Thread(target=self._monitor_memory, daemon=True)
        self.monitoring_active = True
        self.memory_monitor.start()
        
        # Global statistics
        self.global_stats = {
            'startup_time': time.time(),
            'emergency_cleanups': 0,
            'memory_warnings': 0,
            'peak_usage_gb': 0.0
        }
        
        print(f"🚀 Memory Pool Manager initialized:")
        print(f"   • Tensor pool: {self.config.tensor_pool_gb}GB")
        print(f"   • Batch cache: {self.config.batch_cache_gb}GB")
        print(f"   • Compression: {self.config.enable_compression}")
        print(f"   • Memory monitoring: ✅ Active")
    
    def get_optimized_tensor(self, size: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from pool with optimized allocation."""
        return self.tensor_pool.get_tensor(size, dtype)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse."""
        self.tensor_pool.return_tensor(tensor)
    
    def cache_batch_data(self, key: str, data: Any, size_hint: int = None):
        """Cache processed batch data."""
        self.batch_cache.put(key, data, size_hint)
    
    def get_cached_batch_data(self, key: str) -> Optional[Any]:
        """Retrieve cached batch data."""
        return self.batch_cache.get(key)
    
    def _monitor_memory(self):
        """Background memory monitoring thread."""
        while self.monitoring_active:
            try:
                # System memory check
                memory = psutil.virtual_memory()
                usage_pct = memory.percent / 100.0
                
                # Update peak usage
                current_usage_gb = (memory.total - memory.available) / (1024**3)
                self.global_stats['peak_usage_gb'] = max(
                    self.global_stats['peak_usage_gb'], 
                    current_usage_gb
                )
                
                # Check thresholds
                if usage_pct > self.config.emergency_threshold:
                    self._emergency_memory_cleanup()
                elif usage_pct > self.config.warning_threshold:
                    self._memory_warning()
                
                # GPU memory check (if available)
                if torch.cuda.is_available():
                    try:
                        total_gpu, free_gpu = torch.cuda.mem_get_info()
                        gpu_usage_pct = 1.0 - (free_gpu / total_gpu)
                        
                        if gpu_usage_pct > self.config.emergency_threshold:
                            torch.cuda.empty_cache()
                    except Exception:
                        pass  # GPU monitoring is optional
                
                time.sleep(self.config.memory_check_interval)
                
            except Exception as e:
                print(f"⚠️ Memory monitoring error: {e}")
                time.sleep(30)  # Longer interval on error
    
    def _memory_warning(self):
        """Handle memory pressure warning."""
        self.global_stats['memory_warnings'] += 1
        
        if self.global_stats['memory_warnings'] % 10 == 1:  # Only print occasionally
            memory = psutil.virtual_memory()
            print(f"⚠️ Memory pressure warning: {memory.percent:.1f}% usage")
    
    def _emergency_memory_cleanup(self):
        """Emergency memory cleanup procedure."""
        self.global_stats['emergency_cleanups'] += 1
        print("🚨 Emergency memory cleanup initiated")
        
        # Cleanup tensor pool
        self.tensor_pool._emergency_cleanup()
        
        # Clear batch cache partially
        with self.batch_cache.lock:
            cache_size = len(self.batch_cache.cache)
            clear_count = cache_size // 3  # Clear 1/3 of cache
            
            keys_to_clear = list(self.batch_cache.cache.keys())[:clear_count]
            for key in keys_to_clear:
                del self.batch_cache.cache[key]
                if key in self.batch_cache.access_counts:
                    del self.batch_cache.access_counts[key]
                if key in self.batch_cache.timestamps:
                    del self.batch_cache.timestamps[key]
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("✅ Emergency cleanup completed")
    
    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive memory management statistics."""
        system_memory = psutil.virtual_memory()
        
        stats = {
            'system_memory_gb': system_memory.total / (1024**3),
            'available_memory_gb': system_memory.available / (1024**3),
            'memory_usage_pct': system_memory.percent,
            'peak_usage_gb': self.global_stats['peak_usage_gb'],
            'uptime_hours': (time.time() - self.global_stats['startup_time']) / 3600,
            'emergency_cleanups': self.global_stats['emergency_cleanups'],
            'memory_warnings': self.global_stats['memory_warnings'],
            'tensor_pool': self.tensor_pool.get_stats(),
            'batch_cache': self.batch_cache.get_stats()
        }
        
        # Add GPU stats if available
        if torch.cuda.is_available():
            try:
                total_gpu, free_gpu = torch.cuda.mem_get_info()
                stats['gpu_memory_total_gb'] = total_gpu / (1024**3)
                stats['gpu_memory_free_gb'] = free_gpu / (1024**3)
                stats['gpu_memory_usage_pct'] = ((total_gpu - free_gpu) / total_gpu) * 100
            except Exception:
                stats['gpu_memory_available'] = False
        
        return stats
    
    def optimize_for_batch_size(self, batch_size: int):
        """Optimize memory pools for specific batch size."""
        # Pre-allocate tensors for this batch size
        common_shapes = [
            (batch_size, 6),      # Number combinations
            (batch_size, 17),     # Feature vectors
            (batch_size, 10, 6),  # Temporal sequences
        ]
        
        for shape in common_shapes:
            for dtype in [torch.long, torch.float32]:
                # Pre-allocate a few tensors of each type
                for _ in range(2):
                    tensor = self.tensor_pool.get_tensor(shape, dtype)
                    self.tensor_pool.return_tensor(tensor)
    
    def cleanup(self):
        """Clean shutdown of memory manager."""
        print("🧹 Shutting down memory pool manager...")
        self.monitoring_active = False
        
        # Wait for monitor thread
        if self.memory_monitor.is_alive():
            self.memory_monitor.join(timeout=5)
        
        # Final cleanup
        self.tensor_pool._emergency_cleanup()
        
        final_stats = self.get_comprehensive_stats()
        print(f"📊 Final memory stats: {final_stats['memory_usage_pct']:.1f}% system usage")


# Global memory manager instance (initialized when needed)
_global_memory_manager: Optional[MemoryPoolManager] = None

def get_memory_manager(config: MemoryPoolConfig = None) -> MemoryPoolManager:
    """Get the global memory manager instance."""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        _global_memory_manager = MemoryPoolManager(config)
    
    return _global_memory_manager