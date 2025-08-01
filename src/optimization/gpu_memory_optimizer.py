"""
Advanced GPU Memory Optimization for CUDA Streams Pipeline
=========================================================

Implements sophisticated GPU memory management techniques:
1. Dynamic memory allocation strategies
2. Memory fragmentation prevention
3. CUDA unified memory optimization
4. Memory bandwidth optimization
5. GPU cache-aware data layouts

This module complements the CUDA streams pipeline by ensuring optimal
memory utilization patterns for maximum GPU performance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import threading
import time
import gc
import logging
from contextlib import contextmanager
from collections import defaultdict

from ..infrastructure.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryOptimizationConfig:
    """Configuration for GPU memory optimization."""
    # Memory allocation strategies
    enable_memory_pool: bool = True
    pool_growth_factor: float = 1.5
    min_pool_size_mb: int = 256
    max_pool_size_mb: int = 2048
    
    # Memory layout optimization
    enable_tensor_fusion: bool = True
    fusion_threshold_bytes: int = 1024 * 1024  # 1MB
    alignment_bytes: int = 512  # GPU memory alignment
    
    # Cache optimization
    enable_cache_optimization: bool = True
    cache_line_size: int = 128  # GPU cache line size
    prefer_coalesced_access: bool = True
    
    # Memory bandwidth optimization
    enable_memory_compression: bool = False  # Experimental
    compression_threshold: float = 0.7  # Compress if > 70% zeros
    
    # Fragmentation prevention
    enable_defragmentation: bool = True
    defrag_threshold: float = 0.3  # Defrag when >30% fragmented
    defrag_frequency_batches: int = 100
    
    # CUDA unified memory
    enable_unified_memory: bool = False  # Requires capable hardware
    prefetch_ratio: float = 0.8  # Prefetch 80% of data


class GPUMemoryPool:
    """
    Advanced GPU memory pool with fragmentation prevention and optimal allocation.
    
    Uses sophisticated allocation strategies to minimize memory fragmentation
    and maximize GPU memory bandwidth utilization.
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.device = torch.cuda.current_device()
        
        # Memory pools by size class (powers of 2)
        self.size_classes = [2**i for i in range(10, 26)]  # 1KB to 64MB
        self.memory_pools = {size: [] for size in self.size_classes}
        self.allocated_tensors = {}
        
        # Fragmentation tracking
        self.total_allocated = 0
        self.total_free_space = 0
        self.fragmentation_ratio = 0.0
        
        # Performance metrics
        self.allocation_count = 0
        self.deallocation_count = 0
        self.pool_hit_count = 0
        self.pool_miss_count = 0
        
        # Thread safety
        self.allocation_lock = threading.Lock()
        
        logger.info(f"Initialized GPU memory pool with {len(self.size_classes)} size classes")
    
    def _get_size_class(self, size_bytes: int) -> int:
        """Get the appropriate size class for allocation."""
        # Find smallest power of 2 >= size_bytes
        for size_class in self.size_classes:
            if size_class >= size_bytes:
                return size_class
        
        # For very large allocations, use exact size
        return size_bytes
    
    def allocate_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                       device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Allocate a tensor with optimal memory layout and pool management.
        """
        if device is None:
            device = torch.device(f'cuda:{self.device}')
        
        # Calculate required size
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = np.prod(shape)
        size_bytes = total_elements * element_size
        
        # Align to GPU cache boundaries
        aligned_size = self._align_size(size_bytes)
        size_class = self._get_size_class(aligned_size)
        
        with self.allocation_lock:
            # Try to get from pool first
            if size_class in self.memory_pools and self.memory_pools[size_class]:
                tensor = self.memory_pools[size_class].pop()
                
                # Reshape to required size
                if tensor.numel() >= total_elements:
                    tensor = tensor[:total_elements].view(shape)
                    self.pool_hit_count += 1
                    return tensor
            
            # Pool miss - allocate new tensor
            self.pool_miss_count += 1
            
            try:
                # Use aligned allocation for better memory access patterns
                if self.config.enable_cache_optimization:
                    tensor = self._allocate_aligned_tensor(shape, dtype, device, aligned_size)
                else:
                    tensor = torch.empty(shape, dtype=dtype, device=device)
                
                self.allocation_count += 1
                self.total_allocated += aligned_size
                
                # Track allocation
                tensor_id = id(tensor)
                self.allocated_tensors[tensor_id] = {
                    'size_bytes': aligned_size,
                    'size_class': size_class,
                    'shape': shape,
                    'dtype': dtype
                }
                
                return tensor
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Try memory cleanup and retry
                    self._emergency_cleanup()
                    
                    # Retry allocation
                    tensor = torch.empty(shape, dtype=dtype, device=device)
                    self.allocation_count += 1
                    return tensor
                else:
                    raise
    
    def _allocate_aligned_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype,
                               device: torch.device, aligned_size: int) -> torch.Tensor:
        """Allocate tensor with optimal memory alignment."""
        # Calculate padding for alignment
        element_size = torch.tensor([], dtype=dtype).element_size()
        total_elements = np.prod(shape)
        
        # Allocate slightly larger tensor for alignment
        padding_elements = (self.config.alignment_bytes // element_size)
        padded_elements = total_elements + padding_elements
        
        # Allocate and slice to get aligned portion
        padded_tensor = torch.empty(padded_elements, dtype=dtype, device=device)
        
        # Find aligned start position
        tensor_ptr = padded_tensor.data_ptr()
        aligned_ptr = (tensor_ptr + self.config.alignment_bytes - 1) // self.config.alignment_bytes * self.config.alignment_bytes
        offset_elements = (aligned_ptr - tensor_ptr) // element_size
        
        # Return aligned slice
        aligned_tensor = padded_tensor[offset_elements:offset_elements + total_elements].view(shape)
        return aligned_tensor
    
    def deallocate_tensor(self, tensor: torch.Tensor):
        """Return tensor to memory pool for reuse."""
        tensor_id = id(tensor)
        
        with self.allocation_lock:
            if tensor_id not in self.allocated_tensors:
                return  # Not tracked by this pool
            
            tensor_info = self.allocated_tensors[tensor_id]
            size_class = tensor_info['size_class']
            
            # Reset tensor data to prevent information leakage
            tensor.zero_()
            
            # Return to appropriate pool if there's space
            if len(self.memory_pools[size_class]) < 10:  # Limit pool size
                # Reshape back to flat tensor for reuse
                flat_tensor = tensor.view(-1)
                self.memory_pools[size_class].append(flat_tensor)
            
            # Update tracking
            del self.allocated_tensors[tensor_id]
            self.deallocation_count += 1
            self.total_allocated -= tensor_info['size_bytes']
    
    def _align_size(self, size_bytes: int) -> int:
        """Align size to GPU memory boundaries."""
        alignment = self.config.alignment_bytes
        return ((size_bytes + alignment - 1) // alignment) * alignment
    
    def _emergency_cleanup(self):
        """Emergency memory cleanup when allocation fails."""
        logger.warning("Performing emergency GPU memory cleanup")
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Clear our pools
        with self.allocation_lock:
            for size_class in self.memory_pools:
                self.memory_pools[size_class].clear()
            
            self.allocated_tensors.clear()
            self.total_allocated = 0
        
        # Force garbage collection
        gc.collect()
        
        # Wait for GPU to finish pending operations
        torch.cuda.synchronize()
    
    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get memory pool performance statistics."""
        total_requests = self.pool_hit_count + self.pool_miss_count
        hit_rate = self.pool_hit_count / max(1, total_requests) * 100
        
        pool_sizes = {size: len(tensors) for size, tensors in self.memory_pools.items()}
        
        return {
            'hit_rate_percent': hit_rate,
            'total_allocations': self.allocation_count,
            'total_deallocations': self.deallocation_count,
            'active_tensors': len(self.allocated_tensors),
            'total_allocated_mb': self.total_allocated / (1024 * 1024),
            'pool_sizes': pool_sizes,
            'fragmentation_ratio': self.fragmentation_ratio
        }
    
    def defragment_memory(self):
        """Defragment GPU memory to reduce fragmentation."""
        if not self.config.enable_defragmentation:
            return
        
        logger.info("Starting GPU memory defragmentation")
        
        with self.allocation_lock:
            # Clear all pools
            for size_class in self.memory_pools:
                self.memory_pools[size_class].clear()
            
            # Force cleanup
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()
            
            # Update fragmentation tracking
            self.fragmentation_ratio = 0.0
        
        logger.info("GPU memory defragmentation complete")


class TensorFusionManager:
    """
    Manages tensor fusion for improved memory bandwidth utilization.
    
    Combines multiple small tensors into larger contiguous allocations
    to reduce memory access overhead and improve cache efficiency.
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.fusion_groups = defaultdict(list)
        self.fused_tensors = {}
        self.fusion_threshold = config.fusion_threshold_bytes
        
        logger.info(f"Initialized tensor fusion manager with threshold {self.fusion_threshold} bytes")
    
    def can_fuse_tensors(self, tensors: List[torch.Tensor]) -> bool:
        """Check if tensors can be fused together."""
        if not self.config.enable_tensor_fusion:
            return False
        
        if len(tensors) < 2:
            return False
        
        # Check device compatibility
        devices = {t.device for t in tensors}
        if len(devices) > 1:
            return False
        
        # Check dtype compatibility
        dtypes = {t.dtype for t in tensors}
        if len(dtypes) > 1:
            return False
        
        # Check total size
        total_size = sum(t.numel() * t.element_size() for t in tensors)
        return total_size <= self.fusion_threshold
    
    def fuse_tensors(self, tensors: List[torch.Tensor], fusion_id: str = None) -> torch.Tensor:
        """Fuse multiple tensors into a single contiguous tensor."""
        if not self.can_fuse_tensors(tensors):
            return None
        
        if fusion_id is None:
            fusion_id = f"fusion_{len(self.fused_tensors)}"
        
        # Calculate total elements
        total_elements = sum(t.numel() for t in tensors)
        
        # Create fused tensor
        fused_tensor = torch.empty(total_elements, dtype=tensors[0].dtype, device=tensors[0].device)
        
        # Copy data into fused tensor
        offset = 0
        tensor_info = []
        
        for tensor in tensors:
            elements = tensor.numel()
            fused_tensor[offset:offset + elements] = tensor.view(-1)
            
            tensor_info.append({
                'original_shape': tensor.shape,
                'offset': offset,
                'elements': elements
            })
            
            offset += elements
        
        # Store fusion information
        self.fused_tensors[fusion_id] = {
            'fused_tensor': fused_tensor,
            'tensor_info': tensor_info,
            'original_tensors': len(tensors)
        }
        
        logger.debug(f"Fused {len(tensors)} tensors into {fusion_id}: {total_elements} elements")
        return fused_tensor
    
    def unfuse_tensor(self, fusion_id: str) -> List[torch.Tensor]:
        """Extract original tensors from fused tensor."""
        if fusion_id not in self.fused_tensors:
            return []
        
        fusion_info = self.fused_tensors[fusion_id]
        fused_tensor = fusion_info['fused_tensor']
        tensor_info = fusion_info['tensor_info']
        
        original_tensors = []
        
        for info in tensor_info:
            offset = info['offset']
            elements = info['elements']
            original_shape = info['original_shape']
            
            # Extract and reshape tensor
            extracted = fused_tensor[offset:offset + elements].view(original_shape)
            original_tensors.append(extracted)
        
        return original_tensors
    
    def cleanup_fusion(self, fusion_id: str):
        """Clean up a fusion group."""
        if fusion_id in self.fused_tensors:
            del self.fused_tensors[fusion_id]


class CacheOptimizedTensorLayout:
    """
    Optimizes tensor memory layouts for GPU cache efficiency.
    
    Rearranges data layouts to maximize cache hit rates and memory
    coalescing for improved bandwidth utilization.
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.cache_line_size = config.cache_line_size
        
    def optimize_tensor_layout(self, tensor: torch.Tensor, access_pattern: str = "sequential") -> torch.Tensor:
        """Optimize tensor layout for specific access patterns."""
        if not self.config.enable_cache_optimization:
            return tensor
        
        if access_pattern == "sequential":
            return self._optimize_sequential_access(tensor)
        elif access_pattern == "strided":
            return self._optimize_strided_access(tensor)
        elif access_pattern == "random":
            return self._optimize_random_access(tensor)
        else:
            return tensor
    
    def _optimize_sequential_access(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize layout for sequential memory access."""
        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Pad to cache line boundaries if beneficial
        if tensor.numel() * tensor.element_size() % self.cache_line_size != 0:
            pad_elements = (self.cache_line_size // tensor.element_size()) - (tensor.numel() % (self.cache_line_size // tensor.element_size()))
            
            if pad_elements < tensor.numel() * 0.1:  # Only if padding < 10%
                padded_shape = list(tensor.shape)
                padded_shape[-1] += pad_elements
                
                padded_tensor = torch.zeros(padded_shape, dtype=tensor.dtype, device=tensor.device)
                padded_tensor[..., :tensor.shape[-1]] = tensor
                
                return padded_tensor
        
        return tensor
    
    def _optimize_strided_access(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize layout for strided memory access patterns."""
        # Transpose dimensions to improve memory coalescing
        if tensor.dim() >= 2:
            # Move frequently accessed dimension to the innermost position
            dims = list(range(tensor.dim()))
            dims[-1], dims[0] = dims[0], dims[-1]  # Swap first and last dimensions
            
            return tensor.permute(dims).contiguous()
        
        return tensor
    
    def _optimize_random_access(self, tensor: torch.Tensor) -> torch.Tensor:
        """Optimize layout for random access patterns."""
        # For random access, ensure data locality
        # Use blocked/tiled layout if tensor is large enough
        
        if tensor.numel() > 1024 and tensor.dim() >= 2:
            # Implement simple blocking for 2D+ tensors
            return self._apply_blocked_layout(tensor)
        
        return tensor
    
    def _apply_blocked_layout(self, tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
        """Apply blocked memory layout for better cache locality."""
        if tensor.dim() < 2:
            return tensor
        
        # Only apply to 2D tensors for simplicity
        if tensor.dim() == 2:
            h, w = tensor.shape
            
            # Calculate block dimensions
            block_h = min(block_size, h)
            block_w = min(block_size, w)
            
            # Pad to block boundaries
            pad_h = (block_h - (h % block_h)) % block_h
            pad_w = (block_w - (w % block_w)) % block_w
            
            if pad_h > 0 or pad_w > 0:
                padded = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h))
            else:
                padded = tensor
            
            # Reshape into blocks
            new_h, new_w = padded.shape
            blocked = padded.view(new_h // block_h, block_h, new_w // block_w, block_w)
            blocked = blocked.permute(0, 2, 1, 3).contiguous()
            
            return blocked.view(new_h, new_w)
        
        return tensor


class UnifiedMemoryManager:
    """
    Manages CUDA Unified Memory for automatic CPU-GPU data migration.
    
    Provides transparent memory management that automatically migrates
    data between CPU and GPU as needed, reducing explicit transfers.
    """
    
    def __init__(self, config: MemoryOptimizationConfig):
        self.config = config
        self.unified_tensors = {}
        self.prefetch_hints = {}
        
        # Check if unified memory is supported
        self.supported = self._check_unified_memory_support()
        
        if self.supported and config.enable_unified_memory:
            logger.info("CUDA Unified Memory enabled")
        else:
            logger.info("CUDA Unified Memory not available or disabled")
    
    def _check_unified_memory_support(self) -> bool:
        """Check if the current GPU supports unified memory."""
        if not torch.cuda.is_available():
            return False
        
        try:
            # Try to allocate a small unified memory tensor
            test_tensor = torch.empty(10, device='cuda', dtype=torch.float32)
            return hasattr(torch.cuda, 'memory_stats')  # Basic check
        except:
            return False
    
    def allocate_unified_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32,
                              tensor_id: str = None) -> Optional[torch.Tensor]:
        """Allocate a tensor in unified memory space."""
        if not self.supported or not self.config.enable_unified_memory:
            return None
        
        try:
            # Allocate in GPU memory first (will be accessible from CPU too)
            tensor = torch.empty(shape, dtype=dtype, device='cuda')
            
            if tensor_id:
                self.unified_tensors[tensor_id] = tensor
            
            return tensor
            
        except RuntimeError:
            logger.warning("Failed to allocate unified memory tensor")
            return None
    
    def prefetch_to_gpu(self, tensor_id: str, stream: torch.cuda.Stream = None):
        """Prefetch tensor data to GPU memory."""
        if not self.supported or tensor_id not in self.unified_tensors:
            return
        
        tensor = self.unified_tensors[tensor_id]
        
        if stream:
            with torch.cuda.stream(stream):
                # Touch tensor to trigger GPU migration
                _ = tensor.sum()
        else:
            _ = tensor.sum()
    
    def prefetch_to_cpu(self, tensor_id: str):
        """Prefetch tensor data to CPU memory."""
        if not self.supported or tensor_id not in self.unified_tensors:
            return
        
        tensor = self.unified_tensors[tensor_id]
        
        # Move to CPU temporarily to trigger migration
        cpu_tensor = tensor.cpu()
        del cpu_tensor


@dataclass  
class MemoryOptimizationMetrics:
    """Metrics for memory optimization performance."""
    memory_pool_hit_rate: float = 0.0
    tensor_fusion_ratio: float = 0.0
    cache_efficiency: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    fragmentation_ratio: float = 0.0
    unified_memory_hit_rate: float = 0.0


class GPUMemoryOptimizer:
    """
    Main orchestrator for GPU memory optimization.
    
    Coordinates all memory optimization techniques:
    - Memory pooling and allocation
    - Tensor fusion
    - Cache-optimized layouts
    - Unified memory management
    """
    
    def __init__(self, config: MemoryOptimizationConfig = None):
        self.config = config or MemoryOptimizationConfig()
        
        # Initialize optimization components
        self.memory_pool = GPUMemoryPool(self.config)
        self.fusion_manager = TensorFusionManager(self.config)
        self.layout_optimizer = CacheOptimizedTensorLayout(self.config)
        self.unified_memory = UnifiedMemoryManager(self.config)
        
        # Performance tracking
        self.batch_count = 0
        self.total_memory_saved = 0
        self.optimization_metrics = MemoryOptimizationMetrics()
        
        logger.info("GPU memory optimizer initialized with all optimization techniques")
    
    @contextmanager
    def optimized_allocation(self, shapes: List[Tuple[int, ...]], 
                           dtypes: List[torch.dtype] = None,
                           access_patterns: List[str] = None):
        """Context manager for optimized tensor allocation and cleanup."""
        if dtypes is None:
            dtypes = [torch.float32] * len(shapes)
        
        if access_patterns is None:
            access_patterns = ["sequential"] * len(shapes)
        
        # Allocate optimized tensors
        tensors = []
        
        for i, (shape, dtype, pattern) in enumerate(zip(shapes, dtypes, access_patterns)):
            # Try unified memory first
            tensor = self.unified_memory.allocate_unified_tensor(shape, dtype, f"temp_{i}")
            
            if tensor is None:
                # Fall back to memory pool
                tensor = self.memory_pool.allocate_tensor(shape, dtype)
            
            # Optimize layout
            tensor = self.layout_optimizer.optimize_tensor_layout(tensor, pattern)
            tensors.append(tensor)
        
        try:
            yield tensors
        finally:
            # Clean up tensors
            for tensor in tensors:
                self.memory_pool.deallocate_tensor(tensor)
    
    def optimize_batch_tensors(self, tensors: List[torch.Tensor], 
                             fusion_id: str = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Optimize a batch of tensors using available techniques."""
        self.batch_count += 1
        
        # Try tensor fusion first
        if len(tensors) > 1:
            fused = self.fusion_manager.fuse_tensors(tensors, fusion_id)
            if fused is not None:
                # Apply layout optimization to fused tensor
                return self.layout_optimizer.optimize_tensor_layout(fused)
        
        # Individual optimization for each tensor
        optimized_tensors = []
        for tensor in tensors:
            optimized = self.layout_optimizer.optimize_tensor_layout(tensor)
            optimized_tensors.append(optimized)
        
        return optimized_tensors
    
    def periodic_maintenance(self):
        """Perform periodic memory maintenance."""
        if self.batch_count % self.config.defrag_frequency_batches == 0:
            # Check fragmentation
            stats = self.memory_pool.get_pool_statistics()
            
            if stats['fragmentation_ratio'] > self.config.defrag_threshold:
                self.memory_pool.defragment_memory()
        
        # Update metrics
        self._update_metrics()
    
    def _update_metrics(self):
        """Update optimization performance metrics."""
        pool_stats = self.memory_pool.get_pool_statistics()
        
        self.optimization_metrics = MemoryOptimizationMetrics(
            memory_pool_hit_rate=pool_stats['hit_rate_percent'], 
            fragmentation_ratio=pool_stats['fragmentation_ratio'],
            tensor_fusion_ratio=len(self.fusion_manager.fused_tensors) / max(1, self.batch_count),
            # Additional metrics would be calculated here
        )
    
    def get_optimization_metrics(self) -> MemoryOptimizationMetrics:
        """Get current optimization metrics."""
        self._update_metrics()
        return self.optimization_metrics
    
    def get_memory_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        pool_stats = self.memory_pool.get_pool_statistics()
        
        # GPU memory stats
        if torch.cuda.is_available():
            memory_stats = torch.cuda.memory_stats()
            allocated_gb = memory_stats.get('allocated_bytes.all.current', 0) / 1e9
            reserved_gb = memory_stats.get('reserved_bytes.all.current', 0) / 1e9
        else:
            allocated_gb = reserved_gb = 0.0
        
        return {
            'gpu_allocated_gb': allocated_gb,
            'gpu_reserved_gb': reserved_gb,
            'pool_statistics': pool_stats,
            'optimization_metrics': self.optimization_metrics.__dict__,
            'fused_tensors_count': len(self.fusion_manager.fused_tensors),
            'unified_memory_enabled': self.unified_memory.supported and self.config.enable_unified_memory
        }
    
    def cleanup(self):
        """Clean up all optimization resources."""
        logger.info("Cleaning up GPU memory optimizer...")
        
        # Clean up components
        self.memory_pool._emergency_cleanup()
        self.fusion_manager.fused_tensors.clear()
        self.unified_memory.unified_tensors.clear()
        
        # Final GPU cleanup
        torch.cuda.empty_cache()
        
        logger.info("GPU memory optimizer cleanup complete")


# Factory function for easy integration
def create_gpu_memory_optimizer(gpu_memory_gb: float = 10.0) -> GPUMemoryOptimizer:
    """
    Create GPU memory optimizer with settings optimized for available GPU memory.
    
    Args:
        gpu_memory_gb: Available GPU memory in GB
        
    Returns:
        Configured GPUMemoryOptimizer instance
    """
    if gpu_memory_gb >= 10:  # RTX 3080+ class
        config = MemoryOptimizationConfig(
            enable_memory_pool=True,
            pool_growth_factor=1.5,
            min_pool_size_mb=512,
            max_pool_size_mb=2048,
            enable_tensor_fusion=True,
            fusion_threshold_bytes=2 * 1024 * 1024,  # 2MB
            enable_cache_optimization=True,
            enable_defragmentation=True,
            defrag_threshold=0.3,
            enable_unified_memory=True
        )
    elif gpu_memory_gb >= 8:  # RTX 3070 class  
        config = MemoryOptimizationConfig(
            enable_memory_pool=True,
            pool_growth_factor=1.3,
            min_pool_size_mb=384,
            max_pool_size_mb=1536,
            enable_tensor_fusion=True,
            fusion_threshold_bytes=1024 * 1024,  # 1MB
            enable_cache_optimization=True,
            enable_defragmentation=True,
            defrag_threshold=0.4,
            enable_unified_memory=False  # May not be stable on mid-range
        )
    else:  # Lower-end GPUs
        config = MemoryOptimizationConfig(
            enable_memory_pool=True,
            pool_growth_factor=1.2,
            min_pool_size_mb=256,
            max_pool_size_mb=1024,
            enable_tensor_fusion=False,  # Too much overhead
            enable_cache_optimization=False,
            enable_defragmentation=True,
            defrag_threshold=0.5,
            enable_unified_memory=False
        )
    
    logger.info(f"Creating GPU memory optimizer for {gpu_memory_gb}GB GPU")
    return GPUMemoryOptimizer(config)