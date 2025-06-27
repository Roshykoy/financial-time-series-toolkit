"""
Performance monitoring and optimization utilities for MarkSix forecasting system.
Provides real-time monitoring, bottleneck detection, and optimization suggestions.
"""
import time
import psutil
import gc
import warnings
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from collections import deque, defaultdict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    peak_memory_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def is_complete(self) -> bool:
        return self.end_time is not None


class PerformanceMonitor:
    """Real-time performance monitoring with bottleneck detection."""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.bottlenecks: List[Dict[str, Any]] = []
        self.monitoring_enabled = True
        self.monitoring_thread = None
        self._stop_monitoring = threading.Event()
        
    def start_monitoring(self, interval: float = 1.0):
        """Start background monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)
    
    def _monitor_system(self, interval: float):
        """Background system monitoring."""
        while not self._stop_monitoring.wait(interval):
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # GPU metrics
                gpu_memory = 0.0
                gpu_util = 0.0
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    try:
                        gpu_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
                        # GPU utilization requires nvidia-ml-py, skip if not available
                    except Exception:
                        pass
                
                # Check for bottlenecks
                self._detect_bottlenecks(cpu_percent, memory.percent, gpu_memory)
                
            except Exception as e:
                warnings.warn(f"Monitoring error: {e}", RuntimeWarning)
    
    def _detect_bottlenecks(self, cpu_percent: float, memory_percent: float, gpu_memory_mb: float):
        """Detect performance bottlenecks."""
        current_time = time.time()
        
        # CPU bottleneck
        if cpu_percent > 90:
            self._add_bottleneck("high_cpu", cpu_percent, current_time)
        
        # Memory bottleneck
        if memory_percent > 85:
            self._add_bottleneck("high_memory", memory_percent, current_time)
        
        # GPU memory bottleneck
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                gpu_usage_percent = (gpu_memory_mb / total_gpu_memory) * 100
                if gpu_usage_percent > 90:
                    self._add_bottleneck("high_gpu_memory", gpu_usage_percent, current_time)
            except Exception:
                pass
    
    def _add_bottleneck(self, bottleneck_type: str, value: float, timestamp: float):
        """Add detected bottleneck."""
        # Avoid duplicate recent bottlenecks
        recent_threshold = 10.0  # seconds
        for bottleneck in self.bottlenecks[-5:]:  # Check last 5
            if (bottleneck['type'] == bottleneck_type and 
                timestamp - bottleneck['timestamp'] < recent_threshold):
                return
        
        self.bottlenecks.append({
            'type': bottleneck_type,
            'value': value,
            'timestamp': timestamp,
            'suggestion': self._get_bottleneck_suggestion(bottleneck_type, value)
        })
    
    def _get_bottleneck_suggestion(self, bottleneck_type: str, value: float) -> str:
        """Get optimization suggestion for bottleneck."""
        suggestions = {
            'high_cpu': f"CPU usage at {value:.1f}%. Consider reducing batch size or using GPU.",
            'high_memory': f"Memory usage at {value:.1f}%. Consider reducing model size or clearing cache.",
            'high_gpu_memory': f"GPU memory at {value:.1f}%. Consider reducing batch size or using gradient checkpointing."
        }
        return suggestions.get(bottleneck_type, "Performance issue detected.")
    
    @contextmanager
    def monitor_operation(self, operation_name: str, **metadata):
        """Context manager for monitoring specific operations."""
        if not self.monitoring_enabled:
            yield
            return
        
        # Start monitoring
        start_time = time.time()
        start_memory = psutil.virtual_memory().used / (1024**2)  # MB
        start_gpu_memory = 0.0
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                start_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
            except Exception:
                pass
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            memory_mb=start_memory,
            gpu_memory_mb=start_gpu_memory,
            metadata=metadata
        )
        
        self.active_operations[operation_name] = metrics
        
        try:
            yield metrics
        finally:
            # End monitoring
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024**2)
            end_gpu_memory = 0.0
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    end_gpu_memory = torch.cuda.memory_allocated() / (1024**2)
                except Exception:
                    pass
            
            metrics.end_time = end_time
            metrics.cpu_percent = psutil.cpu_percent()
            metrics.peak_memory_mb = max(metrics.memory_mb, end_memory)
            
            # Calculate memory delta
            metrics.metadata.update({
                'memory_delta_mb': end_memory - start_memory,
                'gpu_memory_delta_mb': end_gpu_memory - start_gpu_memory
            })
            
            self.metrics_history.append(metrics)
            self.active_operations.pop(operation_name, None)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_history:
            return {"message": "No performance data available"}
        
        # Calculate statistics
        durations = [m.duration for m in self.metrics_history if m.duration]
        memory_usage = [m.peak_memory_mb for m in self.metrics_history]
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
        else:
            avg_duration = max_duration = min_duration = 0
        
        if memory_usage:
            avg_memory = sum(memory_usage) / len(memory_usage)
            peak_memory = max(memory_usage)
        else:
            avg_memory = peak_memory = 0
        
        # Get operation breakdown
        operation_stats = defaultdict(list)
        for metrics in self.metrics_history:
            if metrics.duration:
                operation_stats[metrics.operation_name].append(metrics.duration)
        
        operation_summary = {}
        for op_name, durations in operation_stats.items():
            operation_summary[op_name] = {
                'count': len(durations),
                'avg_duration': sum(durations) / len(durations),
                'total_duration': sum(durations)
            }
        
        return {
            'total_operations': len(self.metrics_history),
            'avg_duration_seconds': avg_duration,
            'max_duration_seconds': max_duration,
            'min_duration_seconds': min_duration,
            'avg_memory_mb': avg_memory,
            'peak_memory_mb': peak_memory,
            'operation_breakdown': operation_summary,
            'bottlenecks_detected': len(self.bottlenecks),
            'recent_bottlenecks': self.bottlenecks[-5:] if self.bottlenecks else []
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get performance optimization suggestions."""
        suggestions = []
        summary = self.get_performance_summary()
        
        # Duration-based suggestions
        if summary.get('avg_duration_seconds', 0) > 10:
            suggestions.append("⚡ Consider reducing model complexity or batch size for faster operations")
        
        # Memory-based suggestions
        if summary.get('peak_memory_mb', 0) > 8000:  # 8GB
            suggestions.append("💾 High memory usage detected. Consider gradient checkpointing or smaller batch sizes")
        
        # Bottleneck-based suggestions
        bottleneck_types = {b['type'] for b in self.bottlenecks[-10:]}
        if 'high_cpu' in bottleneck_types:
            suggestions.append("🖥️ CPU bottleneck detected. Consider using GPU or reducing computational complexity")
        if 'high_memory' in bottleneck_types:
            suggestions.append("🧠 Memory bottleneck detected. Consider clearing caches or reducing model size")
        if 'high_gpu_memory' in bottleneck_types:
            suggestions.append("🎮 GPU memory bottleneck. Consider mixed precision or gradient accumulation")
        
        # Operation-specific suggestions
        op_breakdown = summary.get('operation_breakdown', {})
        slowest_ops = sorted(op_breakdown.items(), key=lambda x: x[1]['avg_duration'], reverse=True)[:3]
        
        for op_name, stats in slowest_ops:
            if stats['avg_duration'] > 5:
                suggestions.append(f"🐌 Operation '{op_name}' is slow (avg: {stats['avg_duration']:.1f}s). Consider optimization")
        
        if not suggestions:
            suggestions.append("✅ No significant performance issues detected")
        
        return suggestions


class MemoryManager:
    """Advanced memory management utilities."""
    
    @staticmethod
    def clear_all_caches():
        """Clear all possible caches."""
        # Python garbage collection
        gc.collect()
        
        # PyTorch caches
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Additional cleanup
        import sys
        if hasattr(sys, '_clear_type_cache'):
            sys._clear_type_cache()
    
    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get detailed memory usage information."""
        memory_info = {}
        
        # System memory
        memory = psutil.virtual_memory()
        memory_info.update({
            'system_total_gb': memory.total / (1024**3),
            'system_used_gb': memory.used / (1024**3),
            'system_available_gb': memory.available / (1024**3),
            'system_percent': memory.percent
        })
        
        # Process memory
        process = psutil.Process()
        process_memory = process.memory_info()
        memory_info.update({
            'process_rss_mb': process_memory.rss / (1024**2),
            'process_vms_mb': process_memory.vms / (1024**2)
        })
        
        # GPU memory
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.memory_stats()
                total_memory = torch.cuda.get_device_properties(0).total_memory
                
                memory_info.update({
                    'gpu_allocated_mb': gpu_memory.get('allocated_bytes.all.current', 0) / (1024**2),
                    'gpu_reserved_mb': gpu_memory.get('reserved_bytes.all.current', 0) / (1024**2),
                    'gpu_total_mb': total_memory / (1024**2),
                    'gpu_utilization_percent': (gpu_memory.get('allocated_bytes.all.current', 0) / total_memory) * 100
                })
            except Exception:
                memory_info.update({
                    'gpu_allocated_mb': 0,
                    'gpu_reserved_mb': 0,
                    'gpu_total_mb': 0,
                    'gpu_utilization_percent': 0
                })
        
        return memory_info
    
    @staticmethod
    def optimize_memory():
        """Perform comprehensive memory optimization."""
        # Clear caches
        MemoryManager.clear_all_caches()
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # PyTorch specific optimizations
        if TORCH_AVAILABLE:
            # Set memory fraction if on GPU
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use max 80% of GPU memory
            
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except AttributeError:
                pass
    
    @staticmethod
    def monitor_memory_usage(threshold_mb: float = 1000.0) -> Callable:
        """Decorator to monitor memory usage of functions."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Pre-execution memory
                start_memory = MemoryManager.get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    
                    # Post-execution memory
                    end_memory = MemoryManager.get_memory_usage()
                    
                    # Calculate memory delta
                    memory_delta = end_memory['process_rss_mb'] - start_memory['process_rss_mb']
                    
                    if memory_delta > threshold_mb:
                        warnings.warn(
                            f"Function {func.__name__} used {memory_delta:.1f} MB memory "
                            f"(threshold: {threshold_mb} MB)",
                            ResourceWarning
                        )
                    
                    return result
                    
                except Exception as e:
                    # Clear memory on exception
                    MemoryManager.clear_all_caches()
                    raise
            
            return wrapper
        return decorator


class PerformanceOptimizer:
    """Automatic performance optimization."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_history = []
    
    def auto_optimize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically optimize configuration based on system capabilities."""
        optimized_config = config.copy()
        memory_info = MemoryManager.get_memory_usage()
        
        # Optimize based on available memory
        if memory_info['system_available_gb'] < 4:
            # Low memory system
            optimizations = {
                'batch_size': min(config.get('batch_size', 8), 4),
                'latent_dim': min(config.get('latent_dim', 64), 32),
                'graph_hidden_dim': min(config.get('graph_hidden_dim', 64), 32),
                'temporal_hidden_dim': min(config.get('temporal_hidden_dim', 64), 32),
            }
            self._apply_optimizations(optimized_config, optimizations, "Low memory")
            
        elif memory_info['system_available_gb'] > 16:
            # High memory system - can use larger batches
            optimizations = {
                'batch_size': min(config.get('batch_size', 8) * 2, 32),
            }
            self._apply_optimizations(optimized_config, optimizations, "High memory")
        
        # GPU optimizations
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_memory_gb = memory_info.get('gpu_total_mb', 0) / 1024
            
            if gpu_memory_gb < 6:  # Low-end GPU
                optimizations = {
                    'batch_size': min(config.get('batch_size', 8), 4),
                    'use_mixed_precision': False,  # Disable for stability
                }
                self._apply_optimizations(optimized_config, optimizations, "Low-end GPU")
                
            elif gpu_memory_gb > 10:  # High-end GPU
                optimizations = {
                    'batch_size': min(config.get('batch_size', 8) * 2, 16),
                    'use_mixed_precision': True,
                }
                self._apply_optimizations(optimized_config, optimizations, "High-end GPU")
        
        # CPU-only optimizations
        else:
            optimizations = {
                'batch_size': min(config.get('batch_size', 8), 4),
                'epochs': max(config.get('epochs', 10) // 2, 5),  # Fewer epochs on CPU
                'num_gat_layers': min(config.get('num_gat_layers', 2), 1),
            }
            self._apply_optimizations(optimized_config, optimizations, "CPU-only")
        
        return optimized_config
    
    def _apply_optimizations(self, config: Dict[str, Any], optimizations: Dict[str, Any], reason: str):
        """Apply optimizations to configuration."""
        changes = []
        for key, value in optimizations.items():
            if key in config and config[key] != value:
                old_value = config[key]
                config[key] = value
                changes.append(f"{key}: {old_value} → {value}")
        
        if changes:
            self.optimization_history.append({
                'reason': reason,
                'changes': changes,
                'timestamp': time.time()
            })
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization history and suggestions."""
        return {
            'optimizations_applied': len(self.optimization_history),
            'optimization_history': self.optimization_history,
            'current_suggestions': self.monitor.get_optimization_suggestions(),
            'performance_summary': self.monitor.get_performance_summary()
        }


# Global performance monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
        _global_monitor.start_monitoring()
    return _global_monitor


def performance_monitor(operation_name: str = None, **metadata):
    """Decorator for monitoring function performance."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        def wrapper(*args, **kwargs):
            monitor = get_performance_monitor()
            with monitor.monitor_operation(operation_name, **metadata):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


def optimize_for_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize configuration for current system."""
    monitor = get_performance_monitor()
    optimizer = PerformanceOptimizer(monitor)
    return optimizer.auto_optimize_config(config)