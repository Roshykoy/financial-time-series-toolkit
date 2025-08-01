"""
Production-Ready CPU-GPU Hybrid Optimization
============================================

Based on expert panel review, this module implements a simplified, stable approach to:
1. GPU memory optimization with proper resource management
2. CPU-GPU coordination for training pipeline
3. Hardware-aware configuration optimization

Focus: Stability, maintainability, and measurable performance improvements
Architecture: Keep it simple, avoid over-engineering
"""

import torch
import torch.nn as nn
import numpy as np
import threading
import time
import logging
import gc
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import psutil

from .hardware_manager import HardwareResourceManager
from ..infrastructure.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationConfig:
    """Simple configuration for production optimization."""
    enable_gpu_optimization: bool = True
    enable_cpu_gpu_overlap: bool = True
    memory_safety_margin: float = 0.8
    max_cpu_workers: int = 4
    enable_monitoring: bool = True


class SimpleGPUOptimizer:
    """
    Simple, reliable GPU memory optimization.
    
    Based on expert review: Focus on basic memory pooling only,
    remove experimental features that add complexity.
    """
    
    def __init__(self, device: torch.device, config: OptimizationConfig):
        self.device = device
        self.config = config
        self.memory_pool = {}
        self.allocation_stats = {'hits': 0, 'misses': 0, 'total_allocated': 0}
        
        # Initialize GPU memory monitoring
        if torch.cuda.is_available() and device.type == 'cuda':
            self.gpu_memory_total = torch.cuda.get_device_properties(device).total_memory
            self.memory_limit = int(self.gpu_memory_total * config.memory_safety_margin)
        else:
            self.gpu_memory_total = 0
            self.memory_limit = 0
            
        logger.info(f"SimpleGPUOptimizer initialized: {self.memory_limit // (1024**3)}GB limit")
    
    def get_tensor(self, shape: Tuple[int, ...], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get tensor from memory pool or allocate new one."""
        if not self.config.enable_gpu_optimization or self.device.type != 'cuda':
            return torch.empty(shape, dtype=dtype, device=self.device)
        
        size_key = (shape, dtype)
        
        # Try to reuse from pool
        if size_key in self.memory_pool and self.memory_pool[size_key]:
            tensor = self.memory_pool[size_key].pop()
            tensor.zero_()  # Clear data
            self.allocation_stats['hits'] += 1
            return tensor
        
        # Allocate new tensor
        try:
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            self.allocation_stats['misses'] += 1
            self.allocation_stats['total_allocated'] += tensor.numel()
            return tensor
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Emergency cleanup and retry
                self.emergency_cleanup()
                tensor = torch.empty(shape, dtype=dtype, device=self.device)
                return tensor
            raise
    
    def return_tensor(self, tensor: torch.Tensor) -> None:
        """Return tensor to memory pool."""
        if not self.config.enable_gpu_optimization or tensor.device.type != 'cuda':
            return
        
        size_key = (tuple(tensor.shape), tensor.dtype)
        
        if size_key not in self.memory_pool:
            self.memory_pool[size_key] = []
        
        # Limit pool size to prevent memory bloat
        if len(self.memory_pool[size_key]) < 5:
            self.memory_pool[size_key].append(tensor)
    
    def emergency_cleanup(self) -> None:
        """Emergency memory cleanup."""
        logger.warning("GPU memory pressure detected, performing emergency cleanup")
        
        # Clear memory pool
        self.memory_pool.clear()
        
        # Force garbage collection
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.info("Emergency cleanup completed")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        if not torch.cuda.is_available():
            return {'gpu_available': False}
        
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        
        return {
            'gpu_available': True,
            'allocated_gb': allocated / (1024**3),
            'reserved_gb': reserved / (1024**3),
            'pool_stats': self.allocation_stats,
            'pool_sizes': {str(k): len(v) for k, v in self.memory_pool.items()}
        }


class CPUGPUCoordinator:
    """
    Simple CPU-GPU coordination for training pipeline.
    
    Based on expert review: Focus on proven async data loading patterns,
    avoid complex multiprocessing that introduces instability.
    """
    
    def __init__(self, config: OptimizationConfig, device: torch.device):
        self.config = config
        self.device = device
        self.gpu_optimizer = SimpleGPUOptimizer(device, config)
        
        # CPU preprocessing thread pool
        if config.enable_cpu_gpu_overlap:
            self.cpu_executor = ThreadPoolExecutor(
                max_workers=min(config.max_cpu_workers, psutil.cpu_count() // 2)
            )
        else:
            self.cpu_executor = None
        
        # Performance tracking
        self.performance_stats = {
            'batches_processed': 0,
            'total_cpu_time': 0.0,
            'total_gpu_time': 0.0,
            'overlap_time_saved': 0.0
        }
        
        logger.info(f"CPUGPUCoordinator initialized: overlap={config.enable_cpu_gpu_overlap}")
    
    def process_batch_optimized(self, batch_data: Dict[str, Any], 
                               feature_processor: Any) -> Dict[str, Any]:
        """
        Process batch with CPU-GPU optimization.
        
        Simple approach: CPU preprocessing in thread while GPU processes previous batch.
        """
        start_time = time.time()
        
        if not self.config.enable_cpu_gpu_overlap or self.cpu_executor is None:
            # Synchronous processing
            return self._process_batch_sync(batch_data, feature_processor)
        
        # Asynchronous processing
        cpu_start = time.time()
        
        # Submit CPU preprocessing to thread pool
        cpu_future = self.cpu_executor.submit(
            self._preprocess_on_cpu, batch_data, feature_processor
        )
        
        # Move tensors to GPU while CPU preprocessing runs
        gpu_start = time.time()
        gpu_tensors = self._move_tensors_to_gpu(batch_data)
        gpu_transfer_time = time.time() - gpu_start
        
        # Wait for CPU preprocessing to complete
        try:
            cpu_results = cpu_future.result(timeout=30.0)  # 30 second timeout
            cpu_time = time.time() - cpu_start
        except Exception as e:
            logger.error(f"CPU preprocessing failed: {e}")
            # Fallback to synchronous processing
            return self._process_batch_sync(batch_data, feature_processor)
        
        # Combine results
        batch_data.update(cpu_results)
        batch_data.update(gpu_tensors)
        
        # Update performance stats
        total_time = time.time() - start_time
        theoretical_sequential_time = cpu_time + gpu_transfer_time
        overlap_saved = max(0, theoretical_sequential_time - total_time)
        
        self.performance_stats['batches_processed'] += 1
        self.performance_stats['total_cpu_time'] += cpu_time
        self.performance_stats['total_gpu_time'] += gpu_transfer_time
        self.performance_stats['overlap_time_saved'] += overlap_saved
        
        batch_data['_optimization_metrics'] = {
            'cpu_time': cpu_time,
            'gpu_time': gpu_transfer_time,
            'total_time': total_time,
            'overlap_saved': overlap_saved
        }
        
        return batch_data
    
    def _process_batch_sync(self, batch_data: Dict[str, Any], 
                           feature_processor: Any) -> Dict[str, Any]:
        """Synchronous batch processing (fallback)."""
        # CPU preprocessing
        cpu_results = self._preprocess_on_cpu(batch_data, feature_processor)
        batch_data.update(cpu_results)
        
        # GPU transfer
        gpu_tensors = self._move_tensors_to_gpu(batch_data)
        batch_data.update(gpu_tensors)
        
        return batch_data
    
    def _preprocess_on_cpu(self, batch_data: Dict[str, Any], 
                          feature_processor: Any) -> Dict[str, Any]:
        """CPU preprocessing operations."""
        try:
            # Extract combinations for feature processing
            positive_combinations = batch_data.get('positive_combinations', [])
            current_indices = batch_data.get('current_indices', [])
            
            if not positive_combinations or not hasattr(feature_processor, 'transform'):
                return {'cpu_features': torch.zeros(len(positive_combinations) if positive_combinations else 1, 16)}
            
            # Process features on CPU
            features = []
            for i, combo in enumerate(positive_combinations):
                try:
                    idx = current_indices[i] if i < len(current_indices) else 0
                    feature_vec = feature_processor.transform(combo, idx)
                    features.append(feature_vec)
                except Exception as e:
                    logger.warning(f"Feature processing failed for combo {i}: {e}")
                    # Use zero features as fallback
                    features.append(np.zeros(16))
            
            cpu_features = torch.tensor(np.array(features), dtype=torch.float32)
            
            return {'cpu_features': cpu_features}
            
        except Exception as e:
            logger.error(f"CPU preprocessing error: {e}")
            # Return safe fallback
            batch_size = len(batch_data.get('positive_combinations', [1]))
            return {'cpu_features': torch.zeros(batch_size, 16)}
    
    def _move_tensors_to_gpu(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensors to GPU with optimization."""
        gpu_tensors = {}
        
        if self.device.type != 'cuda':
            return gpu_tensors
        
        # Move tensor data to GPU
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                try:
                    gpu_tensors[f"{key}_gpu"] = value.to(self.device, non_blocking=True)
                except Exception as e:
                    logger.warning(f"Failed to move {key} to GPU: {e}")
                    gpu_tensors[f"{key}_gpu"] = value  # Keep on CPU
        
        return gpu_tensors
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance optimization report."""
        stats = self.performance_stats.copy()
        
        if stats['batches_processed'] > 0:
            stats['avg_cpu_time'] = stats['total_cpu_time'] / stats['batches_processed']
            stats['avg_gpu_time'] = stats['total_gpu_time'] / stats['batches_processed']
            stats['avg_overlap_saved'] = stats['overlap_time_saved'] / stats['batches_processed']
            stats['efficiency_improvement'] = (stats['overlap_time_saved'] / 
                                             max(stats['total_cpu_time'] + stats['total_gpu_time'], 1e-6))
        
        # Add GPU memory stats
        stats['gpu_memory'] = self.gpu_optimizer.get_memory_stats()
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        if self.cpu_executor:
            self.cpu_executor.shutdown(wait=True)
        
        self.gpu_optimizer.emergency_cleanup()
        
        logger.info("CPUGPUCoordinator cleanup completed")


class ProductionOptimizer:
    """
    Main production-ready optimizer.
    
    Simplified, stable implementation focusing on:
    1. Hardware-aware configuration
    2. Simple GPU memory optimization  
    3. Basic CPU-GPU coordination
    4. Comprehensive monitoring
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        
        # Hardware detection
        self.hardware_manager = HardwareResourceManager()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Core components
        self.coordinator = CPUGPUCoordinator(self.config, self.device)
        
        # Optimization state
        self.enabled = True
        self.total_batches = 0
        self.start_time = time.time()
        
        logger.info(f"ProductionOptimizer initialized on {self.device}")
        logger.info(f"Hardware: {self.hardware_manager.profile.cpu_cores} CPU cores, "
                   f"{self.hardware_manager.profile.gpu_memory_gb}GB GPU")
    
    def optimize_training_batch(self, batch_data: Dict[str, Any], 
                               feature_processor: Any) -> Dict[str, Any]:
        """
        Optimize a training batch with CPU-GPU coordination.
        
        Args:
            batch_data: Batch data dictionary
            feature_processor: Feature engineering processor
            
        Returns:
            Optimized batch data
        """
        if not self.enabled:
            return batch_data
        
        try:
            optimized_batch = self.coordinator.process_batch_optimized(
                batch_data, feature_processor
            )
            
            self.total_batches += 1
            
            # Periodic monitoring
            if self.total_batches % 50 == 0:
                self._log_performance_update()
            
            return optimized_batch
            
        except Exception as e:
            logger.error(f"Batch optimization failed: {e}")
            # Disable optimization on repeated failures
            self.enabled = False
            logger.warning("Optimization disabled due to repeated failures")
            return batch_data
    
    def _log_performance_update(self):
        """Log periodic performance updates."""
        runtime = time.time() - self.start_time
        report = self.coordinator.get_performance_report()
        
        logger.info(f"Optimization status: {self.total_batches} batches in {runtime:.1f}s")
        
        if 'efficiency_improvement' in report:
            improvement = report['efficiency_improvement'] * 100
            logger.info(f"Efficiency improvement: {improvement:.1f}%")
        
        if report.get('gpu_memory', {}).get('gpu_available'):
            gpu_mem = report['gpu_memory']['allocated_gb']
            logger.info(f"GPU memory usage: {gpu_mem:.1f}GB")
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        runtime = time.time() - self.start_time
        
        return {
            'optimization_enabled': self.enabled,
            'total_batches_processed': self.total_batches,
            'total_runtime_seconds': runtime,
            'batches_per_second': self.total_batches / max(runtime, 1),
            'hardware_profile': {
                'device': str(self.device),
                'cpu_cores': self.hardware_manager.profile.cpu_cores,
                'gpu_memory_gb': self.hardware_manager.profile.gpu_memory_gb,
                'total_memory_gb': self.hardware_manager.profile.total_memory_gb
            },
            'coordinator_performance': self.coordinator.get_performance_report(),
            'configuration': {
                'gpu_optimization': self.config.enable_gpu_optimization,
                'cpu_gpu_overlap': self.config.enable_cpu_gpu_overlap,
                'memory_safety_margin': self.config.memory_safety_margin,
                'max_cpu_workers': self.config.max_cpu_workers
            }
        }
    
    def shutdown(self):
        """Shutdown optimizer and clean up resources."""
        logger.info("Shutting down ProductionOptimizer...")
        
        # Generate final report
        final_report = self.get_comprehensive_report()
        logger.info(f"Final report: {self.total_batches} batches processed")
        
        if final_report['coordinator_performance'].get('efficiency_improvement'):
            improvement = final_report['coordinator_performance']['efficiency_improvement'] * 100
            logger.info(f"Overall efficiency improvement: {improvement:.1f}%")
        
        # Cleanup
        self.coordinator.cleanup()
        
        logger.info("ProductionOptimizer shutdown complete")


# Integration helpers for existing training system
def create_production_optimizer(enable_gpu: bool = True, 
                               enable_overlap: bool = True,
                               max_workers: int = 4) -> ProductionOptimizer:
    """
    Create production optimizer with simple configuration.
    
    Args:
        enable_gpu: Enable GPU memory optimization
        enable_overlap: Enable CPU-GPU overlap processing
        max_workers: Maximum CPU worker threads
        
    Returns:
        Configured ProductionOptimizer
    """
    config = OptimizationConfig(
        enable_gpu_optimization=enable_gpu,
        enable_cpu_gpu_overlap=enable_overlap,
        max_cpu_workers=max_workers,
        memory_safety_margin=0.8,
        enable_monitoring=True
    )
    
    return ProductionOptimizer(config)


def integrate_with_existing_training(training_function: Any) -> Any:
    """
    Simple integration wrapper for existing training functions.
    
    Usage:
        optimized_train = integrate_with_existing_training(original_train_function)
        results = optimized_train(model, data_loader, optimizer)
    """
    def optimized_training_wrapper(*args, **kwargs):
        # Create optimizer
        optimizer = create_production_optimizer()
        
        try:
            # Add optimizer to kwargs
            kwargs['_performance_optimizer'] = optimizer
            
            # Run original training function
            results = training_function(*args, **kwargs)
            
            # Add performance report to results
            if isinstance(results, dict):
                results['optimization_report'] = optimizer.get_comprehensive_report()
            
            return results
            
        finally:
            optimizer.shutdown()
    
    return optimized_training_wrapper


# Context manager for easy integration
class OptimizedTraining:
    """Context manager for optimized training sessions."""
    
    def __init__(self, enable_gpu: bool = True, enable_overlap: bool = True):
        self.optimizer = create_production_optimizer(enable_gpu, enable_overlap)
    
    def __enter__(self):
        return self.optimizer
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.optimizer.shutdown()
    
    def optimize_batch(self, batch_data: Dict[str, Any], feature_processor: Any) -> Dict[str, Any]:
        """Optimize a training batch."""
        return self.optimizer.optimize_training_batch(batch_data, feature_processor)