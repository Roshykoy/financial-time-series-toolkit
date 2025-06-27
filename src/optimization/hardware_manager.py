"""
Hardware resource detection and management for optimization.
Intelligently detects and manages CPU, memory, and GPU resources.
"""

import os
import psutil
import torch
import gc
import time
import threading
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import logging

from ..infrastructure.logging.logger import get_logger
from ..utils.error_handling import safe_execute

logger = get_logger(__name__)


@dataclass
class HardwareProfile:
    """Hardware resource profile."""
    cpu_cores: int
    cpu_frequency: float
    total_memory_gb: float
    available_memory_gb: float
    gpu_available: bool
    gpu_count: int
    gpu_memory_gb: List[float]
    gpu_names: List[str]
    recommended_batch_size: int
    recommended_workers: int
    parallel_jobs: int


@dataclass 
class ResourceConstraints:
    """Resource usage constraints."""
    max_memory_fraction: float = 0.8
    max_gpu_memory_fraction: float = 0.8
    max_cpu_usage: float = 0.9
    min_free_memory_gb: float = 2.0
    enable_memory_monitoring: bool = True
    cleanup_frequency: int = 10


class HardwareResourceManager:
    """Manages hardware resources for optimization."""
    
    def __init__(self, constraints: Optional[ResourceConstraints] = None):
        self.constraints = constraints or ResourceConstraints()
        self.profile = self._detect_hardware()
        self.resource_usage_history = []
        self._monitoring_active = False
        self._monitor_thread = None
        
        logger.info(f"Hardware detected: {self.profile.cpu_cores} CPU cores, "
                   f"{self.profile.total_memory_gb:.1f}GB RAM, "
                   f"{self.profile.gpu_count} GPUs")
    
    def _detect_hardware(self) -> HardwareProfile:
        """Detect available hardware resources."""
        try:
            # CPU information
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 2400.0
            
            # Memory information  
            memory = psutil.virtual_memory()
            total_memory_gb = memory.total / (1024**3)
            available_memory_gb = memory.available / (1024**3)
            
            # GPU information
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            gpu_memory_gb = []
            gpu_names = []
            
            if gpu_available:
                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_memory_gb.append(props.total_memory / (1024**3))
                    gpu_names.append(props.name)
            
            # Calculate recommendations
            recommended_batch_size = self._calculate_batch_size(total_memory_gb, gpu_memory_gb)
            recommended_workers = min(cpu_cores, 8)  # Avoid too many workers
            parallel_jobs = max(1, cpu_cores // 2)  # Conservative parallel jobs
            
            return HardwareProfile(
                cpu_cores=cpu_cores,
                cpu_frequency=cpu_freq,
                total_memory_gb=total_memory_gb,
                available_memory_gb=available_memory_gb,
                gpu_available=gpu_available,
                gpu_count=gpu_count,
                gpu_memory_gb=gpu_memory_gb,
                gpu_names=gpu_names,
                recommended_batch_size=recommended_batch_size,
                recommended_workers=recommended_workers,
                parallel_jobs=parallel_jobs
            )
            
        except Exception as e:
            logger.error(f"Error detecting hardware: {e}")
            # Return minimal fallback profile
            return HardwareProfile(
                cpu_cores=2, cpu_frequency=2400.0,
                total_memory_gb=8.0, available_memory_gb=4.0,
                gpu_available=False, gpu_count=0,
                gpu_memory_gb=[], gpu_names=[],
                recommended_batch_size=8, recommended_workers=2, parallel_jobs=1
            )
    
    def _calculate_batch_size(self, ram_gb: float, gpu_memory_gb: List[float]) -> int:
        """Calculate recommended batch size based on available memory."""
        if gpu_memory_gb:
            # GPU-based calculation
            min_gpu_memory = min(gpu_memory_gb)
            if min_gpu_memory >= 8:
                return 32
            elif min_gpu_memory >= 4:
                return 16
            else:
                return 8
        else:
            # CPU-based calculation
            if ram_gb >= 16:
                return 16
            elif ram_gb >= 8:
                return 8
            else:
                return 4
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource usage status."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'timestamp': time.time()
            }
            
            if self.profile.gpu_available:
                gpu_status = []
                for i in range(self.profile.gpu_count):
                    try:
                        torch.cuda.set_device(i)
                        memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                        memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                        memory_free = self.profile.gpu_memory_gb[i] - memory_reserved
                        
                        gpu_status.append({
                            'device_id': i,
                            'memory_allocated_gb': memory_allocated,
                            'memory_reserved_gb': memory_reserved,
                            'memory_free_gb': memory_free,
                            'utilization_percent': memory_reserved / self.profile.gpu_memory_gb[i] * 100
                        })
                    except Exception as e:
                        logger.warning(f"Could not get GPU {i} status: {e}")
                        
                status['gpu_status'] = gpu_status
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting resource status: {e}")
            return {'error': str(e)}
    
    def check_resource_constraints(self) -> Tuple[bool, List[str]]:
        """Check if current resource usage is within constraints."""
        status = self.get_resource_status()
        issues = []
        
        # Check memory constraints
        if status.get('memory_usage_percent', 0) > self.constraints.max_memory_fraction * 100:
            issues.append(f"Memory usage too high: {status['memory_usage_percent']:.1f}%")
        
        if status.get('memory_available_gb', 0) < self.constraints.min_free_memory_gb:
            issues.append(f"Low free memory: {status['memory_available_gb']:.1f}GB")
        
        # Check CPU constraints
        if status.get('cpu_usage_percent', 0) > self.constraints.max_cpu_usage * 100:
            issues.append(f"CPU usage too high: {status['cpu_usage_percent']:.1f}%")
        
        # Check GPU constraints
        if 'gpu_status' in status:
            for gpu in status['gpu_status']:
                if gpu['utilization_percent'] > self.constraints.max_gpu_memory_fraction * 100:
                    issues.append(f"GPU {gpu['device_id']} memory too high: {gpu['utilization_percent']:.1f}%")
        
        return len(issues) == 0, issues
    
    def optimize_for_hardware(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration parameters for available hardware."""
        optimized_config = base_config.copy()
        
        # Adjust batch size
        if 'batch_size' in optimized_config:
            current_batch = optimized_config['batch_size']
            recommended_batch = self.profile.recommended_batch_size
            
            # Use smaller of current and recommended to be conservative
            optimized_config['batch_size'] = min(current_batch, recommended_batch)
        
        # Adjust number of workers
        if 'num_workers' in optimized_config:
            optimized_config['num_workers'] = self.profile.recommended_workers
        
        # GPU memory management
        if self.profile.gpu_available:
            optimized_config['device'] = 'cuda'
            optimized_config['max_memory_fraction'] = self.constraints.max_gpu_memory_fraction
            
            # Disable mixed precision on older GPUs
            if self.profile.gpu_memory_gb and min(self.profile.gpu_memory_gb) < 6:
                optimized_config['use_mixed_precision'] = False
        else:
            optimized_config['device'] = 'cpu'
            optimized_config['use_mixed_precision'] = False
        
        # Adjust parallelization
        if 'parallel_jobs' not in optimized_config:
            optimized_config['parallel_jobs'] = self.profile.parallel_jobs
        
        logger.info(f"Hardware optimization: batch_size={optimized_config.get('batch_size')}, "
                   f"device={optimized_config.get('device')}, "
                   f"parallel_jobs={optimized_config.get('parallel_jobs')}")
        
        return optimized_config
    
    def cleanup_resources(self) -> None:
        """Clean up system resources."""
        try:
            # Python garbage collection
            gc.collect()
            
            # CUDA cleanup if available
            if self.profile.gpu_available:
                for i in range(self.profile.gpu_count):
                    try:
                        torch.cuda.set_device(i)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except Exception as e:
                        logger.warning(f"Could not cleanup GPU {i}: {e}")
            
            logger.debug("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during resource cleanup: {e}")
    
    def start_monitoring(self, interval: float = 10.0) -> None:
        """Start resource monitoring in background thread."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self, interval: float) -> None:
        """Background resource monitoring loop."""
        while self._monitoring_active:
            try:
                status = self.get_resource_status()
                self.resource_usage_history.append(status)
                
                # Keep only recent history (last hour)
                max_history = int(3600 / interval)
                if len(self.resource_usage_history) > max_history:
                    self.resource_usage_history = self.resource_usage_history[-max_history:]
                
                # Check constraints and warn if necessary
                is_ok, issues = self.check_resource_constraints()
                if not is_ok:
                    logger.warning(f"Resource constraint violations: {', '.join(issues)}")
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(interval)
    
    def get_monitoring_data(self) -> List[Dict[str, Any]]:
        """Get resource monitoring history."""
        return self.resource_usage_history.copy()
    
    def estimate_trial_time(self, config: Dict[str, Any]) -> float:
        """Estimate time for a single optimization trial."""
        base_time = 180.0  # 3 minutes base time
        
        # Adjust for batch size
        batch_factor = config.get('batch_size', 8) / 8.0
        base_time *= (1.0 / batch_factor) ** 0.5  # Larger batches are more efficient
        
        # Adjust for epochs
        epochs_factor = config.get('epochs', 5) / 5.0
        base_time *= epochs_factor
        
        # Adjust for device
        if config.get('device') == 'cpu':
            base_time *= 2.0  # CPU is slower
        
        # Adjust for model complexity
        hidden_size = config.get('hidden_size', 256)
        complexity_factor = (hidden_size / 256.0) ** 1.5
        base_time *= complexity_factor
        
        return base_time
    
    def suggest_optimization_strategy(self, time_budget: Optional[float] = None) -> Dict[str, Any]:
        """Suggest optimization strategy based on available resources."""
        strategy = {
            'recommended_algorithm': 'random_search',
            'max_trials': 20,
            'parallel_jobs': 1,
            'early_stopping': True,
            'resource_monitoring': True
        }
        
        # Adjust based on available resources
        if self.profile.cpu_cores >= 8 and self.profile.total_memory_gb >= 16:
            strategy['max_trials'] = 30
            strategy['parallel_jobs'] = min(2, self.profile.parallel_jobs)
            strategy['recommended_algorithm'] = 'bayesian'
        
        if self.profile.gpu_available and self.profile.gpu_memory_gb:
            if min(self.profile.gpu_memory_gb) >= 8:
                strategy['max_trials'] = 40
                strategy['recommended_algorithm'] = 'optuna'
        
        # Adjust for time budget
        if time_budget:
            estimated_time_per_trial = self.estimate_trial_time({})
            max_trials_by_time = int(time_budget / estimated_time_per_trial)
            strategy['max_trials'] = min(strategy['max_trials'], max_trials_by_time)
        
        return strategy
    
    def save_hardware_profile(self, path: Path) -> None:
        """Save hardware profile to file."""
        try:
            profile_data = {
                'profile': self.profile.__dict__,
                'constraints': self.constraints.__dict__,
                'monitoring_data': self.resource_usage_history[-100:]  # Last 100 points
            }
            
            import json
            with open(path, 'w') as f:
                json.dump(profile_data, f, indent=2, default=str)
                
            logger.info(f"Hardware profile saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving hardware profile: {e}")


def create_hardware_manager(config: Optional[Dict[str, Any]] = None) -> HardwareResourceManager:
    """Factory function to create hardware manager."""
    constraints = None
    if config:
        constraints = ResourceConstraints(
            max_memory_fraction=config.get('max_memory_fraction', 0.8),
            max_gpu_memory_fraction=config.get('max_gpu_memory_fraction', 0.8),
            max_cpu_usage=config.get('max_cpu_usage', 0.9),
            min_free_memory_gb=config.get('min_free_memory_gb', 2.0),
            enable_memory_monitoring=config.get('enable_memory_monitoring', True),
            cleanup_frequency=config.get('cleanup_frequency', 10)
        )
    
    return HardwareResourceManager(constraints)