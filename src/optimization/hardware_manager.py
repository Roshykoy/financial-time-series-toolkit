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
            # GPU-based calculation with minimum viable batch size = 32
            min_gpu_memory = min(gpu_memory_gb)
            if min_gpu_memory >= 10:
                return 128  # RTX 3080+ can handle larger batches
            elif min_gpu_memory >= 8:
                return 64   # Good balance for 8-10GB GPUs
            elif min_gpu_memory >= 6:
                return 32   # Minimum viable for model complexity < 1.0
            else:
                return 32   # Force minimum even on lower VRAM
        else:
            # CPU-based calculation - still respect minimum
            if ram_gb >= 32:
                return 32
            elif ram_gb >= 16:
                return 32
            else:
                return 32   # Always minimum 32 for viable model complexity
    
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
    
    def estimate_gpu_memory_usage(self, config: Dict[str, Any]) -> float:
        """Estimate GPU memory usage in GB for given configuration."""
        try:
            # Base memory for PyTorch overhead
            base_memory_gb = 1.0
            
            # Model parameters memory estimation
            batch_size = config.get('batch_size', 32)
            hidden_size = config.get('hidden_size', 256)
            latent_dim = config.get('latent_dim', 256)
            num_layers = config.get('num_layers', 4)
            decoder_layers = config.get('decoder_layers', 3)
            
            # CVAE encoder memory (scales with batch_size and hidden_size)
            encoder_memory = (batch_size * hidden_size * num_layers * 4) / (1024**3)  # 4 bytes per float32
            
            # CVAE decoder memory (scales with batch_size and decoder complexity)
            decoder_memory = (batch_size * hidden_size * decoder_layers * 4) / (1024**3)
            
            # Latent space memory
            latent_memory = (batch_size * latent_dim * 4) / (1024**3)
            
            # Graph encoder memory (depends on graph size and hidden dimensions)
            graph_memory = (batch_size * hidden_size * 2 * 4) / (1024**3)  # Approximate
            
            # Temporal context memory
            temporal_memory = (batch_size * config.get('temporal_context_dim', 128) * 4) / (1024**3)
            
            # Optimizer states (Adam requires 2x parameter memory)
            model_params = (hidden_size ** 2 * num_layers + hidden_size * latent_dim) * 4 / (1024**3)
            optimizer_memory = model_params * 2
            
            # Gradient memory (same as model parameters)
            gradient_memory = model_params
            
            # Loss computation and intermediate activations (scales significantly with batch size)
            activation_memory = (batch_size * hidden_size * 8) / (1024**3)  # Conservative estimate
            
            total_memory = (base_memory_gb + encoder_memory + decoder_memory + 
                          latent_memory + graph_memory + temporal_memory + 
                          optimizer_memory + gradient_memory + activation_memory)
            
            logger.debug(f"Memory estimation: {total_memory:.2f}GB for batch_size={batch_size}, hidden_size={hidden_size}")
            return total_memory
            
        except Exception as e:
            logger.error(f"Error estimating GPU memory: {e}")
            # Return conservative estimate
            return 8.0
    
    def get_optimal_parameter_combinations(self, max_memory_gb: Optional[float] = None) -> List[Dict[str, Any]]:
        """Get optimal parameter combinations that maximize GPU utilization."""
        if max_memory_gb is None:
            # Use 80% of available GPU memory as safety margin
            if self.profile.gpu_memory_gb:
                max_memory_gb = min(self.profile.gpu_memory_gb) * 0.8
            else:
                max_memory_gb = 6.0  # Conservative fallback
        
        optimal_combinations = []
        
        # Define parameter search space with hardware constraints
        batch_sizes = [8, 16, 32, 48]  # for model complexity
        hidden_sizes = [128, 256, 384, 512, 768]
        num_layers_options = [2, 3, 4, 5, 6]
        
        for batch_size in batch_sizes:
            for hidden_size in hidden_sizes:
                for num_layers in num_layers_options:
                    config = {
                        'batch_size': batch_size,
                        'hidden_size': hidden_size,
                        'num_layers': num_layers,
                        'latent_dim': 256,
                        'decoder_layers': 3,
                        'temporal_context_dim': 128
                    }
                    
                    estimated_memory = self.estimate_gpu_memory_usage(config)
                    
                    if estimated_memory <= max_memory_gb:
                        # Calculate efficiency score (higher batch_size and hidden_size preferred)
                        efficiency_score = (batch_size / 128) * (hidden_size / 768) * (1.0 / num_layers)
                        memory_utilization = estimated_memory / max_memory_gb
                        
                        optimal_combinations.append({
                            'config': config,
                            'estimated_memory_gb': estimated_memory,
                            'memory_utilization': memory_utilization,
                            'efficiency_score': efficiency_score
                        })
        
        # Sort by efficiency score and memory utilization
        optimal_combinations.sort(key=lambda x: (x['efficiency_score'], x['memory_utilization']), reverse=True)
        
        logger.info(f"Found {len(optimal_combinations)} viable parameter combinations for {max_memory_gb:.1f}GB GPU")
        return optimal_combinations
    
    def constrain_parameter_ranges(self, parameter_ranges: Dict[str, Any]) -> Dict[str, Any]:
        """Apply hardware-aware constraints to parameter ranges."""
        constrained_ranges = parameter_ranges.copy()
        
        # Enforce minimum batch size for viable model complexity
        if 'batch_size' in constrained_ranges:
            if 'options' in constrained_ranges['batch_size']:
                # Filter out batch sizes < 32
                original_options = constrained_ranges['batch_size']['options']
                constrained_options = [bs for bs in original_options if bs >= 32]
                constrained_ranges['batch_size']['options'] = constrained_options
                logger.info(f"Constrained batch_size options from {original_options} to {constrained_options}")
            
            elif 'range' in constrained_ranges['batch_size']:
                # Adjust range minimum to 32
                original_range = constrained_ranges['batch_size']['range']
                constrained_range = (max(32, original_range[0]), original_range[1])
                constrained_ranges['batch_size']['range'] = constrained_range
                logger.info(f"Constrained batch_size range from {original_range} to {constrained_range}")
        
        # Adjust hidden_size based on GPU memory
        if 'hidden_size' in constrained_ranges and self.profile.gpu_memory_gb:
            max_gpu_memory = min(self.profile.gpu_memory_gb)
            
            if 'options' in constrained_ranges['hidden_size']:
                original_options = constrained_ranges['hidden_size']['options']
                # Test each hidden_size option with batch_size=32 to see what fits
                viable_options = []
                
                for hidden_size in original_options:
                    test_config = {
                        'batch_size': 32,
                        'hidden_size': hidden_size,
                        'num_layers': 4,
                        'latent_dim': 256,
                        'decoder_layers': 3,
                        'temporal_context_dim': 128
                    }
                    estimated_memory = self.estimate_gpu_memory_usage(test_config)
                    
                    if estimated_memory <= max_gpu_memory * 0.8:  # 80% safety margin
                        viable_options.append(hidden_size)
                
                if viable_options:
                    constrained_ranges['hidden_size']['options'] = viable_options
                    logger.info(f"Constrained hidden_size options from {original_options} to {viable_options}")
        
        return constrained_ranges
    
    def optimize_for_hardware(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration parameters for available hardware."""
        optimized_config = base_config.copy()
        
        # Adjust batch size with minimum constraint for model complexity
        if 'batch_size' in optimized_config:
            current_batch = optimized_config['batch_size']
            recommended_batch = self.profile.recommended_batch_size
            
            # Enforce minimum batch size of 32 for viable model complexity
            min_viable_batch = 32
            optimal_batch = max(min_viable_batch, min(current_batch, recommended_batch))
            optimized_config['batch_size'] = optimal_batch
            
            logger.info(f"Batch size optimization: {current_batch} → {optimal_batch} (min_viable={min_viable_batch})")
        
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
        
        # GPU memory validation and optimization
        if self.profile.gpu_available and self.profile.gpu_memory_gb:
            max_gpu_memory = min(self.profile.gpu_memory_gb) * 0.8  # 80% safety margin
            estimated_memory = self.estimate_gpu_memory_usage(optimized_config)
            
            if estimated_memory > max_gpu_memory:
                logger.warning(f"Configuration exceeds GPU memory ({estimated_memory:.2f}GB > {max_gpu_memory:.2f}GB)")
                
                # Try to reduce hidden_size to fit memory constraint
                if 'hidden_size' in optimized_config:
                    original_hidden_size = optimized_config['hidden_size']
                    
                    # Try smaller hidden sizes
                    hidden_size_options = [256, 128, 384, 512, 768, 1024]
                    for hidden_size in sorted(hidden_size_options):
                        if hidden_size < original_hidden_size:
                            test_config = optimized_config.copy()
                            test_config['hidden_size'] = hidden_size
                            test_memory = self.estimate_gpu_memory_usage(test_config)
                            
                            if test_memory <= max_gpu_memory:
                                optimized_config['hidden_size'] = hidden_size
                                logger.info(f"Reduced hidden_size: {original_hidden_size} → {hidden_size} to fit GPU memory")
                                break
        
        final_memory = self.estimate_gpu_memory_usage(optimized_config)
        logger.info(f"Hardware optimization complete: batch_size={optimized_config.get('batch_size')}, "
                   f"hidden_size={optimized_config.get('hidden_size')}, "
                   f"device={optimized_config.get('device')}, "
                   f"estimated_memory={final_memory:.2f}GB")
        
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
    
    def get_pareto_optimization_constraints(self) -> Dict[str, Any]:
        """Get hardware-optimized constraints for Pareto Front optimization."""
        constraints = {
            'memory_aware': True,
            'gpu_memory_gb': min(self.profile.gpu_memory_gb) if self.profile.gpu_memory_gb else 0,
            'safety_margin': 0.8,  # Use 80% of available GPU memory
            'min_batch_size': 32,  # Minimum for model_complexity < 1.0
            'recommended_batch_sizes': [],
            'viable_hidden_sizes': [],
            'optimal_combinations': []
        }
        
        if self.profile.gpu_available and self.profile.gpu_memory_gb:
            max_memory = min(self.profile.gpu_memory_gb) * constraints['safety_margin']
            
            # Calculate recommended batch sizes for RTX 3080
            gpu_memory = min(self.profile.gpu_memory_gb)
            if gpu_memory >= 10:
                constraints['recommended_batch_sizes'] = [16, 32, 48, 64, 96, 128, 160]
            elif gpu_memory >= 8:
                constraints['recommended_batch_sizes'] = [8, 16, 32, 48, 64, 96]
            else:
                constraints['recommended_batch_sizes'] = [4, 8, 16, 32, 48, 64]
            
            # Test viable hidden sizes
            test_config_base = {
                'batch_size': 32,
                'num_layers': 4,
                'latent_dim': 256,
                'decoder_layers': 3,
                'temporal_context_dim': 128
            }
            
            viable_hidden_sizes = []
            for hidden_size in [128, 256, 384, 512, 640, 768, 896, 1024]:
                test_config = test_config_base.copy()
                test_config['hidden_size'] = hidden_size
                
                estimated_memory = self.estimate_gpu_memory_usage(test_config)
                if estimated_memory <= max_memory:
                    viable_hidden_sizes.append(hidden_size)
            
            constraints['viable_hidden_sizes'] = viable_hidden_sizes
            
            # Get top optimal combinations for Pareto Front
            optimal_combinations = self.get_optimal_parameter_combinations(max_memory)
            constraints['optimal_combinations'] = optimal_combinations[:10]  # Top 10 combinations
            
            logger.info(f"Pareto optimization constraints: batch_sizes={constraints['recommended_batch_sizes']}, "
                       f"hidden_sizes={viable_hidden_sizes}, "
                       f"optimal_combinations={len(constraints['optimal_combinations'])}")
        
        else:
            # CPU fallback constraints
            constraints['recommended_batch_sizes'] = [32, 48, 64]
            constraints['viable_hidden_sizes'] = [128, 256, 384, 512]
            logger.warning("GPU not available, using CPU-optimized constraints")
        
        return constraints


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