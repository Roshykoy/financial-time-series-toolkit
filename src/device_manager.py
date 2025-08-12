"""
Centralized device management for consistent CPU/GPU handling across the system.
"""

import torch
import logging
from typing import Optional, Union, Any, Dict
from functools import wraps
import warnings

logger = logging.getLogger(__name__)


class DeviceManager:
    """Centralized device management with automatic fallback and consistency checks."""
    
    _instance = None
    _device = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._device = self._detect_best_device()
            self._device_properties = self._get_device_properties()
            self._synchronized = True
            DeviceManager._initialized = True
            logger.info(f"DeviceManager initialized with device: {self._device}")
    
    def _detect_best_device(self) -> torch.device:
        """Detect the best available device with fallback logic."""
        try:
            if torch.cuda.is_available():
                # Check if CUDA actually works
                try:
                    test_tensor = torch.randn(2, 2, device='cuda')
                    _ = test_tensor + test_tensor  # Simple operation test
                    del test_tensor  # Clean up
                    torch.cuda.empty_cache()
                    return torch.device('cuda')
                except Exception as e:
                    logger.warning(f"CUDA available but not working: {e}")
                    return torch.device('cpu')
            else:
                return torch.device('cpu')
        except Exception as e:
            logger.error(f"Error detecting device: {e}")
            return torch.device('cpu')  # Safe fallback
    
    def _get_device_properties(self) -> Dict[str, Any]:
        """Get properties of the current device."""
        properties = {
            'type': self._device.type,
            'index': getattr(self._device, 'index', None)
        }
        
        if self._device.type == 'cuda':
            try:
                properties.update({
                    'name': torch.cuda.get_device_name(self._device.index),
                    'memory_total': torch.cuda.get_device_properties(self._device.index).total_memory,
                    'memory_reserved': torch.cuda.memory_reserved(self._device.index),
                    'memory_allocated': torch.cuda.memory_allocated(self._device.index)
                })
            except Exception as e:
                logger.warning(f"Could not get CUDA properties: {e}")
        
        return properties
    
    @property
    def device(self) -> torch.device:
        """Get the managed device."""
        return self._device
    
    @property
    def is_cuda(self) -> bool:
        """Check if the managed device is CUDA."""
        return self._device.type == 'cuda'
    
    @property
    def is_cpu(self) -> bool:
        """Check if the managed device is CPU."""
        return self._device.type == 'cpu'
    
    def to_device(self, obj: Any, non_blocking: bool = False) -> Any:
        """
        Move tensor or model to managed device with error handling.
        
        Args:
            obj: Tensor, model, or other object to move
            non_blocking: Whether to use non-blocking transfer
            
        Returns:
            Object moved to managed device
        """
        try:
            if hasattr(obj, 'to'):
                return obj.to(self._device, non_blocking=non_blocking)
            elif isinstance(obj, (list, tuple)):
                return type(obj)(self.to_device(item, non_blocking) for item in obj)
            elif isinstance(obj, dict):
                return {key: self.to_device(value, non_blocking) for key, value in obj.items()}
            else:
                # Object doesn't support device movement
                return obj
        except Exception as e:
            logger.error(f"Error moving object to device {self._device}: {e}")
            raise
    
    def synchronize(self):
        """Synchronize device operations to ensure completion."""
        try:
            if self.is_cuda:
                torch.cuda.synchronize(self._device)
            # CPU operations are inherently synchronous
            self._synchronized = True
        except Exception as e:
            logger.error(f"Error synchronizing device: {e}")
            self._synchronized = False
    
    def empty_cache(self):
        """Empty device cache if applicable."""
        try:
            if self.is_cuda:
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error emptying cache: {e}")
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        info = {'device': str(self._device)}
        
        if self.is_cuda:
            try:
                info.update({
                    'memory_allocated': torch.cuda.memory_allocated(self._device),
                    'memory_reserved': torch.cuda.memory_reserved(self._device),
                    'memory_free': torch.cuda.get_device_properties(self._device).total_memory - 
                                  torch.cuda.memory_allocated(self._device)
                })
            except Exception as e:
                info['memory_error'] = str(e)
        else:
            info['memory_note'] = 'CPU memory tracking not available'
        
        return info
    
    def check_device_consistency(self, *objects) -> bool:
        """
        Check if all provided objects are on the managed device.
        
        Args:
            *objects: Tensors or models to check
            
        Returns:
            True if all objects are on the correct device
        """
        for obj in objects:
            if hasattr(obj, 'device'):
                if obj.device != self._device:
                    logger.warning(f"Object on device {obj.device}, expected {self._device}")
                    return False
            elif hasattr(obj, 'parameters'):  # Model check
                for param in obj.parameters():
                    if param.device != self._device:
                        logger.warning(f"Model parameter on device {param.device}, expected {self._device}")
                        return False
        return True
    
    def force_device_consistency(self, *objects):
        """
        Force all provided objects to be on the managed device.
        
        Args:
            *objects: Tensors or models to move
        """
        moved_objects = []
        for obj in objects:
            moved_objects.append(self.to_device(obj))
        return moved_objects if len(moved_objects) > 1 else moved_objects[0]
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            'device': str(self._device),
            'properties': self._device_properties.copy(),
            'synchronized': self._synchronized
        }
        
        if self.is_cuda:
            try:
                info['cuda_version'] = torch.version.cuda
                info['cudnn_version'] = torch.backends.cudnn.version()
                info['current_memory'] = self.get_memory_info()
            except Exception as e:
                info['cuda_info_error'] = str(e)
        
        return info


# Global device manager instance
device_manager = DeviceManager()


def device_synchronized(func):
    """
    Decorator to ensure device synchronization before and after function execution.
    
    Usage:
        @device_synchronized
        def my_function():
            # Function that performs device operations
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        device_manager.synchronize()
        try:
            result = func(*args, **kwargs)
            device_manager.synchronize()
            return result
        except Exception as e:
            device_manager.synchronize()  # Ensure sync even on error
            raise
    return wrapper


def ensure_device_consistency(*tensors_or_models):
    """
    Decorator to ensure all tensors/models in function arguments are on the correct device.
    
    Usage:
        @ensure_device_consistency
        def my_function(tensor1, tensor2, model):
            # All arguments will be moved to managed device
            pass
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Move positional arguments to device
            new_args = []
            for arg in args:
                new_args.append(device_manager.to_device(arg) if hasattr(arg, 'to') or hasattr(arg, 'parameters') else arg)
            
            # Move keyword arguments to device
            new_kwargs = {}
            for key, value in kwargs.items():
                new_kwargs[key] = device_manager.to_device(value) if hasattr(value, 'to') or hasattr(value, 'parameters') else value
            
            return func(*new_args, **new_kwargs)
        return wrapper
    return decorator


def get_device() -> torch.device:
    """Get the managed device."""
    return device_manager.device


def to_device(obj: Any, non_blocking: bool = False) -> Any:
    """Move object to managed device."""
    return device_manager.to_device(obj, non_blocking)


def synchronize_device():
    """Synchronize device operations."""
    device_manager.synchronize()


def check_device_memory():
    """Get current device memory information."""
    return device_manager.get_memory_info()


def empty_device_cache():
    """Empty device cache."""
    device_manager.empty_cache()


# Backward compatibility functions
def get_device_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update config with managed device and ensure consistency.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated configuration with device settings
    """
    config_copy = config.copy()
    config_copy['device'] = str(device_manager.device)
    config_copy['device_type'] = device_manager.device.type
    config_copy['is_cuda'] = device_manager.is_cuda
    
    # Add device-specific optimizations
    if device_manager.is_cuda:
        config_copy.setdefault('use_mixed_precision', True)
        config_copy.setdefault('pin_memory', True)
        config_copy.setdefault('non_blocking', True)
    else:
        config_copy.setdefault('use_mixed_precision', False)
        config_copy.setdefault('pin_memory', False)
        config_copy.setdefault('non_blocking', False)
    
    return config_copy