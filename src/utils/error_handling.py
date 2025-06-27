"""
Comprehensive error handling utilities for MarkSix forecasting system.
Provides robust error recovery, graceful degradation, and user-friendly error messages.
"""
import os
import sys
import traceback
import warnings
from functools import wraps
from typing import Optional, Callable, Any, Dict, Union
from contextlib import contextmanager
import logging


class MarkSixError(Exception):
    """Base exception for MarkSix-specific errors."""
    pass


class ConfigurationError(MarkSixError):
    """Raised when configuration is invalid or missing."""
    pass


class DataError(MarkSixError):
    """Raised when data is invalid or corrupted."""
    pass


class ModelError(MarkSixError):
    """Raised when model operations fail."""
    pass


class GPUError(MarkSixError):
    """Raised when GPU operations fail."""
    pass


class ErrorHandler:
    """Centralized error handling with recovery strategies."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.fallback_strategies = {}
    
    def register_fallback(self, error_type: type, fallback_func: Callable):
        """Register a fallback strategy for specific error types."""
        self.fallback_strategies[error_type] = fallback_func
    
    def handle_error(self, error: Exception, context: str = "", critical: bool = False) -> bool:
        """
        Handle error with appropriate logging and recovery.
        Returns True if error was handled, False if it should be re-raised.
        """
        error_key = f"{type(error).__name__}:{context}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log the error
        error_msg = f"Error in {context}: {error}" if context else str(error)
        
        if critical:
            self.logger.critical(error_msg, exc_info=True)
        else:
            self.logger.error(error_msg, exc_info=True)
        
        # Try fallback strategy
        error_type = type(error)
        if error_type in self.fallback_strategies:
            try:
                self.logger.info(f"Attempting fallback for {error_type.__name__}")
                self.fallback_strategies[error_type](error)
                return True
            except Exception as fallback_error:
                self.logger.error(f"Fallback failed: {fallback_error}")
        
        # Check if error occurred too many times
        if self.error_counts[error_key] > 3:
            self.logger.critical(f"Error {error_key} occurred {self.error_counts[error_key]} times. Stopping retries.")
            return False
        
        return not critical


def robust_operation(
    max_retries: int = 3,
    exceptions: tuple = (Exception,),
    delay: float = 1.0,
    backoff: float = 2.0,
    logger: Optional[logging.Logger] = None
):
    """
    Decorator for robust operation execution with retries and exponential backoff.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        _logger.error(f"Operation {func.__name__} failed after {max_retries} retries")
                        raise
                    
                    _logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                    if attempt < max_retries:
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator


@contextmanager
def gpu_error_recovery():
    """Context manager for GPU operations with automatic CPU fallback."""
    try:
        yield
    except RuntimeError as e:
        if "CUDA out of memory" in str(e) or "CUDA" in str(e):
            print("⚠️  GPU error detected, attempting CPU fallback...")
            try:
                import torch
                torch.cuda.empty_cache()
                print("🔄 GPU memory cleared, retrying...")
                yield
            except Exception:
                print("❌ CPU fallback failed")
                raise GPUError(f"GPU operation failed and CPU fallback unsuccessful: {e}")
        else:
            raise


@contextmanager
def safe_file_operation(file_path: str, operation: str = "read"):
    """Context manager for safe file operations with proper error handling."""
    try:
        # Ensure parent directory exists for write operations
        if operation in ["write", "append"]:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        yield
        
    except FileNotFoundError:
        raise DataError(f"File not found: {file_path}")
    except PermissionError:
        raise DataError(f"Permission denied accessing file: {file_path}")
    except OSError as e:
        raise DataError(f"File operation failed for {file_path}: {e}")


def safe_model_operation(operation_name: str = "model operation"):
    """Decorator for safe model operations with memory management."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # Check GPU memory before operation
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                result = func(*args, **kwargs)
                
                # Clean up after operation
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                
                return result
                
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    raise GPUError(f"GPU memory exhausted during {operation_name}: {e}")
                elif "Expected" in str(e) and "tensor" in str(e):
                    raise ModelError(f"Model tensor shape mismatch in {operation_name}: {e}")
                else:
                    raise ModelError(f"Model operation failed in {operation_name}: {e}")
            except Exception as e:
                raise ModelError(f"Unexpected error in {operation_name}: {e}")
        
        return wrapper
    return decorator


def handle_division_by_zero(func: Callable) -> Callable:
    """Decorator to handle division by zero with small epsilon addition."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ZeroDivisionError:
            # Add small epsilon to avoid division by zero
            print("⚠️  Division by zero detected, adding numerical stability epsilon")
            return func(*args, **kwargs, eps=1e-8)
    return wrapper


def validate_tensor_health(tensor, name: str = "tensor", check_finite: bool = True):
    """Validate tensor for NaN, Inf, and other numerical issues."""
    try:
        import torch
        
        if not isinstance(tensor, torch.Tensor):
            return  # Skip validation for non-tensors
        
        if torch.isnan(tensor).any():
            raise ModelError(f"{name} contains NaN values")
        
        if check_finite and torch.isinf(tensor).any():
            raise ModelError(f"{name} contains infinite values")
        
        # Check for FP16 overflow
        if tensor.dtype == torch.float16 and tensor.abs().max() > 65504:
            warnings.warn(f"{name} may have FP16 overflow (max value: {tensor.abs().max()})")
    
    except ImportError:
        pass  # PyTorch not available


def create_error_summary(errors: list) -> str:
    """Create a user-friendly error summary."""
    if not errors:
        return "No errors occurred."
    
    error_types = {}
    for error in errors:
        error_type = type(error).__name__
        error_types[error_type] = error_types.get(error_type, 0) + 1
    
    summary = f"Encountered {len(errors)} error(s):\n"
    for error_type, count in error_types.items():
        summary += f"  • {error_type}: {count} occurrence(s)\n"
    
    return summary


def log_system_info(logger: logging.Logger):
    """Log comprehensive system information for debugging."""
    try:
        import torch
        import platform
        import psutil
        
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"PyTorch: {torch.__version__}")
        
        # GPU Information
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA Available: Yes ({gpu_count} device(s))")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        else:
            logger.info("CUDA Available: No")
        
        # Memory Information
        memory = psutil.virtual_memory()
        logger.info(f"System RAM: {memory.total / (1024**3):.1f} GB")
        logger.info(f"Available RAM: {memory.available / (1024**3):.1f} GB")
        
    except ImportError as e:
        logger.warning(f"Could not gather complete system info: {e}")


def safe_execute(
    func: Callable,
    *args,
    fallback: Any = None,
    log_errors: bool = True,
    context: str = "",
    max_retries: int = 0,
    retry_delay: float = 1.0,
    logger: Optional[logging.Logger] = None,
    **kwargs
) -> Any:
    """
    Safely execute a function with comprehensive error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        fallback: Value to return if function fails (default: None)
        log_errors: Whether to log errors (default: True)
        context: Context string for error logging
        max_retries: Number of retries on failure (default: 0)
        retry_delay: Delay between retries in seconds (default: 1.0)
        logger: Logger instance for error reporting
        **kwargs: Keyword arguments for the function
    
    Returns:
        Function result on success, fallback value on failure
    """
    _logger = logger or logging.getLogger(func.__module__ if hasattr(func, '__module__') else __name__)
    
    last_exception = None
    func_name = getattr(func, '__name__', str(func))
    context_str = f" in {context}" if context else ""
    
    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            
            # Log successful retry if applicable
            if attempt > 0 and log_errors:
                _logger.info(f"Function {func_name} succeeded on attempt {attempt + 1}")
            
            return result
            
        except Exception as e:
            last_exception = e
            
            if log_errors:
                if attempt == max_retries:
                    _logger.error(f"Function {func_name} failed after {max_retries + 1} attempts{context_str}: {e}")
                else:
                    _logger.warning(f"Function {func_name} failed on attempt {attempt + 1}{context_str}: {e}")
            
            # If not the last attempt, wait and retry
            if attempt < max_retries:
                if retry_delay > 0:
                    import time
                    time.sleep(retry_delay)
                continue
            
            # All attempts failed, return fallback
            break
    
    # Log fallback usage
    if log_errors and fallback is not None:
        _logger.info(f"Returning fallback value for {func_name}{context_str}")
    
    return fallback


def safe_execute_with_timeout(
    func: Callable,
    timeout_seconds: float,
    *args,
    fallback: Any = None,
    log_errors: bool = True,
    context: str = "",
    **kwargs
) -> Any:
    """
    Execute function with timeout protection.
    
    Args:
        func: Function to execute
        timeout_seconds: Maximum execution time in seconds
        *args: Positional arguments for the function
        fallback: Value to return on timeout or failure
        log_errors: Whether to log errors
        context: Context string for error logging
        **kwargs: Keyword arguments for the function
    
    Returns:
        Function result on success, fallback value on timeout/failure
    """
    import threading
    import time
    
    result = [None]
    exception = [None]
    
    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Function timed out
        if log_errors:
            logger = logging.getLogger(func.__module__ if hasattr(func, '__module__') else __name__)
            func_name = getattr(func, '__name__', str(func))
            context_str = f" in {context}" if context else ""
            logger.warning(f"Function {func_name} timed out after {timeout_seconds}s{context_str}")
        
        return fallback
    
    if exception[0]:
        # Function raised an exception
        if log_errors:
            logger = logging.getLogger(func.__module__ if hasattr(func, '__module__') else __name__)
            func_name = getattr(func, '__name__', str(func))
            context_str = f" in {context}" if context else ""
            logger.error(f"Function {func_name} failed{context_str}: {exception[0]}")
        
        return fallback
    
    return result[0]


def setup_error_monitoring():
    """Set up comprehensive error monitoring and handling."""
    # Set up warning filters
    warnings.filterwarnings('always', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Set up uncaught exception handler
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow keyboard interrupts to work normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger = logging.getLogger("MarkSix.UncaughtException")
        logger.critical(
            "Uncaught exception occurred",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
        
        # Try to provide helpful error message to user
        if exc_type == MemoryError:
            print("\n❌ Out of memory error. Try reducing batch size or using CPU mode.")
        elif "CUDA" in str(exc_value):
            print("\n❌ GPU error occurred. Check CUDA installation or use CPU mode.")
        else:
            print(f"\n❌ Unexpected error: {exc_value}")
        
        print("Check log files for detailed error information.")
    
    sys.excepthook = handle_uncaught_exception