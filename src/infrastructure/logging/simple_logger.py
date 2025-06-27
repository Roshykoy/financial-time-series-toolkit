"""
Simplified logging system for backward compatibility.
"""
import logging
import sys
from typing import Optional


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "console",
    output_dir: str = "outputs"
) -> None:
    """Configure basic logging for compatibility."""
    
    # Set up basic logging
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            file_handler = logging.FileHandler(f"{output_dir}/{log_file}")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception:
            pass  # Ignore file logging errors for compatibility


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Context manager stub for compatibility
class LogContext:
    def __init__(self, logger, **kwargs):
        self.logger = logger
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass
        
    def info(self, msg, **kwargs):
        self.logger.info(msg)
        
    def debug(self, msg, **kwargs):
        self.logger.debug(msg)
        
    def warning(self, msg, **kwargs):
        self.logger.warning(msg)
        
    def error(self, msg, **kwargs):
        self.logger.error(msg)


def log_with_context(logger, **kwargs):
    """Create logging context for compatibility."""
    return LogContext(logger, **kwargs)