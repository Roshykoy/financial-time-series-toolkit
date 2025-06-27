"""
Centralized logging system for MarkSix forecasting project.
Provides structured logging with different levels and formatters.
"""
import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry['data'] = record.extra_data
            
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)


class ColoredConsoleFormatter(logging.Formatter):
    """Colored console formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format: [TIME] LEVEL - MODULE.FUNCTION:LINE - MESSAGE
        formatted = (
            f"{color}[{datetime.fromtimestamp(record.created).strftime('%H:%M:%S')}] "
            f"{record.levelname:8s}{reset} - "
            f"{record.module}.{record.funcName}:{record.lineno} - "
            f"{record.getMessage()}"
        )
        
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
            
        return formatted


class LoggerManager:
    """Manages logger configuration and instances."""
    
    _loggers: Dict[str, logging.Logger] = {}
    _configured = False
    
    @classmethod
    def configure(
        cls,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        log_format: str = "console",  # "console", "json", or "simple"
        output_dir: str = "outputs"
    ) -> None:
        """Configure the logging system."""
        if cls._configured:
            return
            
        # Create output directory if needed
        if log_file:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            log_file_path = Path(output_dir) / log_file
        else:
            log_file_path = None
            
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Choose formatter
        if log_format == "json":
            console_formatter = StructuredFormatter()
        elif log_format == "console":
            console_formatter = ColoredConsoleFormatter()
        else:  # simple
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler (always uses JSON format)
        if log_file_path:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            file_handler.setFormatter(StructuredFormatter())
            root_logger.addHandler(file_handler)
        
        cls._configured = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance."""
        if not cls._configured:
            cls.configure()
            
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
            
        return cls._loggers[name]


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance (convenience function)."""
    return LoggerManager.get_logger(name)


def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: str = "console",
    output_dir: str = "outputs"
) -> None:
    """Configure the logging system (convenience function)."""
    LoggerManager.configure(log_level, log_file, log_format, output_dir)


class LogContext:
    """Context manager for adding structured data to log records."""
    
    def __init__(self, logger: logging.Logger, **extra_data):
        self.logger = logger
        self.extra_data = extra_data
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
        
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)
        
    def _log(self, level: int, message: str, **kwargs):
        extra_data = {**self.extra_data, **kwargs}
        record = self.logger.makeRecord(
            self.logger.name, level, __file__, 0, message, (), None
        )
        record.extra_data = extra_data
        self.logger.handle(record)


def log_with_context(logger: logging.Logger, **extra_data) -> LogContext:
    """Create a logging context with extra structured data."""
    return LogContext(logger, **extra_data)