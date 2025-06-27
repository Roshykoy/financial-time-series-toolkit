"""
Logging infrastructure for MarkSix forecasting system.
"""
try:
    # Try to import the full logging system
    from .logger import (
        LoggerManager,
        StructuredFormatter,
        ColoredConsoleFormatter,
        LogContext,
        get_logger,
        configure_logging,
        log_with_context
    )
except ImportError:
    # Fallback to simplified logging for compatibility
    from .simple_logger import (
        configure_logging,
        get_logger,
        LogContext,
        log_with_context
    )
    
    # Create dummy classes for compatibility
    class LoggerManager: pass
    class StructuredFormatter: pass
    class ColoredConsoleFormatter: pass

__all__ = [
    'LoggerManager',
    'StructuredFormatter', 
    'ColoredConsoleFormatter',
    'LogContext',
    'get_logger',
    'configure_logging',
    'log_with_context'
]