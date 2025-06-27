"""
Base service classes for the MarkSix forecasting application layer.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.infrastructure.config import ConfigManager
from src.infrastructure.logging import get_logger


@dataclass
class ServiceResult:
    """Standardized service operation result."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseService(ABC):
    """Base class for application services."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.config = config_manager.load_config()
        self.logger = get_logger(self.__class__.__name__)
    
    def get_config(self, section: str = None) -> Any:
        """Get configuration section or full config."""
        if section:
            return self.config.get(section)
        return self.config
    
    def handle_error(self, error: Exception, context: str = "") -> ServiceResult:
        """Standard error handling."""
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.logger.error(error_msg, exc_info=True)
        return ServiceResult(success=False, error=error_msg)
    
    def success_result(self, data: Any = None, metadata: Dict[str, Any] = None) -> ServiceResult:
        """Create success result."""
        return ServiceResult(success=True, data=data, metadata=metadata)