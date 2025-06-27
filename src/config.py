"""
Configuration module - migrated to new structure.
Import from src.infrastructure.config for new code.
"""
from src.config_legacy import CONFIG

# Re-export for compatibility
__all__ = ['CONFIG']
