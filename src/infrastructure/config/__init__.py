"""
Configuration management module for MarkSix forecasting system.
"""
try:
    # Try to import the full configuration system
    from .settings import (
        ModelConfig,
        TrainingConfig,
        InferenceConfig,
        DataConfig,
        SystemConfig,
        PathConfig,
        ConfigManager,
        get_config_manager,
        get_config,
        get_flat_config
    )
except ImportError:
    # Fallback to simplified system for compatibility
    from .simple_settings import (
        ConfigManager,
        get_config_manager,
        get_config,
        get_flat_config,
        configure_logging
    )
    
    # Create dummy classes for compatibility
    class ModelConfig: pass
    class TrainingConfig: pass
    class InferenceConfig: pass
    class DataConfig: pass
    class SystemConfig: pass
    class PathConfig: pass

__all__ = [
    'ModelConfig',
    'TrainingConfig', 
    'InferenceConfig',
    'DataConfig',
    'SystemConfig',
    'PathConfig',
    'ConfigManager',
    'get_config_manager',
    'get_config',
    'get_flat_config'
]