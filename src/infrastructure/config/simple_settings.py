"""
Simplified configuration system for backward compatibility.
This provides the new interface while keeping existing functionality working.
"""
import warnings
from typing import Dict, Any, Optional
from pathlib import Path


def get_flat_config() -> Dict[str, Any]:
    """Get flattened configuration for backward compatibility."""
    # Import the original config to maintain compatibility
    try:
        from src.config_original import CONFIG
        return CONFIG
    except ImportError:
        # Fallback to basic config if original is missing
        import torch
        return {
            # Basic configuration that matches the original structure
            "data_path": "data/raw/Mark_Six.csv",
            "num_lotto_numbers": 49,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "batch_size": 8,
            "learning_rate": 5e-5,
            "epochs": 10,
            "latent_dim": 64,
            "model_save_path": "models/conservative_cvae_model.pth",
            "meta_learner_save_path": "models/conservative_meta_learner.pth",
            "feature_engineer_path": "models/conservative_feature_engineer.pkl"
        }


class ConfigManager:
    """Simplified config manager for compatibility."""
    
    def __init__(self, environment: str = "default"):
        self.environment = environment
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration."""
        return {"flat_config": get_flat_config()}
    
    def get_flat_config(self) -> Dict[str, Any]:
        """Get flattened config."""
        return get_flat_config()


def get_config_manager(environment: str = None) -> ConfigManager:
    """Get configuration manager instance."""
    return ConfigManager(environment or "default")


def get_config(environment: str = None) -> Dict[str, Any]:
    """Get configuration dictionary."""
    return {"flat_config": get_flat_config()}


def configure_logging(*args, **kwargs):
    """Placeholder logging configuration for compatibility."""
    warnings.warn(
        "configure_logging from config is deprecated. Use src.infrastructure.logging.configure_logging",
        DeprecationWarning,
        stacklevel=2
    )