"""
Legacy configuration bridge for backward compatibility.
This file provides the old CONFIG interface while using the new system.
"""
import warnings

# Issue deprecation warning
warnings.warn(
    "Direct import of CONFIG is deprecated. Use get_config() from src.infrastructure.config instead.",
    DeprecationWarning,
    stacklevel=2
)

# Try to use new system, fallback to original
try:
    from src.infrastructure.config import get_flat_config
    CONFIG = get_flat_config()
except ImportError:
    # Fallback to original config if new system not available
    try:
        from src.config_original import CONFIG
    except ImportError:
        # Last resort - basic config
        import torch
        CONFIG = {
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
