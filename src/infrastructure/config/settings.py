"""
Configuration management system for MarkSix forecasting project.
Provides structured configuration with validation and environment support.
"""
import os
import yaml
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Graph Neural Network parameters
    node_embedding_dim: int = 32
    graph_hidden_dim: int = 64
    num_gat_layers: int = 2
    graph_projection_dim: int = 128
    
    # Temporal Context parameters
    temporal_sequence_length: int = 10
    temporal_embedding_dim: int = 16
    temporal_hidden_dim: int = 64
    temporal_lstm_layers: int = 1
    temporal_attention_heads: int = 4
    temporal_context_dim: int = 64
    temporal_projection_dim: int = 128
    trend_hidden_dim: int = 32
    trend_features: int = 16
    
    # VAE Core parameters
    latent_dim: int = 64
    prior_hidden_dim: int = 128
    
    # Decoder parameters
    decoder_hidden_dim: int = 128
    decoder_layers: int = 2
    decoder_projection_dim: int = 256
    number_generation_hidden: int = 64
    constraint_hidden_dim: int = 128
    
    # Meta-learner parameters
    combo_embedding_dim: int = 16
    pattern_hidden_dim: int = 64
    pattern_features: int = 32
    statistical_features: int = 13
    stat_hidden_dim: int = 32
    stat_features: int = 16
    meta_hidden_dim: int = 128
    integrated_features: int = 64
    meta_attention_heads: int = 4
    weight_hidden_dim: int = 32
    confidence_hidden_dim: int = 32


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 5e-5
    dropout: float = 0.1
    gradient_clip_norm: float = 0.5
    weight_decay: float = 1e-6
    
    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 0.01
    contrastive_weight: float = 0.1
    meta_learning_weight: float = 0.05
    
    # Contrastive learning
    negative_samples: int = 8
    hard_negative_ratio: float = 0.5
    negative_pool_size: int = 5000
    contrastive_margin: float = 0.3
    contrastive_temperature: float = 0.2
    
    # Training options
    use_mixed_precision: bool = False
    use_scheduler: bool = False
    scheduler_type: str = "none"
    warmup_epochs: int = 0
    ema_decay: float = 0.0
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 3
    min_improvement: float = 0.001
    
    # Monitoring
    log_interval: int = 10
    save_interval: int = 2
    validation_frequency: int = 2
    plot_latent_space: bool = False
    save_generation_samples: bool = False


@dataclass
class InferenceConfig:
    """Inference configuration parameters."""
    generation_temperature: float = 1.0
    num_generation_samples: int = 10
    top_k_sampling: int = 5
    ensemble_selection_method: str = "fixed_weights"
    evaluation_neg_samples: int = 49
    validation_generation_samples: int = 5
    
    # Ensemble weights
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "generative": 0.7,
        "temporal": 0.2,
        "i_ching": 0.1
    })


@dataclass
class DataConfig:
    """Data configuration parameters."""
    data_path: str = "data/raw/Mark_Six.csv"
    num_lotto_numbers: int = 49
    scorer_types: list = field(default_factory=lambda: ["generative", "temporal", "i_ching"])
    
    # Data augmentation (disabled for stability)
    augment_temporal_noise: float = 0.0
    augment_dropout_prob: float = 0.0


@dataclass
class SystemConfig:
    """System and infrastructure configuration."""
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    max_memory_fraction: float = 0.8
    clear_cache_frequency: int = 5
    fallback_to_cpu: bool = True
    checkpoint_on_error: bool = True
    continue_on_batch_fail: bool = True
    max_failed_batches: int = 10
    numerical_stability_eps: float = 1e-8
    max_grad_norm: float = 0.5
    meta_learner_frequency: int = 20


@dataclass
class PathConfig:
    """File path configuration."""
    model_save_path: str = "models/conservative_cvae_model.pth"
    meta_learner_save_path: str = "models/conservative_meta_learner.pth"
    feature_engineer_path: str = "models/conservative_feature_engineer.pkl"
    output_dir: str = "outputs"
    models_dir: str = "models"
    config_dir: str = "config"


class ConfigManager:
    """Manages application configuration with environment support."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, environment: str = "default"):
        self.environment = environment
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        self._config_cache = {}
        
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path."""
        base_dir = Path(__file__).parent.parent.parent.parent
        return base_dir / "config" / f"{self.environment}.yml"
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file with caching."""
        if self.environment in self._config_cache:
            return self._config_cache[self.environment]
            
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                file_config = yaml.safe_load(f) or {}
        else:
            file_config = {}
        
        # Create configuration objects
        config = {
            'model': ModelConfig(**file_config.get('model', {})),
            'training': TrainingConfig(**file_config.get('training', {})),
            'inference': InferenceConfig(**file_config.get('inference', {})),
            'data': DataConfig(**file_config.get('data', {})),
            'system': SystemConfig(**file_config.get('system', {})),
            'paths': PathConfig(**file_config.get('paths', {})),
        }
        
        self._config_cache[self.environment] = config
        return config
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """Save configuration to file."""
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass objects to dictionaries for serialization
        serializable_config = {}
        for key, value in config.items():
            if hasattr(value, '__dict__'):
                serializable_config[key] = value.__dict__
            else:
                serializable_config[key] = value
        
        with open(self.config_path, 'w') as f:
            yaml.safe_dump(serializable_config, f, default_flow_style=False, indent=2)
    
    def _resolve_device(self, device_string: str) -> str:
        """Resolve device string to valid PyTorch device."""
        if device_string == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device_string
    
    def get_flat_config(self) -> Dict[str, Any]:
        """Get flattened configuration for backward compatibility."""
        config = self.load_config()
        flat_config = {}
        
        for section_name, section_config in config.items():
            if hasattr(section_config, '__dict__'):
                flat_config.update(section_config.__dict__)
            else:
                flat_config.update(section_config)
        
        # Resolve device string
        if 'device' in flat_config:
            flat_config['device'] = self._resolve_device(flat_config['device'])
        
        return flat_config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        config = self.load_config()
        
        # Apply updates to appropriate sections
        for key, value in updates.items():
            # Find which section this key belongs to
            for section_name, section_config in config.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)
                    break
        
        self.save_config(config)
        # Clear cache to force reload
        self._config_cache.pop(self.environment, None)


def get_config_manager(environment: str = None) -> ConfigManager:
    """Get configuration manager instance."""
    env = environment or os.getenv('MARKSIX_ENV', 'default')
    return ConfigManager(environment=env)


def get_config(environment: str = None) -> Dict[str, Any]:
    """Get configuration dictionary."""
    manager = get_config_manager(environment)
    return manager.load_config()


def get_flat_config(environment: str = None) -> Dict[str, Any]:
    """Get flattened configuration for backward compatibility."""
    manager = get_config_manager(environment)
    return manager.get_flat_config()