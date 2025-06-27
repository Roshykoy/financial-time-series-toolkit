"""
Configuration management for hyperparameter optimization.
Provides YAML/JSON configuration support with validation and presets.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict, field
import logging

from .base_optimizer import OptimizationConfig
from ..infrastructure.config.settings import get_config
from ..infrastructure.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SearchSpaceConfig:
    """Configuration for hyperparameter search spaces."""
    # Model architecture parameters
    learning_rate: Dict[str, Any] = field(default_factory=lambda: {
        "type": "loguniform",
        "low": 1e-6,
        "high": 1e-2
    })
    
    batch_size: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "choices": [4, 8, 16, 32, 64]
    })
    
    hidden_size: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "choices": [64, 128, 256, 512, 768]
    })
    
    num_layers: Dict[str, Any] = field(default_factory=lambda: {
        "type": "int",
        "low": 2,
        "high": 8
    })
    
    dropout: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.05,
        "high": 0.5
    })
    
    # Training parameters
    weight_decay: Dict[str, Any] = field(default_factory=lambda: {
        "type": "loguniform",
        "low": 1e-8,
        "high": 1e-3
    })
    
    gradient_clip_norm: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.1,
        "high": 2.0
    })
    
    # Loss weights
    kl_weight: Dict[str, Any] = field(default_factory=lambda: {
        "type": "loguniform",
        "low": 1e-4,
        "high": 1e-1
    })
    
    contrastive_weight: Dict[str, Any] = field(default_factory=lambda: {
        "type": "uniform",
        "low": 0.01,
        "high": 0.5
    })
    
    # Model-specific parameters
    latent_dim: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "choices": [32, 64, 128, 256]
    })
    
    negative_samples: Dict[str, Any] = field(default_factory=lambda: {
        "type": "choice",
        "choices": [4, 8, 16, 32]
    })


@dataclass
class OptimizationPreset:
    """Predefined optimization presets."""
    name: str
    description: str
    optimization_config: OptimizationConfig
    search_space: Dict[str, Any]
    algorithm: str = "random_search"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert preset to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'optimization_config': asdict(self.optimization_config),
            'search_space': self.search_space,
            'algorithm': self.algorithm
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationPreset':
        """Create preset from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            optimization_config=OptimizationConfig(**data['optimization_config']),
            search_space=data['search_space'],
            algorithm=data.get('algorithm', 'random_search')
        )


class OptimizationConfigManager:
    """Manages optimization configurations, presets, and search spaces."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config/optimization")
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration files
        self.presets_file = self.config_dir / "presets.yml"
        self.search_spaces_file = self.config_dir / "search_spaces.yml"
        self.default_config_file = self.config_dir / "default.yml"
        
        # Load configurations
        self.presets = self._load_presets()
        self.search_spaces = self._load_search_spaces()
        self.default_config = self._load_default_config()
        
        logger.info(f"Configuration manager initialized with {len(self.presets)} presets")
    
    def _load_presets(self) -> Dict[str, OptimizationPreset]:
        """Load optimization presets from file."""
        if not self.presets_file.exists():
            presets = self._create_default_presets()
            self._save_presets(presets)
            return presets
        
        try:
            with open(self.presets_file, 'r') as f:
                data = yaml.safe_load(f)
            
            presets = {}
            for preset_data in data.get('presets', []):
                preset = OptimizationPreset.from_dict(preset_data)
                presets[preset.name] = preset
            
            return presets
            
        except Exception as e:
            logger.error(f"Error loading presets: {e}")
            return self._create_default_presets()
    
    def _create_default_presets(self) -> Dict[str, OptimizationPreset]:
        """Create default optimization presets."""
        search_space_config = SearchSpaceConfig()
        basic_search_space = self._extract_basic_search_space(search_space_config)
        advanced_search_space = self._extract_advanced_search_space(search_space_config)
        
        presets = {
            "quick_test": OptimizationPreset(
                name="quick_test",
                description="Quick test optimization (5 trials, 2 epochs each)",
                optimization_config=OptimizationConfig(
                    max_trials=5,
                    max_duration_hours=0.5,
                    early_stopping=False,
                    trial_timeout_minutes=10.0
                ),
                search_space=basic_search_space,
                algorithm="random_search"
            ),
            
            "fast_search": OptimizationPreset(
                name="fast_search",
                description="Fast random search (20 trials, 3 epochs each)",
                optimization_config=OptimizationConfig(
                    max_trials=20,
                    max_duration_hours=2.0,
                    early_stopping=True,
                    early_stopping_patience=5,
                    trial_timeout_minutes=15.0
                ),
                search_space=basic_search_space,
                algorithm="random_search"
            ),
            
            "balanced_search": OptimizationPreset(
                name="balanced_search",
                description="Balanced Bayesian optimization (30 trials, 5 epochs each)",
                optimization_config=OptimizationConfig(
                    max_trials=30,
                    max_duration_hours=4.0,
                    early_stopping=True,
                    early_stopping_patience=8,
                    trial_timeout_minutes=30.0,
                    parallel_jobs=2
                ),
                search_space=advanced_search_space,
                algorithm="bayesian"
            ),
            
            "thorough_search": OptimizationPreset(
                name="thorough_search",
                description="Thorough Optuna optimization (50 trials, 8 epochs each)",
                optimization_config=OptimizationConfig(
                    max_trials=50,
                    max_duration_hours=8.0,
                    early_stopping=True,
                    early_stopping_patience=10,
                    trial_timeout_minutes=60.0,
                    parallel_jobs=2
                ),
                search_space=advanced_search_space,
                algorithm="optuna"
            ),
            
            "grid_exploration": OptimizationPreset(
                name="grid_exploration",
                description="Grid search for systematic exploration",
                optimization_config=OptimizationConfig(
                    max_trials=100,
                    max_duration_hours=6.0,
                    early_stopping=False,
                    trial_timeout_minutes=45.0
                ),
                search_space=basic_search_space,
                algorithm="grid_search"
            )
        }
        
        return presets
    
    def _extract_basic_search_space(self, config: SearchSpaceConfig) -> Dict[str, Any]:
        """Extract basic search space parameters."""
        return {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'hidden_size': config.hidden_size,
            'dropout': config.dropout
        }
    
    def _extract_advanced_search_space(self, config: SearchSpaceConfig) -> Dict[str, Any]:
        """Extract advanced search space parameters."""
        return {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_layers,
            'dropout': config.dropout,
            'weight_decay': config.weight_decay,
            'gradient_clip_norm': config.gradient_clip_norm,
            'kl_weight': config.kl_weight,
            'contrastive_weight': config.contrastive_weight,
            'latent_dim': config.latent_dim,
            'negative_samples': config.negative_samples
        }
    
    def _save_presets(self, presets: Dict[str, OptimizationPreset]) -> None:
        """Save presets to file."""
        try:
            data = {
                'presets': [preset.to_dict() for preset in presets.values()]
            }
            
            with open(self.presets_file, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved {len(presets)} presets to {self.presets_file}")
            
        except Exception as e:
            logger.error(f"Error saving presets: {e}")
    
    def _load_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Load search space configurations."""
        if not self.search_spaces_file.exists():
            search_spaces = self._create_default_search_spaces()
            self._save_search_spaces(search_spaces)
            return search_spaces
        
        try:
            with open(self.search_spaces_file, 'r') as f:
                return yaml.safe_load(f) or {}
            
        except Exception as e:
            logger.error(f"Error loading search spaces: {e}")
            return self._create_default_search_spaces()
    
    def _create_default_search_spaces(self) -> Dict[str, Dict[str, Any]]:
        """Create default search space configurations."""
        config = SearchSpaceConfig()
        
        return {
            'basic': self._extract_basic_search_space(config),
            'advanced': self._extract_advanced_search_space(config),
            'model_architecture': {
                'hidden_size': config.hidden_size,
                'num_layers': config.num_layers,
                'latent_dim': config.latent_dim,
                'dropout': config.dropout
            },
            'training_params': {
                'learning_rate': config.learning_rate,
                'batch_size': config.batch_size,
                'weight_decay': config.weight_decay,
                'gradient_clip_norm': config.gradient_clip_norm
            },
            'loss_weights': {
                'kl_weight': config.kl_weight,
                'contrastive_weight': config.contrastive_weight
            }
        }
    
    def _save_search_spaces(self, search_spaces: Dict[str, Dict[str, Any]]) -> None:
        """Save search spaces to file."""
        try:
            with open(self.search_spaces_file, 'w') as f:
                yaml.safe_dump(search_spaces, f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved search spaces to {self.search_spaces_file}")
            
        except Exception as e:
            logger.error(f"Error saving search spaces: {e}")
    
    def _load_default_config(self) -> OptimizationConfig:
        """Load default optimization configuration."""
        if not self.default_config_file.exists():
            config = OptimizationConfig()
            self._save_default_config(config)
            return config
        
        try:
            with open(self.default_config_file, 'r') as f:
                data = yaml.safe_load(f)
            
            return OptimizationConfig(**data)
            
        except Exception as e:
            logger.error(f"Error loading default config: {e}")
            return OptimizationConfig()
    
    def _save_default_config(self, config: OptimizationConfig) -> None:
        """Save default configuration to file."""
        try:
            with open(self.default_config_file, 'w') as f:
                yaml.safe_dump(asdict(config), f, default_flow_style=False, indent=2)
            
            logger.info(f"Saved default config to {self.default_config_file}")
            
        except Exception as e:
            logger.error(f"Error saving default config: {e}")
    
    def get_preset(self, name: str) -> OptimizationPreset:
        """Get optimization preset by name."""
        if name not in self.presets:
            available = list(self.presets.keys())
            raise ValueError(f"Preset '{name}' not found. Available: {available}")
        
        return self.presets[name]
    
    def get_search_space(self, name: str) -> Dict[str, Any]:
        """Get search space configuration by name."""
        if name not in self.search_spaces:
            available = list(self.search_spaces.keys())
            raise ValueError(f"Search space '{name}' not found. Available: {available}")
        
        return self.search_spaces[name]
    
    def list_presets(self) -> List[str]:
        """List available preset names."""
        return list(self.presets.keys())
    
    def list_search_spaces(self) -> List[str]:
        """List available search space names."""
        return list(self.search_spaces.keys())
    
    def create_custom_preset(
        self,
        name: str,
        description: str,
        optimization_config: OptimizationConfig,
        search_space: Dict[str, Any],
        algorithm: str = "random_search"
    ) -> OptimizationPreset:
        """Create and save a custom preset."""
        preset = OptimizationPreset(
            name=name,
            description=description,
            optimization_config=optimization_config,
            search_space=search_space,
            algorithm=algorithm
        )
        
        self.presets[name] = preset
        self._save_presets(self.presets)
        
        logger.info(f"Created custom preset: {name}")
        return preset
    
    def create_custom_search_space(self, name: str, search_space: Dict[str, Any]) -> None:
        """Create and save a custom search space."""
        self.search_spaces[name] = search_space
        self._save_search_spaces(self.search_spaces)
        
        logger.info(f"Created custom search space: {name}")
    
    def validate_search_space(self, search_space: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate search space configuration."""
        errors = []
        
        for param_name, param_config in search_space.items():
            if not isinstance(param_name, str):
                errors.append(f"Parameter name must be string: {param_name}")
                continue
            
            if isinstance(param_config, (list, tuple)):
                if len(param_config) == 0:
                    errors.append(f"Parameter {param_name}: empty choices list")
                continue
            
            if not isinstance(param_config, dict):
                errors.append(f"Parameter {param_name}: config must be dict or list")
                continue
            
            param_type = param_config.get('type')
            if not param_type:
                errors.append(f"Parameter {param_name}: missing 'type' field")
                continue
            
            # Validate type-specific fields
            if param_type == 'choice':
                if 'choices' not in param_config:
                    errors.append(f"Parameter {param_name}: missing 'choices' for choice type")
                elif not isinstance(param_config['choices'], (list, tuple)):
                    errors.append(f"Parameter {param_name}: 'choices' must be list or tuple")
                elif len(param_config['choices']) == 0:
                    errors.append(f"Parameter {param_name}: empty choices list")
            
            elif param_type in ['uniform', 'loguniform']:
                for field in ['low', 'high']:
                    if field not in param_config:
                        errors.append(f"Parameter {param_name}: missing '{field}' for {param_type} type")
                    elif not isinstance(param_config[field], (int, float)):
                        errors.append(f"Parameter {param_name}: '{field}' must be numeric")
                
                if 'low' in param_config and 'high' in param_config:
                    if param_config['low'] >= param_config['high']:
                        errors.append(f"Parameter {param_name}: 'low' must be less than 'high'")
                    
                    if param_type == 'loguniform' and param_config['low'] <= 0:
                        errors.append(f"Parameter {param_name}: 'low' must be positive for loguniform")
            
            elif param_type == 'int':
                for field in ['low', 'high']:
                    if field not in param_config:
                        errors.append(f"Parameter {param_name}: missing '{field}' for int type")
                    elif not isinstance(param_config[field], int):
                        errors.append(f"Parameter {param_name}: '{field}' must be integer")
                
                if 'low' in param_config and 'high' in param_config:
                    if param_config['low'] >= param_config['high']:
                        errors.append(f"Parameter {param_name}: 'low' must be less than 'high'")
            
            elif param_type == 'bool':
                # No additional validation needed for bool
                pass
            
            else:
                errors.append(f"Parameter {param_name}: unknown type '{param_type}'")
        
        return len(errors) == 0, errors
    
    def export_config(self, filepath: Path, format: str = "yaml") -> None:
        """Export all configurations to file."""
        data = {
            'presets': [preset.to_dict() for preset in self.presets.values()],
            'search_spaces': self.search_spaces,
            'default_config': asdict(self.default_config)
        }
        
        if format.lower() == "yaml":
            with open(filepath, 'w') as f:
                yaml.safe_dump(data, f, default_flow_style=False, indent=2)
        elif format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported configuration to {filepath}")
    
    def import_config(self, filepath: Path) -> None:
        """Import configurations from file."""
        if filepath.suffix.lower() in ['.yml', '.yaml']:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        # Import presets
        if 'presets' in data:
            for preset_data in data['presets']:
                preset = OptimizationPreset.from_dict(preset_data)
                self.presets[preset.name] = preset
        
        # Import search spaces
        if 'search_spaces' in data:
            self.search_spaces.update(data['search_spaces'])
        
        # Import default config
        if 'default_config' in data:
            self.default_config = OptimizationConfig(**data['default_config'])
        
        # Save updated configurations
        self._save_presets(self.presets)
        self._save_search_spaces(self.search_spaces)
        self._save_default_config(self.default_config)
        
        logger.info(f"Imported configuration from {filepath}")


def create_config_manager(config_dir: Optional[Path] = None) -> OptimizationConfigManager:
    """Factory function to create configuration manager."""
    return OptimizationConfigManager(config_dir)