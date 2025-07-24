"""
Integration with existing training pipeline and model architecture.
Provides seamless interface between optimization and model training.
"""

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Callable, List, Tuple
from pathlib import Path
import time
import gc
import logging

from .hardware_manager import HardwareResourceManager
from ..infrastructure.logging.logger import get_logger
from ..utils.error_handling import safe_execute
from ..utils.performance_monitor import PerformanceMonitor

# Import existing components
from ..cvae_model import ConditionalVAE
from ..meta_learner import AttentionMetaLearner  
from ..cvae_engine import train_one_epoch_cvae, evaluate_cvae
from ..cvae_data_loader import create_cvae_data_loaders
from ..feature_engineering import FeatureEngineer
from ..config_legacy import CONFIG

logger = get_logger(__name__)


class ModelTrainingInterface:
    """Interface for training models with hyperparameter optimization."""
    
    def __init__(
        self,
        data_path: str,
        hardware_manager: Optional[HardwareResourceManager] = None,
        base_config: Optional[Dict[str, Any]] = None
    ):
        self.data_path = data_path
        self.hardware_manager = hardware_manager
        self.base_config = base_config or CONFIG
        
        # Components
        self.feature_engineer = None
        self.data_loaders = None
        self.device = None
        
        # Cache for data loading
        self._data_cache = {}
        self._feature_cache = {}
        
        logger.info("Model training interface initialized")
    
    def prepare_data(self, config: Dict[str, Any]) -> Tuple[Any, Any, Any]:
        """Prepare data for training with given configuration."""
        cache_key = self._get_cache_key(config)
        
        if cache_key in self._data_cache:
            logger.debug("Using cached data")
            return self._data_cache[cache_key]
        
        try:
            # Load and preprocess data
            df = self._load_data()
            
            # Initialize feature engineer
            if self.feature_engineer is None:
                self.feature_engineer = FeatureEngineer()
                self.feature_engineer.fit(df)
            
            # Create data loaders with current config
            train_loader, val_loader = create_cvae_data_loaders(
                df, self.feature_engineer, config
            )
            
            # For optimization, we use validation as test set
            test_loader = val_loader
            
            # Cache the results
            data_tuple = (train_loader, val_loader, test_loader)
            self._data_cache[cache_key] = data_tuple
            
            return data_tuple
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise
    
    def _load_data(self) -> pd.DataFrame:
        """Load Mark Six data."""
        try:
            col_names = [
                'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
                'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
                'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
                '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
                'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
                'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
                'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
            ]
            
            df = pd.read_csv(self.data_path, header=None, skiprows=33, names=col_names)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} records from {self.data_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {self.data_path}: {e}")
            raise
    
    def _get_cache_key(self, config: Dict[str, Any]) -> str:
        """Generate cache key for configuration."""
        # Include only data-relevant parameters
        data_params = {
            'batch_size': config.get('batch_size', 32),
            'negative_samples': config.get('negative_samples', 16),
            'validation_split': config.get('validation_split', 0.2)
        }
        return str(hash(frozenset(data_params.items())))
    
    def create_model(self, config: Dict[str, Any]) -> Tuple[ConditionalVAE, AttentionMetaLearner]:
        """Create model instances with given configuration."""
        try:
            # Ensure device is set
            device = self._get_device(config)
            
            # Create CVAE model
            cvae_model = ConditionalVAE(config).to(device)
            
            # Create meta-learner
            meta_learner = AttentionMetaLearner(config).to(device)
            
            logger.debug(f"Created models on device: {device}")
            return cvae_model, meta_learner
            
        except Exception as e:
            logger.error(f"Error creating models: {e}")
            raise
    
    def _get_device(self, config: Dict[str, Any]) -> torch.device:
        """Get appropriate device for training."""
        if self.device is not None:
            return self.device
        
        if config.get('device') == 'cpu':
            self.device = torch.device('cpu')
        elif torch.cuda.is_available() and config.get('device') != 'cpu':
            self.device = torch.device('cuda')
            # Set memory fraction if specified
            if 'max_memory_fraction' in config:
                torch.cuda.set_per_process_memory_fraction(config['max_memory_fraction'])
        else:
            self.device = torch.device('cpu')
        
        logger.info(f"Using device: {self.device}")
        return self.device
    
    def train_model(self, config: Dict[str, Any]) -> float:
        """Train model with given configuration and return validation score."""
        start_time = time.time()
        
        try:
            # Prepare data
            train_loader, val_loader, test_loader = self.prepare_data(config)
            
            # Create models
            cvae_model, meta_learner = self.create_model(config)
            device = self._get_device(config)
            
            # Create optimizers dictionary
            optimizers = {
                'cvae': self._create_optimizer(cvae_model, config),
                'meta': self._create_optimizer(meta_learner, config)
            }
            
            # Training configuration
            epochs = config.get('epochs', 5)  # Reduced for optimization
            best_val_loss = float('inf')
            patience_counter = 0
            patience = config.get('early_stopping_patience', 3)
            
            # Training loop
            for epoch in range(epochs):
                # Check for early termination
                if self._should_stop_training(start_time, config):
                    logger.info(f"Training stopped early at epoch {epoch}")
                    break
                
                # Train for one epoch
                train_losses = train_one_epoch_cvae(
                    cvae_model, meta_learner, train_loader,
                    optimizers, device, config, epoch
                )
                
                # Extract main loss from dictionary
                train_loss = train_losses.get('total_cvae_loss', 0.0)
                
                # Check for invalid training loss
                if not np.isfinite(train_loss) or np.isnan(train_loss):
                    logger.warning(f"Invalid training loss: {train_loss}, stopping training")
                    break
                
                # Validate
                val_loss, val_metrics = evaluate_cvae(
                    cvae_model, meta_learner, val_loader, device, config
                )
                
                # Check for invalid validation loss
                if not np.isfinite(val_loss) or np.isnan(val_loss):
                    logger.warning(f"Invalid validation loss: {val_loss}, stopping training")
                    break
                
                logger.debug(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.debug(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Resource cleanup
                if (epoch + 1) % 2 == 0:
                    self._cleanup_resources()
            
            # Return negative validation loss as score (higher is better)
            if np.isfinite(best_val_loss) and not np.isnan(best_val_loss):
                score = -best_val_loss
                # Clamp extremely large scores to prevent overflow
                score = max(score, -1e6)
            else:
                logger.warning(f"Invalid best validation loss: {best_val_loss}, returning minimum score")
                score = -1e6
            
            # Cleanup
            self._cleanup_resources()
            
            duration = time.time() - start_time
            logger.debug(f"Training completed in {duration:.1f}s, score: {score:.6f}")
            
            return score
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            self._cleanup_resources()
            raise
    
    def _create_optimizer(self, model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
        """Create optimizer for model."""
        optimizer_type = config.get('optimizer_type', 'adamw').lower()
        lr = config.get('learning_rate', 5e-5)
        weight_decay = config.get('weight_decay', 1e-6)
        
        if optimizer_type == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            momentum = config.get('momentum', 0.9)
            return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        else:
            logger.warning(f"Unknown optimizer: {optimizer_type}, using AdamW")
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def _should_stop_training(self, start_time: float, config: Dict[str, Any]) -> bool:
        """Check if training should be stopped early."""
        # Check time limit - allow much longer for thorough search
        max_duration = config.get('max_training_duration_minutes', 180) * 60  # 3 hours default
        elapsed = time.time() - start_time
        
        if elapsed > max_duration:
            logger.warning(f"Training time limit exceeded: {elapsed/60:.1f} minutes")
            return True
        
        # Check resource constraints if hardware manager available
        if self.hardware_manager:
            is_ok, issues = self.hardware_manager.check_resource_constraints()
            if not is_ok:
                logger.warning(f"Resource constraints violated: {issues}")
                return True
        
        return False
    
    def _cleanup_resources(self) -> None:
        """Clean up system resources."""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Error during resource cleanup: {e}")
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration parameters."""
        errors = []
        warnings = []
        
        # Check required parameters
        required_params = ['learning_rate', 'batch_size', 'epochs']
        for param in required_params:
            if param not in config:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate parameter ranges
        if 'learning_rate' in config:
            lr = config['learning_rate']
            if not (1e-7 <= lr <= 1e-1):
                warnings.append(f"Learning rate {lr} is outside typical range [1e-7, 1e-1]")
        
        if 'batch_size' in config:
            batch_size = config['batch_size']
            if not (1 <= batch_size <= 512):
                errors.append(f"Batch size {batch_size} is outside valid range [1, 512]")
        
        if 'epochs' in config:
            epochs = config['epochs']
            if not (1 <= epochs <= 100):
                warnings.append(f"Epochs {epochs} is outside typical range [1, 100]")
        
        if 'dropout' in config:
            dropout = config['dropout']
            if not (0.0 <= dropout <= 0.9):
                errors.append(f"Dropout {dropout} is outside valid range [0.0, 0.9]")
        
        # Hardware-specific validation
        if self.hardware_manager:
            optimized_config = self.hardware_manager.optimize_for_hardware(config)
            if optimized_config != config:
                warnings.append("Configuration was adjusted for hardware constraints")
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
        
        return len(errors) == 0, errors
    
    def estimate_training_time(self, config: Dict[str, Any]) -> float:
        """Estimate training time in seconds."""
        base_time = 120.0  # 2 minutes base time
        
        # Adjust for epochs
        epochs = config.get('epochs', 5)
        base_time *= epochs / 5.0
        
        # Adjust for batch size
        batch_size = config.get('batch_size', 32)
        base_time *= (32.0 / batch_size) ** 0.5
        
        # Adjust for model complexity
        hidden_size = config.get('hidden_size', 256)
        complexity_factor = (hidden_size / 256.0) ** 1.2
        base_time *= complexity_factor
        
        latent_dim = config.get('latent_dim', 64)
        latent_factor = (latent_dim / 64.0) ** 0.8
        base_time *= latent_factor
        
        # Adjust for device
        device = config.get('device', 'auto')
        if device == 'cpu' or (device == 'auto' and not torch.cuda.is_available()):
            base_time *= 2.5  # CPU is slower
        
        return base_time
    
    def get_search_space(self) -> Dict[str, Any]:
        """Get default search space for this model."""
        return {
            # Core training parameters
            'learning_rate': {
                'type': 'loguniform',
                'low': 1e-6,
                'high': 1e-2
            },
            'batch_size': {
                'type': 'choice',
                'choices': [4, 8, 16, 32]
            },
            'epochs': {
                'type': 'int',
                'low': 3,
                'high': 10
            },
            
            # Model architecture
            'latent_dim': {
                'type': 'choice',
                'choices': [32, 64, 128, 256]
            },
            'dropout': {
                'type': 'uniform',
                'low': 0.05,
                'high': 0.3
            },
            
            # Loss weights
            'kl_weight': {
                'type': 'loguniform',
                'low': 1e-4,
                'high': 1e-1
            },
            'contrastive_weight': {
                'type': 'uniform',
                'low': 0.01,
                'high': 0.5
            },
            
            # Training hyperparameters
            'weight_decay': {
                'type': 'loguniform',
                'low': 1e-8,
                'high': 1e-3
            },
            'gradient_clip_norm': {
                'type': 'uniform',
                'low': 0.1,
                'high': 2.0
            },
            
            # Data parameters
            'negative_samples': {
                'type': 'choice',
                'choices': [4, 8, 16, 32]
            }
        }
    
    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._data_cache.clear()
        self._feature_cache.clear()
        self._cleanup_resources()
        logger.info("Cleared training interface caches")


class OptimizationObjective:
    """Objective function wrapper for hyperparameter optimization."""
    
    def __init__(
        self,
        training_interface: ModelTrainingInterface,
        base_config: Dict[str, Any],
        validation_strategy: str = "holdout"
    ):
        self.training_interface = training_interface
        self.base_config = base_config
        self.validation_strategy = validation_strategy
        
        # Performance tracking
        self.call_count = 0
        self.total_time = 0.0
        self.best_score = float('-inf')
        
        logger.info(f"Optimization objective initialized with {validation_strategy} validation")
    
    def __call__(self, parameters: Dict[str, Any]) -> float:
        """Evaluate objective function with given parameters."""
        self.call_count += 1
        start_time = time.time()
        
        try:
            # Merge parameters with base config
            config = self.base_config.copy()
            config.update(parameters)
            
            # Validate configuration
            is_valid, errors = self.training_interface.validate_config(config)
            if not is_valid:
                logger.error(f"Invalid configuration: {errors}")
                return float('-inf')
            
            # Train and evaluate
            if self.validation_strategy == "holdout":
                score = self.training_interface.train_model(config)
            elif self.validation_strategy == "cross_validation":
                score = self._cross_validation_score(config)
            else:
                raise ValueError(f"Unknown validation strategy: {self.validation_strategy}")
            
            # Update tracking
            duration = time.time() - start_time
            self.total_time += duration
            
            if score > self.best_score:
                self.best_score = score
                logger.info(f"New best score: {score:.6f} (call {self.call_count})")
            
            logger.debug(f"Objective evaluation {self.call_count}: score={score:.6f}, time={duration:.1f}s")
            
            return score
            
        except Exception as e:
            logger.error(f"Objective evaluation failed: {e}")
            return float('-inf')
    
    def _cross_validation_score(self, config: Dict[str, Any]) -> float:
        """Perform cross-validation scoring (simplified implementation)."""
        # For time series data, use temporal splits
        scores = []
        
        try:
            # Simple 3-fold temporal validation
            for fold in range(3):
                fold_config = config.copy()
                fold_config['cv_fold'] = fold
                score = self.training_interface.train_model(fold_config)
                scores.append(score)
            
            return np.mean(scores)
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return float('-inf')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get objective function statistics."""
        return {
            'call_count': self.call_count,
            'total_time_seconds': self.total_time,
            'average_time_seconds': self.total_time / max(1, self.call_count),
            'best_score': self.best_score
        }


def create_training_interface(
    data_path: str,
    hardware_manager: Optional[HardwareResourceManager] = None,
    base_config: Optional[Dict[str, Any]] = None
) -> ModelTrainingInterface:
    """Factory function to create training interface."""
    return ModelTrainingInterface(data_path, hardware_manager, base_config)


def create_objective_function(
    training_interface: ModelTrainingInterface,
    base_config: Dict[str, Any],
    validation_strategy: str = "holdout"
) -> OptimizationObjective:
    """Factory function to create objective function."""
    return OptimizationObjective(training_interface, base_config, validation_strategy)