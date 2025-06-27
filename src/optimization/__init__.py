"""
Comprehensive hyperparameter optimization module for MarkSix forecasting.

This module provides intelligent hyperparameter optimization with hardware-aware
resource management, multiple optimization algorithms, and seamless integration
with the existing training pipeline.
"""

__version__ = "1.0.0"

# Lazy imports to avoid immediate dependency loading
def _get_base_optimizer():
    from .base_optimizer import BaseHyperparameterOptimizer
    return BaseHyperparameterOptimizer

def _get_algorithms():
    from .algorithms import (
        GridSearchOptimizer,
        RandomSearchOptimizer, 
        BayesianOptimizer,
        OptunaOptimizer
    )
    return GridSearchOptimizer, RandomSearchOptimizer, BayesianOptimizer, OptunaOptimizer

def _get_hardware_manager():
    from .hardware_manager import HardwareResourceManager
    return HardwareResourceManager

def _get_config_manager():
    from .config_manager import OptimizationConfigManager
    return OptimizationConfigManager

def _get_monitoring():
    from .monitoring import OptimizationMonitor, OptimizationVisualizer
    return OptimizationMonitor, OptimizationVisualizer

def _get_integration():
    from .integration import ModelTrainingInterface
    return ModelTrainingInterface

def _get_utils():
    from .utils import OptimizationUtils
    return OptimizationUtils

# Public API with lazy loading
def __getattr__(name):
    """Lazy loading of module components."""
    if name == 'BaseHyperparameterOptimizer':
        return _get_base_optimizer()
    elif name in ['GridSearchOptimizer', 'RandomSearchOptimizer', 'BayesianOptimizer', 'OptunaOptimizer']:
        algorithms = _get_algorithms()
        mapping = {
            'GridSearchOptimizer': algorithms[0],
            'RandomSearchOptimizer': algorithms[1],
            'BayesianOptimizer': algorithms[2],
            'OptunaOptimizer': algorithms[3]
        }
        return mapping[name]
    elif name == 'HardwareResourceManager':
        return _get_hardware_manager()
    elif name == 'OptimizationConfigManager':
        return _get_config_manager()
    elif name in ['OptimizationMonitor', 'OptimizationVisualizer']:
        monitoring = _get_monitoring()
        mapping = {
            'OptimizationMonitor': monitoring[0],
            'OptimizationVisualizer': monitoring[1]
        }
        return mapping[name]
    elif name == 'ModelTrainingInterface':
        return _get_integration()
    elif name == 'OptimizationUtils':
        return _get_utils()
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'BaseHyperparameterOptimizer',
    'GridSearchOptimizer',
    'RandomSearchOptimizer',
    'BayesianOptimizer', 
    'OptunaOptimizer',
    'HardwareResourceManager',
    'OptimizationConfigManager',
    'OptimizationMonitor',
    'OptimizationVisualizer',
    'ModelTrainingInterface',
    'OptimizationUtils'
]