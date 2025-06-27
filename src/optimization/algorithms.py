"""
Hyperparameter optimization algorithms.
Implements Grid Search, Random Search, Bayesian Optimization, and Optuna integration.
"""

import random
import itertools
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import logging

from .base_optimizer import BaseHyperparameterOptimizer, OptimizationConfig, OptimizationTrial
from ..infrastructure.logging.logger import get_logger

logger = get_logger(__name__)

# Optional imports for advanced algorithms
try:
    import optuna
    from optuna.samplers import TPESampler, RandomSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
    from scipy.optimize import minimize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Scikit-learn not available for Gaussian Process optimization")


class SearchSpaceHandler:
    """Handles different types of search space definitions."""
    
    @staticmethod
    def validate_search_space(search_space: Dict[str, Any]) -> bool:
        """Validate search space format."""
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                if 'type' not in param_config:
                    return False
                param_type = param_config['type']
                
                if param_type == 'choice' and 'choices' not in param_config:
                    return False
                elif param_type in ['uniform', 'loguniform'] and ('low' not in param_config or 'high' not in param_config):
                    return False
                elif param_type == 'int' and ('low' not in param_config or 'high' not in param_config):
                    return False
            elif not isinstance(param_config, (list, tuple)):
                return False
        
        return True
    
    @staticmethod
    def sample_parameter(param_name: str, param_config: Union[Dict, List, Tuple]) -> Any:
        """Sample a parameter value from its configuration."""
        if isinstance(param_config, (list, tuple)):
            return random.choice(param_config)
        
        param_type = param_config.get('type', 'choice')
        
        if param_type == 'choice':
            return random.choice(param_config['choices'])
        elif param_type == 'uniform':
            return random.uniform(param_config['low'], param_config['high'])
        elif param_type == 'loguniform':
            return np.exp(random.uniform(np.log(param_config['low']), np.log(param_config['high'])))
        elif param_type == 'int':
            return random.randint(param_config['low'], param_config['high'])
        elif param_type == 'bool':
            return random.choice([True, False])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    @staticmethod
    def get_grid_values(param_config: Union[Dict, List, Tuple], num_values: int = 5) -> List[Any]:
        """Get grid values for a parameter."""
        if isinstance(param_config, (list, tuple)):
            return list(param_config)
        
        param_type = param_config.get('type', 'choice')
        
        if param_type == 'choice':
            return param_config['choices']
        elif param_type == 'uniform':
            return list(np.linspace(param_config['low'], param_config['high'], num_values))
        elif param_type == 'loguniform':
            log_values = np.linspace(np.log(param_config['low']), np.log(param_config['high']), num_values)
            return list(np.exp(log_values))
        elif param_type == 'int':
            return list(range(param_config['low'], param_config['high'] + 1))
        elif param_type == 'bool':
            return [True, False]
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")


class GridSearchOptimizer(BaseHyperparameterOptimizer):
    """Grid Search hyperparameter optimization."""
    
    def __init__(self, *args, grid_size: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_size = grid_size or 5
        self.parameter_grid = self._generate_parameter_grid()
        self.grid_index = 0
        
        # Update max_trials to match grid size if not specified
        if len(self.parameter_grid) < self.config.max_trials:
            self.config.max_trials = len(self.parameter_grid)
        
        logger.info(f"Grid search initialized with {len(self.parameter_grid)} combinations")
    
    def _generate_parameter_grid(self) -> List[Dict[str, Any]]:
        """Generate the complete parameter grid."""
        if not SearchSpaceHandler.validate_search_space(self.search_space):
            raise ValueError("Invalid search space format")
        
        param_names = list(self.search_space.keys())
        param_values = []
        
        for param_name in param_names:
            param_config = self.search_space[param_name]
            values = SearchSpaceHandler.get_grid_values(param_config, self.grid_size)
            param_values.append(values)
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        parameter_grid = []
        for combination in combinations:
            param_dict = dict(zip(param_names, combination))
            parameter_grid.append(param_dict)
        
        # Shuffle for randomness
        random.shuffle(parameter_grid)
        
        return parameter_grid
    
    def _generate_trial_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Generate parameters for the next trial."""
        if self.grid_index >= len(self.parameter_grid):
            # If we've exhausted the grid, generate random parameters
            return self._generate_random_parameters()
        
        parameters = self.parameter_grid[self.grid_index].copy()
        self.grid_index += 1
        return parameters
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters when grid is exhausted."""
        parameters = {}
        for param_name, param_config in self.search_space.items():
            parameters[param_name] = SearchSpaceHandler.sample_parameter(param_name, param_config)
        return parameters
    
    def _update_algorithm_state(self, trial: OptimizationTrial) -> None:
        """Update algorithm state after trial completion."""
        # Grid search doesn't need to update state
        pass


class RandomSearchOptimizer(BaseHyperparameterOptimizer):
    """Random Search hyperparameter optimization."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Random search optimizer initialized")
    
    def _generate_trial_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Generate random parameters for the next trial."""
        if not SearchSpaceHandler.validate_search_space(self.search_space):
            raise ValueError("Invalid search space format")
        
        parameters = {}
        for param_name, param_config in self.search_space.items():
            parameters[param_name] = SearchSpaceHandler.sample_parameter(param_name, param_config)
        
        return parameters
    
    def _update_algorithm_state(self, trial: OptimizationTrial) -> None:
        """Update algorithm state after trial completion."""
        # Random search doesn't need to update state
        pass


class BayesianOptimizer(BaseHyperparameterOptimizer):
    """Simple Bayesian Optimization using Gaussian Process (if scikit-learn available)."""
    
    def __init__(self, *args, acquisition_function: str = "expected_improvement", **kwargs):
        super().__init__(*args, **kwargs)
        self.acquisition_function = acquisition_function
        self.gp_model = None
        self.X_observed = []
        self.y_observed = []
        self.exploration_trials = max(5, self.config.max_trials // 4)
        
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available. Falling back to intelligent random search.")
        
        logger.info(f"Bayesian optimizer initialized with {self.exploration_trials} exploration trials")
    
    def _generate_trial_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Generate parameters using Bayesian optimization."""
        # Initial exploration phase or fallback to random
        if (trial_number < self.exploration_trials or 
            not SKLEARN_AVAILABLE or 
            len(self.y_observed) < 3):
            return self._generate_random_parameters()
        
        # Bayesian optimization phase
        try:
            return self._generate_bayesian_parameters()
        except Exception as e:
            logger.warning(f"Bayesian optimization failed, using random: {e}")
            return self._generate_random_parameters()
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters."""
        parameters = {}
        for param_name, param_config in self.search_space.items():
            parameters[param_name] = SearchSpaceHandler.sample_parameter(param_name, param_config)
        return parameters
    
    def _generate_bayesian_parameters(self) -> Dict[str, Any]:
        """Generate parameters using Gaussian Process surrogate model."""
        if not self.gp_model or len(self.y_observed) < 2:
            return self._generate_random_parameters()
        
        # Generate candidate points
        num_candidates = 1000
        candidates = []
        for _ in range(num_candidates):
            candidate = self._generate_random_parameters()
            candidates.append(candidate)
        
        # Convert to arrays for GP
        candidate_arrays = [self._parameters_to_array(c) for c in candidates]
        
        # Predict with GP
        try:
            means, stds = self.gp_model.predict(candidate_arrays, return_std=True)
            
            # Calculate acquisition function
            if self.acquisition_function == "expected_improvement":
                acquisition_values = self._expected_improvement(means, stds)
            elif self.acquisition_function == "upper_confidence_bound":
                acquisition_values = self._upper_confidence_bound(means, stds)
            else:  # probability_of_improvement
                acquisition_values = self._probability_of_improvement(means, stds)
            
            # Select best candidate
            best_idx = np.argmax(acquisition_values)
            return candidates[best_idx]
            
        except Exception as e:
            logger.warning(f"GP prediction failed: {e}")
            return self._generate_random_parameters()
    
    def _parameters_to_array(self, parameters: Dict[str, Any]) -> np.ndarray:
        """Convert parameter dictionary to array for GP model."""
        # Simple encoding - can be improved for categorical variables
        values = []
        for param_name in sorted(self.search_space.keys()):
            value = parameters[param_name]
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)):
                values.append(float(value))
            else:
                # For categorical, use hash or encoding
                values.append(float(hash(str(value)) % 1000))
        
        return np.array(values)
    
    def _expected_improvement(self, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Calculate Expected Improvement acquisition function."""
        if len(self.y_observed) == 0:
            return stds
        
        best_y = max(self.y_observed)
        z = (means - best_y) / (stds + 1e-8)
        
        from scipy.stats import norm
        ei = (means - best_y) * norm.cdf(z) + stds * norm.pdf(z)
        return ei
    
    def _upper_confidence_bound(self, means: np.ndarray, stds: np.ndarray, kappa: float = 2.0) -> np.ndarray:
        """Calculate Upper Confidence Bound acquisition function."""
        return means + kappa * stds
    
    def _probability_of_improvement(self, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
        """Calculate Probability of Improvement acquisition function."""
        if len(self.y_observed) == 0:
            return np.ones_like(means)
        
        best_y = max(self.y_observed)
        z = (means - best_y) / (stds + 1e-8)
        
        from scipy.stats import norm
        return norm.cdf(z)
    
    def _update_algorithm_state(self, trial: OptimizationTrial) -> None:
        """Update Gaussian Process with new observation."""
        if trial.status != "completed" or trial.score is None:
            return
        
        # Add observation
        x_new = self._parameters_to_array(trial.parameters)
        self.X_observed.append(x_new)
        self.y_observed.append(trial.score)
        
        # Retrain GP model
        if SKLEARN_AVAILABLE and len(self.y_observed) >= 2:
            try:
                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                self.gp_model = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=5
                )
                
                X = np.array(self.X_observed)
                y = np.array(self.y_observed)
                self.gp_model.fit(X, y)
                
                logger.debug(f"GP model updated with {len(self.y_observed)} observations")
                
            except Exception as e:
                logger.warning(f"Failed to update GP model: {e}")


class OptunaOptimizer(BaseHyperparameterOptimizer):
    """Optuna-based hyperparameter optimization."""
    
    def __init__(self, *args, sampler_type: str = "tpe", **kwargs):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for OptunaOptimizer. Install with: pip install optuna")
        
        super().__init__(*args, **kwargs)
        self.sampler_type = sampler_type
        
        # Create Optuna study
        sampler = self._create_sampler()
        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            study_name=f"marksix_optimization_{int(time.time())}"
        )
        
        logger.info(f"Optuna optimizer initialized with {sampler_type} sampler")
    
    def _create_sampler(self):
        """Create Optuna sampler."""
        if self.sampler_type == "tpe":
            return TPESampler(seed=self.config.random_seed)
        elif self.sampler_type == "random":
            return RandomSampler(seed=self.config.random_seed)
        else:
            logger.warning(f"Unknown sampler type: {self.sampler_type}, using TPE")
            return TPESampler(seed=self.config.random_seed)
    
    def _generate_trial_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Generate parameters using Optuna."""
        # Create Optuna trial
        optuna_trial = self.study.ask()
        
        # Sample parameters
        parameters = {}
        for param_name, param_config in self.search_space.items():
            parameters[param_name] = self._sample_optuna_parameter(optuna_trial, param_name, param_config)
        
        # Store trial for later update
        self._current_optuna_trial = optuna_trial
        
        return parameters
    
    def _sample_optuna_parameter(self, trial, param_name: str, param_config: Union[Dict, List, Tuple]) -> Any:
        """Sample parameter using Optuna trial."""
        if isinstance(param_config, (list, tuple)):
            return trial.suggest_categorical(param_name, param_config)
        
        param_type = param_config.get('type', 'choice')
        
        if param_type == 'choice':
            return trial.suggest_categorical(param_name, param_config['choices'])
        elif param_type == 'uniform':
            return trial.suggest_float(param_name, param_config['low'], param_config['high'])
        elif param_type == 'loguniform':
            return trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
        elif param_type == 'int':
            return trial.suggest_int(param_name, param_config['low'], param_config['high'])
        elif param_type == 'bool':
            return trial.suggest_categorical(param_name, [True, False])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def _update_algorithm_state(self, trial: OptimizationTrial) -> None:
        """Update Optuna study with trial result."""
        if hasattr(self, '_current_optuna_trial'):
            if trial.status == "completed" and trial.score is not None:
                self.study.tell(self._current_optuna_trial, trial.score)
            else:
                # Mark trial as failed in Optuna
                self.study.tell(self._current_optuna_trial, float('-inf'))
            
            delattr(self, '_current_optuna_trial')


def create_optimizer(
    algorithm: str,
    objective_function,
    search_space: Dict[str, Any],
    config: OptimizationConfig,
    **kwargs
) -> BaseHyperparameterOptimizer:
    """Factory function to create optimizer instances."""
    
    algorithm = algorithm.lower()
    
    if algorithm == "grid_search":
        return GridSearchOptimizer(objective_function, search_space, config, **kwargs)
    elif algorithm == "random_search":
        return RandomSearchOptimizer(objective_function, search_space, config, **kwargs)
    elif algorithm == "bayesian":
        return BayesianOptimizer(objective_function, search_space, config, **kwargs)
    elif algorithm == "optuna":
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to Bayesian optimization")
            return BayesianOptimizer(objective_function, search_space, config, **kwargs)
        return OptunaOptimizer(objective_function, search_space, config, **kwargs)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. "
                        f"Available: grid_search, random_search, bayesian, optuna")


# Example search space definitions
EXAMPLE_SEARCH_SPACES = {
    "basic": {
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-5,
            "high": 1e-2
        },
        "batch_size": {
            "type": "choice",
            "choices": [8, 16, 32, 64]
        },
        "hidden_size": {
            "type": "choice", 
            "choices": [128, 256, 512, 768]
        },
        "dropout": {
            "type": "uniform",
            "low": 0.1,
            "high": 0.5
        }
    },
    
    "advanced": {
        "learning_rate": {
            "type": "loguniform",
            "low": 1e-6,
            "high": 1e-2
        },
        "batch_size": {
            "type": "choice",
            "choices": [4, 8, 16, 32, 64, 128]
        },
        "hidden_size": {
            "type": "choice",
            "choices": [64, 128, 256, 512, 768, 1024]
        },
        "num_layers": {
            "type": "int",
            "low": 2,
            "high": 8
        },
        "dropout": {
            "type": "uniform",
            "low": 0.05,
            "high": 0.5
        },
        "weight_decay": {
            "type": "loguniform",
            "low": 1e-8,
            "high": 1e-3
        },
        "use_batch_norm": {
            "type": "bool"
        },
        "optimizer_type": {
            "type": "choice",
            "choices": ["adam", "adamw", "sgd"]
        }
    }
}