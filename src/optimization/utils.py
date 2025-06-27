"""
Utility functions for hyperparameter optimization.
Provides helper functions, validation, and common operations.
"""

import os
import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd

from ..infrastructure.logging.logger import get_logger
from ..utils.error_handling import safe_execute

logger = get_logger(__name__)


class OptimizationUtils:
    """Utility functions for optimization operations."""
    
    @staticmethod
    def hash_config(config: Dict[str, Any]) -> str:
        """Generate a hash for configuration dictionary."""
        # Sort and stringify for consistent hashing
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], update_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        merged = base_config.copy()
        merged.update(update_config)
        return merged
    
    @staticmethod
    def validate_parameter_types(parameters: Dict[str, Any], expected_types: Dict[str, type]) -> Tuple[bool, List[str]]:
        """Validate parameter types."""
        errors = []
        
        for param_name, expected_type in expected_types.items():
            if param_name in parameters:
                value = parameters[param_name]
                if not isinstance(value, expected_type):
                    errors.append(f"Parameter '{param_name}' should be {expected_type.__name__}, got {type(value).__name__}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def normalize_parameters(parameters: Dict[str, Any], parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Normalize parameters to [0, 1] range."""
        normalized = {}
        
        for param_name, value in parameters.items():
            if param_name in parameter_ranges:
                min_val, max_val = parameter_ranges[param_name]
                if max_val > min_val:
                    normalized[param_name] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[param_name] = 0.0
            else:
                normalized[param_name] = value
        
        return normalized
    
    @staticmethod
    def denormalize_parameters(normalized_params: Dict[str, Any], parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
        """Denormalize parameters from [0, 1] range."""
        denormalized = {}
        
        for param_name, norm_value in normalized_params.items():
            if param_name in parameter_ranges:
                min_val, max_val = parameter_ranges[param_name]
                denormalized[param_name] = norm_value * (max_val - min_val) + min_val
            else:
                denormalized[param_name] = norm_value
        
        return denormalized
    
    @staticmethod
    def sample_around_best(best_params: Dict[str, Any], search_space: Dict[str, Any], 
                          noise_factor: float = 0.1, num_samples: int = 5) -> List[Dict[str, Any]]:
        """Sample parameters around the best configuration."""
        samples = []
        
        for _ in range(num_samples):
            sample = {}
            
            for param_name, param_config in search_space.items():
                best_value = best_params.get(param_name)
                
                if best_value is None:
                    # If no best value, sample randomly
                    sample[param_name] = OptimizationUtils.sample_parameter(param_name, param_config)
                    continue
                
                if isinstance(param_config, (list, tuple)):
                    # For choice parameters, sometimes keep the best, sometimes sample
                    if np.random.random() < 0.7:
                        sample[param_name] = best_value
                    else:
                        sample[param_name] = np.random.choice(param_config)
                
                elif isinstance(param_config, dict):
                    param_type = param_config.get('type', 'choice')
                    
                    if param_type == 'choice':
                        if np.random.random() < 0.7:
                            sample[param_name] = best_value
                        else:
                            sample[param_name] = np.random.choice(param_config['choices'])
                    
                    elif param_type in ['uniform', 'loguniform']:
                        # Add noise to continuous parameters
                        low, high = param_config['low'], param_config['high']
                        range_size = high - low
                        noise = np.random.normal(0, range_size * noise_factor)
                        
                        if param_type == 'loguniform':
                            # Work in log space
                            log_best = np.log(best_value)
                            log_noise = np.random.normal(0, np.log(high/low) * noise_factor)
                            new_value = np.exp(log_best + log_noise)
                        else:
                            new_value = best_value + noise
                        
                        # Clip to bounds
                        sample[param_name] = np.clip(new_value, low, high)
                    
                    elif param_type == 'int':
                        # Add discrete noise
                        low, high = param_config['low'], param_config['high']
                        noise = np.random.randint(-max(1, int((high-low) * noise_factor)), 
                                                 max(1, int((high-low) * noise_factor)) + 1)
                        new_value = best_value + noise
                        sample[param_name] = np.clip(new_value, low, high)
                    
                    else:
                        sample[param_name] = best_value
                
                else:
                    sample[param_name] = best_value
            
            samples.append(sample)
        
        return samples
    
    @staticmethod
    def sample_parameter(param_name: str, param_config: Union[Dict, List, Tuple]) -> Any:
        """Sample a single parameter value."""
        if isinstance(param_config, (list, tuple)):
            return np.random.choice(param_config)
        
        param_type = param_config.get('type', 'choice')
        
        if param_type == 'choice':
            return np.random.choice(param_config['choices'])
        elif param_type == 'uniform':
            return np.random.uniform(param_config['low'], param_config['high'])
        elif param_type == 'loguniform':
            return np.exp(np.random.uniform(np.log(param_config['low']), np.log(param_config['high'])))
        elif param_type == 'int':
            return np.random.randint(param_config['low'], param_config['high'] + 1)
        elif param_type == 'bool':
            return np.random.choice([True, False])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    @staticmethod
    def calculate_parameter_importance(trials: List[Dict[str, Any]], scores: List[float]) -> Dict[str, float]:
        """Calculate parameter importance using correlation analysis."""
        if len(trials) < 5:
            logger.warning("Not enough trials for importance analysis")
            return {}
        
        # Convert to DataFrame for analysis
        df_data = []
        for trial, score in zip(trials, scores):
            row = trial.copy()
            row['score'] = score
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Calculate correlations for numeric parameters
        importance = {}
        for column in df.columns:
            if column == 'score':
                continue
            
            try:
                # Convert to numeric if possible
                numeric_values = pd.to_numeric(df[column], errors='coerce')
                if numeric_values.notna().sum() > len(df) * 0.5:  # At least 50% numeric
                    correlation = numeric_values.corr(df['score'])
                    if not np.isnan(correlation):
                        importance[column] = abs(correlation)
                        
            except Exception:
                continue
        
        return importance
    
    @staticmethod
    def suggest_parameter_bounds(trials: List[Dict[str, Any]], scores: List[float], 
                                percentile: float = 0.8) -> Dict[str, Tuple[float, float]]:
        """Suggest parameter bounds based on top-performing trials."""
        if len(trials) < 10:
            return {}
        
        # Get top trials
        sorted_indices = np.argsort(scores)
        top_count = max(1, int(len(trials) * (1 - percentile)))
        top_indices = sorted_indices[-top_count:]
        top_trials = [trials[i] for i in top_indices]
        
        bounds = {}
        
        # Analyze each parameter
        for param_name in top_trials[0].keys():
            values = []
            for trial in top_trials:
                value = trial.get(param_name)
                if isinstance(value, (int, float)):
                    values.append(value)
            
            if len(values) >= 3:
                values = np.array(values)
                bounds[param_name] = (np.min(values), np.max(values))
        
        return bounds
    
    @staticmethod
    def export_results(trials: List[Dict[str, Any]], scores: List[float], 
                      metadata: Dict[str, Any], output_path: Path) -> None:
        """Export optimization results to file."""
        results = {
            'metadata': metadata,
            'trials': [
                {
                    'parameters': trial,
                    'score': score,
                    'trial_id': i
                }
                for i, (trial, score) in enumerate(zip(trials, scores))
            ],
            'summary': {
                'total_trials': len(trials),
                'best_score': max(scores) if scores else None,
                'best_trial': trials[np.argmax(scores)] if scores else None,
                'parameter_importance': OptimizationUtils.calculate_parameter_importance(trials, scores)
            },
            'exported_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results exported to {output_path}")
    
    @staticmethod
    def load_results(input_path: Path) -> Tuple[List[Dict[str, Any]], List[float], Dict[str, Any]]:
        """Load optimization results from file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        trials = [trial_data['parameters'] for trial_data in data['trials']]
        scores = [trial_data['score'] for trial_data in data['trials']]
        metadata = data.get('metadata', {})
        
        logger.info(f"Loaded {len(trials)} trials from {input_path}")
        return trials, scores, metadata
    
    @staticmethod
    def create_parameter_grid(search_space: Dict[str, Any], max_combinations: int = 100) -> List[Dict[str, Any]]:
        """Create parameter grid for grid search."""
        param_names = list(search_space.keys())
        param_value_lists = []
        
        for param_name in param_names:
            param_config = search_space[param_name]
            
            if isinstance(param_config, (list, tuple)):
                param_value_lists.append(list(param_config))
            elif isinstance(param_config, dict):
                param_type = param_config.get('type', 'choice')
                
                if param_type == 'choice':
                    param_value_lists.append(param_config['choices'])
                elif param_type in ['uniform', 'loguniform']:
                    low, high = param_config['low'], param_config['high']
                    num_values = min(5, int(max_combinations ** (1/len(param_names))))
                    
                    if param_type == 'uniform':
                        values = np.linspace(low, high, num_values).tolist()
                    else:  # loguniform
                        values = np.exp(np.linspace(np.log(low), np.log(high), num_values)).tolist()
                    
                    param_value_lists.append(values)
                elif param_type == 'int':
                    low, high = param_config['low'], param_config['high']
                    values = list(range(low, min(high + 1, low + 10)))  # Limit range
                    param_value_lists.append(values)
                else:
                    param_value_lists.append([True, False])  # Default for bool or unknown
        
        # Generate all combinations
        import itertools
        combinations = list(itertools.product(*param_value_lists))
        
        # Limit combinations
        if len(combinations) > max_combinations:
            combinations = np.random.choice(len(combinations), max_combinations, replace=False)
            combinations = [list(itertools.product(*param_value_lists))[i] for i in combinations]
        
        # Convert to dictionaries
        parameter_grid = []
        for combination in combinations:
            param_dict = dict(zip(param_names, combination))
            parameter_grid.append(param_dict)
        
        return parameter_grid
    
    @staticmethod
    def estimate_optimization_time(search_space: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
        """Estimate optimization time for different algorithms."""
        # Base time per trial (in minutes)
        base_time_per_trial = config.get('estimated_trial_time_minutes', 5.0)
        
        # Calculate search space size
        space_size = 1
        for param_config in search_space.values():
            if isinstance(param_config, (list, tuple)):
                space_size *= len(param_config)
            elif isinstance(param_config, dict):
                param_type = param_config.get('type', 'choice')
                if param_type == 'choice':
                    space_size *= len(param_config['choices'])
                else:
                    space_size *= 10  # Estimate for continuous parameters
        
        estimates = {
            'grid_search': min(space_size, config.get('max_trials', 50)) * base_time_per_trial,
            'random_search': config.get('max_trials', 30) * base_time_per_trial,
            'bayesian': config.get('max_trials', 25) * base_time_per_trial * 1.2,  # Slightly longer per trial
            'optuna': config.get('max_trials', 40) * base_time_per_trial * 1.1
        }
        
        return estimates
    
    @staticmethod
    def validate_search_space(search_space: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate search space configuration."""
        errors = []
        
        if not isinstance(search_space, dict):
            errors.append("Search space must be a dictionary")
            return False, errors
        
        if len(search_space) == 0:
            errors.append("Search space cannot be empty")
            return False, errors
        
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
            
            # Type-specific validation
            if param_type == 'choice':
                if 'choices' not in param_config:
                    errors.append(f"Parameter {param_name}: missing 'choices'")
                elif not isinstance(param_config['choices'], (list, tuple)):
                    errors.append(f"Parameter {param_name}: 'choices' must be list")
                elif len(param_config['choices']) == 0:
                    errors.append(f"Parameter {param_name}: empty choices")
            
            elif param_type in ['uniform', 'loguniform']:
                for field in ['low', 'high']:
                    if field not in param_config:
                        errors.append(f"Parameter {param_name}: missing '{field}'")
                    elif not isinstance(param_config[field], (int, float)):
                        errors.append(f"Parameter {param_name}: '{field}' must be numeric")
                
                if ('low' in param_config and 'high' in param_config and 
                    param_config['low'] >= param_config['high']):
                    errors.append(f"Parameter {param_name}: 'low' must be < 'high'")
                
                if param_type == 'loguniform' and param_config.get('low', 1) <= 0:
                    errors.append(f"Parameter {param_name}: 'low' must be positive for loguniform")
            
            elif param_type == 'int':
                for field in ['low', 'high']:
                    if field not in param_config:
                        errors.append(f"Parameter {param_name}: missing '{field}'")
                    elif not isinstance(param_config[field], int):
                        errors.append(f"Parameter {param_name}: '{field}' must be integer")
                
                if ('low' in param_config and 'high' in param_config and 
                    param_config['low'] >= param_config['high']):
                    errors.append(f"Parameter {param_name}: 'low' must be < 'high'")
            
            elif param_type == 'bool':
                pass  # No additional validation for bool
            
            else:
                errors.append(f"Parameter {param_name}: unknown type '{param_type}'")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def create_directory_structure(base_dir: Path) -> Dict[str, Path]:
        """Create standard directory structure for optimization."""
        directories = {
            'results': base_dir / 'results',
            'plots': base_dir / 'plots', 
            'configs': base_dir / 'configs',
            'logs': base_dir / 'logs',
            'checkpoints': base_dir / 'checkpoints',
            'exports': base_dir / 'exports'
        }
        
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return directories


class ConfigTemplate:
    """Template generator for optimization configurations."""
    
    @staticmethod
    def create_quick_test_config() -> Dict[str, Any]:
        """Create configuration for quick testing."""
        return {
            'max_trials': 5,
            'max_duration_hours': 0.5,
            'early_stopping': False,
            'parallel_jobs': 1,
            'trial_timeout_minutes': 10.0,
            'algorithm': 'random_search'
        }
    
    @staticmethod
    def create_balanced_config() -> Dict[str, Any]:
        """Create balanced configuration."""
        return {
            'max_trials': 30,
            'max_duration_hours': 4.0,
            'early_stopping': True,
            'early_stopping_patience': 8,
            'parallel_jobs': 2,
            'trial_timeout_minutes': 30.0,
            'algorithm': 'bayesian'
        }
    
    @staticmethod
    def create_thorough_config() -> Dict[str, Any]:
        """Create thorough optimization configuration."""
        return {
            'max_trials': 50,
            'max_duration_hours': 8.0,
            'early_stopping': True,
            'early_stopping_patience': 12,
            'parallel_jobs': 3,
            'trial_timeout_minutes': 60.0,
            'algorithm': 'optuna'
        }
    
    @staticmethod
    def create_basic_search_space() -> Dict[str, Any]:
        """Create basic search space."""
        return {
            'learning_rate': {
                'type': 'loguniform',
                'low': 1e-5,
                'high': 1e-2
            },
            'batch_size': {
                'type': 'choice',
                'choices': [8, 16, 32]
            },
            'dropout': {
                'type': 'uniform',
                'low': 0.1,
                'high': 0.3
            }
        }
    
    @staticmethod
    def create_advanced_search_space() -> Dict[str, Any]:
        """Create advanced search space."""
        return {
            'learning_rate': {
                'type': 'loguniform',
                'low': 1e-6,
                'high': 1e-2
            },
            'batch_size': {
                'type': 'choice',
                'choices': [4, 8, 16, 32, 64]
            },
            'hidden_size': {
                'type': 'choice',
                'choices': [128, 256, 512, 768]
            },
            'num_layers': {
                'type': 'int',
                'low': 2,
                'high': 8
            },
            'dropout': {
                'type': 'uniform',
                'low': 0.05,
                'high': 0.5
            },
            'weight_decay': {
                'type': 'loguniform',
                'low': 1e-8,
                'high': 1e-3
            },
            'kl_weight': {
                'type': 'loguniform',
                'low': 1e-4,
                'high': 1e-1
            }
        }