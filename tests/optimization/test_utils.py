"""
Tests for optimization utility functions.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from src.optimization.utils import (
    OptimizationUtils,
    ConfigTemplate
)


class TestOptimizationUtils:
    """Test OptimizationUtils functionality."""
    
    def test_hash_config(self):
        """Test configuration hashing."""
        config1 = {'a': 1, 'b': 2, 'c': 3}
        config2 = {'c': 3, 'a': 1, 'b': 2}  # Same content, different order
        config3 = {'a': 1, 'b': 2, 'c': 4}  # Different content
        
        hash1 = OptimizationUtils.hash_config(config1)
        hash2 = OptimizationUtils.hash_config(config2)
        hash3 = OptimizationUtils.hash_config(config3)
        
        # Same content should produce same hash
        assert hash1 == hash2
        # Different content should produce different hash
        assert hash1 != hash3
        # Hash should be string
        assert isinstance(hash1, str)
        assert len(hash1) == 12  # Should be 12 characters
    
    def test_merge_configs(self):
        """Test configuration merging."""
        base_config = {'a': 1, 'b': 2, 'c': 3}
        update_config = {'b': 20, 'd': 4}
        
        merged = OptimizationUtils.merge_configs(base_config, update_config)
        
        expected = {'a': 1, 'b': 20, 'c': 3, 'd': 4}
        assert merged == expected
        
        # Original configs should not be modified
        assert base_config == {'a': 1, 'b': 2, 'c': 3}
        assert update_config == {'b': 20, 'd': 4}
    
    def test_validate_parameter_types(self):
        """Test parameter type validation."""
        parameters = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'use_dropout': True,
            'model_name': 'test'
        }
        
        expected_types = {
            'learning_rate': float,
            'batch_size': int,
            'use_dropout': bool,
            'model_name': str
        }
        
        is_valid, errors = OptimizationUtils.validate_parameter_types(parameters, expected_types)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_parameter_types_invalid(self):
        """Test parameter type validation with invalid types."""
        parameters = {
            'learning_rate': '0.001',  # Should be float
            'batch_size': 32.5,        # Should be int
            'use_dropout': 'true'      # Should be bool
        }
        
        expected_types = {
            'learning_rate': float,
            'batch_size': int,
            'use_dropout': bool
        }
        
        is_valid, errors = OptimizationUtils.validate_parameter_types(parameters, expected_types)
        
        assert is_valid is False
        assert len(errors) == 3
    
    def test_normalize_parameters(self):
        """Test parameter normalization."""
        parameters = {
            'x': 0.5,
            'y': 75,
            'z': 'category'  # Non-numeric
        }
        
        parameter_ranges = {
            'x': (0.0, 1.0),
            'y': (50, 100),
            # 'z' not in ranges
        }
        
        normalized = OptimizationUtils.normalize_parameters(parameters, parameter_ranges)
        
        assert normalized['x'] == 0.5  # Already in [0, 1]
        assert normalized['y'] == 0.5  # (75 - 50) / (100 - 50) = 0.5
        assert normalized['z'] == 'category'  # Unchanged
    
    def test_denormalize_parameters(self):
        """Test parameter denormalization."""
        normalized_params = {
            'x': 0.5,
            'y': 0.3,
            'z': 'category'
        }
        
        parameter_ranges = {
            'x': (0.0, 1.0),
            'y': (10, 20)
        }
        
        denormalized = OptimizationUtils.denormalize_parameters(normalized_params, parameter_ranges)
        
        assert denormalized['x'] == 0.5  # 0.5 * (1.0 - 0.0) + 0.0 = 0.5
        assert denormalized['y'] == 13.0  # 0.3 * (20 - 10) + 10 = 13.0
        assert denormalized['z'] == 'category'  # Unchanged
    
    def test_sample_around_best_choice_parameters(self):
        """Test sampling around best parameters for choice type."""
        best_params = {'optimizer': 'adam'}
        search_space = {
            'optimizer': {'type': 'choice', 'choices': ['adam', 'sgd', 'adamw']}
        }
        
        samples = OptimizationUtils.sample_around_best(
            best_params, search_space, noise_factor=0.1, num_samples=10
        )
        
        assert len(samples) == 10
        for sample in samples:
            assert 'optimizer' in sample
            assert sample['optimizer'] in ['adam', 'sgd', 'adamw']
        
        # Should favor best value (adam) but sometimes pick others
        optimizers = [s['optimizer'] for s in samples]
        assert 'adam' in optimizers  # Should appear at least once
    
    def test_sample_around_best_uniform_parameters(self):
        """Test sampling around best parameters for uniform type."""
        best_params = {'learning_rate': 0.001}
        search_space = {
            'learning_rate': {'type': 'uniform', 'low': 0.0001, 'high': 0.01}
        }
        
        samples = OptimizationUtils.sample_around_best(
            best_params, search_space, noise_factor=0.2, num_samples=20
        )
        
        assert len(samples) == 20
        for sample in samples:
            assert 'learning_rate' in sample
            lr = sample['learning_rate']
            assert 0.0001 <= lr <= 0.01
    
    def test_sample_parameter(self):
        """Test individual parameter sampling."""
        # Test choice type
        choice_config = {'type': 'choice', 'choices': [1, 2, 3]}
        value = OptimizationUtils.sample_parameter('test', choice_config)
        assert value in [1, 2, 3]
        
        # Test uniform type
        uniform_config = {'type': 'uniform', 'low': 0, 'high': 1}
        value = OptimizationUtils.sample_parameter('test', uniform_config)
        assert 0 <= value <= 1
        
        # Test int type
        int_config = {'type': 'int', 'low': 1, 'high': 10}
        value = OptimizationUtils.sample_parameter('test', int_config)
        assert isinstance(value, int)
        assert 1 <= value <= 10
        
        # Test bool type
        bool_config = {'type': 'bool'}
        value = OptimizationUtils.sample_parameter('test', bool_config)
        assert isinstance(value, bool)
        
        # Test list format
        list_config = ['a', 'b', 'c']
        value = OptimizationUtils.sample_parameter('test', list_config)
        assert value in ['a', 'b', 'c']
    
    def test_calculate_parameter_importance(self):
        """Test parameter importance calculation."""
        trials = [
            {'x': 0.1, 'y': 1},
            {'x': 0.5, 'y': 2},
            {'x': 0.9, 'y': 3},
            {'x': 0.2, 'y': 1},
            {'x': 0.8, 'y': 3}
        ]
        scores = [0.1, 0.5, 0.9, 0.2, 0.8]
        
        importance = OptimizationUtils.calculate_parameter_importance(trials, scores)
        
        # x should be highly correlated with scores
        # y should also be correlated
        assert 'x' in importance
        assert 'y' in importance
        assert importance['x'] > 0.8  # Strong correlation
        assert importance['y'] > 0.8  # Strong correlation
    
    def test_calculate_parameter_importance_insufficient_data(self):
        """Test parameter importance with insufficient data."""
        trials = [{'x': 1}, {'x': 2}]  # Only 2 trials
        scores = [0.1, 0.2]
        
        importance = OptimizationUtils.calculate_parameter_importance(trials, scores)
        
        assert importance == {}  # Should return empty dict
    
    def test_suggest_parameter_bounds(self):
        """Test parameter bounds suggestion."""
        trials = [
            {'x': 0.1, 'y': 10, 'z': 'a'},
            {'x': 0.3, 'y': 20, 'z': 'b'},
            {'x': 0.5, 'y': 30, 'z': 'a'},
            {'x': 0.7, 'y': 40, 'z': 'c'},
            {'x': 0.9, 'y': 50, 'z': 'b'}
        ]
        scores = [0.1, 0.8, 0.9, 0.7, 0.2]  # Top performers: trials 1, 2, 3
        
        bounds = OptimizationUtils.suggest_parameter_bounds(trials, scores, percentile=0.6)
        
        # Should suggest bounds based on top 40% (2 trials: indices 2, 1)
        assert 'x' in bounds
        assert 'y' in bounds
        assert 'z' not in bounds  # Non-numeric parameter
        
        # Bounds should cover range of top performers
        x_bounds = bounds['x']
        y_bounds = bounds['y']
        assert x_bounds[0] <= 0.3  # Min of top performers
        assert x_bounds[1] >= 0.5  # Max of top performers
    
    def test_export_and_load_results(self):
        """Test exporting and loading optimization results."""
        trials = [
            {'learning_rate': 0.001, 'batch_size': 32},
            {'learning_rate': 0.01, 'batch_size': 16}
        ]
        scores = [0.8, 0.6]
        metadata = {'algorithm': 'random_search', 'total_time': 3600}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            # Export results
            OptimizationUtils.export_results(trials, scores, metadata, output_path)
            
            # Verify file exists and has content
            assert output_path.exists()
            
            # Load results back
            loaded_trials, loaded_scores, loaded_metadata = OptimizationUtils.load_results(output_path)
            
            assert loaded_trials == trials
            assert loaded_scores == scores
            assert loaded_metadata == metadata
            
        finally:
            output_path.unlink()
    
    def test_create_parameter_grid(self):
        """Test parameter grid creation."""
        search_space = {
            'learning_rate': {'type': 'choice', 'choices': [0.001, 0.01]},
            'batch_size': {'type': 'choice', 'choices': [16, 32]},
            'optimizer': ['adam', 'sgd']
        }
        
        grid = OptimizationUtils.create_parameter_grid(search_space, max_combinations=10)
        
        # Should create all combinations: 2 * 2 * 2 = 8
        assert len(grid) == 8
        
        # Each combination should have all parameters
        for combination in grid:
            assert 'learning_rate' in combination
            assert 'batch_size' in combination
            assert 'optimizer' in combination
        
        # All combinations should be unique
        unique_combinations = set(str(combo) for combo in grid)
        assert len(unique_combinations) == len(grid)
    
    def test_create_parameter_grid_limit_combinations(self):
        """Test parameter grid creation with combination limit."""
        # Create search space with many combinations
        search_space = {
            'param1': {'type': 'choice', 'choices': list(range(10))},
            'param2': {'type': 'choice', 'choices': list(range(10))}
        }
        # This would create 10 * 10 = 100 combinations
        
        grid = OptimizationUtils.create_parameter_grid(search_space, max_combinations=20)
        
        # Should be limited to 20
        assert len(grid) <= 20
    
    def test_estimate_optimization_time(self):
        """Test optimization time estimation."""
        search_space = {
            'param1': {'type': 'choice', 'choices': [1, 2, 3]},
            'param2': {'type': 'uniform', 'low': 0, 'high': 1}
        }
        
        config = {
            'max_trials': 30,
            'estimated_trial_time_minutes': 5.0
        }
        
        estimates = OptimizationUtils.estimate_optimization_time(search_space, config)
        
        # Should contain estimates for different algorithms
        expected_algorithms = ['grid_search', 'random_search', 'bayesian', 'optuna']
        for algorithm in expected_algorithms:
            assert algorithm in estimates
            assert estimates[algorithm] > 0
        
        # Random search should use max_trials
        assert estimates['random_search'] == 30 * 5.0
    
    def test_validate_search_space_valid(self):
        """Test search space validation with valid space."""
        search_space = {
            'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
            'batch_size': {'type': 'choice', 'choices': [16, 32, 64]},
            'dropout': {'type': 'uniform', 'low': 0.1, 'high': 0.5},
            'num_layers': {'type': 'int', 'low': 2, 'high': 8},
            'use_batch_norm': {'type': 'bool'},
            'optimizer': ['adam', 'sgd']  # List format
        }
        
        is_valid, errors = OptimizationUtils.validate_search_space(search_space)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_search_space_invalid(self):
        """Test search space validation with invalid space."""
        invalid_search_space = {
            'param1': {'type': 'unknown_type'},  # Unknown type
            'param2': {'type': 'choice'},         # Missing choices
            'param3': {'type': 'uniform', 'low': 1, 'high': 0},  # Invalid bounds
            'param4': {'type': 'loguniform', 'low': -1, 'high': 1},  # Negative for loguniform
            'param5': [],  # Empty choices
            123: {'type': 'uniform', 'low': 0, 'high': 1}  # Non-string parameter name
        }
        
        is_valid, errors = OptimizationUtils.validate_search_space(invalid_search_space)
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_create_directory_structure(self):
        """Test directory structure creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir) / "optimization"
            
            directories = OptimizationUtils.create_directory_structure(base_dir)
            
            # Should return dict with expected keys
            expected_keys = ['results', 'plots', 'configs', 'logs', 'checkpoints', 'exports']
            assert set(directories.keys()) == set(expected_keys)
            
            # All directories should exist
            for dir_path in directories.values():
                assert dir_path.exists()
                assert dir_path.is_dir()


class TestConfigTemplate:
    """Test ConfigTemplate functionality."""
    
    def test_create_quick_test_config(self):
        """Test quick test configuration creation."""
        config = ConfigTemplate.create_quick_test_config()
        
        assert config['max_trials'] == 5
        assert config['max_duration_hours'] == 0.5
        assert config['early_stopping'] is False
        assert config['algorithm'] == 'random_search'
    
    def test_create_balanced_config(self):
        """Test balanced configuration creation."""
        config = ConfigTemplate.create_balanced_config()
        
        assert config['max_trials'] == 30
        assert config['max_duration_hours'] == 4.0
        assert config['early_stopping'] is True
        assert config['algorithm'] == 'bayesian'
    
    def test_create_thorough_config(self):
        """Test thorough configuration creation."""
        config = ConfigTemplate.create_thorough_config()
        
        assert config['max_trials'] == 50
        assert config['max_duration_hours'] == 8.0
        assert config['early_stopping'] is True
        assert config['algorithm'] == 'optuna'
    
    def test_create_basic_search_space(self):
        """Test basic search space creation."""
        search_space = ConfigTemplate.create_basic_search_space()
        
        # Should contain fundamental parameters
        expected_params = {'learning_rate', 'batch_size', 'dropout'}
        assert expected_params.issubset(set(search_space.keys()))
        
        # Validate structure
        is_valid, errors = OptimizationUtils.validate_search_space(search_space)
        assert is_valid is True
    
    def test_create_advanced_search_space(self):
        """Test advanced search space creation."""
        search_space = ConfigTemplate.create_advanced_search_space()
        
        # Should contain more parameters than basic
        basic_space = ConfigTemplate.create_basic_search_space()
        assert len(search_space) > len(basic_space)
        
        # Should include advanced parameters
        advanced_params = {'num_layers', 'weight_decay', 'kl_weight'}
        assert advanced_params.issubset(set(search_space.keys()))
        
        # Validate structure
        is_valid, errors = OptimizationUtils.validate_search_space(search_space)
        assert is_valid is True


if __name__ == '__main__':
    pytest.main([__file__])