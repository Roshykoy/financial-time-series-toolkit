"""
Tests for optimization algorithms.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from src.optimization.algorithms import (
    SearchSpaceHandler,
    GridSearchOptimizer,
    RandomSearchOptimizer,
    BayesianOptimizer,
    OptunaOptimizer,
    create_optimizer,
    EXAMPLE_SEARCH_SPACES
)
from src.optimization.base_optimizer import OptimizationConfig, OptimizationTrial


class TestSearchSpaceHandler:
    """Test SearchSpaceHandler functionality."""
    
    def test_validate_search_space_valid(self):
        """Test validation of valid search space."""
        search_space = {
            'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2},
            'batch_size': {'type': 'choice', 'choices': [16, 32, 64]},
            'dropout': {'type': 'uniform', 'low': 0.1, 'high': 0.5},
            'num_layers': {'type': 'int', 'low': 2, 'high': 8},
            'use_batch_norm': {'type': 'bool'},
            'optimizer': ['adam', 'sgd', 'adamw']  # List format
        }
        
        is_valid = SearchSpaceHandler.validate_search_space(search_space)
        assert is_valid is True
    
    def test_validate_search_space_invalid(self):
        """Test validation of invalid search space."""
        invalid_spaces = [
            # Missing type
            {'param1': {'low': 0, 'high': 1}},
            # Missing choices for choice type
            {'param1': {'type': 'choice'}},
            # Missing bounds for uniform
            {'param1': {'type': 'uniform', 'low': 0}},
            # Invalid bounds
            {'param1': {'type': 'uniform', 'low': 1, 'high': 0}},
            # Negative value for loguniform
            {'param1': {'type': 'loguniform', 'low': -1, 'high': 1}}
        ]
        
        for invalid_space in invalid_spaces:
            is_valid = SearchSpaceHandler.validate_search_space(invalid_space)
            assert is_valid is False
    
    def test_sample_parameter_choice(self):
        """Test parameter sampling for choice type."""
        param_config = {'type': 'choice', 'choices': [1, 2, 3, 4, 5]}
        
        # Sample multiple times to test randomness
        samples = [SearchSpaceHandler.sample_parameter('test', param_config) for _ in range(100)]
        
        # All samples should be from choices
        assert all(sample in param_config['choices'] for sample in samples)
        # Should have some variety (not all the same)
        assert len(set(samples)) > 1
    
    def test_sample_parameter_uniform(self):
        """Test parameter sampling for uniform type."""
        param_config = {'type': 'uniform', 'low': 0.1, 'high': 0.9}
        
        samples = [SearchSpaceHandler.sample_parameter('test', param_config) for _ in range(100)]
        
        # All samples should be in range
        assert all(0.1 <= sample <= 0.9 for sample in samples)
        # Should have variety
        assert len(set(samples)) > 10
    
    def test_sample_parameter_loguniform(self):
        """Test parameter sampling for loguniform type."""
        param_config = {'type': 'loguniform', 'low': 1e-5, 'high': 1e-2}
        
        samples = [SearchSpaceHandler.sample_parameter('test', param_config) for _ in range(100)]
        
        # All samples should be in range
        assert all(1e-5 <= sample <= 1e-2 for sample in samples)
        # Should have variety
        assert len(set(samples)) > 10
    
    def test_sample_parameter_int(self):
        """Test parameter sampling for int type."""
        param_config = {'type': 'int', 'low': 2, 'high': 8}
        
        samples = [SearchSpaceHandler.sample_parameter('test', param_config) for _ in range(100)]
        
        # All samples should be integers in range
        assert all(isinstance(sample, int) and 2 <= sample <= 8 for sample in samples)
        # Should cover the range
        unique_samples = set(samples)
        assert len(unique_samples) >= 3
    
    def test_sample_parameter_list_format(self):
        """Test parameter sampling for list format."""
        param_config = ['option1', 'option2', 'option3']
        
        samples = [SearchSpaceHandler.sample_parameter('test', param_config) for _ in range(100)]
        
        assert all(sample in param_config for sample in samples)
        assert len(set(samples)) > 1
    
    def test_get_grid_values(self):
        """Test grid value generation."""
        # Choice type
        choice_config = {'type': 'choice', 'choices': [1, 2, 3]}
        values = SearchSpaceHandler.get_grid_values(choice_config)
        assert values == [1, 2, 3]
        
        # Uniform type
        uniform_config = {'type': 'uniform', 'low': 0, 'high': 1}
        values = SearchSpaceHandler.get_grid_values(uniform_config, num_values=5)
        assert len(values) == 5
        assert values[0] == 0
        assert values[-1] == 1
        
        # Int type
        int_config = {'type': 'int', 'low': 2, 'high': 5}
        values = SearchSpaceHandler.get_grid_values(int_config)
        assert values == [2, 3, 4, 5]


class TestGridSearchOptimizer:
    """Test GridSearchOptimizer."""
    
    @pytest.fixture
    def mock_objective(self):
        """Create mock objective function."""
        def objective(params):
            # Simple quadratic function with known optimum
            x = params.get('x', 0)
            y = params.get('y', 0)
            return -(x - 0.5)**2 - (y - 0.3)**2
        return objective
    
    @pytest.fixture
    def simple_search_space(self):
        """Create simple search space for testing."""
        return {
            'x': {'type': 'uniform', 'low': 0, 'high': 1},
            'y': {'type': 'uniform', 'low': 0, 'high': 1}
        }
    
    def test_grid_search_initialization(self, mock_objective, simple_search_space):
        """Test grid search optimizer initialization."""
        config = OptimizationConfig(max_trials=10)
        
        optimizer = GridSearchOptimizer(
            mock_objective, simple_search_space, config, grid_size=3
        )
        
        assert len(optimizer.parameter_grid) > 0
        assert optimizer.grid_index == 0
        assert optimizer.grid_size == 3
    
    def test_grid_search_parameter_generation(self, mock_objective, simple_search_space):
        """Test parameter generation in grid search."""
        config = OptimizationConfig(max_trials=10)
        
        optimizer = GridSearchOptimizer(
            mock_objective, simple_search_space, config, grid_size=2
        )
        
        # Generate parameters for first few trials
        params1 = optimizer._generate_trial_parameters(0)
        params2 = optimizer._generate_trial_parameters(1)
        
        assert 'x' in params1 and 'y' in params1
        assert 'x' in params2 and 'y' in params2
        assert params1 != params2  # Should be different
    
    @patch('src.optimization.base_optimizer.BaseHyperparameterOptimizer._execute_trial_with_monitoring')
    def test_grid_search_exhausts_grid(self, mock_execute, mock_objective, simple_search_space):
        """Test that grid search handles grid exhaustion."""
        config = OptimizationConfig(max_trials=100)  # More than grid size
        
        # Mock trial execution to return dummy scores
        mock_execute.return_value = 0.5
        
        optimizer = GridSearchOptimizer(
            mock_objective, simple_search_space, config, grid_size=2
        )
        
        initial_grid_size = len(optimizer.parameter_grid)
        
        # Generate more parameters than grid size
        for i in range(initial_grid_size + 5):
            params = optimizer._generate_trial_parameters(i)
            assert params is not None


class TestRandomSearchOptimizer:
    """Test RandomSearchOptimizer."""
    
    @pytest.fixture
    def mock_objective(self):
        """Create mock objective function."""
        return lambda params: np.random.random()
    
    @pytest.fixture
    def search_space(self):
        """Create search space for testing."""
        return EXAMPLE_SEARCH_SPACES['basic']
    
    def test_random_search_initialization(self, mock_objective, search_space):
        """Test random search optimizer initialization."""
        config = OptimizationConfig(max_trials=20)
        
        optimizer = RandomSearchOptimizer(mock_objective, search_space, config)
        
        assert optimizer.search_space == search_space
        assert optimizer.config.max_trials == 20
    
    def test_random_search_parameter_generation(self, mock_objective, search_space):
        """Test parameter generation in random search."""
        config = OptimizationConfig(max_trials=20)
        
        optimizer = RandomSearchOptimizer(mock_objective, search_space, config)
        
        # Generate multiple parameter sets
        param_sets = [optimizer._generate_trial_parameters(i) for i in range(10)]
        
        # Should have variety
        assert len(set(str(params) for params in param_sets)) > 1
        
        # All should contain expected parameters
        for params in param_sets:
            assert 'learning_rate' in params
            assert 'batch_size' in params
            assert 'hidden_size' in params
            assert 'dropout' in params


class TestBayesianOptimizer:
    """Test BayesianOptimizer."""
    
    @pytest.fixture
    def mock_objective(self):
        """Create mock objective function."""
        def objective(params):
            # Simple function for testing
            return params.get('x', 0) * params.get('y', 1)
        return objective
    
    @pytest.fixture
    def simple_search_space(self):
        """Create simple search space."""
        return {
            'x': {'type': 'uniform', 'low': 0, 'high': 1},
            'y': {'type': 'uniform', 'low': 0, 'high': 1}
        }
    
    def test_bayesian_optimizer_initialization(self, mock_objective, simple_search_space):
        """Test Bayesian optimizer initialization."""
        config = OptimizationConfig(max_trials=20)
        
        optimizer = BayesianOptimizer(mock_objective, simple_search_space, config)
        
        assert optimizer.acquisition_function == "expected_improvement"
        assert optimizer.exploration_trials > 0
        assert optimizer.gp_model is None  # Not yet trained
    
    def test_bayesian_exploration_phase(self, mock_objective, simple_search_space):
        """Test exploration phase of Bayesian optimization."""
        config = OptimizationConfig(max_trials=20)
        
        optimizer = BayesianOptimizer(mock_objective, simple_search_space, config)
        
        # During exploration, should generate random parameters
        params = optimizer._generate_trial_parameters(0)  # First trial
        
        assert 'x' in params and 'y' in params
        assert 0 <= params['x'] <= 1
        assert 0 <= params['y'] <= 1
    
    def test_parameters_to_array_conversion(self, mock_objective, simple_search_space):
        """Test parameter to array conversion."""
        config = OptimizationConfig(max_trials=20)
        
        optimizer = BayesianOptimizer(mock_objective, simple_search_space, config)
        
        params = {'x': 0.5, 'y': 0.3}
        array = optimizer._parameters_to_array(params)
        
        assert len(array) == 2  # Two parameters
        assert isinstance(array, np.ndarray)
    
    def test_acquisition_functions(self, mock_objective, simple_search_space):
        """Test acquisition function calculations."""
        config = OptimizationConfig(max_trials=20)
        
        optimizer = BayesianOptimizer(mock_objective, simple_search_space, config)
        optimizer.y_observed = [0.1, 0.5, 0.3]  # Some observed values
        
        means = np.array([0.4, 0.6, 0.2])
        stds = np.array([0.1, 0.2, 0.3])
        
        # Test expected improvement
        ei = optimizer._expected_improvement(means, stds)
        assert len(ei) == 3
        assert all(ei >= 0)
        
        # Test upper confidence bound
        ucb = optimizer._upper_confidence_bound(means, stds)
        assert len(ucb) == 3
        assert all(ucb >= means)
        
        # Test probability of improvement
        pi = optimizer._probability_of_improvement(means, stds)
        assert len(pi) == 3
        assert all((pi >= 0) & (pi <= 1))


@pytest.mark.skipif(not hasattr(pytest, 'importorskip'), reason="Optuna tests require special handling")
class TestOptunaOptimizer:
    """Test OptunaOptimizer (if Optuna is available)."""
    
    def setup_method(self):
        """Setup for Optuna tests."""
        try:
            import optuna
            self.optuna_available = True
        except ImportError:
            self.optuna_available = False
    
    @pytest.fixture
    def mock_objective(self):
        """Create mock objective function."""
        return lambda params: np.random.random()
    
    @pytest.fixture
    def search_space(self):
        """Create search space for testing."""
        return {
            'x': {'type': 'uniform', 'low': 0, 'high': 1},
            'y': {'type': 'choice', 'choices': [1, 2, 3]},
            'z': {'type': 'int', 'low': 2, 'high': 10}
        }
    
    def test_optuna_optimizer_initialization(self, mock_objective, search_space):
        """Test Optuna optimizer initialization."""
        if not self.optuna_available:
            pytest.skip("Optuna not available")
        
        config = OptimizationConfig(max_trials=20)
        
        optimizer = OptunaOptimizer(mock_objective, search_space, config)
        
        assert optimizer.sampler_type == "tpe"
        assert optimizer.study is not None
    
    def test_optuna_parameter_sampling(self, mock_objective, search_space):
        """Test Optuna parameter sampling."""
        if not self.optuna_available:
            pytest.skip("Optuna not available")
        
        config = OptimizationConfig(max_trials=20)
        
        optimizer = OptunaOptimizer(mock_objective, search_space, config)
        
        params = optimizer._generate_trial_parameters(0)
        
        assert 'x' in params
        assert 'y' in params  
        assert 'z' in params
        assert 0 <= params['x'] <= 1
        assert params['y'] in [1, 2, 3]
        assert 2 <= params['z'] <= 10


class TestCreateOptimizer:
    """Test optimizer factory function."""
    
    @pytest.fixture
    def mock_objective(self):
        """Create mock objective function."""
        return lambda params: 0.5
    
    @pytest.fixture
    def search_space(self):
        """Create search space for testing."""
        return {'x': {'type': 'uniform', 'low': 0, 'high': 1}}
    
    @pytest.fixture
    def config(self):
        """Create optimization config."""
        return OptimizationConfig(max_trials=10)
    
    def test_create_grid_search_optimizer(self, mock_objective, search_space, config):
        """Test creating grid search optimizer."""
        optimizer = create_optimizer('grid_search', mock_objective, search_space, config)
        assert isinstance(optimizer, GridSearchOptimizer)
    
    def test_create_random_search_optimizer(self, mock_objective, search_space, config):
        """Test creating random search optimizer."""
        optimizer = create_optimizer('random_search', mock_objective, search_space, config)
        assert isinstance(optimizer, RandomSearchOptimizer)
    
    def test_create_bayesian_optimizer(self, mock_objective, search_space, config):
        """Test creating Bayesian optimizer."""
        optimizer = create_optimizer('bayesian', mock_objective, search_space, config)
        assert isinstance(optimizer, BayesianOptimizer)
    
    def test_create_optuna_optimizer_fallback(self, mock_objective, search_space, config):
        """Test creating Optuna optimizer with fallback."""
        # This should either create OptunaOptimizer or fall back to BayesianOptimizer
        optimizer = create_optimizer('optuna', mock_objective, search_space, config)
        assert isinstance(optimizer, (OptunaOptimizer, BayesianOptimizer))
    
    def test_create_optimizer_invalid_algorithm(self, mock_objective, search_space, config):
        """Test creating optimizer with invalid algorithm."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            create_optimizer('invalid_algorithm', mock_objective, search_space, config)


class TestExampleSearchSpaces:
    """Test example search spaces."""
    
    def test_basic_search_space_validity(self):
        """Test that basic search space is valid."""
        search_space = EXAMPLE_SEARCH_SPACES['basic']
        is_valid = SearchSpaceHandler.validate_search_space(search_space)
        assert is_valid is True
    
    def test_advanced_search_space_validity(self):
        """Test that advanced search space is valid."""
        search_space = EXAMPLE_SEARCH_SPACES['advanced']
        is_valid = SearchSpaceHandler.validate_search_space(search_space)
        assert is_valid is True
    
    def test_example_search_spaces_completeness(self):
        """Test that example search spaces contain expected parameters."""
        basic = EXAMPLE_SEARCH_SPACES['basic']
        advanced = EXAMPLE_SEARCH_SPACES['advanced']
        
        # Basic should contain fundamental parameters
        basic_params = {'learning_rate', 'batch_size', 'hidden_size', 'dropout'}
        assert basic_params.issubset(set(basic.keys()))
        
        # Advanced should contain basic + more
        assert basic_params.issubset(set(advanced.keys()))
        assert len(advanced) > len(basic)


if __name__ == '__main__':
    pytest.main([__file__])