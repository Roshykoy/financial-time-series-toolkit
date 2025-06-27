"""
Tests for optimization integration with training pipeline.
"""

import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.optimization.integration import (
    ModelTrainingInterface,
    OptimizationObjective,
    create_training_interface,
    create_objective_function
)
from src.optimization.hardware_manager import HardwareResourceManager


class TestModelTrainingInterface:
    """Test ModelTrainingInterface functionality."""
    
    @pytest.fixture
    def mock_data_path(self):
        """Create mock data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write minimal CSV data
            f.write("Draw,Date,Winning_Num_1,Winning_Num_2,Winning_Num_3,Winning_Num_4,Winning_Num_5,Winning_Num_6,Extra_Num\n")
            for i in range(50):
                f.write(f"{i},2024-01-{i+1:02d},1,2,3,4,5,6,7\n")
            return Path(f.name)
    
    @pytest.fixture
    def mock_hardware_manager(self):
        """Create mock hardware manager."""
        manager = Mock(spec=HardwareResourceManager)
        manager.optimize_for_hardware.return_value = {'device': 'cpu', 'batch_size': 8}
        manager.check_resource_constraints.return_value = (True, [])
        manager.cleanup_resources.return_value = None
        return manager
    
    @pytest.fixture
    def mock_base_config(self):
        """Create mock base configuration."""
        return {
            'learning_rate': 1e-4,
            'batch_size': 16,
            'epochs': 3,
            'device': 'cpu',
            'num_lotto_numbers': 49
        }
    
    def test_interface_initialization(self, mock_data_path, mock_hardware_manager, mock_base_config):
        """Test training interface initialization."""
        interface = ModelTrainingInterface(
            str(mock_data_path), mock_hardware_manager, mock_base_config
        )
        
        assert interface.data_path == str(mock_data_path)
        assert interface.hardware_manager == mock_hardware_manager
        assert interface.base_config == mock_base_config
        assert interface.feature_engineer is None
        assert interface.device is None
    
    @patch('src.optimization.integration.pd.read_csv')
    def test_load_data(self, mock_read_csv, mock_data_path, mock_hardware_manager, mock_base_config):
        """Test data loading functionality."""
        # Mock pandas read_csv
        mock_df = pd.DataFrame({
            'Draw': range(100),
            'Date': pd.date_range('2024-01-01', periods=100),
            'Winning_Num_1': np.random.randint(1, 50, 100),
            'Winning_Num_2': np.random.randint(1, 50, 100),
            'Winning_Num_3': np.random.randint(1, 50, 100),
            'Winning_Num_4': np.random.randint(1, 50, 100),
            'Winning_Num_5': np.random.randint(1, 50, 100),
            'Winning_Num_6': np.random.randint(1, 50, 100),
            'Extra_Num': np.random.randint(1, 50, 100)
        })
        mock_read_csv.return_value = mock_df
        
        interface = ModelTrainingInterface(
            str(mock_data_path), mock_hardware_manager, mock_base_config
        )
        
        df = interface._load_data()
        
        assert len(df) == 100
        assert 'Date' in df.columns
        mock_read_csv.assert_called_once()
    
    def test_get_device_cpu(self, mock_data_path, mock_hardware_manager, mock_base_config):
        """Test device selection for CPU."""
        config = mock_base_config.copy()
        config['device'] = 'cpu'
        
        interface = ModelTrainingInterface(
            str(mock_data_path), mock_hardware_manager, mock_base_config
        )
        
        device = interface._get_device(config)
        assert device.type == 'cpu'
    
    def test_get_device_auto_no_cuda(self, mock_data_path, mock_hardware_manager, mock_base_config):
        """Test device selection when CUDA not available."""
        config = mock_base_config.copy()
        config['device'] = 'auto'
        
        with patch('torch.cuda.is_available', return_value=False):
            interface = ModelTrainingInterface(
                str(mock_data_path), mock_hardware_manager, mock_base_config
            )
            
            device = interface._get_device(config)
            assert device.type == 'cpu'
    
    @patch('src.optimization.integration.ConditionalVAE')
    @patch('src.optimization.integration.AttentionMetaLearner')
    def test_create_model(self, mock_meta_learner, mock_cvae, mock_data_path, mock_hardware_manager, mock_base_config):
        """Test model creation."""
        # Mock model constructors
        mock_cvae_instance = Mock()
        mock_cvae_instance.to.return_value = mock_cvae_instance
        mock_cvae.return_value = mock_cvae_instance
        
        mock_meta_instance = Mock()
        mock_meta_instance.to.return_value = mock_meta_instance
        mock_meta_learner.return_value = mock_meta_instance
        
        interface = ModelTrainingInterface(
            str(mock_data_path), mock_hardware_manager, mock_base_config
        )
        
        config = mock_base_config.copy()
        cvae_model, meta_learner = interface.create_model(config)
        
        assert cvae_model == mock_cvae_instance
        assert meta_learner == mock_meta_instance
        mock_cvae.assert_called_once()
        mock_meta_learner.assert_called_once()
    
    def test_validate_config_valid(self, mock_data_path, mock_hardware_manager, mock_base_config):
        """Test configuration validation with valid config."""
        interface = ModelTrainingInterface(
            str(mock_data_path), mock_hardware_manager, mock_base_config
        )
        
        config = {
            'learning_rate': 1e-4,
            'batch_size': 16,
            'epochs': 5,
            'dropout': 0.2
        }
        
        is_valid, errors = interface.validate_config(config)
        
        assert is_valid is True
        assert len(errors) == 0
    
    def test_validate_config_invalid(self, mock_data_path, mock_hardware_manager, mock_base_config):
        """Test configuration validation with invalid config."""
        interface = ModelTrainingInterface(
            str(mock_data_path), mock_hardware_manager, mock_base_config
        )
        
        config = {
            'learning_rate': 10.0,  # Too high
            'batch_size': 1000,     # Too high
            'epochs': -1,           # Invalid
            'dropout': 1.5          # Too high
        }
        
        is_valid, errors = interface.validate_config(config)
        
        assert is_valid is False
        assert len(errors) > 0
    
    def test_estimate_training_time(self, mock_data_path, mock_hardware_manager, mock_base_config):
        """Test training time estimation."""
        interface = ModelTrainingInterface(
            str(mock_data_path), mock_hardware_manager, mock_base_config
        )
        
        config = {
            'epochs': 5,
            'batch_size': 16,
            'hidden_size': 256,
            'latent_dim': 64,
            'device': 'cpu'
        }
        
        estimated_time = interface.estimate_training_time(config)
        
        assert isinstance(estimated_time, float)
        assert estimated_time > 0
    
    def test_get_search_space(self, mock_data_path, mock_hardware_manager, mock_base_config):
        """Test getting default search space."""
        interface = ModelTrainingInterface(
            str(mock_data_path), mock_hardware_manager, mock_base_config
        )
        
        search_space = interface.get_search_space()
        
        # Should contain key parameters
        expected_params = {
            'learning_rate', 'batch_size', 'epochs', 'latent_dim', 'dropout'
        }
        assert expected_params.issubset(set(search_space.keys()))
        
        # Check structure
        for param_name, param_config in search_space.items():
            assert isinstance(param_config, dict)
            assert 'type' in param_config
    
    def test_clear_cache(self, mock_data_path, mock_hardware_manager, mock_base_config):
        """Test cache clearing."""
        interface = ModelTrainingInterface(
            str(mock_data_path), mock_hardware_manager, mock_base_config
        )
        
        # Add some dummy cache data
        interface._data_cache['test'] = Mock()
        interface._feature_cache['test'] = Mock()
        
        interface.clear_cache()
        
        assert len(interface._data_cache) == 0
        assert len(interface._feature_cache) == 0
    
    @patch('src.optimization.integration.create_cvae_data_loaders')
    @patch('src.optimization.integration.train_one_epoch_cvae')
    @patch('src.optimization.integration.evaluate_cvae')
    def test_train_model_simple(self, mock_evaluate, mock_train_epoch, mock_create_loaders,
                               mock_data_path, mock_hardware_manager, mock_base_config):
        """Test model training (simplified mock version)."""
        # Mock data loaders
        mock_train_loader = Mock()
        mock_val_loader = Mock()
        mock_test_loader = Mock()
        mock_create_loaders.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)
        
        # Mock training functions
        mock_train_epoch.return_value = 0.5
        mock_evaluate.return_value = (0.3, {})  # val_loss, metrics
        
        # Mock model creation
        with patch.object(ModelTrainingInterface, 'create_model') as mock_create_model:
            mock_cvae = Mock()
            mock_meta = Mock()
            mock_create_model.return_value = (mock_cvae, mock_meta)
            
            interface = ModelTrainingInterface(
                str(mock_data_path), mock_hardware_manager, mock_base_config
            )
            
            config = {
                'epochs': 2,
                'batch_size': 8,
                'learning_rate': 1e-4,
                'device': 'cpu',
                'early_stopping_patience': 3
            }
            
            # Mock data preparation
            with patch.object(interface, 'prepare_data') as mock_prepare:
                mock_prepare.return_value = (mock_train_loader, mock_val_loader, mock_test_loader)
                
                score = interface.train_model(config)
                
                assert isinstance(score, float)
                # Score should be negative val_loss
                assert score == -0.3


class TestOptimizationObjective:
    """Test OptimizationObjective functionality."""
    
    @pytest.fixture
    def mock_training_interface(self):
        """Create mock training interface."""
        interface = Mock(spec=ModelTrainingInterface)
        interface.validate_config.return_value = (True, [])
        interface.train_model.return_value = 0.8
        return interface
    
    @pytest.fixture
    def base_config(self):
        """Create base configuration."""
        return {
            'epochs': 5,
            'device': 'cpu',
            'num_lotto_numbers': 49
        }
    
    def test_objective_initialization(self, mock_training_interface, base_config):
        """Test objective function initialization."""
        objective = OptimizationObjective(
            mock_training_interface, base_config, "holdout"
        )
        
        assert objective.training_interface == mock_training_interface
        assert objective.base_config == base_config
        assert objective.validation_strategy == "holdout"
        assert objective.call_count == 0
        assert objective.best_score == float('-inf')
    
    def test_objective_call_success(self, mock_training_interface, base_config):
        """Test successful objective function call."""
        objective = OptimizationObjective(
            mock_training_interface, base_config, "holdout"
        )
        
        parameters = {
            'learning_rate': 1e-4,
            'batch_size': 16
        }
        
        score = objective(parameters)
        
        assert score == 0.8
        assert objective.call_count == 1
        assert objective.best_score == 0.8
        
        # Verify config merging
        expected_config = base_config.copy()
        expected_config.update(parameters)
        mock_training_interface.train_model.assert_called_once_with(expected_config)
    
    def test_objective_call_invalid_config(self, mock_training_interface, base_config):
        """Test objective function call with invalid config."""
        # Mock validation to return invalid
        mock_training_interface.validate_config.return_value = (False, ["Error"])
        
        objective = OptimizationObjective(
            mock_training_interface, base_config, "holdout"
        )
        
        parameters = {'learning_rate': -1}  # Invalid
        
        score = objective(parameters)
        
        assert score == float('-inf')
        assert objective.call_count == 1
        mock_training_interface.train_model.assert_not_called()
    
    def test_objective_call_training_failure(self, mock_training_interface, base_config):
        """Test objective function call with training failure."""
        # Mock training to raise exception
        mock_training_interface.train_model.side_effect = Exception("Training failed")
        
        objective = OptimizationObjective(
            mock_training_interface, base_config, "holdout"
        )
        
        parameters = {'learning_rate': 1e-4}
        
        score = objective(parameters)
        
        assert score == float('-inf')
        assert objective.call_count == 1
    
    def test_objective_best_score_tracking(self, mock_training_interface, base_config):
        """Test best score tracking across calls."""
        objective = OptimizationObjective(
            mock_training_interface, base_config, "holdout"
        )
        
        # Mock different scores
        mock_training_interface.train_model.side_effect = [0.5, 0.8, 0.3, 0.9]
        
        scores = []
        for i in range(4):
            score = objective({'param': i})
            scores.append(score)
        
        assert scores == [0.5, 0.8, 0.3, 0.9]
        assert objective.best_score == 0.9
        assert objective.call_count == 4
    
    def test_objective_statistics(self, mock_training_interface, base_config):
        """Test objective function statistics."""
        objective = OptimizationObjective(
            mock_training_interface, base_config, "holdout"
        )
        
        # Make a few calls
        for i in range(3):
            objective({'param': i})
        
        stats = objective.get_statistics()
        
        assert stats['call_count'] == 3
        assert 'total_time_seconds' in stats
        assert 'average_time_seconds' in stats
        assert 'best_score' in stats
        assert stats['average_time_seconds'] >= 0


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_training_interface(self):
        """Test training interface factory function."""
        interface = create_training_interface("dummy_path.csv")
        
        assert isinstance(interface, ModelTrainingInterface)
        assert interface.data_path == "dummy_path.csv"
    
    def test_create_objective_function(self):
        """Test objective function factory function."""
        mock_interface = Mock(spec=ModelTrainingInterface)
        base_config = {'param': 'value'}
        
        objective = create_objective_function(mock_interface, base_config)
        
        assert isinstance(objective, OptimizationObjective)
        assert objective.training_interface == mock_interface
        assert objective.base_config == base_config
        assert objective.validation_strategy == "holdout"


class TestIntegrationWithRealComponents:
    """Integration tests with real components (mocked external dependencies)."""
    
    @pytest.fixture
    def sample_csv_data(self):
        """Create sample CSV data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write header
            headers = [
                'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
                'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num'
            ] + [f'Col_{i}' for i in range(25)]  # Additional columns
            
            f.write(','.join(headers) + '\n')
            
            # Write 50 rows of sample data
            for i in range(50):
                row = [
                    str(i),  # Draw
                    f'2024-01-{i+1:02d}',  # Date
                    '1', '2', '3', '4', '5', '6',  # Winning numbers
                    '7',  # Extra number
                ] + ['0'] * 25  # Additional columns
                
                f.write(','.join(row) + '\n')
            
            return Path(f.name)
    
    @patch('src.optimization.integration.create_cvae_data_loaders')
    @patch('src.optimization.integration.ConditionalVAE')
    @patch('src.optimization.integration.AttentionMetaLearner')
    @patch('src.optimization.integration.train_one_epoch_cvae')
    @patch('src.optimization.integration.evaluate_cvae')
    def test_full_training_pipeline_mock(self, mock_evaluate, mock_train_epoch, 
                                        mock_meta_learner, mock_cvae, mock_create_loaders,
                                        sample_csv_data):
        """Test full training pipeline with mocked components."""
        # Mock data loaders
        mock_create_loaders.return_value = (Mock(), Mock(), Mock())
        
        # Mock models
        mock_cvae_instance = Mock()
        mock_cvae_instance.to.return_value = mock_cvae_instance
        mock_cvae.return_value = mock_cvae_instance
        
        mock_meta_instance = Mock()
        mock_meta_instance.to.return_value = mock_meta_instance
        mock_meta_learner.return_value = mock_meta_instance
        
        # Mock training functions
        mock_train_epoch.return_value = 0.5
        mock_evaluate.return_value = (0.3, {})
        
        # Create interface
        interface = ModelTrainingInterface(str(sample_csv_data))
        
        # Test configuration
        config = {
            'learning_rate': 1e-4,
            'batch_size': 8,
            'epochs': 2,
            'device': 'cpu',
            'latent_dim': 32,
            'dropout': 0.1
        }
        
        # Train model
        score = interface.train_model(config)
        
        # Verify calls were made
        assert isinstance(score, float)
        mock_cvae.assert_called_once()
        mock_meta_learner.assert_called_once()
        mock_train_epoch.assert_called()
        mock_evaluate.assert_called()
        
        # Clean up
        sample_csv_data.unlink()


if __name__ == '__main__':
    pytest.main([__file__])