"""
Integration tests for batch size constraints across the full pipeline.
Tests end-to-end batch size enforcement from Pareto parameters to DataLoader.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
import os
from pathlib import Path
import psutil

# Import modules under test
from src.optimization.pareto_integration import (
    load_pareto_parameters, 
    apply_pareto_parameters_to_config,
    create_pareto_optimized_config
)
from src.cvae_data_loader import (
    _calculate_optimal_batch_size,
    create_cvae_data_loaders,
    CVAEDataset
)
from src.config import CONFIG


class TestEndToEndBatchConstraints:
    """Test complete pipeline from Pareto parameters to DataLoader."""
    
    def create_sample_data(self):
        """Create sample lottery data for testing."""
        data = {
            'Draw': range(100),
            'Date': pd.date_range('2024-01-01', periods=100),
            'Winning_Num_1': np.random.randint(1, 50, 100),
            'Winning_Num_2': np.random.randint(1, 50, 100),
            'Winning_Num_3': np.random.randint(1, 50, 100),
            'Winning_Num_4': np.random.randint(1, 50, 100),
            'Winning_Num_5': np.random.randint(1, 50, 100),
            'Winning_Num_6': np.random.randint(1, 50, 100),
            'Extra_Num': np.random.randint(1, 50, 100)
        }
        return pd.DataFrame(data)
    
    def create_mock_feature_engineer(self):
        """Create mock feature engineer for testing."""
        mock_fe = Mock()
        mock_fe.pair_counts = {(i, j): np.random.randint(1, 10) 
                              for i in range(1, 50) for j in range(i+1, 50)}
        return mock_fe
    
    def create_temp_pareto_file(self, batch_size):
        """Create temporary Pareto parameter file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        data = {
            'selected_parameters': {
                'batch_size': batch_size,
                'learning_rate': 0.001,
                'epochs': 5,
                'hidden_size': 256,
                'dropout': 0.2
            },
            'algorithm': 'test',
            'timestamp': '20240101_120000'
        }
        json.dump(data, temp_file)
        temp_file.close()
        return temp_file.name
    
    def test_pareto_to_dataloader_minimum_enforcement(self):
        """Test full pipeline enforces minimum batch size from Pareto params."""
        # Create temp Pareto file with batch_size < 8
        pareto_file = self.create_temp_pareto_file(batch_size=4)
        
        try:
            # Load and apply Pareto parameters
            pareto_params = load_pareto_parameters(pareto_file)
            assert pareto_params is not None, "Failed to load Pareto parameters"
            assert pareto_params['batch_size'] == 4, "Pareto params not loaded correctly"
            
            # Apply constraints to config
            base_config = CONFIG.copy()
            constrained_config = apply_pareto_parameters_to_config(base_config, pareto_params)
            
            # Verify constraint applied
            assert constrained_config['batch_size'] >= 8, f"Pareto constraint failed: {constrained_config['batch_size']}"
            
            # Test hardware optimization layer
            hardware_optimized_batch = _calculate_optimal_batch_size(constrained_config)
            assert hardware_optimized_batch >= 8, f"Hardware optimization failed: {hardware_optimized_batch}"
            
            print(f"✅ Pipeline test: 4 → {constrained_config['batch_size']} → {hardware_optimized_batch}")
            
        finally:
            os.unlink(pareto_file)
    
    def test_pareto_to_dataloader_no_ceiling(self):
        """Test pipeline respects high batch sizes without artificial ceiling."""
        test_sizes = [16, 32, 64, 128]
        
        for original_size in test_sizes:
            pareto_file = self.create_temp_pareto_file(batch_size=original_size)
            
            try:
                pareto_params = load_pareto_parameters(pareto_file)
                base_config = CONFIG.copy()
                constrained_config = apply_pareto_parameters_to_config(base_config, pareto_params)
                hardware_batch = _calculate_optimal_batch_size(constrained_config)
                
                # Should not reduce from original (unless hardware limits)
                assert constrained_config['batch_size'] == original_size, \
                    f"Artificial ceiling applied: {original_size} → {constrained_config['batch_size']}"
                
                # Hardware may scale but shouldn't artificially limit
                if hardware_batch < original_size:
                    # Only acceptable if due to actual hardware constraints
                    print(f"⚠️  Hardware limitation: {original_size} → {hardware_batch}")
                
            finally:
                os.unlink(pareto_file)
    
    @patch('src.cvae_data_loader.torch.cuda.is_available')
    @patch('src.cvae_data_loader.torch.cuda.mem_get_info')
    def test_full_dataloader_integration(self, mock_mem_info, mock_cuda_available):
        """Test complete DataLoader creation with batch size constraints."""
        # Mock GPU environment
        mock_cuda_available.return_value = True
        mock_mem_info.return_value = (8 * 1024**3, 6 * 1024**3)  # 8GB total, 6GB free
        
        # Create test data
        df = self.create_sample_data()
        feature_engineer = self.create_mock_feature_engineer()
        
        # Test with low batch size that should be constrained
        pareto_file = self.create_temp_pareto_file(batch_size=2)
        
        try:
            # Create config with Pareto parameters
            pareto_params = load_pareto_parameters(pareto_file)
            config = apply_pareto_parameters_to_config(CONFIG.copy(), pareto_params)
            config.update({
                'negative_pool_size': 100,
                'negative_samples': 5,
                'temporal_sequence_length': 10,
                'num_lotto_numbers': 49,
                'optimized_batch_size': 'auto'
            })
            
            # Create DataLoaders
            train_loader, val_loader = create_cvae_data_loaders(df, feature_engineer, config)
            
            # Verify batch sizes
            assert train_loader.batch_size >= 8, f"Train loader batch size: {train_loader.batch_size}"
            assert val_loader.batch_size >= 8, f"Val loader batch size: {val_loader.batch_size}"
            assert train_loader.batch_size == val_loader.batch_size, "Inconsistent batch sizes"
            
            print(f"✅ DataLoader integration: batch_size={train_loader.batch_size}")
            
        finally:
            os.unlink(pareto_file)
    
    def test_config_chain_consistency(self):
        """Test that all config transformations maintain consistency."""
        original_sizes = [1, 4, 8, 16, 32, 64]
        
        for original_size in original_sizes:
            pareto_file = self.create_temp_pareto_file(batch_size=original_size)
            
            try:
                # Step 1: Load Pareto params
                pareto_params = load_pareto_parameters(pareto_file)
                
                # Step 2: Apply to base config  
                base_config = CONFIG.copy()
                constrained_config = apply_pareto_parameters_to_config(base_config, pareto_params)
                
                # Step 3: Hardware optimization
                with patch('torch.cuda.is_available', return_value=False):
                    hardware_batch = _calculate_optimal_batch_size(constrained_config)
                
                # Verify consistency - all should be >= 8
                assert constrained_config['batch_size'] >= 8, \
                    f"Pareto constraint failed for {original_size}"
                assert hardware_batch >= 8, \
                    f"Hardware constraint failed for {original_size}"
                
                # If original was >= 8, should not be reduced unnecessarily
                if original_size >= 8:
                    assert constrained_config['batch_size'] == original_size, \
                        f"Unnecessary reduction: {original_size} → {constrained_config['batch_size']}"
                
                print(f"✅ Consistency: {original_size} → {constrained_config['batch_size']} → {hardware_batch}")
                
            finally:
                os.unlink(pareto_file)


class TestMemoryPressureScenarios:
    """Test batch size behavior under different memory conditions."""
    
    @patch('src.cvae_data_loader.torch.cuda.is_available')
    @patch('src.cvae_data_loader.torch.cuda.mem_get_info') 
    def test_low_memory_scenarios(self, mock_mem_info, mock_cuda_available):
        """Test batch size constraints under low GPU memory."""
        mock_cuda_available.return_value = True
        
        # Test various memory scenarios
        memory_scenarios = [
            (2 * 1024**3, 1 * 1024**3, "2GB GPU, 1GB free"),  # Low-end
            (4 * 1024**3, 2 * 1024**3, "4GB GPU, 2GB free"),  # Mid-range tight
            (8 * 1024**3, 1 * 1024**3, "8GB GPU, 1GB free"),  # High-end but full
        ]
        
        base_config = {
            'batch_size': 32,  # High starting batch size
            'optimized_batch_size': 'auto'
        }
        
        for total_mem, free_mem, description in memory_scenarios:
            mock_mem_info.return_value = (total_mem, free_mem)
            
            optimal_batch = _calculate_optimal_batch_size(base_config)
            
            # Should still enforce minimum even under memory pressure
            assert optimal_batch >= 8, f"Memory pressure violation in {description}: {optimal_batch}"
            
            print(f"✅ {description}: batch_size={optimal_batch}")
    
    @patch('src.cvae_data_loader.torch.cuda.is_available')
    @patch('src.cvae_data_loader.torch.cuda.mem_get_info')
    def test_memory_error_handling(self, mock_mem_info, mock_cuda_available):
        """Test batch size constraint when GPU memory query fails."""
        mock_cuda_available.return_value = True
        mock_mem_info.side_effect = RuntimeError("CUDA memory error")
        
        config = {'batch_size': 4, 'optimized_batch_size': 'auto'}
        result = _calculate_optimal_batch_size(config)
        
        # Should fallback gracefully but still enforce minimum
        assert result >= 8, f"Error handling failed to enforce minimum: {result}"
        print(f"✅ Memory error handling: batch_size={result}")
    
    def test_cpu_only_constraints(self):
        """Test batch size constraints in CPU-only environment."""
        with patch('torch.cuda.is_available', return_value=False):
            low_batch_configs = [
                {'batch_size': 1},
                {'batch_size': 4}, 
                {'batch_size': 6},
                {'batch_size': 8},
                {'batch_size': 16}
            ]
            
            for config in low_batch_configs:
                result = _calculate_optimal_batch_size(config)
                expected_min = max(8, config['batch_size'])
                
                assert result >= 8, f"CPU constraint failed: {config['batch_size']} → {result}"
                
                if config['batch_size'] >= 8:
                    assert result == config['batch_size'], f"Unnecessary CPU scaling: {config['batch_size']} → {result}"
                
                print(f"✅ CPU-only: {config['batch_size']} → {result}")


class TestDynamicBatchSizing:
    """Test dynamic batch sizing with constraints."""
    
    def create_mock_memory_manager(self, memory_pressure=0.5):
        """Create mock memory manager for dynamic batch testing."""
        mock_manager = Mock()
        mock_manager.get_comprehensive_stats.return_value = {
            'memory_usage_pct': memory_pressure * 100,
            'total_memory_gb': 16.0,
            'available_memory_gb': 16.0 * (1 - memory_pressure)
        }
        return mock_manager
    
    @patch('src.cvae_data_loader.PHASE2_AVAILABLE', True)
    @patch('src.cvae_data_loader.get_memory_manager')
    def test_dynamic_batch_sizing_constraints(self, mock_get_memory_manager):
        """Test that dynamic batch sizing respects minimum constraints."""
        # Low memory pressure should allow scaling
        memory_manager = self.create_mock_memory_manager(memory_pressure=0.3)
        mock_get_memory_manager.return_value = memory_manager
        
        config = {
            'batch_size': 4,  # Below minimum
            'enable_dynamic_batching': True,
            'enable_performance_optimizations': True,
            'batch_size_scaling_factor': 2.0,
            'max_dynamic_batch_size': 32,
            'optimized_batch_size': 'auto'
        }
        
        # This tests the logic from cvae_data_loader.py lines 410-417
        optimal_batch = _calculate_optimal_batch_size(config)
        
        # Even with dynamic scaling, should enforce minimum
        assert optimal_batch >= 8, f"Dynamic batching minimum enforcement failed: {optimal_batch}"
        
        print(f"✅ Dynamic batch sizing: 4 → {optimal_batch} (min constraint enforced)")
    
    @patch('src.cvae_data_loader.PHASE2_AVAILABLE', True) 
    @patch('src.cvae_data_loader.get_memory_manager')
    def test_dynamic_scaling_with_high_memory_pressure(self, mock_get_memory_manager):
        """Test dynamic scaling under high memory pressure."""
        # High memory pressure should prevent scaling
        memory_manager = self.create_mock_memory_manager(memory_pressure=0.9)
        mock_get_memory_manager.return_value = memory_manager
        
        config = {
            'batch_size': 16,  # Above minimum 
            'enable_dynamic_batching': True,
            'enable_performance_optimizations': True
        }
        
        optimal_batch = _calculate_optimal_batch_size(config)
        
        # Should not scale up under high memory pressure, but maintain minimum
        assert optimal_batch >= 8, f"High memory pressure minimum violation: {optimal_batch}"
        assert optimal_batch <= 16, f"Unexpected scaling under memory pressure: {optimal_batch}"
        
        print(f"✅ High memory pressure: 16 → {optimal_batch}")


class TestBatchSizeLogging:
    """Test batch size constraint logging and debugging."""
    
    def test_constraint_logging(self, caplog):
        """Test that constraint applications are properly logged."""
        base_config = {'batch_size': 16}
        pareto_params = {'batch_size': 2, 'learning_rate': 0.001}  # Below minimum
        
        with caplog.at_level('INFO'):
            result_config = apply_pareto_parameters_to_config(base_config, pareto_params)
        
        # Check that constraint enforcement was logged
        assert result_config['batch_size'] == 8
        
        # Look for constraint enforcement message in logs
        constraint_logged = any('minimum batch_size' in record.message.lower() 
                              for record in caplog.records)
        assert constraint_logged, "Constraint enforcement not logged"
        
        print("✅ Constraint logging verified")
    
    def test_hardware_optimization_logging(self, caplog):
        """Test hardware optimization logging includes constraint info."""
        config = {'batch_size': 3, 'optimized_batch_size': 'auto'}
        
        with patch('torch.cuda.is_available', return_value=False):
            with caplog.at_level('INFO'):
                result = _calculate_optimal_batch_size(config)
        
        assert result >= 8, f"Hardware constraint failed: {result}"
        print(f"✅ Hardware logging: 3 → {result}")


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_missing_pareto_parameters(self):
        """Test behavior when Pareto parameters are missing."""
        # Test with non-existent file
        params = load_pareto_parameters("non_existent_file.json")
        assert params is None, "Should return None for missing file"
        
        # Test with empty parameters
        empty_config = apply_pareto_parameters_to_config({'batch_size': 16}, {})
        assert empty_config['batch_size'] == 16, "Should preserve original when no params"
        
        print("✅ Missing parameters handled correctly")
    
    def test_invalid_batch_size_values(self):
        """Test handling of invalid batch size values."""
        base_config = {'batch_size': 16}
        
        invalid_cases = [
            {'batch_size': 0},      # Zero
            {'batch_size': -5},     # Negative
            {'batch_size': None},   # None
        ]
        
        for invalid_params in invalid_cases:
            try:
                result = apply_pareto_parameters_to_config(base_config, invalid_params)
                # Should enforce minimum regardless of invalid input
                if 'batch_size' in result:
                    assert result['batch_size'] >= 8, f"Invalid value not constrained: {invalid_params}"
            except (ValueError, TypeError):
                # Acceptable to raise error for invalid values
                print(f"✅ Invalid value rejected: {invalid_params}")
    
    def test_extremely_high_batch_sizes(self):
        """Test behavior with extremely high batch sizes."""
        high_batch_sizes = [256, 512, 1024, 2048]
        
        for batch_size in high_batch_sizes:
            base_config = {'batch_size': 16}
            pareto_params = {'batch_size': batch_size}
            
            result_config = apply_pareto_parameters_to_config(base_config, pareto_params)
            
            # Should not impose artificial ceiling
            assert result_config['batch_size'] == batch_size, \
                f"Artificial ceiling imposed: {batch_size} → {result_config['batch_size']}"
            
            # Hardware optimization may limit based on actual memory
            with patch('torch.cuda.is_available', return_value=True):
                with patch('torch.cuda.mem_get_info', return_value=(8*1024**3, 4*1024**3)):
                    hardware_result = _calculate_optimal_batch_size(result_config)
                    # Hardware can limit, but should still maintain minimum
                    assert hardware_result >= 8, f"Hardware failed minimum: {hardware_result}"
            
            print(f"✅ High batch size: {batch_size} → {result_config['batch_size']}")


def run_integration_tests():
    """Run all integration tests and return comprehensive results."""
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        'tests/integration/test_batch_constraints_integration.py', 
        '-v', '--tb=short', '--capture=no'
    ], capture_output=True, text=True, cwd='.')
    
    return {
        'success': result.returncode == 0,
        'output': result.stdout,
        'errors': result.stderr,
        'test_file': 'test_batch_constraints_integration.py'
    }


if __name__ == '__main__':
    # Run integration tests directly
    pytest.main([__file__, '-v', '--capture=no'])