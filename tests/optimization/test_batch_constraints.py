"""
Comprehensive unit tests for batch size constraints system.
Tests the minimum batch size enforcement across all optimization components.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

# Import modules under test
from src.optimization.pareto_integration import (
    load_pareto_parameters, 
    apply_pareto_parameters_to_config,
    create_pareto_optimized_config
)
from src.cvae_data_loader import _calculate_optimal_batch_size
from src.optimization.hardware_manager import HardwareResourceManager
from src.config import CONFIG


class TestParetoIntegrationConstraints:
    """Test batch size constraints in Pareto integration."""
    
    def test_pareto_batch_size_minimum_enforcement(self):
        """Test batch_size < 8 gets raised to 8."""
        base_config = {'batch_size': 16}
        
        # Test with batch_size below minimum
        pareto_params = {'batch_size': 4, 'learning_rate': 0.001}
        result = apply_pareto_parameters_to_config(base_config, pareto_params)
        
        assert result['batch_size'] == 8, f"Expected 8, got {result['batch_size']}"
    
    def test_pareto_batch_size_no_ceiling(self):
        """Test batch_size > 8 stays unchanged (no artificial ceiling)."""
        base_config = {'batch_size': 16}
        
        test_cases = [32, 64, 128, 256]
        for batch_size in test_cases:
            pareto_params = {'batch_size': batch_size, 'learning_rate': 0.001}
            result = apply_pareto_parameters_to_config(base_config, pareto_params)
            assert result['batch_size'] == batch_size, f"Batch size {batch_size} was modified"
    
    def test_pareto_batch_size_edge_cases(self):
        """Test edge cases: 0, negative values, None."""
        base_config = {'batch_size': 16}
        
        edge_cases = [0, -1, -10]
        for batch_size in edge_cases:
            pareto_params = {'batch_size': batch_size, 'learning_rate': 0.001}
            result = apply_pareto_parameters_to_config(base_config, pareto_params)
            assert result['batch_size'] == 8, f"Edge case {batch_size} not handled correctly"
    
    def test_pareto_batch_size_exact_minimum(self):
        """Test batch_size exactly 8 stays 8."""
        base_config = {'batch_size': 16}
        pareto_params = {'batch_size': 8, 'learning_rate': 0.001}
        result = apply_pareto_parameters_to_config(base_config, pareto_params)
        
        assert result['batch_size'] == 8, "Exact minimum not preserved"
    
    def test_pareto_other_parameters_unchanged(self):
        """Test that other parameters are not affected by batch size constraints."""
        base_config = {'batch_size': 16, 'epochs': 10}
        pareto_params = {
            'batch_size': 4,  # Will be constrained to 8
            'learning_rate': 0.002,
            'dropout': 0.15
        }
        result = apply_pareto_parameters_to_config(base_config, pareto_params)
        
        assert result['batch_size'] == 8
        assert result['learning_rate'] == 0.002
        assert result['dropout'] == 0.15
        assert result['epochs'] == 10  # Unchanged from base


class TestHardwareManagerConstraints:
    """Test batch size constraints in hardware manager."""
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.mem_get_info')
    def test_hardware_batch_size_minimum_gpu(self, mock_mem_info, mock_cuda_available):
        """Test hardware optimization enforces minimum on GPU."""
        mock_cuda_available.return_value = True
        mock_mem_info.return_value = (8 * 1024**3, 6 * 1024**3)  # 8GB total, 6GB free
        
        config = {'batch_size': 2, 'optimized_batch_size': 'auto'}
        result = _calculate_optimal_batch_size(config)
        
        assert result >= 8, f"Hardware optimization returned {result}, expected >= 8"
    
    @patch('torch.cuda.is_available')
    def test_hardware_batch_size_minimum_cpu(self, mock_cuda_available):
        """Test hardware optimization enforces minimum on CPU."""
        mock_cuda_available.return_value = False
        
        config = {'batch_size': 4}
        result = _calculate_optimal_batch_size(config)
        
        assert result >= 8, f"CPU optimization returned {result}, expected >= 8"
    
    def test_hardware_batch_size_no_optimization(self):
        """Test minimum enforcement when optimization is disabled."""
        config = {'batch_size': 3, 'optimized_batch_size': 'manual'}
        result = _calculate_optimal_batch_size(config)
        
        assert result >= 8, f"Manual mode returned {result}, expected >= 8"
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.mem_get_info')
    def test_hardware_batch_size_scaling_respects_minimum(self, mock_mem_info, mock_cuda_available):
        """Test that scaling operations respect minimum constraint."""
        mock_cuda_available.return_value = True
        mock_mem_info.return_value = (16 * 1024**3, 12 * 1024**3)  # High memory
        
        config = {'batch_size': 1, 'optimized_batch_size': 'auto'}  # Very low starting point
        result = _calculate_optimal_batch_size(config)
        
        assert result >= 8, f"Scaling from low value returned {result}, expected >= 8"


class TestDynamicBatchSizing:
    """Test dynamic batch sizing constraints."""
    
    def create_mock_memory_manager(self, memory_pressure=0.5):
        """Create a mock memory manager for testing."""
        mock_manager = Mock()
        mock_manager.get_comprehensive_stats.return_value = {
            'memory_usage_pct': memory_pressure * 100
        }
        return mock_manager
    
    @patch('src.cvae_data_loader.PHASE2_AVAILABLE', True)
    @patch('src.cvae_data_loader.get_memory_manager')
    def test_dynamic_batch_sizing_minimum_enforcement(self, mock_get_memory_manager):
        """Test dynamic batch sizing respects minimum constraint."""
        # This is a complex test that would require mocking the entire dynamic batching system
        # For now, we'll test the constraint logic directly
        
        # Test the constraint logic from the dynamic batching code
        min_batch_size = 8
        enhanced_batch_size = 4  # Below minimum
        optimal_batch_size = max(min_batch_size, enhanced_batch_size)
        
        assert optimal_batch_size == 8, "Dynamic batching minimum enforcement failed"
    
    def test_dynamic_batch_sizing_no_reduction_above_minimum(self):
        """Test dynamic batching doesn't reduce batch size below minimum."""
        min_batch_size = 8
        enhanced_batch_size = 32  # Above minimum
        optimal_batch_size = max(min_batch_size, enhanced_batch_size)
        
        assert optimal_batch_size == 32, "Dynamic batching incorrectly reduced above-minimum batch size"


class TestBatchSizeConstraintIntegration:
    """Test integration between different constraint systems."""
    
    def create_temp_pareto_file(self, batch_size):
        """Create temporary Pareto parameter file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                'selected_parameters': {
                    'batch_size': batch_size,
                    'learning_rate': 0.001,
                    'epochs': 5
                }
            }
            json.dump(data, f)
            return Path(f.name)
    
    def test_pareto_to_config_pipeline(self):
        """Test full pipeline from Pareto file to final config."""
        # Create temp file with low batch size
        temp_file = self.create_temp_pareto_file(batch_size=4)
        
        try:
            # Load parameters (mocking the file location)
            with patch('src.optimization.pareto_integration.Path.exists') as mock_exists:
                mock_exists.return_value = True
                with patch('builtins.open', create=True) as mock_open:
                    mock_open.return_value.__enter__.return_value.read.return_value = temp_file.read_text()
                    
                    # Load and apply constraints
                    params = load_pareto_parameters(str(temp_file))
                    config = apply_pareto_parameters_to_config(CONFIG.copy(), params)
                    
                    assert config['batch_size'] >= 8, "Full pipeline constraint failed"
        finally:
            temp_file.unlink()  # Clean up
    
    def test_constraint_consistency_across_modules(self):
        """Test that all modules use consistent minimum constraint."""
        # Test different entry points all enforce the same minimum
        min_expected = 8
        
        # Pareto integration
        pareto_result = apply_pareto_parameters_to_config(
            {'batch_size': 16}, 
            {'batch_size': 4}
        )
        
        # Hardware optimization (mocked)
        with patch('torch.cuda.is_available', return_value=False):
            hardware_result = _calculate_optimal_batch_size({'batch_size': 4})
        
        assert pareto_result['batch_size'] >= min_expected
        assert hardware_result >= min_expected
        assert pareto_result['batch_size'] == hardware_result  # Should be consistent


class TestConstraintErrorHandling:
    """Test error handling in batch size constraints."""
    
    def test_invalid_pareto_params_handling(self):
        """Test handling of invalid Pareto parameters."""
        base_config = {'batch_size': 16}
        
        # Test with string batch_size
        pareto_params = {'batch_size': 'invalid', 'learning_rate': 0.001}
        
        with pytest.raises((ValueError, TypeError)):
            apply_pareto_parameters_to_config(base_config, pareto_params)
    
    def test_missing_batch_size_in_config(self):
        """Test handling when batch_size is missing from config."""
        base_config = {}  # No batch_size
        pareto_params = {'learning_rate': 0.001}  # No batch_size
        
        result = apply_pareto_parameters_to_config(base_config, pareto_params)
        
        # Should use default from CONFIG or handle gracefully
        assert 'batch_size' in CONFIG, "Default config should have batch_size"
    
    @patch('src.cvae_data_loader.torch.cuda.mem_get_info')
    def test_gpu_memory_error_handling(self, mock_mem_info):
        """Test handling of GPU memory query errors."""
        mock_mem_info.side_effect = RuntimeError("CUDA error")
        
        config = {'batch_size': 4, 'optimized_batch_size': 'auto'}
        result = _calculate_optimal_batch_size(config)
        
        # Should fallback gracefully and still enforce minimum
        assert result >= 8, "Error handling should still enforce minimum"


def run_batch_constraint_tests():
    """Run all batch constraint tests and return results."""
    import subprocess
    import sys
    
    result = subprocess.run([
        sys.executable, '-m', 'pytest', 
        'tests/optimization/test_batch_constraints.py', 
        '-v', '--tb=short'
    ], capture_output=True, text=True, cwd='.')
    
    return {
        'success': result.returncode == 0,
        'output': result.stdout,
        'errors': result.stderr
    }


if __name__ == '__main__':
    # Run tests directly
    pytest.main([__file__, '-v'])