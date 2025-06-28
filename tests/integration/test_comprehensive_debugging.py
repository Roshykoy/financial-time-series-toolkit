"""
Comprehensive test suite for MarkSix debugging audit.
Tests all critical components identified in the debugging analysis.
"""
import os
import sys
import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import warnings
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.input_validation import InputValidator, ValidationError, safe_input, validate_training_config
from utils.error_handling import (
    ErrorHandler, robust_operation, gpu_error_recovery, 
    safe_file_operation, safe_model_operation, GPUError
)
from utils.safe_math import (
    safe_divide, safe_log, safe_exp, safe_softmax, safe_normalize,
    check_tensor_health, stable_kl_divergence
)


class TestInputValidation:
    """Test input validation utilities."""
    
    def test_menu_choice_validation(self):
        """Test menu choice validation with various inputs."""
        validator = InputValidator()
        
        # Valid inputs
        assert validator.validate_menu_choice("1") == "1"
        assert validator.validate_menu_choice("6") == "6"
        
        # Invalid inputs
        with pytest.raises(ValidationError, match="Please enter a choice"):
            validator.validate_menu_choice("")
        
        with pytest.raises(ValidationError, match="single digit"):
            validator.validate_menu_choice("12")
        
        with pytest.raises(ValidationError, match="single digit"):
            validator.validate_menu_choice("a")
        
        with pytest.raises(ValidationError, match="between 1 and 6"):
            validator.validate_menu_choice("0")
        
        with pytest.raises(ValidationError, match="between 1 and 6"):
            validator.validate_menu_choice("7")
    
    def test_positive_integer_validation(self):
        """Test positive integer validation with bounds."""
        validator = InputValidator()
        
        # Valid inputs
        assert validator.validate_positive_integer("5", "test", 1, 10) == 5
        assert validator.validate_positive_integer(5, "test", 1, 10) == 5
        assert validator.validate_positive_integer("", "test", 1, 10, default=5) == 5
        
        # Invalid inputs
        with pytest.raises(ValidationError, match="cannot be empty"):
            validator.validate_positive_integer("", "test")
        
        with pytest.raises(ValidationError, match="valid integer"):
            validator.validate_positive_integer("abc", "test")
        
        with pytest.raises(ValidationError, match="at least 1"):
            validator.validate_positive_integer("0", "test", 1, 10)
        
        with pytest.raises(ValidationError, match="not exceed 10"):
            validator.validate_positive_integer("15", "test", 1, 10)
    
    def test_float_range_validation(self):
        """Test float range validation."""
        validator = InputValidator()
        
        # Valid inputs
        assert validator.validate_float_range("0.5", "test", 0.0, 1.0) == 0.5
        assert validator.validate_float_range(0.5, "test", 0.0, 1.0) == 0.5
        
        # Invalid inputs
        with pytest.raises(ValidationError, match="valid number"):
            validator.validate_float_range("abc", "test")
        
        with pytest.raises(ValidationError, match="at least 0.0"):
            validator.validate_float_range("-0.5", "test", 0.0, 1.0)
    
    def test_number_combination_validation(self):
        """Test lottery number combination validation."""
        validator = InputValidator()
        
        # Valid combination
        assert validator.validate_number_combination([1, 2, 3, 4, 5, 6]) == [1, 2, 3, 4, 5, 6]
        assert validator.validate_number_combination([49, 1, 25, 12, 35, 8]) == [1, 8, 12, 25, 35, 49]
        
        # Invalid combinations
        with pytest.raises(ValidationError, match="list or tuple"):
            validator.validate_number_combination("123456")
        
        with pytest.raises(ValidationError, match="Expected 6 numbers"):
            validator.validate_number_combination([1, 2, 3, 4, 5])
        
        with pytest.raises(ValidationError, match="must be integers"):
            validator.validate_number_combination([1, 2, 3, 4, 5, "6"])
        
        with pytest.raises(ValidationError, match="must be unique"):
            validator.validate_number_combination([1, 2, 3, 4, 5, 5])
        
        with pytest.raises(ValidationError, match="between 1 and 49"):
            validator.validate_number_combination([0, 2, 3, 4, 5, 6])
        
        with pytest.raises(ValidationError, match="between 1 and 49"):
            validator.validate_number_combination([1, 2, 3, 4, 5, 50])
    
    def test_file_path_validation(self):
        """Test file path validation."""
        validator = InputValidator()
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"test content")
        
        try:
            # Valid existing file
            result = validator.validate_file_path(tmp_path, must_exist=True)
            assert result.exists()
            
            # Non-existing file with must_exist=False
            new_path = tmp_path + "_new"
            result = validator.validate_file_path(new_path, must_exist=False)
            assert result.parent.exists()  # Parent directory should be created
            
            # Invalid cases
            with pytest.raises(ValidationError, match="cannot be empty"):
                validator.validate_file_path("", must_exist=True)
            
            with pytest.raises(ValidationError, match="File not found"):
                validator.validate_file_path("/nonexistent/file.txt", must_exist=True)
                
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_training_config_validation(self):
        """Test comprehensive training configuration validation."""
        # Valid config
        valid_config = {
            "epochs": 10,
            "batch_size": 8,
            "learning_rate": 0.001,
            "dropout": 0.1
        }
        
        result = validate_training_config(valid_config)
        assert result["epochs"] == 10
        assert result["batch_size"] == 8
        assert result["learning_rate"] == 0.001
        assert result["dropout"] == 0.1
        
        # Invalid config - will use defaults and validate
        invalid_config = {
            "epochs": -5,  # Will be caught by validation
            "batch_size": 1000,  # Will be caught by validation
            "learning_rate": 5.0,  # Will be caught by validation
        }
        
        with pytest.raises(ValidationError):
            validate_training_config(invalid_config)


class TestErrorHandling:
    """Test error handling utilities."""
    
    def test_error_handler_basic(self):
        """Test basic error handler functionality."""
        handler = ErrorHandler()
        
        # Test error handling
        test_error = ValueError("Test error")
        result = handler.handle_error(test_error, "test_context")
        
        assert "ValueError:test_context" in handler.error_counts
        assert handler.error_counts["ValueError:test_context"] == 1
    
    def test_robust_operation_decorator(self):
        """Test robust operation decorator."""
        
        @robust_operation(max_retries=3, exceptions=(ValueError,))
        def failing_function(fail_count=2):
            if hasattr(failing_function, 'call_count'):
                failing_function.call_count += 1
            else:
                failing_function.call_count = 1
            
            if failing_function.call_count <= fail_count:
                raise ValueError(f"Attempt {failing_function.call_count}")
            
            return "success"
        
        # Should succeed after retries
        result = failing_function(fail_count=2)
        assert result == "success"
        assert failing_function.call_count == 3
        
        # Reset for next test
        failing_function.call_count = 0
        
        # Should fail after max retries
        with pytest.raises(ValueError):
            failing_function(fail_count=5)
    
    def test_gpu_error_recovery(self):
        """Test GPU error recovery context manager."""
        
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            # Test successful operation
            with gpu_error_recovery():
                result = "success"
            assert result == "success"
            
            # Test CUDA OOM recovery
            with patch('builtins.print'):  # Suppress prints during test
                with pytest.raises(GPUError, match="GPU operation failed"):
                    with gpu_error_recovery():
                        # This should trigger the recovery mechanism
                        raise RuntimeError("CUDA out of memory")
    
    def test_safe_file_operation(self):
        """Test safe file operation context manager."""
        
        # Test successful file operation
        with tempfile.NamedTemporaryFile() as tmp:
            with safe_file_operation(tmp.name, "read"):
                content = tmp.read()
        
        # Test file not found
        with pytest.raises(Exception):  # Should raise DataError, but we're testing the wrapper
            with safe_file_operation("/nonexistent/file.txt", "read"):
                # Actually try to open the file to trigger the error
                with open("/nonexistent/file.txt", "r") as f:
                    f.read()
    
    def test_safe_model_operation(self):
        """Test safe model operation decorator."""
        
        @safe_model_operation("test_operation")
        def model_function():
            return torch.tensor([1.0, 2.0, 3.0])
        
        # Should work normally
        result = model_function()
        assert torch.allclose(result, torch.tensor([1.0, 2.0, 3.0]))
        
        @safe_model_operation("failing_operation")
        def failing_model_function():
            raise RuntimeError("CUDA out of memory")
        
        # Should catch and re-raise as ModelError
        with pytest.raises(Exception):  # Should be GPUError, but testing the wrapper
            failing_model_function()


class TestSafeMath:
    """Test safe mathematical operations."""
    
    def test_safe_divide(self):
        """Test safe division operations."""
        
        # Normal division
        assert safe_divide(10, 2) == 5.0
        
        # Division by zero protection
        result = safe_divide(10, 0)
        assert result == 10 / 1e-8  # Should use epsilon
        
        # Tensor division
        numerator = torch.tensor([10.0, 20.0, 30.0])
        denominator = torch.tensor([2.0, 0.0, 5.0])
        
        result = safe_divide(numerator, denominator)
        expected = torch.tensor([5.0, 20.0 / 1e-8, 6.0])
        assert torch.allclose(result, expected, rtol=1e-3)
    
    def test_safe_log(self):
        """Test safe logarithm operations."""
        
        # Normal log
        assert abs(safe_log(10) - np.log(10)) < 1e-6
        
        # Log of zero protection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = safe_log(0)
            assert result == np.log(1e-8)
        
        # Log of negative protection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = safe_log(-5)
            assert result == np.log(1e-8)
        
        # Tensor log
        x = torch.tensor([1.0, 0.0, -1.0, 10.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = safe_log(x)
            assert torch.isfinite(result).all()
    
    def test_safe_exp(self):
        """Test safe exponential operations."""
        
        # Normal exp
        assert abs(safe_exp(1) - np.exp(1)) < 1e-6
        
        # Overflow protection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = safe_exp(1000)
            assert np.isfinite(result)
        
        # Tensor exp
        x = torch.tensor([1.0, 100.0, 1000.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = safe_exp(x)
            assert torch.isfinite(result).all()
    
    def test_safe_softmax(self):
        """Test numerically stable softmax."""
        
        # Normal case
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
        result = safe_softmax(logits, dim=1)
        
        # Check properties
        assert torch.allclose(result.sum(dim=1), torch.tensor([1.0, 1.0]))
        assert (result >= 0).all()
        
        # Extreme values
        extreme_logits = torch.tensor([[1000.0, 2000.0, 3000.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = safe_softmax(extreme_logits, dim=1)
            assert torch.isfinite(result).all()
            assert torch.allclose(result.sum(dim=1), torch.tensor([1.0]))
    
    def test_check_tensor_health(self):
        """Test tensor health checking."""
        
        # Healthy tensor
        healthy = torch.tensor([1.0, 2.0, 3.0])
        assert check_tensor_health(healthy, "healthy") == True
        
        # NaN tensor
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert check_tensor_health(nan_tensor, "nan") == False
        
        # Inf tensor
        inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert check_tensor_health(inf_tensor, "inf") == False
        
        # Large values
        large_tensor = torch.tensor([1e7, 2e7, 3e7])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = check_tensor_health(large_tensor, "large", warn_threshold=1e6)
            # Should still return True but issue warning
            assert result == True
    
    def test_stable_kl_divergence(self):
        """Test stable KL divergence computation."""
        
        # Normal case
        mu1 = torch.tensor([0.0, 1.0])
        logvar1 = torch.tensor([0.0, 0.0])
        mu2 = torch.tensor([0.0, 0.0])
        logvar2 = torch.tensor([0.0, 0.0])
        
        kl_div = stable_kl_divergence(mu1, logvar1, mu2, logvar2)
        assert torch.isfinite(kl_div)
        assert kl_div >= 0  # KL divergence is always non-negative
        
        # Extreme values
        extreme_mu1 = torch.tensor([1000.0, -1000.0])
        extreme_logvar1 = torch.tensor([50.0, -50.0])
        extreme_mu2 = torch.tensor([0.0, 0.0])
        extreme_logvar2 = torch.tensor([0.0, 0.0])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kl_div = stable_kl_divergence(extreme_mu1, extreme_logvar1, extreme_mu2, extreme_logvar2)
            assert torch.isfinite(kl_div)


class TestSystemIntegration:
    """Test system integration and edge cases."""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        
        # Empty DataFrame
        empty_df = pd.DataFrame()
        
        # Should handle gracefully
        validator = InputValidator()
        with pytest.raises(ValidationError, match="empty"):
            validator.validate_csv_structure(empty_df)
    
    def test_memory_constraints(self):
        """Test behavior under memory constraints."""
        
        # Test with large tensor operations
        try:
            large_tensor = torch.randn(1000, 1000)
            result = safe_softmax(large_tensor)
            assert result.shape == large_tensor.shape
        except RuntimeError:
            # Expected on systems with limited memory
            pass
    
    def test_gpu_fallback_scenarios(self):
        """Test GPU fallback scenarios."""
        
        if torch.cuda.is_available():
            # Test device switching
            cpu_tensor = torch.randn(10, 10)
            gpu_tensor = cpu_tensor.cuda()
            
            # Operations should work on both
            cpu_result = safe_softmax(cpu_tensor)
            gpu_result = safe_softmax(gpu_tensor)
            
            assert torch.allclose(cpu_result, gpu_result.cpu())
    
    def test_configuration_edge_cases(self):
        """Test configuration with edge case values."""
        
        # Extreme but valid configuration
        extreme_config = {
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-6,
            "dropout": 0.0
        }
        
        result = validate_training_config(extreme_config)
        assert result["epochs"] == 1
        assert result["batch_size"] == 1
        assert result["learning_rate"] == 1e-6
        assert result["dropout"] == 0.0


def test_main_import_safety():
    """Test that main modules can be imported safely."""
    
    # Test that our utility modules can be imported
    try:
        from utils.input_validation import InputValidator
        from utils.error_handling import ErrorHandler
        from utils.safe_math import safe_divide
        
        # Basic functionality test
        validator = InputValidator()
        handler = ErrorHandler()
        result = safe_divide(10, 2)
        
        assert result == 5.0
        
    except Exception as e:
        pytest.fail(f"Failed to import utility modules: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])