"""
Comprehensive input validation utilities for MarkSix forecasting system.
Provides robust validation functions with user-friendly error messages.
"""
import os
import sys
from typing import Optional, Union, List, Tuple, Any
from pathlib import Path
import warnings


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class InputValidator:
    """Comprehensive input validation class."""
    
    @staticmethod
    def validate_menu_choice(choice_input: str) -> str:
        """Validate menu choice input with comprehensive error handling."""
        if not choice_input:
            raise ValidationError("Please enter a choice.")
        
        if len(choice_input) != 1 or not choice_input.isdigit():
            raise ValidationError("Please enter a single digit (1-6).")
        
        choice = int(choice_input)
        if not (1 <= choice <= 6):
            raise ValidationError("Please enter a number between 1 and 6.")
        
        return choice_input
    
    @staticmethod
    def validate_positive_integer(
        value: Union[str, int], 
        name: str, 
        min_val: int = 1, 
        max_val: Optional[int] = None,
        default: Optional[int] = None
    ) -> int:
        """Validate positive integer with bounds checking."""
        if isinstance(value, str):
            if not value.strip():
                if default is not None:
                    return default
                raise ValidationError(f"{name} cannot be empty.")
            
            try:
                value = int(value.strip())
            except ValueError:
                raise ValidationError(f"{name} must be a valid integer.")
        
        if value < min_val:
            raise ValidationError(f"{name} must be at least {min_val}.")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"{name} must not exceed {max_val}.")
        
        return value
    
    @staticmethod
    def validate_float_range(
        value: Union[str, float], 
        name: str, 
        min_val: float = 0.0, 
        max_val: Optional[float] = None,
        default: Optional[float] = None
    ) -> float:
        """Validate float within specified range."""
        if isinstance(value, str):
            if not value.strip():
                if default is not None:
                    return default
                raise ValidationError(f"{name} cannot be empty.")
            
            try:
                value = float(value.strip())
            except ValueError:
                raise ValidationError(f"{name} must be a valid number.")
        
        if value < min_val:
            raise ValidationError(f"{name} must be at least {min_val}.")
        
        if max_val is not None and value > max_val:
            raise ValidationError(f"{name} must not exceed {max_val}.")
        
        return value
    
    @staticmethod
    def validate_file_path(file_path: str, must_exist: bool = True) -> Path:
        """Validate file path with existence and permission checks."""
        if not file_path or not file_path.strip():
            raise ValidationError("File path cannot be empty.")
        
        path = Path(file_path.strip())
        
        if must_exist and not path.exists():
            raise ValidationError(f"File not found: {path}")
        
        if must_exist and not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        
        if must_exist and not os.access(path, os.R_OK):
            raise ValidationError(f"No read permission for file: {path}")
        
        # Check parent directory for write operations
        if not must_exist:
            parent_dir = path.parent
            if not parent_dir.exists():
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise ValidationError(f"Cannot create directory {parent_dir}: {e}")
        
        return path
    
    @staticmethod
    def validate_number_combination(numbers: List[int]) -> List[int]:
        """Validate lottery number combination."""
        if not isinstance(numbers, (list, tuple)):
            raise ValidationError("Number combination must be a list or tuple.")
        
        if len(numbers) != 6:
            raise ValidationError(f"Expected 6 numbers, got {len(numbers)}.")
        
        # Check that all elements are already integers
        for i, n in enumerate(numbers):
            if not isinstance(n, int):
                raise ValidationError("All numbers must be integers.")
        
        validated_numbers = list(numbers)
        
        if len(set(validated_numbers)) != 6:
            raise ValidationError("Numbers must be unique.")
        
        if not all(1 <= n <= 49 for n in validated_numbers):
            raise ValidationError("Numbers must be between 1 and 49.")
        
        return sorted(validated_numbers)
    
    @staticmethod
    def validate_gpu_memory_request(batch_size: int, model_size_mb: float = 0) -> None:
        """Validate GPU memory requirements."""
        try:
            import torch
            if torch.cuda.is_available():
                # Estimate memory requirements
                estimated_mb = batch_size * 10 + model_size_mb  # Rough estimate
                available_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                
                if estimated_mb > available_mb * 0.8:  # Use 80% as safety margin
                    warnings.warn(
                        f"Requested batch size ({batch_size}) may exceed GPU memory. "
                        f"Consider reducing batch size or using CPU.",
                        UserWarning
                    )
        except ImportError:
            pass  # PyTorch not available, skip validation
    
    @staticmethod
    def validate_csv_structure(df, required_columns: Optional[List[str]] = None) -> None:
        """Validate CSV structure and content."""
        if df.empty:
            raise ValidationError("CSV file is empty or contains no valid data.")
        
        if len(df) < 10:
            warnings.warn(
                f"Dataset is very small ({len(df)} rows). Results may be unreliable.",
                UserWarning
            )
        
        if required_columns:
            missing_cols = set(required_columns) - set(df.columns)
            if missing_cols:
                raise ValidationError(f"CSV missing required columns: {missing_cols}")


def safe_input(prompt: str, validator_func, max_attempts: int = 3) -> Any:
    """Safely get user input with validation and retry logic."""
    for attempt in range(max_attempts):
        try:
            user_input = input(prompt).strip()
            return validator_func(user_input)
        except ValidationError as e:
            print(f"❌ {e}")
            if attempt == max_attempts - 1:
                print("Maximum attempts reached. Using default value or exiting.")
                raise
        except (KeyboardInterrupt, EOFError):
            print("\n\nOperation cancelled by user.")
            sys.exit(0)
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            if attempt == max_attempts - 1:
                raise


def validate_training_config(config: dict) -> dict:
    """Validate training configuration parameters."""
    validator = InputValidator()
    validated_config = config.copy()
    
    # Validate epochs
    if 'epochs' in config:
        validated_config['epochs'] = validator.validate_positive_integer(
            config['epochs'], 'epochs', min_val=1, max_val=1000
        )
    
    # Validate batch size
    if 'batch_size' in config:
        batch_size = validator.validate_positive_integer(
            config['batch_size'], 'batch_size', min_val=1, max_val=512
        )
        validator.validate_gpu_memory_request(batch_size)
        validated_config['batch_size'] = batch_size
    
    # Validate learning rate
    if 'learning_rate' in config:
        validated_config['learning_rate'] = validator.validate_float_range(
            config['learning_rate'], 'learning_rate', min_val=1e-6, max_val=1.0
        )
    
    # Validate dropout
    if 'dropout' in config:
        validated_config['dropout'] = validator.validate_float_range(
            config['dropout'], 'dropout', min_val=0.0, max_val=0.9
        )
    
    return validated_config


def estimate_training_time(epochs: int, batch_size: int, dataset_size: int) -> str:
    """Estimate training time and provide user warning for long operations."""
    # Rough estimates based on typical hardware
    samples_per_second = 1000  # Conservative estimate
    batches_per_epoch = max(1, dataset_size // batch_size)
    total_samples = epochs * dataset_size
    estimated_seconds = total_samples / samples_per_second
    
    if estimated_seconds > 3600:  # More than 1 hour
        hours = estimated_seconds / 3600
        return f"⚠️  Estimated training time: {hours:.1f} hours"
    elif estimated_seconds > 300:  # More than 5 minutes
        minutes = estimated_seconds / 60
        return f"⚠️  Estimated training time: {minutes:.1f} minutes"
    else:
        return f"Estimated training time: {estimated_seconds:.0f} seconds"