"""
Data structures for hyperparameter optimization.
Separate module to avoid circular imports.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class OptimizationTrial:
    """Represents a single optimization trial."""
    trial_id: str
    parameters: Dict[str, Any]
    score: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None
    resource_usage: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.resource_usage is None:
            self.resource_usage = {}


@dataclass
class OptimizationConfig:
    """Configuration for optimization process."""
    # General settings
    max_trials: int = 50
    max_duration_hours: float = 12.0
    parallel_jobs: int = 1
    random_seed: Optional[int] = None
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 10
    min_improvement: float = 0.001
    
    # Resource management
    enable_hardware_monitoring: bool = True
    cleanup_frequency: int = 5
    memory_limit_gb: Optional[float] = None
    gpu_memory_fraction: float = 0.8
    
    # Trial management
    trial_timeout_minutes: float = 60.0
    retry_failed_trials: bool = False
    max_retries: int = 2
    
    # Checkpointing
    save_intermediate_results: bool = True
    checkpoint_frequency: int = 5
    backup_results: bool = True
    
    # Validation
    validation_strategy: str = "holdout"  # holdout, kfold, timeseries
    validation_split: float = 0.2
    cross_validation_folds: int = 5