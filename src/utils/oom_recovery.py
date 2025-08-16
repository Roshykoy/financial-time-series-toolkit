"""
OOM (Out of Memory) Recovery System for Conservative Batch Size Management.

This module provides automatic batch size reduction when CUDA OOM errors occur,
with stepped reduction strategies that are compatible with Pareto Front optimization.
"""

import torch
import logging
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import json
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OOMEvent:
    """Record of an OOM event and recovery action."""
    timestamp: float
    original_batch_size: int
    reduced_batch_size: int
    error_message: str
    recovery_step: int
    location: str
    

class BatchSizeOOMRecovery:
    """
    Automatic batch size reduction system for OOM recovery.
    
    Uses stepped reduction strategy that preserves Pareto Front optimization
    while providing graceful degradation when memory limits are hit.
    """
    
    # Standard reduction steps (multiply batch size by these factors)
    DEFAULT_REDUCTION_STEPS = [0.75, 0.5, 0.375, 0.25, 0.125]
    
    def __init__(self, 
                 min_batch_size: int = 8,
                 reduction_steps: Optional[List[float]] = None,
                 max_retries: int = 3,
                 recovery_log_file: Optional[str] = None):
        """
        Initialize OOM recovery system.
        
        Args:
            min_batch_size: Absolute minimum batch size (never go below this)
            reduction_steps: List of multiplication factors for batch size reduction
            max_retries: Maximum number of reduction attempts before giving up
            recovery_log_file: Optional file to log recovery events
        """
        self.min_batch_size = min_batch_size
        self.reduction_steps = reduction_steps or self.DEFAULT_REDUCTION_STEPS
        self.max_retries = max_retries
        self.recovery_log_file = recovery_log_file
        
        # Track OOM events for analysis
        self.oom_events: List[OOMEvent] = []
        self.current_retry = 0
        
        # Setup logging
        if self.recovery_log_file:
            self._setup_file_logging()
    
    def _setup_file_logging(self):
        """Setup file logging for OOM recovery events."""
        try:
            Path(self.recovery_log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.recovery_log_file, mode='a')
            formatter = logging.Formatter(
                '%(asctime)s [OOMRecovery] %(levelname)s: %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
        except Exception as e:
            print(f"Warning: Could not setup OOM recovery file logging: {e}")
    
    def is_oom_error(self, error: Exception) -> bool:
        """
        Check if the error is a CUDA OOM error.
        
        Args:
            error: Exception to check
            
        Returns:
            bool: True if this is an OOM error
        """
        error_str = str(error).lower()
        oom_indicators = [
            "cuda out of memory",
            "out of memory",
            "cudnn_status_not_supported",
            "insufficient memory",
            "memory error"
        ]
        
        return any(indicator in error_str for indicator in oom_indicators)
    
    def calculate_reduced_batch_size(self, 
                                   current_batch_size: int,
                                   step: int = 0) -> Optional[int]:
        """
        Calculate the next reduced batch size.
        
        Args:
            current_batch_size: Current batch size that caused OOM
            step: Which reduction step to use (0-based index)
            
        Returns:
            int: New reduced batch size, or None if below minimum
        """
        if step >= len(self.reduction_steps):
            return None
        
        reduction_factor = self.reduction_steps[step]
        new_batch_size = max(
            self.min_batch_size,
            int(current_batch_size * reduction_factor)
        )
        
        # If we're at minimum and it's still the same, try half of minimum
        if new_batch_size == current_batch_size and new_batch_size > self.min_batch_size:
            new_batch_size = max(self.min_batch_size, new_batch_size // 2)
        
        return new_batch_size if new_batch_size >= self.min_batch_size else None
    
    def attempt_recovery(self,
                        current_batch_size: int,
                        error: Exception,
                        location: str = "unknown") -> Tuple[Optional[int], bool]:
        """
        Attempt to recover from OOM by reducing batch size.
        
        Args:
            current_batch_size: Batch size that caused the OOM
            error: The OOM error that occurred
            location: Where the error occurred (for logging)
            
        Returns:
            Tuple[Optional[int], bool]: (new_batch_size, should_retry)
                - new_batch_size: Reduced batch size, or None if can't recover
                - should_retry: Whether to retry training with new batch size
        """
        if not self.is_oom_error(error):
            # Not an OOM error, don't handle it
            return None, False
        
        if self.current_retry >= self.max_retries:
            logger.error(f"Max OOM recovery retries ({self.max_retries}) exceeded at {location}")
            return None, False
        
        # Calculate reduced batch size
        reduced_batch_size = self.calculate_reduced_batch_size(
            current_batch_size, 
            self.current_retry
        )
        
        if reduced_batch_size is None:
            logger.error(f"Cannot reduce batch size further from {current_batch_size} at {location}")
            return None, False
        
        # Record the OOM event
        oom_event = OOMEvent(
            timestamp=time.time(),
            original_batch_size=current_batch_size,
            reduced_batch_size=reduced_batch_size,
            error_message=str(error),
            recovery_step=self.current_retry,
            location=location
        )
        self.oom_events.append(oom_event)
        
        # Log the recovery attempt
        reduction_pct = (1 - reduced_batch_size / current_batch_size) * 100
        logger.warning(f"OOM Recovery at {location}: {current_batch_size} → {reduced_batch_size} "
                      f"({reduction_pct:.1f}% reduction, attempt {self.current_retry + 1}/{self.max_retries})")
        
        print(f"⚠️  CUDA OOM detected - reducing batch size: {current_batch_size} → {reduced_batch_size}")
        
        self.current_retry += 1
        return reduced_batch_size, True
    
    def reset_recovery_state(self):
        """Reset recovery state after successful training."""
        self.current_retry = 0
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about OOM recovery events."""
        if not self.oom_events:
            return {
                'total_events': 0,
                'successful_recoveries': 0,
                'average_reduction_pct': 0,
                'locations': {}
            }
        
        successful = len([e for e in self.oom_events if e.reduced_batch_size >= self.min_batch_size])
        total_reduction = sum(
            (1 - e.reduced_batch_size / e.original_batch_size) * 100 
            for e in self.oom_events
        )
        
        # Count events by location
        location_counts = {}
        for event in self.oom_events:
            location_counts[event.location] = location_counts.get(event.location, 0) + 1
        
        return {
            'total_events': len(self.oom_events),
            'successful_recoveries': successful,
            'average_reduction_pct': total_reduction / len(self.oom_events),
            'locations': location_counts,
            'last_event': {
                'timestamp': self.oom_events[-1].timestamp,
                'original_batch_size': self.oom_events[-1].original_batch_size,
                'reduced_batch_size': self.oom_events[-1].reduced_batch_size,
                'location': self.oom_events[-1].location
            } if self.oom_events else None
        }
    
    def save_recovery_report(self, filename: Optional[str] = None) -> str:
        """Save detailed recovery report to JSON file."""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"oom_recovery_report_{timestamp}.json"
        
        report = {
            'metadata': {
                'min_batch_size': self.min_batch_size,
                'reduction_steps': self.reduction_steps,
                'max_retries': self.max_retries,
                'report_timestamp': time.time()
            },
            'statistics': self.get_recovery_statistics(),
            'events': [
                {
                    'timestamp': event.timestamp,
                    'original_batch_size': event.original_batch_size,
                    'reduced_batch_size': event.reduced_batch_size,
                    'error_message': event.error_message,
                    'recovery_step': event.recovery_step,
                    'location': event.location
                }
                for event in self.oom_events
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"OOM recovery report saved to: {filename}")
        return filename


# Global OOM recovery instance
_global_oom_recovery: Optional[BatchSizeOOMRecovery] = None


def get_oom_recovery_manager() -> BatchSizeOOMRecovery:
    """Get or create the global OOM recovery manager."""
    global _global_oom_recovery
    if _global_oom_recovery is None:
        # Try to get config for initialization
        try:
            from src.config import CONFIG
            min_batch_size = CONFIG.get('min_batch_size', 8)
            reduction_steps = CONFIG.get('oom_reduction_steps', None)
            max_retries = CONFIG.get('max_oom_retries', 3)
        except ImportError:
            # Fallback to defaults if config not available
            min_batch_size = 8
            reduction_steps = None
            max_retries = 3
        
        _global_oom_recovery = BatchSizeOOMRecovery(
            min_batch_size=min_batch_size,
            reduction_steps=reduction_steps,
            max_retries=max_retries,
            recovery_log_file="logs/oom_recovery.log"
        )
    return _global_oom_recovery


def handle_oom_error(current_batch_size: int,
                    error: Exception,
                    location: str = "training") -> Tuple[Optional[int], bool]:
    """
    Convenience function to handle OOM errors.
    
    Args:
        current_batch_size: Batch size that caused OOM
        error: The exception that occurred
        location: Where the error occurred
        
    Returns:
        Tuple[Optional[int], bool]: (new_batch_size, should_retry)
    """
    recovery_manager = get_oom_recovery_manager()
    return recovery_manager.attempt_recovery(current_batch_size, error, location)


def reset_oom_recovery():
    """Reset global OOM recovery state after successful training."""
    recovery_manager = get_oom_recovery_manager()
    recovery_manager.reset_recovery_state()