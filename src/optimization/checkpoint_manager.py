"""
Checkpoint and recovery system for long-running hyperparameter optimization.
Ensures that 8+ hour thorough_search runs can be safely interrupted and resumed.
"""

import os
import json
import pickle
import shutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

from .data_structures import OptimizationTrial, OptimizationConfig

# Handle logging import - try relative first, then absolute
try:
    from ..infrastructure.logging.logger import get_logger
except ImportError:
    try:
        from src.infrastructure.logging.logger import get_logger
    except ImportError:
        import logging
        def get_logger(name):
            return logging.getLogger(name)

logger = get_logger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for optimization checkpoint."""
    checkpoint_version: str = "1.0"
    created_at: str = ""
    optimization_start_time: str = ""
    algorithm: str = ""
    total_trials_planned: int = 0
    trials_completed: int = 0
    trials_failed: int = 0
    best_score: Optional[float] = None
    best_trial_id: Optional[str] = None
    estimated_completion_time: Optional[str] = None
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class OptimizationCheckpoint:
    """Complete optimization checkpoint data."""
    metadata: CheckpointMetadata
    optimization_config: Dict[str, Any]
    search_space: Dict[str, Any]
    completed_trials: List[Dict[str, Any]]
    best_trial: Optional[Dict[str, Any]]
    algorithm_state: Dict[str, Any]
    hardware_profile: Dict[str, Any]
    random_state: Optional[Dict[str, Any]] = None


class CheckpointManager:
    """Manages checkpointing and recovery for optimization runs."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_frequency: int = 5,
        max_checkpoints: int = 10,
        compression: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints
        self.compression = compression
        
        # State tracking
        self.current_checkpoint_id: Optional[str] = None
        self.last_save_time = datetime.now()
        self.trials_since_last_save = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
    
    def create_checkpoint(
        self,
        metadata: CheckpointMetadata,
        optimization_config: OptimizationConfig,
        search_space: Dict[str, Any],
        completed_trials: List[OptimizationTrial],
        best_trial: Optional[OptimizationTrial],
        algorithm_state: Dict[str, Any],
        hardware_profile: Dict[str, Any],
        random_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new checkpoint."""
        
        with self._lock:
            try:
                # Generate checkpoint ID
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_id = f"checkpoint_{timestamp}_{metadata.trials_completed:04d}"
                
                # Convert trials to serializable format
                trials_data = [asdict(trial) for trial in completed_trials]
                best_trial_data = asdict(best_trial) if best_trial else None
                
                # Create checkpoint object
                checkpoint = OptimizationCheckpoint(
                    metadata=metadata,
                    optimization_config=asdict(optimization_config),
                    search_space=search_space,
                    completed_trials=trials_data,
                    best_trial=best_trial_data,
                    algorithm_state=algorithm_state,
                    hardware_profile=hardware_profile,
                    random_state=random_state
                )
                
                # Save checkpoint
                checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.json"
                
                if self.compression:
                    # Save compressed checkpoint
                    checkpoint_path = checkpoint_path.with_suffix('.json.gz')
                    import gzip
                    with gzip.open(checkpoint_path, 'wt', encoding='utf-8') as f:
                        json.dump(asdict(checkpoint), f, indent=2, default=str)
                else:
                    # Save regular JSON
                    with open(checkpoint_path, 'w') as f:
                        json.dump(asdict(checkpoint), f, indent=2, default=str)
                
                # Update state
                self.current_checkpoint_id = checkpoint_id
                self.last_save_time = datetime.now()
                self.trials_since_last_save = 0
                
                # Cleanup old checkpoints
                self._cleanup_old_checkpoints()
                
                # Save checkpoint metadata for quick access
                self._save_checkpoint_index()
                
                logger.info(f"Checkpoint created: {checkpoint_id}")
                return checkpoint_id
                
            except Exception as e:
                logger.error(f"Error creating checkpoint: {e}")
                raise
    
    def should_save_checkpoint(self, trials_completed: int) -> bool:
        """Determine if a checkpoint should be saved."""
        # Save based on frequency
        if trials_completed % self.save_frequency == 0:
            return True
        
        # Save based on time (every 30 minutes)
        time_since_last = datetime.now() - self.last_save_time
        if time_since_last > timedelta(minutes=30):
            return True
        
        return False
    
    def load_latest_checkpoint(self) -> Optional[OptimizationCheckpoint]:
        """Load the most recent checkpoint."""
        try:
            checkpoints = self._list_checkpoints()
            if not checkpoints:
                logger.info("No checkpoints found")
                return None
            
            # Get latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda x: x['created_at'])
            checkpoint_file = latest_checkpoint['file_path']
            
            logger.info(f"Loading checkpoint: {checkpoint_file}")
            
            # Load checkpoint data
            if checkpoint_file.endswith('.gz'):
                import gzip
                with gzip.open(checkpoint_file, 'rt', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
            else:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
            
            # Convert back to dataclass
            checkpoint = self._deserialize_checkpoint(checkpoint_data)
            
            logger.info(f"Checkpoint loaded: {checkpoint.metadata.trials_completed} trials completed")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def load_specific_checkpoint(self, checkpoint_id: str) -> Optional[OptimizationCheckpoint]:
        """Load a specific checkpoint by ID."""
        try:
            checkpoint_files = [
                self.checkpoint_dir / f"{checkpoint_id}.json",
                self.checkpoint_dir / f"{checkpoint_id}.json.gz"
            ]
            
            checkpoint_file = None
            for file_path in checkpoint_files:
                if file_path.exists():
                    checkpoint_file = file_path
                    break
            
            if not checkpoint_file:
                logger.error(f"Checkpoint not found: {checkpoint_id}")
                return None
            
            logger.info(f"Loading specific checkpoint: {checkpoint_file}")
            
            # Load checkpoint data
            if checkpoint_file.suffix == '.gz':
                import gzip
                with gzip.open(checkpoint_file, 'rt', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
            else:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
            
            # Convert back to dataclass
            checkpoint = self._deserialize_checkpoint(checkpoint_data)
            
            logger.info(f"Checkpoint loaded: {checkpoint.metadata.trials_completed} trials completed")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Error loading checkpoint {checkpoint_id}: {e}")
            return None
    
    def list_available_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with metadata."""
        return self._list_checkpoints()
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        try:
            checkpoint_files = [
                self.checkpoint_dir / f"{checkpoint_id}.json",
                self.checkpoint_dir / f"{checkpoint_id}.json.gz"
            ]
            
            deleted = False
            for file_path in checkpoint_files:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Deleted checkpoint: {file_path}")
                    deleted = True
            
            if deleted:
                self._save_checkpoint_index()
            
            return deleted
            
        except Exception as e:
            logger.error(f"Error deleting checkpoint {checkpoint_id}: {e}")
            return False
    
    def get_recovery_info(self) -> Optional[Dict[str, Any]]:
        """Get information about recovery options."""
        try:
            checkpoints = self._list_checkpoints()
            if not checkpoints:
                return None
            
            latest = max(checkpoints, key=lambda x: x['created_at'])
            
            return {
                'latest_checkpoint': latest,
                'total_checkpoints': len(checkpoints),
                'can_resume': True,
                'estimated_progress': latest.get('trials_completed', 0) / latest.get('total_trials_planned', 1),
                'recommendation': f"Resume from checkpoint with {latest.get('trials_completed', 0)} completed trials"
            }
            
        except Exception as e:
            logger.error(f"Error getting recovery info: {e}")
            return None
    
    def _list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all checkpoint files with metadata."""
        checkpoints = []
        
        try:
            for file_path in self.checkpoint_dir.glob("checkpoint_*.json*"):
                try:
                    # Load metadata
                    if file_path.suffix == '.gz':
                        import gzip
                        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                            data = json.load(f)
                    else:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                    
                    metadata = data.get('metadata', {})
                    
                    checkpoint_info = {
                        'checkpoint_id': file_path.stem.replace('.json', ''),
                        'file_path': str(file_path),
                        'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                        'created_at': metadata.get('created_at', ''),
                        'trials_completed': metadata.get('trials_completed', 0),
                        'total_trials_planned': metadata.get('total_trials_planned', 0),
                        'best_score': metadata.get('best_score'),
                        'algorithm': metadata.get('algorithm', ''),
                    }
                    
                    checkpoints.append(checkpoint_info)
                    
                except Exception as e:
                    logger.warning(f"Error reading checkpoint metadata from {file_path}: {e}")
                    continue
            
            # Sort by creation time
            checkpoints.sort(key=lambda x: x['created_at'])
            
        except Exception as e:
            logger.error(f"Error listing checkpoints: {e}")
        
        return checkpoints
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to stay within max_checkpoints limit."""
        try:
            checkpoints = self._list_checkpoints()
            
            if len(checkpoints) <= self.max_checkpoints:
                return
            
            # Sort by creation time and remove oldest
            checkpoints.sort(key=lambda x: x['created_at'])
            checkpoints_to_remove = checkpoints[:-self.max_checkpoints]
            
            for checkpoint in checkpoints_to_remove:
                file_path = Path(checkpoint['file_path'])
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint['checkpoint_id']}")
            
        except Exception as e:
            logger.error(f"Error cleaning up old checkpoints: {e}")
    
    def _save_checkpoint_index(self):
        """Save an index of available checkpoints for quick access."""
        try:
            checkpoints = self._list_checkpoints()
            
            index_data = {
                'last_updated': datetime.now().isoformat(),
                'total_checkpoints': len(checkpoints),
                'checkpoints': checkpoints
            }
            
            index_file = self.checkpoint_dir / "checkpoint_index.json"
            with open(index_file, 'w') as f:
                json.dump(index_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.warning(f"Error saving checkpoint index: {e}")
    
    def _deserialize_checkpoint(self, data: Dict[str, Any]) -> OptimizationCheckpoint:
        """Convert JSON data back to OptimizationCheckpoint object."""
        # Convert metadata
        metadata = CheckpointMetadata(**data['metadata'])
        
        # Convert trials back to OptimizationTrial objects
        completed_trials = []
        for trial_data in data.get('completed_trials', []):
            # Handle datetime fields
            if trial_data.get('start_time'):
                trial_data['start_time'] = datetime.fromisoformat(trial_data['start_time'])
            if trial_data.get('end_time'):
                trial_data['end_time'] = datetime.fromisoformat(trial_data['end_time'])
            
            trial = OptimizationTrial(**trial_data)
            completed_trials.append(trial)
        
        # Convert best trial
        best_trial = None
        if data.get('best_trial'):
            best_trial_data = data['best_trial']
            if best_trial_data.get('start_time'):
                best_trial_data['start_time'] = datetime.fromisoformat(best_trial_data['start_time'])
            if best_trial_data.get('end_time'):
                best_trial_data['end_time'] = datetime.fromisoformat(best_trial_data['end_time'])
            best_trial = OptimizationTrial(**best_trial_data)
        
        return OptimizationCheckpoint(
            metadata=metadata,
            optimization_config=data['optimization_config'],
            search_space=data['search_space'],
            completed_trials=completed_trials,
            best_trial=best_trial,
            algorithm_state=data.get('algorithm_state', {}),
            hardware_profile=data.get('hardware_profile', {}),
            random_state=data.get('random_state')
        )


class OptimizationRecovery:
    """Handles recovery from checkpoints."""
    
    def __init__(self, checkpoint_manager: CheckpointManager):
        self.checkpoint_manager = checkpoint_manager
        self.logger = get_logger(__name__)
    
    def can_resume_optimization(self) -> bool:
        """Check if optimization can be resumed from checkpoint."""
        recovery_info = self.checkpoint_manager.get_recovery_info()
        return recovery_info is not None and recovery_info['can_resume']
    
    def get_resume_options(self) -> Dict[str, Any]:
        """Get available resume options."""
        checkpoints = self.checkpoint_manager.list_available_checkpoints()
        
        if not checkpoints:
            return {'can_resume': False, 'message': 'No checkpoints available'}
        
        latest = max(checkpoints, key=lambda x: x['created_at'])
        
        return {
            'can_resume': True,
            'latest_checkpoint': latest,
            'total_checkpoints': len(checkpoints),
            'estimated_progress': latest['trials_completed'] / max(latest['total_trials_planned'], 1),
            'estimated_time_saved': f"~{latest['trials_completed']} trials worth of compute time",
            'recommendation': f"Resume from latest checkpoint ({latest['trials_completed']} trials completed)"
        }
    
    def resume_from_latest(self) -> Optional[OptimizationCheckpoint]:
        """Resume optimization from latest checkpoint."""
        return self.checkpoint_manager.load_latest_checkpoint()
    
    def resume_from_checkpoint(self, checkpoint_id: str) -> Optional[OptimizationCheckpoint]:
        """Resume optimization from specific checkpoint."""
        return self.checkpoint_manager.load_specific_checkpoint(checkpoint_id)
    
    def validate_checkpoint_integrity(self, checkpoint: OptimizationCheckpoint) -> Tuple[bool, List[str]]:
        """Validate checkpoint data integrity."""
        issues = []
        
        try:
            # Check required fields
            if not checkpoint.metadata:
                issues.append("Missing metadata")
            
            if not checkpoint.optimization_config:
                issues.append("Missing optimization config")
            
            if not checkpoint.search_space:
                issues.append("Missing search space")
            
            # Validate trials data
            if checkpoint.completed_trials:
                trial_ids = [t.trial_id if hasattr(t, 'trial_id') else str(i) 
                           for i, t in enumerate(checkpoint.completed_trials)]
                if len(set(trial_ids)) != len(trial_ids):
                    issues.append("Duplicate trial IDs found")
            
            # Check metadata consistency
            if checkpoint.metadata.trials_completed != len(checkpoint.completed_trials):
                issues.append("Trial count mismatch in metadata")
            
            # Validate best trial
            if checkpoint.best_trial and checkpoint.completed_trials:
                best_score = checkpoint.best_trial.score if hasattr(checkpoint.best_trial, 'score') else None
                trial_scores = [getattr(t, 'score', None) for t in checkpoint.completed_trials 
                              if hasattr(t, 'score') and t.score is not None]
                
                if best_score is not None and trial_scores:
                    if best_score not in trial_scores:
                        issues.append("Best trial score not found in completed trials")
            
            is_valid = len(issues) == 0
            
            if is_valid:
                self.logger.info("Checkpoint validation passed")
            else:
                self.logger.warning(f"Checkpoint validation issues: {issues}")
            
            return is_valid, issues
            
        except Exception as e:
            self.logger.error(f"Error validating checkpoint: {e}")
            return False, [f"Validation error: {str(e)}"]


def create_checkpoint_manager(
    output_dir: str,
    save_frequency: int = 5,
    max_checkpoints: int = 10
) -> CheckpointManager:
    """Factory function to create checkpoint manager."""
    checkpoint_dir = Path(output_dir) / "checkpoints"
    return CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        save_frequency=save_frequency,
        max_checkpoints=max_checkpoints,
        compression=True
    )


def create_recovery_manager(checkpoint_manager: CheckpointManager) -> OptimizationRecovery:
    """Factory function to create recovery manager."""
    return OptimizationRecovery(checkpoint_manager)