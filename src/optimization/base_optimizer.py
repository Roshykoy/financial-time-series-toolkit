"""
Base hyperparameter optimization framework.
Provides foundation for all optimization algorithms with hardware-aware capabilities.
"""

import os
import time
import json
import pickle
import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import torch
import numpy as np
from tqdm import tqdm

from .hardware_manager import HardwareResourceManager, create_hardware_manager
from .checkpoint_manager import CheckpointManager, CheckpointMetadata, OptimizationRecovery, create_checkpoint_manager
from ..infrastructure.logging.logger import get_logger
from ..utils.error_handling import safe_execute
from ..utils.performance_monitor import PerformanceMonitor

logger = get_logger(__name__)


from .data_structures import OptimizationTrial, OptimizationConfig


class BaseHyperparameterOptimizer(ABC):
    """
    Abstract base class for hyperparameter optimization.
    
    Provides common functionality for all optimization algorithms including:
    - Hardware resource management
    - Trial execution and monitoring
    - Result tracking and checkpointing
    - Early stopping and timeout handling
    """
    
    def __init__(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        search_space: Dict[str, Any],
        config: OptimizationConfig,
        results_dir: Union[str, Path] = "optimization_results",
        hardware_manager: Optional[HardwareResourceManager] = None,
        enable_checkpointing: bool = True,
        checkpoint_frequency: int = 5
    ):
        self.objective_function = objective_function
        self.search_space = search_space
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.hardware_manager = hardware_manager or create_hardware_manager()
        self.performance_monitor = PerformanceMonitor()
        
        # Checkpoint management
        self.enable_checkpointing = enable_checkpointing
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.recovery_manager: Optional[OptimizationRecovery] = None
        
        if enable_checkpointing:
            self.checkpoint_manager = create_checkpoint_manager(
                str(self.results_dir),
                save_frequency=checkpoint_frequency,
                max_checkpoints=10
            )
            self.recovery_manager = OptimizationRecovery(self.checkpoint_manager)
        
        # Trial tracking
        self.trials: List[OptimizationTrial] = []
        self.best_trial: Optional[OptimizationTrial] = None
        self.optimization_start_time: Optional[datetime] = None
        self.optimization_end_time: Optional[datetime] = None
        
        # Resume state
        self._resumed_from_checkpoint = False
        self._starting_trial_number = 0
        
        # State management
        self._is_running = False
        self._should_stop = False
        self._lock = threading.Lock()
        
        # Setup random seed
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            torch.manual_seed(config.random_seed)
        
        logger.info(f"Initialized {self.__class__.__name__} with {config.max_trials} max trials")
    
    @abstractmethod
    def _generate_trial_parameters(self, trial_number: int) -> Dict[str, Any]:
        """Generate parameters for next trial. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _update_algorithm_state(self, trial: OptimizationTrial) -> None:
        """Update algorithm-specific state after trial completion."""
        pass
    
    @abstractmethod
    def _get_algorithm_state(self) -> Dict[str, Any]:
        """Get algorithm-specific state for checkpointing."""
        pass
    
    @abstractmethod
    def _restore_algorithm_state(self, state: Dict[str, Any]) -> None:
        """Restore algorithm-specific state from checkpoint."""
        pass
    
    def can_resume_from_checkpoint(self) -> bool:
        """Check if optimization can be resumed from checkpoint."""
        if not self.enable_checkpointing or not self.recovery_manager:
            return False
        return self.recovery_manager.can_resume_optimization()
    
    def get_resume_options(self) -> Dict[str, Any]:
        """Get available resume options."""
        if not self.enable_checkpointing or not self.recovery_manager:
            return {'can_resume': False, 'message': 'Checkpointing not enabled'}
        return self.recovery_manager.get_resume_options()
    
    def resume_from_checkpoint(self, checkpoint_id: Optional[str] = None) -> bool:
        """Resume optimization from checkpoint."""
        if not self.enable_checkpointing or not self.recovery_manager:
            logger.warning("Checkpointing not enabled, cannot resume")
            return False
        
        try:
            # Load checkpoint
            if checkpoint_id:
                checkpoint = self.recovery_manager.resume_from_checkpoint(checkpoint_id)
            else:
                checkpoint = self.recovery_manager.resume_from_latest()
            
            if not checkpoint:
                logger.error("Failed to load checkpoint")
                return False
            
            # Validate checkpoint integrity
            is_valid, issues = self.recovery_manager.validate_checkpoint_integrity(checkpoint)
            if not is_valid:
                logger.error(f"Checkpoint validation failed: {issues}")
                return False
            
            # Restore state
            self.trials = checkpoint.completed_trials
            self.best_trial = checkpoint.best_trial
            self.optimization_start_time = datetime.fromisoformat(checkpoint.metadata.optimization_start_time)
            self._starting_trial_number = checkpoint.metadata.trials_completed
            self._resumed_from_checkpoint = True
            
            # Restore algorithm-specific state
            self._restore_algorithm_state(checkpoint.algorithm_state)
            
            logger.info(f"Resumed from checkpoint: {checkpoint.metadata.trials_completed} trials completed")
            return True
            
        except Exception as e:
            logger.error(f"Error resuming from checkpoint: {e}")
            return False
    
    def _save_checkpoint(self, force: bool = False) -> bool:
        """Save current optimization state to checkpoint."""
        if not self.enable_checkpointing or not self.checkpoint_manager:
            return False
        
        try:
            # Check if we should save
            trials_completed = len([t for t in self.trials if t.status == "completed"])
            
            if not force and not self.checkpoint_manager.should_save_checkpoint(trials_completed):
                return False
            
            # Create metadata
            metadata = CheckpointMetadata(
                optimization_start_time=self.optimization_start_time.isoformat() if self.optimization_start_time else "",
                algorithm=self.__class__.__name__,
                total_trials_planned=self.config.max_trials,
                trials_completed=trials_completed,
                trials_failed=len([t for t in self.trials if t.status == "failed"]),
                best_score=self.best_trial.score if self.best_trial else None,
                best_trial_id=self.best_trial.trial_id if self.best_trial else None
            )
            
            # Calculate ETA
            if self.optimization_start_time and trials_completed > 0:
                elapsed = datetime.now() - self.optimization_start_time
                trials_remaining = self.config.max_trials - trials_completed
                avg_time_per_trial = elapsed.total_seconds() / trials_completed
                eta = datetime.now() + timedelta(seconds=avg_time_per_trial * trials_remaining)
                metadata.estimated_completion_time = eta.isoformat()
            
            # Get algorithm state
            algorithm_state = self._get_algorithm_state()
            
            # Get hardware profile
            hardware_profile = asdict(self.hardware_manager.profile) if self.hardware_manager else {}
            
            # Save checkpoint
            checkpoint_id = self.checkpoint_manager.create_checkpoint(
                metadata=metadata,
                optimization_config=self.config,
                search_space=self.search_space,
                completed_trials=self.trials,
                best_trial=self.best_trial,
                algorithm_state=algorithm_state,
                hardware_profile=hardware_profile,
                random_state=self._get_random_state()
            )
            
            logger.info(f"Checkpoint saved: {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
            return False
    
    def _get_random_state(self) -> Dict[str, Any]:
        """Get current random state for reproducibility."""
        try:
            return {
                'numpy_state': np.random.get_state(),
                'torch_state': torch.get_rng_state().tolist() if torch.is_tensor(torch.get_rng_state()) else None,
                'python_state': random.getstate()
            }
        except Exception:
            return {}
    
    def _restore_random_state(self, random_state: Dict[str, Any]) -> None:
        """Restore random state for reproducibility."""
        try:
            if 'numpy_state' in random_state:
                np.random.set_state(random_state['numpy_state'])
            if 'torch_state' in random_state and random_state['torch_state']:
                torch.set_rng_state(torch.tensor(random_state['torch_state']))
            if 'python_state' in random_state:
                random.setstate(random_state['python_state'])
        except Exception as e:
            logger.warning(f"Could not restore random state: {e}")

    def optimize(self) -> Tuple[Dict[str, Any], float]:
        """
        Main optimization loop with checkpoint support.
        
        Returns:
            Tuple of (best_parameters, best_score)
        """
        logger.info("Starting hyperparameter optimization")
        
        # Check for resume opportunity
        if not self._resumed_from_checkpoint and self.can_resume_from_checkpoint():
            resume_options = self.get_resume_options()
            logger.info(f"Found existing checkpoint with {resume_options.get('latest_checkpoint', {}).get('trials_completed', 0)} completed trials")
            logger.info("To resume, use resume_from_checkpoint() before calling optimize()")
        
        if not self.optimization_start_time:
            self.optimization_start_time = datetime.now()
        
        self._is_running = True
        self._should_stop = False
        
        try:
            # Start hardware monitoring
            if self.config.enable_hardware_monitoring:
                self.hardware_manager.start_monitoring()
            
            # Run optimization based on parallelization strategy
            if self.config.parallel_jobs > 1:
                self._run_parallel_optimization()
            else:
                self._run_sequential_optimization()
            
            self.optimization_end_time = datetime.now()
            self._is_running = False
            
            # Save final checkpoint
            if self.enable_checkpointing:
                self._save_checkpoint(force=True)
            
            # Final cleanup and reporting
            self._cleanup_and_finalize()
            
            if self.best_trial:
                logger.info(f"Optimization completed. Best score: {self.best_trial.score:.6f}")
                return self.best_trial.parameters, self.best_trial.score
            else:
                logger.warning("No successful trials completed")
                return {}, float('-inf')
                
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            self._should_stop = True
            # Save emergency checkpoint
            if self.enable_checkpointing:
                logger.info("Saving emergency checkpoint...")
                self._save_checkpoint(force=True)
            return self._handle_early_termination()
            
        except Exception as e:
            logger.error(f"Optimization failed with error: {e}")
            self._should_stop = True
            # Save emergency checkpoint
            if self.enable_checkpointing:
                logger.info("Saving emergency checkpoint...")
                self._save_checkpoint(force=True)
            return self._handle_early_termination()
        
        finally:
            self._is_running = False
            if self.config.enable_hardware_monitoring:
                self.hardware_manager.stop_monitoring()
    
    def _run_sequential_optimization(self) -> None:
        """Run optimization trials sequentially with checkpoint support."""
        # Adjust progress bar for resumed optimization
        completed_trials = len([t for t in self.trials if t.status == "completed"])
        start_trial = self._starting_trial_number if self._resumed_from_checkpoint else 0
        
        with tqdm(
            total=self.config.max_trials, 
            initial=completed_trials,
            desc="Optimization Progress"
        ) as pbar:
            
            for trial_num in range(start_trial, self.config.max_trials):
                if self._should_stop or self._check_termination_conditions():
                    break
                
                # Generate and execute trial
                trial = self._create_and_execute_trial(trial_num)
                
                # Update progress bar
                if trial.status == "completed":
                    pbar.update(1)
                    if self.best_trial:
                        pbar.set_postfix(best_score=f"{self.best_trial.score:.6f}")
                
                # Checkpoint saving
                if self.enable_checkpointing:
                    self._save_checkpoint()
                
                # Periodic cleanup
                if (trial_num + 1) % self.config.cleanup_frequency == 0:
                    self._periodic_maintenance(trial_num + 1)
    
    def _run_parallel_optimization(self) -> None:
        """Run optimization trials in parallel."""
        max_workers = min(self.config.parallel_jobs, self.hardware_manager.profile.parallel_jobs)
        logger.info(f"Running parallel optimization with {max_workers} workers")
        
        # Use ThreadPoolExecutor for I/O bound tasks (most ML training is GPU bound)
        # For CPU-intensive tasks, consider ProcessPoolExecutor
        executor_class = ThreadPoolExecutor
        
        with executor_class(max_workers=max_workers) as executor:
            futures = {}
            trial_num = 0
            completed_trials = 0
            
            with tqdm(total=self.config.max_trials, desc="Parallel Optimization") as pbar:
                while completed_trials < self.config.max_trials and not self._should_stop:
                    # Submit new trials up to max_trials
                    while (len(futures) < max_workers and 
                           trial_num < self.config.max_trials and 
                           not self._should_stop):
                        
                        future = executor.submit(self._create_and_execute_trial, trial_num)
                        futures[future] = trial_num
                        trial_num += 1
                    
                    # Wait for at least one trial to complete
                    if futures:
                        done_futures = as_completed(futures, timeout=60.0)
                        for future in done_futures:
                            try:
                                trial = future.result()
                                completed_trials += 1
                                pbar.update(1)
                                
                                if self.best_trial:
                                    pbar.set_postfix(best_score=f"{self.best_trial.score:.6f}")
                                
                                # Periodic maintenance
                                if completed_trials % self.config.cleanup_frequency == 0:
                                    self._periodic_maintenance(completed_trials)
                                
                                # Check early stopping
                                if self._check_termination_conditions():
                                    self._should_stop = True
                                    break
                                    
                            except Exception as e:
                                logger.error(f"Trial execution failed: {e}")
                            
                            finally:
                                futures.pop(future, None)
                    
                    if self._check_termination_conditions():
                        self._should_stop = True
                        break
            
            # Cancel remaining futures
            for future in futures:
                future.cancel()
    
    def _create_and_execute_trial(self, trial_num: int) -> OptimizationTrial:
        """Create and execute a single optimization trial."""
        trial_id = f"trial_{trial_num:04d}_{int(time.time())}"
        
        try:
            # Generate parameters
            parameters = self._generate_trial_parameters(trial_num)
            
            # Apply hardware optimizations
            optimized_params = self.hardware_manager.optimize_for_hardware(parameters)
            
            # Create trial object
            trial = OptimizationTrial(
                trial_id=trial_id,
                parameters=optimized_params,
                start_time=datetime.now(),
                status="running"
            )
            
            # Execute trial with timeout and monitoring
            score = self._execute_trial_with_monitoring(trial)
            
            # Update trial
            trial.end_time = datetime.now()
            trial.duration_seconds = (trial.end_time - trial.start_time).total_seconds()
            trial.score = score
            trial.status = "completed"
            
            # Update best trial
            with self._lock:
                self.trials.append(trial)
                if self.best_trial is None or score > self.best_trial.score:
                    self.best_trial = trial
                    logger.info(f"New best trial {trial_id}: score={score:.6f}")
            
            # Update algorithm state
            self._update_algorithm_state(trial)
            
            return trial
            
        except Exception as e:
            # Handle failed trial
            trial = OptimizationTrial(
                trial_id=trial_id,
                parameters=parameters if 'parameters' in locals() else {},
                start_time=datetime.now() if 'trial' not in locals() else trial.start_time,
                end_time=datetime.now(),
                status="failed",
                error_message=str(e)
            )
            
            with self._lock:
                self.trials.append(trial)
            
            logger.error(f"Trial {trial_id} failed: {e}")
            return trial
    
    def _execute_trial_with_monitoring(self, trial: OptimizationTrial) -> float:
        """Execute trial with resource monitoring and timeout."""
        # Set timeout
        timeout_seconds = self.config.trial_timeout_minutes * 60
        start_time = time.time()
        
        # Monitor resources before trial
        resource_status_before = self.hardware_manager.get_resource_status()
        
        # Execute objective function with timeout
        def target():
            return self.objective_function(trial.parameters)
        
        # Simple timeout implementation (can be enhanced with threading)
        try:
            score = target()
            
            # Check if trial exceeded timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(f"Trial exceeded timeout of {timeout_seconds}s")
            
            # Monitor resources after trial
            resource_status_after = self.hardware_manager.get_resource_status()
            trial.resource_usage = {
                'before': resource_status_before,
                'after': resource_status_after,
                'duration_seconds': elapsed
            }
            
            return score
            
        except Exception as e:
            logger.error(f"Trial execution error: {e}")
            raise
    
    def _check_termination_conditions(self) -> bool:
        """Check if optimization should be terminated."""
        if self._should_stop:
            return True
        
        # Check time limit
        if self.optimization_start_time:
            elapsed_hours = (datetime.now() - self.optimization_start_time).total_seconds() / 3600
            if elapsed_hours >= self.config.max_duration_hours:
                logger.info(f"Time limit reached: {elapsed_hours:.2f} hours")
                return True
        
        # Check early stopping
        if self.config.early_stopping and len(self.trials) >= self.config.early_stopping_patience:
            recent_trials = sorted(
                [t for t in self.trials if t.status == "completed"],
                key=lambda x: x.start_time, 
                reverse=True
            )[:self.config.early_stopping_patience]
            
            if len(recent_trials) >= self.config.early_stopping_patience:
                recent_scores = [t.score for t in recent_trials]
                score_improvement = max(recent_scores) - min(recent_scores)
                
                if score_improvement < self.config.min_improvement:
                    logger.info(f"Early stopping: no improvement > {self.config.min_improvement}")
                    return True
        
        return False
    
    def _periodic_maintenance(self, completed_trials: int) -> None:
        """Perform periodic maintenance tasks."""
        try:
            # Resource cleanup
            self.hardware_manager.cleanup_resources()
            
            # Save intermediate results
            if self.config.save_intermediate_results:
                self._save_intermediate_results(completed_trials)
            
            # Check resource constraints
            is_ok, issues = self.hardware_manager.check_resource_constraints()
            if not is_ok:
                logger.warning(f"Resource issues detected: {issues}")
            
        except Exception as e:
            logger.error(f"Error during periodic maintenance: {e}")
    
    def _save_intermediate_results(self, trial_count: int) -> None:
        """Save intermediate optimization results."""
        try:
            results = {
                'optimization_config': asdict(self.config),
                'search_space': self.search_space,
                'trials': [asdict(trial) for trial in self.trials],
                'best_trial': asdict(self.best_trial) if self.best_trial else None,
                'trial_count': trial_count,
                'timestamp': datetime.now().isoformat(),
                'hardware_profile': asdict(self.hardware_manager.profile)
            }
            
            # Save main results
            results_file = self.results_dir / f"intermediate_results_{trial_count:04d}.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save best parameters separately for easy access
            if self.best_trial:
                best_params_file = self.results_dir / "best_parameters.json"
                with open(best_params_file, 'w') as f:
                    json.dump(self.best_trial.parameters, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
    
    def _cleanup_and_finalize(self) -> None:
        """Final cleanup and result saving."""
        try:
            # Final resource cleanup
            self.hardware_manager.cleanup_resources()
            
            # Save final results
            self._save_final_results()
            
            # Generate optimization report
            self._generate_optimization_report()
            
            logger.info("Optimization cleanup and finalization completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup and finalization: {e}")
    
    def _save_final_results(self) -> None:
        """Save final optimization results."""
        try:
            final_results = {
                'optimization_config': asdict(self.config),
                'search_space': self.search_space,
                'all_trials': [asdict(trial) for trial in self.trials],
                'best_trial': asdict(self.best_trial) if self.best_trial else None,
                'optimization_summary': self._generate_summary(),
                'hardware_profile': asdict(self.hardware_manager.profile),
                'monitoring_data': self.hardware_manager.get_monitoring_data(),
                'start_time': self.optimization_start_time.isoformat() if self.optimization_start_time else None,
                'end_time': self.optimization_end_time.isoformat() if self.optimization_end_time else None
            }
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            final_file = self.results_dir / f"final_results_{timestamp}.json"
            
            with open(final_file, 'w') as f:
                json.dump(final_results, f, indent=2, default=str)
            
            logger.info(f"Final results saved to {final_file}")
            
        except Exception as e:
            logger.error(f"Error saving final results: {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate optimization summary statistics."""
        completed_trials = [t for t in self.trials if t.status == "completed"]
        failed_trials = [t for t in self.trials if t.status == "failed"]
        
        summary = {
            'total_trials': len(self.trials),
            'completed_trials': len(completed_trials),
            'failed_trials': len(failed_trials),
            'success_rate': len(completed_trials) / len(self.trials) if self.trials else 0,
            'best_score': self.best_trial.score if self.best_trial else None,
            'algorithm': self.__class__.__name__
        }
        
        if completed_trials:
            scores = [t.score for t in completed_trials]
            durations = [t.duration_seconds for t in completed_trials if t.duration_seconds]
            
            summary.update({
                'score_statistics': {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores)
                },
                'duration_statistics': {
                    'mean_seconds': np.mean(durations) if durations else 0,
                    'total_hours': sum(durations) / 3600 if durations else 0
                }
            })
        
        return summary
    
    def _generate_optimization_report(self) -> None:
        """Generate human-readable optimization report."""
        try:
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("HYPERPARAMETER OPTIMIZATION REPORT")
            report_lines.append("=" * 80)
            report_lines.append("")
            
            # Basic information
            report_lines.append(f"Algorithm: {self.__class__.__name__}")
            report_lines.append(f"Start Time: {self.optimization_start_time}")
            report_lines.append(f"End Time: {self.optimization_end_time}")
            
            if self.optimization_start_time and self.optimization_end_time:
                duration = self.optimization_end_time - self.optimization_start_time
                report_lines.append(f"Total Duration: {duration}")
            
            report_lines.append("")
            
            # Trial summary
            summary = self._generate_summary()
            report_lines.append("TRIAL SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"Total Trials: {summary['total_trials']}")
            report_lines.append(f"Completed: {summary['completed_trials']}")
            report_lines.append(f"Failed: {summary['failed_trials']}")
            report_lines.append(f"Success Rate: {summary['success_rate']:.2%}")
            report_lines.append("")
            
            # Best result
            if self.best_trial:
                report_lines.append("BEST RESULT")
                report_lines.append("-" * 40)
                report_lines.append(f"Score: {self.best_trial.score:.6f}")
                report_lines.append(f"Trial ID: {self.best_trial.trial_id}")
                report_lines.append("Parameters:")
                for key, value in self.best_trial.parameters.items():
                    report_lines.append(f"  {key}: {value}")
                report_lines.append("")
            
            # Hardware summary
            report_lines.append("HARDWARE PROFILE")
            report_lines.append("-" * 40)
            profile = self.hardware_manager.profile
            report_lines.append(f"CPU Cores: {profile.cpu_cores}")
            report_lines.append(f"Total Memory: {profile.total_memory_gb:.1f} GB")
            report_lines.append(f"GPU Available: {profile.gpu_available}")
            if profile.gpu_available:
                report_lines.append(f"GPU Count: {profile.gpu_count}")
                for i, (name, memory) in enumerate(zip(profile.gpu_names, profile.gpu_memory_gb)):
                    report_lines.append(f"  GPU {i}: {name} ({memory:.1f} GB)")
            
            # Save report
            report_file = self.results_dir / "optimization_report.txt"
            with open(report_file, 'w') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"Optimization report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating optimization report: {e}")
    
    def _handle_early_termination(self) -> Tuple[Dict[str, Any], float]:
        """Handle early termination of optimization."""
        logger.info("Handling early termination")
        
        # Save current state
        self._save_final_results()
        
        if self.best_trial:
            return self.best_trial.parameters, self.best_trial.score
        else:
            return {}, float('-inf')
    
    def stop_optimization(self) -> None:
        """Request optimization to stop gracefully."""
        logger.info("Stop requested for optimization")
        self._should_stop = True
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            'is_running': self._is_running,
            'should_stop': self._should_stop,
            'total_trials': len(self.trials),
            'completed_trials': len([t for t in self.trials if t.status == "completed"]),
            'failed_trials': len([t for t in self.trials if t.status == "failed"]),
            'best_score': self.best_trial.score if self.best_trial else None,
            'start_time': self.optimization_start_time.isoformat() if self.optimization_start_time else None,
            'elapsed_time': (datetime.now() - self.optimization_start_time).total_seconds() 
                           if self.optimization_start_time else None
        }