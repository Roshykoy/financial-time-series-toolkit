"""
Main entry point for the comprehensive hyperparameter optimization system.
Provides CLI interface and orchestrates the optimization process.
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time
import numpy as np
from datetime import datetime

from .algorithms import create_optimizer
from .base_optimizer import OptimizationConfig
from .config_manager import create_config_manager, OptimizationPreset
from .hardware_manager import create_hardware_manager
from .integration import create_training_interface, create_objective_function
from .monitoring import create_monitor, create_visualizer
from .utils import OptimizationUtils, ConfigTemplate
from .pareto_interface import create_pareto_interface
from ..infrastructure.logging.logger import get_logger
from ..config_legacy import CONFIG

logger = get_logger(__name__)


class OptimizationOrchestrator:
    """Main orchestrator for hyperparameter optimization."""
    
    def __init__(self, data_path: str, output_dir: str = "optimization_results"):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.config_manager = create_config_manager()
        self.hardware_manager = create_hardware_manager()
        self.training_interface = create_training_interface(
            data_path, self.hardware_manager, CONFIG
        )
        self.monitor = create_monitor()
        self.visualizer = create_visualizer(self.output_dir / "plots")
        self.pareto_interface = create_pareto_interface(self.training_interface, CONFIG)
        
        logger.info(f"Optimization orchestrator initialized for {data_path}")
    
    def run_optimization(
        self,
        preset_name: Optional[str] = None,
        algorithm: Optional[str] = None,
        max_trials: Optional[int] = None,
        max_duration_hours: Optional[float] = None,
        parallel_jobs: Optional[int] = None,
        custom_search_space: Optional[Dict[str, Any]] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization with specified parameters."""
        
        logger.info("Starting hyperparameter optimization")
        start_time = datetime.now()
        
        try:
            # Load configuration
            if preset_name:
                preset = self.config_manager.get_preset(preset_name)
                opt_config = preset.optimization_config
                search_space = preset.search_space
                algorithm = algorithm or preset.algorithm
            else:
                opt_config = self.config_manager.default_config
                search_space = custom_search_space or self.training_interface.get_search_space()
                algorithm = algorithm or "random_search"
            
            # Override configuration parameters
            if max_trials:
                opt_config.max_trials = max_trials
            if max_duration_hours:
                opt_config.max_duration_hours = max_duration_hours
            if parallel_jobs:
                opt_config.parallel_jobs = parallel_jobs
            
            # Apply custom configuration
            if custom_config:
                for key, value in custom_config.items():
                    if hasattr(opt_config, key):
                        setattr(opt_config, key, value)
            
            # Validate search space
            is_valid, errors = OptimizationUtils.validate_search_space(search_space)
            if not is_valid:
                raise ValueError(f"Invalid search space: {errors}")
            
            # Create base config with trial timeout
            base_config = CONFIG.copy()
            base_config['max_training_duration_minutes'] = opt_config.trial_timeout_minutes
            
            # Create objective function
            objective = create_objective_function(
                self.training_interface, base_config, opt_config.validation_strategy
            )
            
            # Get hardware recommendations
            hardware_strategy = self.hardware_manager.suggest_optimization_strategy(
                opt_config.max_duration_hours
            )
            logger.info(f"Hardware recommendations: {hardware_strategy}")
            
            # Create optimizer
            optimizer = create_optimizer(
                algorithm=algorithm,
                objective_function=objective,
                search_space=search_space,
                config=opt_config,
                results_dir=self.output_dir,
                hardware_manager=self.hardware_manager
            )
            
            # Start monitoring
            self.monitor.start_monitoring()
            
            # Run optimization
            best_params, best_score = optimizer.optimize()
            
            # Stop monitoring
            self.monitor.stop_monitoring()
            
            # Generate results
            results = self._generate_results(
                optimizer, best_params, best_score, start_time, 
                algorithm, search_space, opt_config
            )
            
            # Create visualizations
            self._create_visualizations(optimizer.trials)
            
            # Save results
            self._save_results(results)
            
            logger.info(f"Optimization completed successfully. Best score: {best_score:.6f}")
            return results
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            self.monitor.stop_monitoring()
            raise
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self.monitor.stop_monitoring()
            raise
    
    def _generate_results(
        self,
        optimizer,
        best_params: Dict[str, Any],
        best_score: float,
        start_time: datetime,
        algorithm: str,
        search_space: Dict[str, Any],
        opt_config: OptimizationConfig
    ) -> Dict[str, Any]:
        """Generate comprehensive results summary."""
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Get optimization statistics
        summary_stats = self.monitor.get_summary_stats()
        objective_stats = optimizer.objective_function.get_statistics()
        
        results = {
            'optimization_summary': {
                'algorithm': algorithm,
                'best_score': best_score,
                'best_parameters': best_params,
                'total_trials': summary_stats['total_trials'],
                'successful_trials': summary_stats['completed_trials'],
                'failed_trials': summary_stats['failed_trials'],
                'success_rate': summary_stats['success_rate'],
                'total_duration': str(duration),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat()
            },
            'configuration': {
                'algorithm': algorithm,
                'optimization_config': opt_config.__dict__,
                'search_space': search_space
            },
            'performance_metrics': {
                'average_trial_duration': summary_stats.get('avg_trial_duration_seconds', 0),
                'total_compute_hours': summary_stats.get('total_compute_hours', 0),
                'trials_per_hour': summary_stats['completed_trials'] / max(duration.total_seconds() / 3600, 0.1),
                'objective_calls': objective_stats['call_count'],
                'objective_total_time': objective_stats['total_time_seconds']
            },
            'hardware_profile': {
                'cpu_cores': self.hardware_manager.profile.cpu_cores,
                'total_memory_gb': self.hardware_manager.profile.total_memory_gb,
                'gpu_available': self.hardware_manager.profile.gpu_available,
                'gpu_count': self.hardware_manager.profile.gpu_count,
                'recommended_batch_size': self.hardware_manager.profile.recommended_batch_size
            },
            'parameter_analysis': self._analyze_parameters(optimizer.trials),
            'recommendations': self._generate_recommendations(
                optimizer.trials, best_params, search_space
            )
        }
        
        return results
    
    def _analyze_parameters(self, trials) -> Dict[str, Any]:
        """Analyze parameter importance and correlations."""
        completed_trials = [t for t in trials if t.status == "completed" and t.score is not None]
        
        if len(completed_trials) < 5:
            return {'message': 'Not enough trials for parameter analysis'}
        
        # Extract data
        trial_params = [t.parameters for t in completed_trials]
        trial_scores = [t.score for t in completed_trials]
        
        # Calculate importance
        importance = OptimizationUtils.calculate_parameter_importance(trial_params, trial_scores)
        
        # Suggest parameter bounds
        bounds = OptimizationUtils.suggest_parameter_bounds(trial_params, trial_scores)
        
        return {
            'parameter_importance': importance,
            'suggested_bounds': bounds,
            'total_analyzed_trials': len(completed_trials)
        }
    
    def _generate_recommendations(
        self,
        trials,
        best_params: Dict[str, Any],
        search_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimization recommendations."""
        
        completed_trials = [t for t in trials if t.status == "completed" and t.score is not None]
        
        recommendations = {
            'next_steps': [],
            'parameter_adjustments': {},
            'search_space_refinements': {}
        }
        
        if len(completed_trials) < 10:
            recommendations['next_steps'].append(
                "Run more trials (at least 20) for better parameter analysis"
            )
        
        # Analyze score distribution
        if completed_trials:
            scores = [t.score for t in completed_trials]
            score_std = np.std(scores)
            score_range = max(scores) - min(scores)
            
            if score_std < score_range * 0.1:
                recommendations['next_steps'].append(
                    "Consider expanding search space - limited score variation observed"
                )
            
            if len(set(scores)) == 1:
                recommendations['next_steps'].append(
                    "All trials have same score - check objective function and search space"
                )
        
        # Check for parameter convergence
        if len(completed_trials) >= 10:
            trial_params = [t.parameters for t in completed_trials[-10:]]  # Last 10 trials
            
            for param_name in best_params.keys():
                if param_name in search_space:
                    recent_values = [p.get(param_name) for p in trial_params if param_name in p]
                    if len(set(recent_values)) <= 2:
                        recommendations['parameter_adjustments'][param_name] = (
                            f"Consider expanding range - converging to {recent_values[0]}"
                        )
        
        # Hardware recommendations
        avg_duration = np.mean([t.duration_seconds for t in completed_trials if t.duration_seconds])
        if avg_duration > 600:  # 10 minutes
            recommendations['next_steps'].append(
                "Consider reducing epochs or batch size to speed up trials"
            )
        
        return recommendations
    
    def _create_visualizations(self, trials) -> None:
        """Create optimization visualizations."""
        try:
            logger.info("Creating optimization visualizations")
            
            # Create dashboard
            self.visualizer.create_optimization_dashboard(trials, self.monitor)
            
            logger.info(f"Visualizations saved to {self.visualizer.output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save optimization results to files."""
        try:
            # Save main results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = self.output_dir / f"optimization_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save best parameters separately
            best_params_file = self.output_dir / "best_parameters.json"
            with open(best_params_file, 'w') as f:
                json.dump(results['optimization_summary']['best_parameters'], f, indent=2)
            
            # Export monitoring data
            self.monitor.export_data(self.output_dir / "monitoring")
            
            logger.info(f"Results saved to {results_file}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def list_presets(self) -> List[str]:
        """List available optimization presets."""
        return self.config_manager.list_presets()
    
    def get_preset_info(self, preset_name: str) -> Dict[str, Any]:
        """Get information about a specific preset."""
        preset = self.config_manager.get_preset(preset_name)
        return {
            'name': preset.name,
            'description': preset.description,
            'algorithm': preset.algorithm,
            'max_trials': preset.optimization_config.max_trials,
            'max_duration_hours': preset.optimization_config.max_duration_hours,
            'search_space_size': len(preset.search_space)
        }
    
    def run_pareto_optimization(
        self,
        search_space: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Run Pareto Front multi-objective optimization."""
        
        logger.info("Starting Pareto Front optimization")
        
        try:
            # Show algorithm selection menu
            choice = self.pareto_interface.show_algorithm_selection()
            
            if choice == '1':
                algorithm = "nsga2"
                algorithm_name = "NSGA-II"
            elif choice == '2':
                algorithm = "tpe"
                algorithm_name = "TPE/Optuna"
            elif choice == '3':
                # Show saved results
                result = self.pareto_interface.show_saved_results_menu()
                if result:
                    self.pareto_interface.display_pareto_results(result)
                    
                    # Allow selection from Pareto Front
                    pareto_front = result.get_pareto_front()
                    selected_params = self.pareto_interface.select_pareto_solution(pareto_front)
                    
                    if selected_params:
                        return {
                            'type': 'pareto_selection',
                            'algorithm': result.algorithm,
                            'selected_parameters': selected_params,
                            'pareto_front_size': len(pareto_front)
                        }
                return None
            elif choice == '4':
                logger.info("Pareto Front optimization cancelled")
                return None
            else:
                logger.error("Invalid choice for Pareto Front optimization")
                return None
            
            # Get optimization parameters
            search_space = search_space or self.training_interface.get_search_space()
            config, kwargs = self.pareto_interface.get_optimization_parameters(algorithm_name)
            
            # Run Pareto Front optimization
            result = self.pareto_interface.run_pareto_optimization(
                algorithm, search_space, config, **kwargs
            )
            
            # Display results
            self.pareto_interface.display_pareto_results(result)
            
            # Allow user to select solution from Pareto Front
            pareto_front = result.get_pareto_front()
            selected_params = self.pareto_interface.select_pareto_solution(pareto_front)
            
            return {
                'type': 'pareto_optimization',
                'algorithm': result.algorithm,
                'total_evaluations': result.total_evaluations,
                'computation_time': result.computation_time,
                'pareto_front_size': len(pareto_front),
                'selected_parameters': selected_params,
                'result': result
            }
            
        except Exception as e:
            logger.error(f"Pareto Front optimization failed: {e}")
            raise


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Hyperparameter Optimization for MarkSix Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 5 trials
  python -m src.optimization.main --preset quick_test
  
  # Balanced optimization with custom parameters
  python -m src.optimization.main --preset balanced_search --max-trials 40
  
  # Custom optimization
  python -m src.optimization.main --algorithm bayesian --max-trials 30 --max-duration 4
  
  # List available presets
  python -m src.optimization.main --list-presets
        """
    )
    
    parser.add_argument(
        '--data-path',
        default='data/raw/Mark_Six.csv',
        help='Path to Mark Six data file'
    )
    
    parser.add_argument(
        '--output-dir',
        default='optimization_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--preset',
        help='Optimization preset to use'
    )
    
    parser.add_argument(
        '--algorithm',
        choices=['grid_search', 'random_search', 'bayesian', 'optuna'],
        help='Optimization algorithm'
    )
    
    parser.add_argument(
        '--max-trials',
        type=int,
        help='Maximum number of trials'
    )
    
    parser.add_argument(
        '--max-duration',
        type=float,
        help='Maximum duration in hours'
    )
    
    parser.add_argument(
        '--parallel-jobs',
        type=int,
        help='Number of parallel jobs'
    )
    
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List available presets and exit'
    )
    
    parser.add_argument(
        '--preset-info',
        help='Show information about specific preset'
    )
    
    parser.add_argument(
        '--config',
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser


def main():
    """Main entry point for CLI."""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create orchestrator
        orchestrator = OptimizationOrchestrator(args.data_path, args.output_dir)
        
        # Handle special commands
        if args.list_presets:
            presets = orchestrator.list_presets()
            print("Available optimization presets:")
            for preset in presets:
                info = orchestrator.get_preset_info(preset)
                print(f"  {preset}: {info['description']}")
                print(f"    Algorithm: {info['algorithm']}, "
                      f"Max Trials: {info['max_trials']}, "
                      f"Duration: {info['max_duration_hours']}h")
            return
        
        if args.preset_info:
            info = orchestrator.get_preset_info(args.preset_info)
            print(f"Preset: {info['name']}")
            print(f"Description: {info['description']}")
            print(f"Algorithm: {info['algorithm']}")
            print(f"Max Trials: {info['max_trials']}")
            print(f"Max Duration: {info['max_duration_hours']} hours")
            print(f"Search Space Parameters: {info['search_space_size']}")
            return
        
        # Load custom configuration if provided
        custom_config = None
        if args.config:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
        
        # Run optimization
        results = orchestrator.run_optimization(
            preset_name=args.preset,
            algorithm=args.algorithm,
            max_trials=args.max_trials,
            max_duration_hours=args.max_duration,
            parallel_jobs=args.parallel_jobs,
            custom_config=custom_config
        )
        
        # Print summary
        summary = results['optimization_summary']
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED")
        print("="*60)
        print(f"Algorithm: {summary['algorithm']}")
        print(f"Best Score: {summary['best_score']:.6f}")
        print(f"Total Trials: {summary['total_trials']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Duration: {summary['total_duration']}")
        print(f"\nResults saved to: {args.output_dir}")
        print("\nBest Parameters:")
        for param, value in summary['best_parameters'].items():
            print(f"  {param}: {value}")
        
        # Print recommendations
        if 'recommendations' in results:
            recommendations = results['recommendations']
            if recommendations['next_steps']:
                print("\nRecommendations:")
                for step in recommendations['next_steps']:
                    print(f"  - {step}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()