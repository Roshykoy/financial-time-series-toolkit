"""
User interface for Pareto Front hyperparameter optimization.
Provides algorithm selection and result management.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from .pareto_front import (
    create_pareto_optimizer, 
    MultiObjectiveFunction, 
    ParetoFrontResult,
    DEFAULT_MULTI_OBJECTIVE_DEFINITIONS
)
from .base_optimizer import OptimizationConfig

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


class ParetoFrontInterface:
    """User-friendly interface for Pareto Front optimization."""
    
    def __init__(self, training_interface, base_config: Dict[str, Any]):
        self.training_interface = training_interface
        self.base_config = base_config
        self.output_dir = Path("models/pareto_front")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Pareto Front interface initialized")
    
    def show_algorithm_selection(self) -> str:
        """Display algorithm selection menu and get user choice."""
        print("\n" + "="*80)
        print("PARETO FRONT MULTI-OBJECTIVE OPTIMIZATION")
        print("="*80)
        print("\nSelect optimization algorithm:")
        print("\n1. NSGA-II (Evolutionary Algorithm)")
        print("   ✅ Pros:")
        print("      • Global search - explores entire hyperparameter space")
        print("      • True Pareto Front - generates well-distributed solutions")
        print("      • Population-based - provides multiple solutions simultaneously")
        print("      • Robust - less sensitive to hyperparameter space topology")
        print("      • Parallel-friendly - easy to parallelize across cores/GPUs")
        print("   ❌ Cons:")
        print("      • Computational cost - requires many evaluations (100s-1000s)")
        print("      • Time intensive - slower convergence")
        print("      • Memory usage - maintains entire population in memory")
        print("      • No learning - doesn't learn from previous evaluations efficiently")
        print("   🎯 Best for: Thorough exploration, production optimization")
        
        print("\n2. TPE/Optuna (Multi-Objective Bayesian Optimization)")
        print("   ✅ Pros:")
        print("      • Sample efficient - fewer evaluations needed for good results")
        print("      • Fast convergence - learns from previous trials")
        print("      • Adaptive - automatically focuses on promising regions")
        print("      • Built-in framework - robust, tested implementation")
        print("      • Memory efficient - sequential optimization with learning")
        print("   ❌ Cons:")
        print("      • Local optima risk - may miss global optima in complex spaces")
        print("      • Limited front coverage - may not explore full Pareto Front")
        print("      • Hyperparameter sensitive - depends on acquisition function")
        print("      • Sequential - less naturally parallel than population methods")
        print("   🎯 Best for: Quick optimization, limited computational budget")
        
        print("\n3. Show current Pareto Front results")
        print("4. Cancel")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-4): ").strip()
                if choice in ['1', '2', '3', '4']:
                    return choice
                else:
                    print("Please enter 1, 2, 3, or 4")
            except KeyboardInterrupt:
                return '4'
    
    def get_optimization_parameters(self, algorithm: str) -> Tuple[OptimizationConfig, Dict[str, Any]]:
        """Get optimization parameters based on algorithm choice."""
        print(f"\nConfiguring {algorithm} optimization:")
        
        # Get basic parameters
        try:
            max_trials = int(input("Maximum trials (default 50): ") or "50")
            max_hours = float(input("Maximum hours (default 2.0): ") or "2.0")
        except ValueError:
            print("Using default values")
            max_trials = 50
            max_hours = 2.0
        
        # Algorithm-specific parameters
        kwargs = {}
        if algorithm.lower() in ["nsga2", "nsga-ii"]:
            try:
                population_size = int(input("Population size (default 20): ") or "20")
                kwargs['population_size'] = population_size
            except ValueError:
                kwargs['population_size'] = 20
        
        config = OptimizationConfig(
            max_trials=max_trials,
            max_duration_hours=max_hours,
            trial_timeout_minutes=30,
            parallel_jobs=1,
            validation_strategy="holdout",
            random_seed=42
        )
        
        return config, kwargs
    
    def run_pareto_optimization(
        self, 
        algorithm: str, 
        search_space: Dict[str, Any],
        config: OptimizationConfig,
        **kwargs
    ) -> ParetoFrontResult:
        """Run Pareto Front optimization with specified algorithm."""
        
        # Create multi-objective function
        objective_function = MultiObjectiveFunction(
            self.training_interface,
            self.base_config,
            DEFAULT_MULTI_OBJECTIVE_DEFINITIONS
        )
        
        # Determine output directory
        algo_name = "nsga2" if algorithm.lower() in ["nsga2", "nsga-ii"] else "tpe"
        output_dir = self.output_dir / algo_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nStarting {algorithm} optimization...")
        print(f"Objectives: {list(DEFAULT_MULTI_OBJECTIVE_DEFINITIONS.keys())}")
        print(f"Max trials: {config.max_trials}")
        print(f"Max duration: {config.max_duration_hours} hours")
        print(f"Output directory: {output_dir}")
        
        # Create and run optimizer
        optimizer = create_pareto_optimizer(
            algorithm=algorithm,
            objective_function=objective_function,
            search_space=search_space,
            config=config,
            output_dir=output_dir,
            **kwargs
        )
        
        result = optimizer.optimize()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = output_dir / f"pareto_results_{timestamp}.json"
        result.save_to_file(result_file)
        
        # Save best parameters for easy access
        pareto_front = result.get_pareto_front()
        if pareto_front:
            best_params_dir = Path("models/best_parameters")
            best_params_dir.mkdir(parents=True, exist_ok=True)
            
            # Save all Pareto Front solutions
            pareto_params_file = best_params_dir / f"pareto_front_{algo_name}_{timestamp}.json"
            pareto_data = {
                'algorithm': algorithm,
                'timestamp': timestamp,
                'pareto_front': [
                    {
                        'parameters': sol.parameters,
                        'objectives': sol.objectives,
                        'trial_number': sol.trial_number
                    }
                    for sol in pareto_front
                ]
            }
            
            with open(pareto_params_file, 'w') as f:
                json.dump(pareto_data, f, indent=2)
            
            logger.info(f"Pareto Front solutions saved to {pareto_params_file}")
        
        return result
    
    def display_pareto_results(self, result: ParetoFrontResult) -> None:
        """Display Pareto Front optimization results."""
        pareto_front = result.get_pareto_front()
        
        print("\n" + "="*80)
        print("PARETO FRONT OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Algorithm: {result.algorithm}")
        print(f"Total evaluations: {result.total_evaluations}")
        print(f"Computation time: {result.computation_time:.2f} seconds")
        print(f"Pareto Front size: {len(pareto_front)} solutions")
        
        if pareto_front:
            print(f"\nPareto Front Solutions:")
            print("-" * 80)
            
            # Get objective names
            obj_names = list(pareto_front[0].objectives.keys())
            
            # Header
            header = f"{'#':<3} "
            for obj_name in obj_names:
                header += f"{obj_name:<12} "
            header += "Key Parameters"
            print(header)
            print("-" * 80)
            
            # Solutions
            for i, solution in enumerate(pareto_front[:10]):  # Show top 10
                row = f"{i+1:<3} "
                for obj_name in obj_names:
                    value = solution.objectives[obj_name]
                    row += f"{value:<12.4f} "
                
                # Show key parameters
                key_params = []
                for param in ['learning_rate', 'batch_size', 'hidden_size', 'dropout']:
                    if param in solution.parameters:
                        value = solution.parameters[param]
                        if isinstance(value, float):
                            key_params.append(f"{param}={value:.4f}")
                        else:
                            key_params.append(f"{param}={value}")
                
                row += ", ".join(key_params[:3])  # Limit to 3 params
                print(row)
            
            if len(pareto_front) > 10:
                print(f"... and {len(pareto_front) - 10} more solutions")
        
        print("\n" + "="*80)
    
    def select_pareto_solution(self, pareto_front: List) -> Optional[Dict[str, Any]]:
        """Allow user to select a solution from the Pareto Front."""
        if not pareto_front:
            print("No Pareto Front solutions available")
            return None
        
        print(f"\nSelect a solution from the Pareto Front (1-{len(pareto_front)}):")
        print("0. Cancel selection")
        
        while True:
            try:
                choice = input(f"\nEnter choice (0-{len(pareto_front)}): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    return None
                elif 1 <= choice_num <= len(pareto_front):
                    selected_solution = pareto_front[choice_num - 1]
                    
                    print(f"\nSelected solution #{choice_num}:")
                    print("Objectives:")
                    for obj_name, obj_value in selected_solution.objectives.items():
                        print(f"  {obj_name}: {obj_value:.4f}")
                    
                    print("Parameters:")
                    for param_name, param_value in selected_solution.parameters.items():
                        if isinstance(param_value, float):
                            print(f"  {param_name}: {param_value:.6f}")
                        else:
                            print(f"  {param_name}: {param_value}")
                    
                    confirm = input("\nUse this solution? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        return selected_solution.parameters
                    
                else:
                    print(f"Please enter a number between 0 and {len(pareto_front)}")
                    
            except (ValueError, KeyboardInterrupt):
                print("Invalid input or cancelled")
                return None
    
    def list_saved_pareto_results(self) -> List[Path]:
        """List all saved Pareto Front result files."""
        results = []
        
        for algo_dir in ["nsga2", "tpe"]:
            algo_path = self.output_dir / algo_dir
            if algo_path.exists():
                for result_file in algo_path.glob("pareto_results_*.json"):
                    results.append(result_file)
        
        # Sort by modification time (newest first)
        results.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return results
    
    def load_pareto_result(self, result_file: Path) -> ParetoFrontResult:
        """Load a saved Pareto Front result."""
        return ParetoFrontResult.load_from_file(result_file)
    
    def show_saved_results_menu(self) -> Optional[ParetoFrontResult]:
        """Show menu for selecting saved Pareto Front results."""
        saved_results = self.list_saved_pareto_results()
        
        if not saved_results:
            print("No saved Pareto Front results found")
            return None
        
        print("\n" + "="*80)
        print("SAVED PARETO FRONT RESULTS")
        print("="*80)
        
        for i, result_file in enumerate(saved_results[:10]):  # Show latest 10
            # Extract info from filename
            filename = result_file.name
            algorithm = result_file.parent.name
            timestamp = filename.replace("pareto_results_", "").replace(".json", "")
            mod_time = datetime.fromtimestamp(result_file.stat().st_mtime)
            
            print(f"{i+1}. {algorithm.upper()} - {timestamp} ({mod_time.strftime('%Y-%m-%d %H:%M')})")
        
        print("0. Cancel")
        
        while True:
            try:
                choice = input(f"\nSelect result (0-{len(saved_results[:10])}): ").strip()
                choice_num = int(choice)
                
                if choice_num == 0:
                    return None
                elif 1 <= choice_num <= len(saved_results[:10]):
                    selected_file = saved_results[choice_num - 1]
                    print(f"Loading {selected_file.name}...")
                    return self.load_pareto_result(selected_file)
                else:
                    print(f"Please enter a number between 0 and {len(saved_results[:10])}")
                    
            except (ValueError, KeyboardInterrupt):
                print("Invalid input or cancelled")
                return None


def create_pareto_interface(training_interface, base_config: Dict[str, Any]) -> ParetoFrontInterface:
    """Factory function to create Pareto Front interface."""
    return ParetoFrontInterface(training_interface, base_config)