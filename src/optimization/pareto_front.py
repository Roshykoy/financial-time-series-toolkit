"""
Pareto Front Multi-Objective Hyperparameter Optimization.
Implements both NSGA-II (Evolutionary Algorithm) and TPE-based (MOBO) approaches.
"""

import numpy as np
import random
import time
import json
import signal
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

from .data_structures import OptimizationConfig, OptimizationTrial
from .checkpoint_manager import CheckpointManager, CheckpointMetadata

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

# Optional imports for advanced algorithms
try:
    import optuna
    from optuna.samplers import NSGAIISampler, TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("Matplotlib not available for Pareto Front visualization")


@dataclass
class ParetoSolution:
    """Represents a single solution on the Pareto Front."""
    parameters: Dict[str, Any]
    objectives: Dict[str, float]  # e.g., {'accuracy': 0.85, 'training_time': 120, 'model_complexity': 0.3}
    trial_number: int
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    def dominates(self, other: 'ParetoSolution') -> bool:
        """Check if this solution dominates another (all objectives better or equal, at least one strictly better)."""
        better_in_any = False
        for obj_name, obj_value in self.objectives.items():
            other_value = other.objectives.get(obj_name, float('-inf'))
            
            # Assume higher is better for all objectives (we'll negate minimization objectives)
            if obj_value < other_value:
                return False
            elif obj_value > other_value:
                better_in_any = True
        
        return better_in_any


@dataclass
class ParetoFrontResult:
    """Result containing the entire Pareto Front."""
    solutions: List[ParetoSolution]
    algorithm: str
    total_evaluations: int
    computation_time: float
    hypervolume: Optional[float] = None
    convergence_metrics: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def get_pareto_front(self) -> List[ParetoSolution]:
        """Return only non-dominated solutions."""
        return compute_pareto_front(self.solutions)

    def save_to_file(self, filepath: Path) -> None:
        """Save Pareto Front results to JSON file."""
        data = {
            'solutions': [asdict(sol) for sol in self.solutions],
            'algorithm': self.algorithm,
            'total_evaluations': self.total_evaluations,
            'computation_time': self.computation_time,
            'hypervolume': self.hypervolume,
            'convergence_metrics': self.convergence_metrics,
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat(),
            'pareto_front_size': len(self.get_pareto_front())
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Pareto Front results saved to {filepath}")

    @classmethod
    def load_from_file(cls, filepath: Path) -> 'ParetoFrontResult':
        """Load Pareto Front results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        solutions = [ParetoSolution(**sol_data) for sol_data in data['solutions']]
        
        return cls(
            solutions=solutions,
            algorithm=data['algorithm'],
            total_evaluations=data['total_evaluations'],
            computation_time=data['computation_time'],
            hypervolume=data.get('hypervolume'),
            convergence_metrics=data.get('convergence_metrics'),
            metadata=data.get('metadata')
        )


class MultiObjectiveFunction:
    """Multi-objective function wrapper that returns multiple objectives."""
    
    def __init__(
        self,
        training_interface,
        base_config: Dict[str, Any],
        objective_definitions: Dict[str, Dict[str, Any]]
    ):
        """
        Initialize multi-objective function.
        
        Args:
            training_interface: Interface for model training
            base_config: Base configuration
            objective_definitions: Dict defining objectives, e.g.:
                {
                    'accuracy': {'direction': 'maximize', 'weight': 1.0},
                    'training_time': {'direction': 'minimize', 'weight': 1.0},
                    'model_complexity': {'direction': 'minimize', 'weight': 0.5}
                }
        """
        self.training_interface = training_interface
        self.base_config = base_config
        self.objective_definitions = objective_definitions
        
        # Performance tracking
        self.call_count = 0
        self.total_time = 0.0
        self.evaluation_history = []
        
        logger.info(f"Multi-objective function initialized with objectives: {list(objective_definitions.keys())}")
    
    def __call__(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate all objectives for given parameters."""
        self.call_count += 1
        start_time = time.time()
        
        try:
            # Merge parameters with base config
            config = self.base_config.copy()
            config.update(parameters)
            
            # Validate configuration
            is_valid, errors = self.training_interface.validate_config(config)
            if not is_valid:
                logger.error(f"Invalid configuration: {errors}")
                return {obj: float('-inf') for obj in self.objective_definitions.keys()}
            
            # Train model and collect metrics
            training_start = time.time()
            accuracy_score = self.training_interface.train_model(config)
            training_time = time.time() - training_start
            
            # Calculate model complexity (simplified)
            model_complexity = self._calculate_model_complexity(config)
            
            # Build objectives dictionary
            objectives = {}
            
            for obj_name, obj_config in self.objective_definitions.items():
                if obj_name == 'accuracy':
                    value = accuracy_score
                elif obj_name == 'training_time':
                    # Convert to minutes and negate for maximization (lower is better)
                    value = -(training_time / 60.0)
                elif obj_name == 'model_complexity':
                    # Negate for maximization (lower complexity is better)
                    value = -model_complexity
                else:
                    logger.warning(f"Unknown objective: {obj_name}")
                    value = 0.0
                
                # Apply direction (negate if minimizing)
                if obj_config.get('direction', 'maximize') == 'minimize':
                    value = -value
                
                objectives[obj_name] = value
            
            # Update tracking
            duration = time.time() - start_time
            self.total_time += duration
            
            evaluation_record = {
                'call': self.call_count,
                'parameters': parameters.copy(),
                'objectives': objectives.copy(),
                'duration': duration
            }
            self.evaluation_history.append(evaluation_record)
            
            logger.debug(f"Multi-objective evaluation {self.call_count}: {objectives}, time={duration:.1f}s")
            
            return objectives
            
        except Exception as e:
            logger.error(f"Multi-objective evaluation failed: {e}")
            return {obj: float('-inf') for obj in self.objective_definitions.keys()}
    
    def _calculate_model_complexity(self, config: Dict[str, Any]) -> float:
        """Calculate model complexity score (0-1, higher = more complex)."""
        complexity = 0.0
        
        # Factor in hidden size
        hidden_size = config.get('hidden_size', 256)
        complexity += (hidden_size / 1024) * 0.3
        
        # Factor in number of layers
        num_layers = config.get('num_layers', 4) 
        complexity += (num_layers / 8) * 0.3
        
        # Factor in dropout (lower dropout = higher complexity)
        dropout = config.get('dropout', 0.2)
        complexity += (1 - dropout) * 0.2
        
        # Factor in batch size (smaller batch = more complex training)
        batch_size = config.get('batch_size', 32)
        complexity += (64 / max(batch_size, 1)) * 0.2
        
        return min(complexity, 1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            'call_count': self.call_count,
            'total_time_seconds': self.total_time,
            'average_time_seconds': self.total_time / max(1, self.call_count),
            'evaluation_history': self.evaluation_history
        }


class BaseParetoOptimizer(ABC):
    """Base class for Pareto Front optimization algorithms."""
    
    def __init__(
        self,
        objective_function: MultiObjectiveFunction,
        search_space: Dict[str, Any],
        config: OptimizationConfig,
        output_dir: Path
    ):
        self.objective_function = objective_function
        self.search_space = search_space
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.solutions = []
        self.start_time = None
        
    @abstractmethod
    def optimize(self) -> ParetoFrontResult:
        """Run Pareto Front optimization and return results."""
        pass
    
    def _sample_parameter(self, param_name: str, param_config: Union[Dict, List, Tuple]) -> Any:
        """Sample a parameter value from its configuration."""
        if isinstance(param_config, (list, tuple)):
            return random.choice(param_config)
        
        param_type = param_config.get('type', 'choice')
        
        if param_type == 'choice':
            return random.choice(param_config['choices'])
        elif param_type == 'uniform':
            return random.uniform(param_config['low'], param_config['high'])
        elif param_type == 'loguniform':
            return np.exp(random.uniform(np.log(param_config['low']), np.log(param_config['high'])))
        elif param_type == 'int':
            return random.randint(param_config['low'], param_config['high'])
        elif param_type == 'bool':
            return random.choice([True, False])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
    
    def _generate_random_parameters(self) -> Dict[str, Any]:
        """Generate random parameters from search space."""
        parameters = {}
        for param_name, param_config in self.search_space.items():
            parameters[param_name] = self._sample_parameter(param_name, param_config)
        return parameters


class NSGAIIParetoOptimizer(BaseParetoOptimizer):
    """NSGA-II based Pareto Front optimization with checkpoint support."""
    
    def __init__(
        self,
        objective_function: MultiObjectiveFunction,
        search_space: Dict[str, Any],
        config: OptimizationConfig,
        output_dir: Path,
        population_size: int = 50,
        crossover_probability: float = 0.9,
        mutation_probability: float = 0.1
    ):
        super().__init__(objective_function, search_space, config, output_dir)
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        
        # Initialize checkpoint manager
        checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            save_frequency=3,  # Save every 3 evaluations
            max_checkpoints=20,
            compression=True
        )
        
        # Interrupt handling
        self.interrupted = False
        self.trials_completed = 0
        
        logger.info(f"NSGA-II optimizer initialized with population_size={population_size} and checkpoint system")
    
    def can_resume(self) -> bool:
        """Check if optimization can be resumed from checkpoint."""
        try:
            return len(self.checkpoint_manager.list_available_checkpoints()) > 0
        except:
            return False
    
    def resume_from_checkpoint(self, checkpoint_id: str = None) -> Tuple[bool, Optional[List[ParetoSolution]], int]:
        """Resume NSGA-II optimization from checkpoint. Returns (success, population, generation)."""
        try:
            if checkpoint_id:
                checkpoint = self.checkpoint_manager.load_specific_checkpoint(checkpoint_id)
            else:
                checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            
            if not checkpoint:
                print("❌ No checkpoint found to resume from")
                return False, None, 0
            
            print(f"🔄 Resuming NSGA-II optimization from checkpoint with {len(checkpoint.completed_trials)} completed trials")
            
            # Restore solutions from checkpoint
            self.solutions = []
            for trial_data in checkpoint.completed_trials:
                solution = ParetoSolution(
                    parameters=trial_data.parameters,
                    objectives=trial_data.metadata.get('objectives', {}),
                    trial_number=int(trial_data.trial_id.replace('trial_', '').lstrip('0') or '0'),
                    timestamp=trial_data.start_time,
                    metadata=trial_data.metadata
                )
                self.solutions.append(solution)
            
            # Restore state
            self.trials_completed = len(self.solutions)
            self.interrupted = False
            
            # Restore population if available in algorithm state
            population = []
            algorithm_state = checkpoint.algorithm_state
            generation = algorithm_state.get('generation', 0)
            
            if 'current_population' in algorithm_state:
                for pop_data in algorithm_state['current_population']:
                    solution = ParetoSolution(
                        parameters=pop_data['parameters'],
                        objectives=pop_data['objectives'],
                        trial_number=pop_data['trial_number'],
                        timestamp=datetime.now().isoformat()
                    )
                    population.append(solution)
            else:
                # If no population saved, use recent solutions as population
                population = self.solutions[-self.population_size:] if len(self.solutions) >= self.population_size else self.solutions[:]
            
            print(f"✅ Resumed with {self.trials_completed} completed trials, generation {generation}")
            return True, population, generation
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            print(f"❌ Resume failed: {e}")
            return False, None, 0
    
    def optimize(self) -> ParetoFrontResult:
        """Run NSGA-II optimization with checkpoint support."""
        self.start_time = time.time()
        
        # Set up keyboard interrupt handler
        def signal_handler(signum, frame):
            print(f"\n⚠️  Keyboard interrupt received. Finishing current generation and saving checkpoint...")
            self.interrupted = True
        
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Check for resumable checkpoints
            population = []
            generation = 0
            
            if self.can_resume():
                checkpoints = self.checkpoint_manager.list_available_checkpoints()
                if checkpoints:
                    latest = max(checkpoints, key=lambda x: x['created_at'])
                    print(f"\n🔄 Found checkpoint with {latest['trials_completed']} completed trials")
                    print(f"   Created: {latest['created_at']}")
                    
                    resume_choice = input("Resume from checkpoint? (y/n): ").strip().lower()
                    if resume_choice in ['y', 'yes']:
                        success, resumed_population, resumed_generation = self.resume_from_checkpoint()
                        if success:
                            population = resumed_population or []
                            generation = resumed_generation
                        else:
                            print("Resume failed, starting fresh optimization")
            
            # Initialize population if not resumed
            if not population:
                print(f"\n🧬 Creating initial population of {self.population_size} individuals...")
            
            for i in range(self.population_size):
                if self.interrupted:
                    break
                    
                parameters = self._generate_random_parameters()
                objectives = self.objective_function(parameters)
                
                solution = ParetoSolution(
                    parameters=parameters,
                    objectives=objectives,
                    trial_number=i,
                    timestamp=datetime.now().isoformat()
                )
                population.append(solution)
                self.solutions.append(solution)
                self.trials_completed += 1
                
                # Checkpoint every 3 evaluations
                if self.trials_completed % 3 == 0:
                    self._create_checkpoint(generation=0, population=population)
                
                print(f"   Individual {i+1}/{self.population_size} evaluated")
            
            if not self.interrupted:
                logger.info(f"Initial population of {len(population)} created")
            
            # Evolution loop (generation already set from resume or 0)
            max_generations = max(10, self.config.max_trials // self.population_size)
            
            while generation < max_generations and len(self.solutions) < self.config.max_trials and not self.interrupted:
                generation += 1
                print(f"\n🔄 Generation {generation}/{max_generations}")
                
                # Selection, crossover, mutation (simplified NSGA-II)
                new_population = []
                
                for child_idx in range(self.population_size):
                    if len(self.solutions) >= self.config.max_trials or self.interrupted:
                        break
                    
                    # Tournament selection
                    parent1 = self._tournament_selection(population)
                    parent2 = self._tournament_selection(population)
                    
                    # Crossover
                    if random.random() < self.crossover_probability:
                        child_params = self._crossover(parent1.parameters, parent2.parameters)
                    else:
                        child_params = parent1.parameters.copy()
                    
                    # Mutation
                    if random.random() < self.mutation_probability:
                        child_params = self._mutate(child_params)
                    
                    # Evaluate child
                    objectives = self.objective_function(child_params)
                    
                    child = ParetoSolution(
                        parameters=child_params,
                        objectives=objectives,
                        trial_number=len(self.solutions),
                        timestamp=datetime.now().isoformat()
                    )
                    
                    new_population.append(child)
                    self.solutions.append(child)
                    self.trials_completed += 1
                    
                    # Checkpoint every 3 evaluations
                    if self.trials_completed % 3 == 0:
                        self._create_checkpoint(generation=generation, population=population + new_population)
                    
                    print(f"   Child {child_idx+1}/{self.population_size} evaluated - Trial {self.trials_completed}")
                
                if not self.interrupted:
                    # Environmental selection (simplified)
                    combined_population = population + new_population
                    population = self._environmental_selection(combined_population, self.population_size)
                    
                    if generation % 5 == 0 or generation == max_generations:
                        pareto_front = compute_pareto_front(population)
                        print(f"   📊 Generation {generation}: {len(pareto_front)} solutions on Pareto Front")
                        logger.info(f"Generation {generation}: {len(pareto_front)} solutions on Pareto Front")
            
            # Create final checkpoint if not recently saved
            if not self.interrupted or self.trials_completed % 3 != 0:
                self._create_checkpoint(generation=generation, population=population)
            
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
        
        # Create final result
        computation_time = time.time() - self.start_time
        status = "interrupted" if self.interrupted else "completed"
        
        result = ParetoFrontResult(
            solutions=self.solutions,
            algorithm="NSGA-II",
            total_evaluations=len(self.solutions),
            computation_time=computation_time,
            metadata={
                'population_size': self.population_size,
                'generations': generation,
                'crossover_probability': self.crossover_probability,
                'mutation_probability': self.mutation_probability,
                'status': status,
                'trials_completed': self.trials_completed,
                'interrupted': self.interrupted
            }
        )
        
        # Save final results
        self._save_final_results(result, status)
        
        logger.info(f"NSGA-II {status}: {len(result.get_pareto_front())} solutions on final Pareto Front")
        if self.interrupted:
            print(f"\n✅ Optimization interrupted after {self.trials_completed} evaluations")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"🔄 Checkpoints available for resuming")
        
        return result
    
    def _tournament_selection(self, population: List[ParetoSolution], tournament_size: int = 3) -> ParetoSolution:
        """Tournament selection for parent selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Select best from tournament (simplified ranking)
        pareto_front = compute_pareto_front(tournament)
        if pareto_front:
            return random.choice(pareto_front)
        else:
            return random.choice(tournament)
    
    def _crossover(self, parent1_params: Dict[str, Any], parent2_params: Dict[str, Any]) -> Dict[str, Any]:
        """Simple crossover between two parameter sets."""
        child_params = {}
        
        for param_name in parent1_params.keys():
            if random.random() < 0.5:
                child_params[param_name] = parent1_params[param_name]
            else:
                child_params[param_name] = parent2_params[param_name]
        
        return child_params
    
    def _mutate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate parameters."""
        mutated_params = parameters.copy()
        
        # Mutate with small probability
        for param_name, param_config in self.search_space.items():
            if random.random() < 0.1:  # 10% chance to mutate each parameter
                mutated_params[param_name] = self._sample_parameter(param_name, param_config)
        
        return mutated_params
    
    def _environmental_selection(self, population: List[ParetoSolution], target_size: int) -> List[ParetoSolution]:
        """Environmental selection to maintain population size."""
        if len(population) <= target_size:
            return population
        
        # Simplified selection: prefer Pareto Front solutions
        pareto_front = compute_pareto_front(population)
        
        if len(pareto_front) >= target_size:
            return random.sample(pareto_front, target_size)
        else:
            # Fill remaining with random from non-dominated
            remaining = target_size - len(pareto_front)
            others = [sol for sol in population if sol not in pareto_front]
            return pareto_front + random.sample(others, min(remaining, len(others)))
    
    def _create_checkpoint(self, generation: int, population: List[ParetoSolution]):
        """Create a checkpoint of current NSGA-II optimization state."""
        try:
            # Convert solutions to trial format for checkpoint manager
            completed_trials = []
            for i, solution in enumerate(self.solutions):
                trial = OptimizationTrial(
                    trial_id=f"trial_{i:04d}",
                    parameters=solution.parameters,
                    score=sum(solution.objectives.values()) / len(solution.objectives),  # Average score
                    metadata={**(solution.metadata or {}), 'objectives': solution.objectives},
                    start_time=solution.timestamp,
                    end_time=solution.timestamp,
                    status="completed"
                )
                completed_trials.append(trial)
            
            # Create checkpoint metadata
            metadata = CheckpointMetadata(
                optimization_start_time=datetime.fromtimestamp(self.start_time).isoformat(),
                algorithm="NSGA-II",
                total_trials_planned=self.config.max_trials,
                trials_completed=self.trials_completed,
                trials_failed=0,
                best_score=max(sum(s.objectives.values()) / len(s.objectives) for s in self.solutions) if self.solutions else None,
                best_trial_id=f"trial_{len(self.solutions)-1:04d}" if self.solutions else None
            )
            
            # Get best trial for checkpoint
            best_trial = None
            if self.solutions:
                best_idx = max(range(len(self.solutions)), key=lambda i: sum(self.solutions[i].objectives.values()) / len(self.solutions[i].objectives))
                best_solution = self.solutions[best_idx]
                best_trial = OptimizationTrial(
                    trial_id=f"trial_{best_idx:04d}",
                    parameters=best_solution.parameters,
                    score=sum(best_solution.objectives.values()) / len(best_solution.objectives),
                    metadata=best_solution.metadata or {},
                    start_time=best_solution.timestamp,
                    end_time=best_solution.timestamp,
                    status="completed"
                )
            
            # Create checkpoint
            checkpoint_id = self.checkpoint_manager.create_checkpoint(
                metadata=metadata,
                optimization_config=self.config,
                search_space=self.search_space,
                completed_trials=completed_trials,
                best_trial=best_trial,
                algorithm_state={
                    'algorithm': 'NSGA-II',
                    'generation': generation,
                    'population_size': self.population_size,
                    'solutions_count': len(self.solutions),
                    'interrupted': self.interrupted,
                    'current_population': [
                        {
                            'parameters': sol.parameters,
                            'objectives': sol.objectives,
                            'trial_number': sol.trial_number
                        }
                        for sol in population
                    ]
                },
                hardware_profile={'checkpoint_time': datetime.now().isoformat()}
            )
            
            print(f"💾 Checkpoint saved: {checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            print(f"⚠️  Checkpoint creation failed: {e}")
    
    def _save_final_results(self, result: ParetoFrontResult, status: str):
        """Save final NSGA-II optimization results in compatible format."""
        try:
            # Save Pareto Front result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = self.output_dir / f"nsga2_result_{status}_{timestamp}.json"
            
            # Convert result to JSON-serializable format
            result_data = {
                'algorithm': result.algorithm,
                'status': status,
                'total_evaluations': result.total_evaluations,
                'computation_time': result.computation_time,
                'trials_completed': self.trials_completed,
                'interrupted': self.interrupted,
                'metadata': result.metadata,
                'pareto_front': [
                    {
                        'parameters': sol.parameters,
                        'objectives': sol.objectives,
                        'trial_number': sol.trial_number,
                        'timestamp': sol.timestamp,
                        'metadata': sol.metadata
                    }
                    for sol in result.get_pareto_front()
                ],
                'all_solutions': [
                    {
                        'parameters': sol.parameters,
                        'objectives': sol.objectives,
                        'trial_number': sol.trial_number,
                        'timestamp': sol.timestamp,
                        'metadata': sol.metadata
                    }
                    for sol in result.solutions
                ]
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            
            # Also save in training-compatible format if we have solutions
            if result.solutions:
                pareto_front = result.get_pareto_front()
                if pareto_front:
                    # Save best solution for training integration
                    best_solution = max(pareto_front, key=lambda s: sum(s.objectives.values()) / len(s.objectives))
                    
                    training_params = {
                        'optimization_method': 'pareto_front_nsga2',
                        'timestamp': timestamp,
                        'status': status,
                        'total_trials': self.trials_completed,
                        'best_parameters': best_solution.parameters,
                        'best_score': sum(best_solution.objectives.values()) / len(best_solution.objectives),
                        'objectives': best_solution.objectives,
                        'pareto_front_size': len(pareto_front)
                    }
                    
                    # Save to best_parameters directory for training integration
                    best_params_dir = Path("models/best_parameters")
                    best_params_dir.mkdir(parents=True, exist_ok=True)
                    best_params_file = best_params_dir / f"pareto_selected_{status}_{timestamp}.json"
                    
                    with open(best_params_file, 'w') as f:
                        json.dump(training_params, f, indent=2, default=str)
                    
                    print(f"📊 Results saved: {result_file.name}")
                    print(f"🎯 Training parameters saved: {best_params_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
            print(f"⚠️  Failed to save results: {e}")


class TPEParetoOptimizer(BaseParetoOptimizer):
    """TPE-based (Optuna) Pareto Front optimization using NSGA-II sampler."""
    
    def __init__(
        self,
        objective_function: MultiObjectiveFunction,
        search_space: Dict[str, Any],
        config: OptimizationConfig,
        output_dir: Path
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required for TPE Pareto optimization. Install with: pip install optuna")
        
        super().__init__(objective_function, search_space, config, output_dir)
        
        # Create Optuna study for multi-objective optimization
        self.study = optuna.create_study(
            directions=["maximize"] * len(objective_function.objective_definitions),
            sampler=NSGAIISampler(population_size=50),
            study_name=f"marksix_pareto_{int(time.time())}"
        )
        
        # Initialize checkpoint manager
        checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            save_frequency=3,  # Save every 3 trials
            max_checkpoints=20,
            compression=True
        )
        
        # Interrupt handling
        self.interrupted = False
        self.trials_completed = 0
        
        logger.info("TPE-based Pareto optimizer initialized with Optuna NSGA-II sampler and checkpoint system")
    
    def can_resume(self) -> bool:
        """Check if optimization can be resumed from checkpoint."""
        try:
            return len(self.checkpoint_manager.list_available_checkpoints()) > 0
        except:
            return False
    
    def resume_from_checkpoint(self, checkpoint_id: str = None) -> bool:
        """Resume optimization from checkpoint."""
        try:
            if checkpoint_id:
                checkpoint = self.checkpoint_manager.load_specific_checkpoint(checkpoint_id)
            else:
                checkpoint = self.checkpoint_manager.load_latest_checkpoint()
            
            if not checkpoint:
                print("❌ No checkpoint found to resume from")
                return False
            
            print(f"🔄 Resuming TPE optimization from checkpoint with {len(checkpoint.completed_trials)} completed trials")
            
            # Restore solutions from checkpoint
            self.solutions = []
            for trial_data in checkpoint.completed_trials:
                solution = ParetoSolution(
                    parameters=trial_data.parameters,
                    objectives=trial_data.metadata.get('objectives', {}),
                    trial_number=trial_data.trial_id.replace('trial_', '').lstrip('0') or '0',
                    timestamp=trial_data.start_time,
                    metadata=trial_data.metadata
                )
                self.solutions.append(solution)
            
            # Restore state
            self.trials_completed = len(self.solutions)
            self.interrupted = False
            
            # Update Optuna study with previous trials
            for trial_data in checkpoint.completed_trials:
                # Optuna will handle the trial history internally
                pass
            
            print(f"✅ Resumed with {self.trials_completed} completed trials")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            print(f"❌ Resume failed: {e}")
            return False
    
    def optimize(self) -> ParetoFrontResult:
        """Run TPE-based Pareto optimization with checkpoint support."""
        self.start_time = time.time()
        
        # Set up keyboard interrupt handler
        def signal_handler(signum, frame):
            print(f"\n⚠️  Keyboard interrupt received. Finishing current trial and saving checkpoint...")
            self.interrupted = True
        
        original_handler = signal.signal(signal.SIGINT, signal_handler)
        
        try:
            # Check for resumable checkpoints
            if self.can_resume():
                checkpoints = self.checkpoint_manager.list_available_checkpoints()
                if checkpoints:
                    latest = max(checkpoints, key=lambda x: x['created_at'])
                    print(f"\n🔄 Found checkpoint with {latest['trials_completed']} completed trials")
                    print(f"   Created: {latest['created_at']}")
                    
                    resume_choice = input("Resume from checkpoint? (y/n): ").strip().lower()
                    if resume_choice in ['y', 'yes']:
                        if self.resume_from_checkpoint():
                            print(f"🚀 Continuing TPE optimization from trial {self.trials_completed + 1}")
                        else:
                            print("Resume failed, starting fresh optimization")
            
            def optuna_objective(trial):
                # Check for interruption before starting trial
                if self.interrupted:
                    raise optuna.TrialPruned()
                
                print(f"\n🔍 Starting trial {self.trials_completed + 1}/{self.config.max_trials}")
                trial_start_time = time.time()
                
                # Sample parameters using Optuna
                parameters = {}
                for param_name, param_config in self.search_space.items():
                    parameters[param_name] = self._sample_optuna_parameter(trial, param_name, param_config)
                
                # Evaluate objectives
                objectives = self.objective_function(parameters)
                
                # Store solution
                solution = ParetoSolution(
                    parameters=parameters,
                    objectives=objectives,
                    trial_number=self.trials_completed,
                    timestamp=datetime.now().isoformat(),
                    metadata={'trial_time': time.time() - trial_start_time}
                )
                self.solutions.append(solution)
                self.trials_completed += 1
                
                # Progress display
                elapsed = time.time() - self.start_time
                avg_trial_time = elapsed / self.trials_completed if self.trials_completed > 0 else 0
                remaining_trials = self.config.max_trials - self.trials_completed
                eta = avg_trial_time * remaining_trials
                
                print(f"✅ Trial {self.trials_completed} completed in {time.time() - trial_start_time:.1f}s")
                print(f"   Objectives: {', '.join(f'{k}={v:.4f}' for k, v in objectives.items())}")
                print(f"   Progress: {self.trials_completed}/{self.config.max_trials} ({100*self.trials_completed/self.config.max_trials:.1f}%)")
                print(f"   ETA: {eta/60:.1f} minutes")
                
                # Checkpoint every 3 trials
                if self.trials_completed % 3 == 0:
                    self._create_checkpoint()
                
                # Check for interruption after trial completion
                if self.interrupted:
                    print("🔄 Creating final checkpoint before exit...")
                    self._create_checkpoint()
                    raise optuna.TrialPruned()
                
                # Return objectives as tuple for Optuna
                return tuple(objectives.values())
            
            # Run optimization with interrupt handling
            try:
                self.study.optimize(optuna_objective, n_trials=self.config.max_trials)
            except KeyboardInterrupt:
                # This should not happen due to signal handler, but just in case
                self.interrupted = True
                print("\n⚠️  Optimization interrupted by user")
            
            # Create final checkpoint if not interrupted or if interrupted without recent checkpoint
            if not self.interrupted or self.trials_completed % 3 != 0:
                self._create_checkpoint()
            
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGINT, original_handler)
        
        # Create final result
        computation_time = time.time() - self.start_time
        status = "interrupted" if self.interrupted else "completed"
        
        result = ParetoFrontResult(
            solutions=self.solutions,
            algorithm="TPE-NSGA-II",
            total_evaluations=len(self.solutions),
            computation_time=computation_time,
            metadata={
                'optuna_study_name': self.study.study_name,
                'n_trials': len(self.study.trials),
                'best_trials': len(self.study.best_trials) if hasattr(self.study, 'best_trials') else 0,
                'status': status,
                'trials_completed': self.trials_completed,
                'interrupted': self.interrupted
            }
        )
        
        # Save final results
        self._save_final_results(result, status)
        
        logger.info(f"TPE optimization {status}: {len(result.get_pareto_front())} solutions on Pareto Front")
        if self.interrupted:
            print(f"\n✅ Optimization interrupted after {self.trials_completed} trials")
            print(f"📁 Results saved to: {self.output_dir}")
            print(f"🔄 Checkpoints available for resuming")
        
        return result
    
    def _create_checkpoint(self):
        """Create a checkpoint of current optimization state."""
        try:
            # Convert solutions to trial format for checkpoint manager
            completed_trials = []
            for i, solution in enumerate(self.solutions):
                trial = OptimizationTrial(
                    trial_id=f"trial_{i:04d}",
                    parameters=solution.parameters,
                    score=sum(solution.objectives.values()) / len(solution.objectives),  # Average score
                    metadata={**(solution.metadata or {}), 'objectives': solution.objectives},
                    start_time=solution.timestamp,
                    end_time=solution.timestamp,
                    status="completed"
                )
                completed_trials.append(trial)
            
            # Create checkpoint metadata
            metadata = CheckpointMetadata(
                optimization_start_time=datetime.fromtimestamp(self.start_time).isoformat(),
                algorithm="TPE-NSGA-II",
                total_trials_planned=self.config.max_trials,
                trials_completed=self.trials_completed,
                trials_failed=0,
                best_score=max(sum(s.objectives.values()) / len(s.objectives) for s in self.solutions) if self.solutions else None,
                best_trial_id=f"trial_{len(self.solutions)-1:04d}" if self.solutions else None
            )
            
            # Get best trial for checkpoint
            best_trial = None
            if self.solutions:
                best_idx = max(range(len(self.solutions)), key=lambda i: sum(self.solutions[i].objectives.values()) / len(self.solutions[i].objectives))
                best_solution = self.solutions[best_idx]
                best_trial = OptimizationTrial(
                    trial_id=f"trial_{best_idx:04d}",
                    parameters=best_solution.parameters,
                    score=sum(best_solution.objectives.values()) / len(best_solution.objectives),
                    metadata=best_solution.metadata or {},
                    start_time=best_solution.timestamp,
                    end_time=best_solution.timestamp,
                    status="completed"
                )
            
            # Create checkpoint
            checkpoint_id = self.checkpoint_manager.create_checkpoint(
                metadata=metadata,
                optimization_config=self.config,
                search_space=self.search_space,
                completed_trials=completed_trials,
                best_trial=best_trial,
                algorithm_state={
                    'study_name': self.study.study_name,
                    'directions': ["maximize"] * len(self.objective_function.objective_definitions),
                    'solutions_count': len(self.solutions),
                    'interrupted': self.interrupted
                },
                hardware_profile={'checkpoint_time': datetime.now().isoformat()}
            )
            
            print(f"💾 Checkpoint saved: {checkpoint_id}")
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            print(f"⚠️  Checkpoint creation failed: {e}")
    
    def _save_final_results(self, result: ParetoFrontResult, status: str):
        """Save final optimization results in compatible format."""
        try:
            # Save Pareto Front result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = self.output_dir / f"pareto_result_{status}_{timestamp}.json"
            
            # Convert result to JSON-serializable format
            result_data = {
                'algorithm': result.algorithm,
                'status': status,
                'total_evaluations': result.total_evaluations,
                'computation_time': result.computation_time,
                'trials_completed': self.trials_completed,
                'interrupted': self.interrupted,
                'metadata': result.metadata,
                'pareto_front': [
                    {
                        'parameters': sol.parameters,
                        'objectives': sol.objectives,
                        'trial_number': sol.trial_number,
                        'timestamp': sol.timestamp,
                        'metadata': sol.metadata
                    }
                    for sol in result.get_pareto_front()
                ],
                'all_solutions': [
                    {
                        'parameters': sol.parameters,
                        'objectives': sol.objectives,
                        'trial_number': sol.trial_number,
                        'timestamp': sol.timestamp,
                        'metadata': sol.metadata
                    }
                    for sol in result.solutions
                ]
            }
            
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            
            # Also save in training-compatible format if we have solutions
            if result.solutions:
                pareto_front = result.get_pareto_front()
                if pareto_front:
                    # Save best solution for training integration
                    best_solution = max(pareto_front, key=lambda s: sum(s.objectives.values()) / len(s.objectives))
                    
                    training_params = {
                        'optimization_method': 'pareto_front_tpe',
                        'timestamp': timestamp,
                        'status': status,
                        'total_trials': self.trials_completed,
                        'best_parameters': best_solution.parameters,
                        'best_score': sum(best_solution.objectives.values()) / len(best_solution.objectives),
                        'objectives': best_solution.objectives,
                        'pareto_front_size': len(pareto_front)
                    }
                    
                    # Save to best_parameters directory for training integration
                    best_params_dir = Path("models/best_parameters")
                    best_params_dir.mkdir(parents=True, exist_ok=True)
                    best_params_file = best_params_dir / f"pareto_selected_{status}_{timestamp}.json"
                    
                    with open(best_params_file, 'w') as f:
                        json.dump(training_params, f, indent=2, default=str)
                    
                    print(f"📊 Results saved: {result_file.name}")
                    print(f"🎯 Training parameters saved: {best_params_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to save final results: {e}")
            print(f"⚠️  Failed to save results: {e}")
    
    def _sample_optuna_parameter(self, trial, param_name: str, param_config: Union[Dict, List, Tuple]) -> Any:
        """Sample parameter using Optuna trial."""
        if isinstance(param_config, (list, tuple)):
            return trial.suggest_categorical(param_name, param_config)
        
        param_type = param_config.get('type', 'choice')
        
        if param_type == 'choice':
            return trial.suggest_categorical(param_name, param_config['choices'])
        elif param_type == 'uniform':
            return trial.suggest_float(param_name, param_config['low'], param_config['high'])
        elif param_type == 'loguniform':
            return trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
        elif param_type == 'int':
            return trial.suggest_int(param_name, param_config['low'], param_config['high'])
        elif param_type == 'bool':
            return trial.suggest_categorical(param_name, [True, False])
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")


def compute_pareto_front(solutions: List[ParetoSolution]) -> List[ParetoSolution]:
    """Compute Pareto Front from a list of solutions."""
    if not solutions:
        return []
    
    pareto_front = []
    
    for candidate in solutions:
        is_dominated = False
        
        for other in solutions:
            if other is candidate:
                continue
            
            if other.dominates(candidate):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_front.append(candidate)
    
    return pareto_front


def create_pareto_optimizer(
    algorithm: str,
    objective_function: MultiObjectiveFunction,
    search_space: Dict[str, Any],
    config: OptimizationConfig,
    output_dir: Path,
    **kwargs
) -> BaseParetoOptimizer:
    """Factory function to create Pareto Front optimizers."""
    
    algorithm = algorithm.lower()
    
    if algorithm in ["nsga2", "nsga-ii", "evolutionary", "ea"]:
        return NSGAIIParetoOptimizer(
            objective_function, search_space, config, output_dir, **kwargs
        )
    elif algorithm in ["tpe", "optuna", "mobo", "bayesian"]:
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, falling back to NSGA-II")
            return NSGAIIParetoOptimizer(
                objective_function, search_space, config, output_dir, **kwargs
            )
        return TPEParetoOptimizer(
            objective_function, search_space, config, output_dir, **kwargs
        )
    else:
        raise ValueError(f"Unknown Pareto algorithm: {algorithm}. "
                        f"Available: nsga2, tpe")


# Multi-objective search space example
DEFAULT_MULTI_OBJECTIVE_DEFINITIONS = {
    'accuracy': {
        'direction': 'maximize',
        'weight': 1.0,
        'description': 'Model prediction accuracy'
    },
    'training_time': {
        'direction': 'minimize', 
        'weight': 0.8,
        'description': 'Training time in minutes'
    },
    'model_complexity': {
        'direction': 'minimize',
        'weight': 0.6,
        'description': 'Model complexity score (0-1)'
    }
}