# Hyperparameter Optimization Guide

## Overview

The MarkSix Probabilistic Forecasting system includes a comprehensive hyperparameter optimization module that intelligently balances hardware constraints, model performance, and computational efficiency. This guide provides detailed instructions for using the optimization system.

## Features

### Core Capabilities
- **Multiple Algorithms**: Grid Search, Random Search, Bayesian Optimization, and Optuna integration
- **Hardware-Aware Optimization**: Automatic detection and optimization for available CPU, memory, and GPU resources
- **Intelligent Resource Management**: Dynamic batch size adjustment, memory monitoring, and resource cleanup
- **Real-time Monitoring**: Progress tracking, visualization, and performance analytics
- **Configuration Management**: YAML/JSON configuration with presets and templates
- **Robust Error Handling**: Retry mechanisms, fallback strategies, and comprehensive logging

### Advanced Features
- **Parallel Optimization**: Multi-threaded trial execution when hardware allows
- **Early Stopping**: Convergence detection and time-based termination
- **Checkpointing**: Resume interrupted optimization runs
- **Parameter Analysis**: Importance scoring and correlation analysis
- **Interactive Visualizations**: Progress plots, parameter relationships, and performance dashboards

## Quick Start

### Basic Usage

```python
from src.optimization.main import OptimizationOrchestrator

# Initialize orchestrator
orchestrator = OptimizationOrchestrator(
    data_path="data/raw/Mark_Six.csv",
    output_dir="my_optimization_results"
)

# Run optimization with preset
results = orchestrator.run_optimization(preset_name="balanced_search")

# Print results
print(f"Best score: {results['optimization_summary']['best_score']}")
print(f"Best parameters: {results['optimization_summary']['best_parameters']}")
```

### Command Line Interface

```bash
# Quick test (5 trials)
python -m src.optimization.main --preset quick_test

# Balanced optimization (30 trials, 4 hours max)
python -m src.optimization.main --preset balanced_search

# Custom optimization
python -m src.optimization.main --algorithm bayesian --max-trials 40 --max-duration 6

# List available presets
python -m src.optimization.main --list-presets
```

## Configuration System

### Available Presets

#### `quick_test`
- **Purpose**: Fast testing and validation
- **Trials**: 5 trials, 2 epochs each
- **Duration**: ~30 minutes
- **Algorithm**: Random Search
- **Use Case**: Development and debugging

#### `fast_search`
- **Purpose**: Quick optimization for initial exploration
- **Trials**: 20 trials, 3 epochs each
- **Duration**: ~2 hours
- **Algorithm**: Random Search
- **Use Case**: Initial hyperparameter exploration

#### `balanced_search`
- **Purpose**: Good balance of speed and thoroughness
- **Trials**: 30 trials, 5 epochs each
- **Duration**: ~4 hours
- **Algorithm**: Bayesian Optimization
- **Use Case**: Standard optimization runs

#### `thorough_search`
- **Purpose**: Comprehensive optimization
- **Trials**: 50 trials, 8 epochs each
- **Duration**: ~8 hours
- **Algorithm**: Optuna (TPE sampler)
- **Use Case**: Final optimization for production models

#### `grid_exploration`
- **Purpose**: Systematic parameter space exploration
- **Trials**: Variable (based on grid size)
- **Duration**: ~6 hours
- **Algorithm**: Grid Search
- **Use Case**: Understanding parameter interactions

### Custom Configuration

Create custom optimization configurations using YAML:

```yaml
# config/optimization/custom.yml
optimization_config:
  max_trials: 40
  max_duration_hours: 6.0
  early_stopping: true
  early_stopping_patience: 10
  parallel_jobs: 2
  trial_timeout_minutes: 45.0

search_space:
  learning_rate:
    type: loguniform
    low: 1e-6
    high: 1e-2
  batch_size:
    type: choice
    choices: [8, 16, 32, 64]
  hidden_size:
    type: choice
    choices: [128, 256, 512, 768]
  dropout:
    type: uniform
    low: 0.05
    high: 0.3

algorithm: bayesian
```

## Search Space Definition

### Parameter Types

#### Choice Parameters
```python
'optimizer': {
    'type': 'choice',
    'choices': ['adam', 'adamw', 'sgd']
}
```

#### Uniform Distribution
```python
'dropout': {
    'type': 'uniform',
    'low': 0.1,
    'high': 0.5
}
```

#### Log-Uniform Distribution
```python
'learning_rate': {
    'type': 'loguniform',
    'low': 1e-6,
    'high': 1e-2
}
```

#### Integer Parameters
```python
'num_layers': {
    'type': 'int',
    'low': 2,
    'high': 8
}
```

#### Boolean Parameters
```python
'use_batch_norm': {
    'type': 'bool'
}
```

#### List Format (Shorthand)
```python
'activation': ['relu', 'tanh', 'gelu']
```

### Default Search Spaces

The system provides several predefined search spaces:

#### Basic Search Space
- `learning_rate`: Log-uniform from 1e-5 to 1e-2
- `batch_size`: Choice from [8, 16, 32]
- `hidden_size`: Choice from [128, 256, 512]
- `dropout`: Uniform from 0.1 to 0.3

#### Advanced Search Space
- All basic parameters plus:
- `num_layers`: Integer from 2 to 8
- `weight_decay`: Log-uniform from 1e-8 to 1e-3
- `kl_weight`: Log-uniform from 1e-4 to 1e-1
- `contrastive_weight`: Uniform from 0.01 to 0.5
- `latent_dim`: Choice from [32, 64, 128, 256]

## Optimization Algorithms

### Grid Search
- **Best for**: Systematic exploration of discrete parameter spaces
- **Advantages**: Guaranteed to find global optimum in discrete space
- **Disadvantages**: Exponential growth with parameter count
- **Recommended**: Small search spaces (< 100 combinations)

```python
orchestrator.run_optimization(
    algorithm="grid_search",
    max_trials=50  # Will be limited by grid size
)
```

### Random Search
- **Best for**: Initial exploration and continuous parameters
- **Advantages**: Simple, parallelizable, good baseline
- **Disadvantages**: No learning from previous trials
- **Recommended**: First optimization runs, time-constrained scenarios

```python
orchestrator.run_optimization(
    algorithm="random_search",
    max_trials=30
)
```

### Bayesian Optimization
- **Best for**: Expensive function evaluations, continuous parameters
- **Advantages**: Learns from previous trials, efficient exploration
- **Disadvantages**: Overhead for simple functions, requires more trials for warmup
- **Recommended**: Standard optimization runs

```python
orchestrator.run_optimization(
    algorithm="bayesian",
    max_trials=40
)
```

### Optuna (TPE)
- **Best for**: Complex search spaces, production optimization
- **Advantages**: State-of-the-art algorithm, handles mixed parameter types
- **Disadvantages**: Additional dependency, complexity
- **Recommended**: Final optimization runs, complex parameter spaces

```python
orchestrator.run_optimization(
    algorithm="optuna",
    max_trials=50
)
```

## Hardware Management

### Automatic Detection
The system automatically detects and optimizes for:
- **CPU**: Core count, frequency, usage monitoring
- **Memory**: Total, available, usage constraints
- **GPU**: CUDA availability, memory, utilization

### Resource Optimization
- **Batch Size**: Automatically adjusted based on available memory
- **Parallel Jobs**: Limited by CPU cores and memory
- **Device Selection**: Automatic CPU/GPU selection with fallback
- **Memory Management**: Cleanup, monitoring, and constraint checking

### Manual Override
```python
# Override hardware recommendations
orchestrator.run_optimization(
    preset_name="balanced_search",
    parallel_jobs=1,  # Force single-threaded
    custom_config={
        'device': 'cpu',  # Force CPU usage
        'batch_size': 8   # Override batch size
    }
)
```

## Monitoring and Visualization

### Real-time Monitoring
- **Progress Tracking**: Trial completion, success rates, timing
- **Resource Usage**: CPU, memory, GPU utilization
- **Best Score Tracking**: Real-time updates of best performance
- **Early Stopping**: Automatic termination based on convergence

### Visualization Dashboard
The system generates comprehensive visualizations:

#### Optimization Progress
- Score progression over trials
- Best score evolution
- Trial duration analysis
- Success/failure rates

#### Parameter Analysis
- Parameter importance ranking
- Correlation matrices
- Parameter distribution plots
- Convergence analysis

#### Resource Monitoring
- CPU and memory usage over time
- GPU utilization (if available)
- Resource constraint violations
- Hardware performance metrics

### Accessing Results
```python
# Get optimization status during run
status = orchestrator.get_optimization_status()

# Access final results
results = orchestrator.run_optimization(preset_name="balanced_search")

# Generate visualizations
visualizations_dir = "optimization_results/plots"
# Check generated files: progress.png, importance.png, correlations.png, dashboard.html
```

## Advanced Usage

### Custom Objective Functions
```python
from src.optimization.integration import ModelTrainingInterface, OptimizationObjective

# Create custom training interface
interface = ModelTrainingInterface(
    data_path="data/raw/Mark_Six.csv",
    base_config=custom_config
)

# Define custom objective
def custom_objective(params):
    # Custom training logic
    score = interface.train_model(params)
    return score

# Use with optimizer
from src.optimization.algorithms import create_optimizer
from src.optimization.base_optimizer import OptimizationConfig

config = OptimizationConfig(max_trials=20)
optimizer = create_optimizer("bayesian", custom_objective, search_space, config)
best_params, best_score = optimizer.optimize()
```

### Parallel Optimization
```python
# Enable parallel optimization
orchestrator.run_optimization(
    preset_name="balanced_search",
    parallel_jobs=4,  # Use 4 parallel threads
    custom_config={
        'trial_timeout_minutes': 30.0,  # Timeout per trial
        'cleanup_frequency': 5          # Cleanup every 5 trials
    }
)
```

### Resume Interrupted Optimization
```python
# Optimization automatically saves checkpoints
# To resume, run with same output directory
orchestrator = OptimizationOrchestrator(
    data_path="data/raw/Mark_Six.csv",
    output_dir="previous_optimization_results"  # Same directory
)

# System will detect and resume from checkpoint
results = orchestrator.run_optimization(preset_name="thorough_search")
```

### Configuration Management
```python
from src.optimization.config_manager import create_config_manager

# Load configuration manager
config_manager = create_config_manager()

# List available presets
presets = config_manager.list_presets()
print("Available presets:", presets)

# Create custom preset
from src.optimization.base_optimizer import OptimizationConfig

custom_config = OptimizationConfig(
    max_trials=25,
    max_duration_hours=3.0,
    early_stopping=True,
    parallel_jobs=2
)

custom_search_space = {
    'learning_rate': {'type': 'loguniform', 'low': 1e-5, 'high': 1e-3},
    'batch_size': {'type': 'choice', 'choices': [16, 32]},
    'dropout': {'type': 'uniform', 'low': 0.1, 'high': 0.2}
}

config_manager.create_custom_preset(
    name="my_preset",
    description="Custom optimization for my use case",
    optimization_config=custom_config,
    search_space=custom_search_space,
    algorithm="bayesian"
)
```

## Performance Tips

### Optimization Strategy
1. **Start Small**: Begin with `quick_test` to validate setup
2. **Use Random Search**: For initial exploration of parameter space
3. **Progress to Bayesian**: For more efficient optimization
4. **Finish with Optuna**: For final fine-tuning

### Hardware Optimization
1. **GPU Usage**: Ensure CUDA is available for faster training
2. **Memory Management**: Monitor memory usage, reduce batch size if needed
3. **Parallel Jobs**: Use 2-4 parallel jobs on multi-core systems
4. **Storage**: Use fast storage (SSD) for data and checkpoints

### Parameter Space Design
1. **Start Broad**: Use wide parameter ranges initially
2. **Narrow Down**: Focus on promising regions in subsequent runs
3. **Validate Ranges**: Ensure parameter ranges make sense for your model
4. **Balance Exploration**: Don't make ranges too narrow too early

### Time Management
1. **Set Realistic Timeouts**: Allow sufficient time per trial
2. **Use Early Stopping**: Enable to avoid wasting time on poor configurations
3. **Monitor Progress**: Check intermediate results to adjust strategy
4. **Plan Resources**: Estimate total time based on hardware and trials

## Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Reduce batch size
python -m src.optimization.main --preset fast_search \
    --config '{"batch_size": 8, "max_memory_fraction": 0.7}'
```

#### Slow Training
```bash
# Reduce epochs per trial
python -m src.optimization.main --preset quick_test \
    --config '{"epochs": 2, "trial_timeout_minutes": 15}'
```

#### GPU Not Detected
```python
# Force CPU usage
orchestrator.run_optimization(
    preset_name="balanced_search",
    custom_config={'device': 'cpu'}
)
```

#### Failed Trials
- Check log files in `optimization_results/logs/`
- Verify data path and format
- Ensure sufficient disk space
- Check parameter ranges for validity

### Debug Mode
```bash
# Enable verbose logging
python -m src.optimization.main --preset quick_test --verbose
```

### Hardware Diagnostics
```python
from src.optimization.hardware_manager import create_hardware_manager

# Check hardware profile
manager = create_hardware_manager()
print("CPU Cores:", manager.profile.cpu_cores)
print("Total Memory:", manager.profile.total_memory_gb, "GB")
print("GPU Available:", manager.profile.gpu_available)
print("Recommended Batch Size:", manager.profile.recommended_batch_size)

# Check resource status
status = manager.get_resource_status()
print("Current Memory Usage:", status['memory_usage_percent'], "%")
print("Current CPU Usage:", status['cpu_usage_percent'], "%")
```

## Integration Examples

### With Jupyter Notebooks
```python
# notebook_optimization.py
import sys
sys.path.append('/path/to/marksix/project')

from src.optimization.main import OptimizationOrchestrator
import matplotlib.pyplot as plt

# Run optimization
orchestrator = OptimizationOrchestrator("data/raw/Mark_Six.csv")
results = orchestrator.run_optimization(preset_name="balanced_search")

# Plot results
trials = results['all_trials']
scores = [t['score'] for t in trials if t['status'] == 'completed']

plt.figure(figsize=(10, 6))
plt.plot(scores)
plt.title('Optimization Progress')
plt.xlabel('Trial')
plt.ylabel('Score')
plt.show()
```

### With MLflow
```python
import mlflow
from src.optimization.main import OptimizationOrchestrator

# Start MLflow experiment
mlflow.set_experiment("marksix_optimization")

with mlflow.start_run():
    orchestrator = OptimizationOrchestrator("data/raw/Mark_Six.csv")
    results = orchestrator.run_optimization(preset_name="balanced_search")
    
    # Log results
    mlflow.log_params(results['optimization_summary']['best_parameters'])
    mlflow.log_metric("best_score", results['optimization_summary']['best_score'])
    mlflow.log_artifacts("optimization_results")
```

### Automated Deployment
```python
# production_optimization.py
from src.optimization.main import OptimizationOrchestrator
import json

def optimize_for_production():
    orchestrator = OptimizationOrchestrator(
        data_path="data/raw/Mark_Six.csv",
        output_dir="production_optimization"
    )
    
    # Run thorough optimization
    results = orchestrator.run_optimization(preset_name="thorough_search")
    
    # Save best configuration for deployment
    best_config = results['optimization_summary']['best_parameters']
    with open('production_config.json', 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"Production optimization completed!")
    print(f"Best score: {results['optimization_summary']['best_score']}")
    
    return best_config

if __name__ == "__main__":
    optimize_for_production()
```

## API Reference

For detailed API documentation, see:
- [Base Optimizer API](../src/optimization/base_optimizer.py)
- [Algorithms API](../src/optimization/algorithms.py)
- [Hardware Manager API](../src/optimization/hardware_manager.py)
- [Integration API](../src/optimization/integration.py)
- [Configuration API](../src/optimization/config_manager.py)

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review log files in the optimization results directory
3. Consult the test files for usage examples
4. Refer to the source code documentation

The optimization system is designed to be robust and user-friendly. Most common issues can be resolved by adjusting configuration parameters or hardware settings.