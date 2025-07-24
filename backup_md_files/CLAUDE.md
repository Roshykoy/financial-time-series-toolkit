# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a sophisticated probabilistic forecasting system for Mark Six lottery analysis using deep learning. The system combines a Conditional Variational Autoencoder (CVAE) with graph neural networks, temporal context modeling, and meta-learning to generate probabilistically-informed number combinations.

## Development Commands

### Environment Setup
```bash
# Automated setup (recommended)
python setup.py

# Manual setup
conda env create -f environment.yml
conda activate marksix_ai
```

### Main Application
```bash
# Primary entry point - interactive CLI with 7 options
python main.py
```

### Testing and Validation
```bash
# Test hyperparameter optimization functionality
python test_hyperparameter_optimization.py

# Debug model architecture and training pipeline
python test_model_debug.py
```

### Development Workflow
1. **First-time setup**: Run `python setup.py` to create environment and verify system
2. **Before development**: Run `python test_hyperparameter_optimization.py` to verify functionality
3. **Training**: Use `python main.py` → Option 1 (Train New Model)
4. **Optimization**: Use `python main.py` → Option 4 (Optimize Hyperparameters)
5. **Evaluation**: Use `python main.py` → Option 3 (Evaluate Trained Model)

## Architecture Overview

### Core Components

1. **CVAE Model** (`src/cvae_model.py`): Conditional Variational Autoencoder for number generation
2. **Graph Encoder** (`src/graph_encoder.py`): Graph neural network for modeling number relationships
3. **Temporal Context** (`src/temporal_context.py`): LSTM-based temporal pattern modeling
4. **Meta-Learner** (`src/meta_learner.py`): Attention-based ensemble weight optimization
5. **Feature Engineering** (`src/feature_engineering.py`): Rich feature extraction from number combinations

### Training Pipeline
- **CVAE Engine** (`src/cvae_engine.py`): Core training loop with contrastive learning
- **Training Pipeline** (`src/training_pipeline.py`): Orchestrates the complete training process
- **Hyperparameter Optimizer** (`src/hyperparameter_optimizer.py`): Automated parameter tuning

### Inference Pipeline
- **Inference Pipeline** (`src/inference_pipeline.py`): Number generation and ensemble scoring
- **Evaluation Pipeline** (`src/evaluation_pipeline.py`): Model performance assessment

## Configuration System

### Main Configuration
- All parameters are centralized in `src/config.py`
- Conservative settings are used for stability (reduced from original values)
- Device auto-detection with CUDA/CPU fallback

### Hyperparameter Optimization
- **Random Search**: 15-30 minutes, good for beginners
- **Grid Search**: 30-60 minutes, thorough exploration
- **Bayesian Optimization**: 20-40 minutes, intelligent search
- **Quick Search**: 5-10 minutes, for testing

### Configuration Presets
- `fast_training`: Quick results, lower quality
- `balanced`: Good balance of speed and quality (default)
- `high_quality`: Best results, longer training time
- `experimental`: Cutting-edge parameters for research

## Data Requirements

### Input Data
- **Required**: `data/raw/Mark_Six.csv` - Historical Mark Six lottery data
- **Format**: CSV with columns for Draw, Date, Winning numbers, Extra number, and statistics
- **Minimum**: 100+ historical draws for meaningful training

### Generated Data
- `data/processed/`: Automatically processed data files
- `models/`: Trained model artifacts (.pth files)
- `outputs/`: Training logs, plots, and debug reports
- `hyperparameter_results/`: Optimization results and trial data

## Key Features

### Advanced Model Architecture
- **CVAE**: Conditional generation with latent space modeling
- **Graph Networks**: Model relationships between numbers
- **Temporal Context**: LSTM with attention for historical patterns
- **Meta-Learning**: Dynamic ensemble weight optimization

### Hyperparameter Optimization
- **Multiple Algorithms**: Grid Search, Random Search, Bayesian Optimization
- **Intelligent Configuration**: Save/load parameter presets
- **Performance Tracking**: Comprehensive evaluation metrics
- **Early Stopping**: Prevent overfitting with patience-based stopping

### Conservative Training Approach
- **Stability-First**: Reduced parameters to prevent overfitting
- **Memory Management**: CUDA cache clearing and memory fraction limits
- **Error Handling**: Fallback to CPU, checkpoint on error, batch failure tolerance
- **Gradient Clipping**: Aggressive norm clipping (0.5) for stability

## Development Notes

### No Traditional Build System
- This is a research-oriented ML project
- No Make, npm, or similar build tools
- Environment managed through conda
- Interactive CLI for user interface

### GPU Considerations
- CUDA auto-detection with CPU fallback
- Mixed precision training disabled for stability
- Memory fraction limited to 80% to prevent OOM
- Gradient accumulation and cache clearing implemented

### Model Saving Strategy
- Conservative models saved with `conservative_` prefix
- Checkpoints saved every 2 epochs
- Feature engineer pickled separately
- Configuration presets saved in JSON format

### Code Structure Patterns
- Ensemble approach with multiple scorers
- Pipeline architecture for training/inference
- Configuration-driven design
- Comprehensive error handling and logging

## Common Issues and Solutions

### Memory Issues
- Reduce `batch_size` in config (default: 8)
- Enable CPU fallback in config
- Reduce model parameters (already conservative)

### Training Instability
- Use conservative learning rate (5e-5)
- Enable gradient clipping (0.5)
- Reduce KL and contrastive loss weights

### GPU Issues
- System auto-detects and falls back to CPU
- Test GPU functionality with test scripts
- Check CUDA memory with diagnostic tools

## Debugging and Monitoring

### Debug Tools
- `src/debug_utils.py`: Comprehensive debugging utilities
- `outputs/cvae_debug_report.txt`: Detailed training reports
- `test_model_debug.py`: Model architecture validation

### Monitoring
- Real-time loss tracking during training
- Epoch-by-epoch validation metrics
- Hyperparameter optimization progress
- Memory usage and GPU utilization

### Log Files
- Training progress in `outputs/`
- Optimization results in `hyperparameter_results/`
- Configuration changes tracked in `configurations/`

## Best Practices for Development

1. **Always test first**: Run test scripts before making changes
2. **Use hyperparameter optimization**: Don't manually tune parameters
3. **Monitor memory usage**: Watch for OOM errors on GPU
4. **Validate data**: Ensure CSV format matches expected structure
5. **Backup models**: Save successful configurations as presets
6. **Conservative changes**: Start with small modifications to stable config