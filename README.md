# MarkSix AI: A Probabilistic Forecasting System

This project implements a sophisticated, multi-stage pipeline to analyze historical Mark Six lottery data and generate probabilistically-informed number combinations. It leverages a **Conditional Variational Autoencoder (CVAE)** with graph neural networks, temporal context modeling, meta-learning, and **automated hyperparameter optimization** to identify high-scoring number sets through advanced generative modeling.

## Project Overview

The core of the system is a **CVAE-based Generative Model** that learns complex latent representations of lottery number patterns. Combined with graph neural networks for number relationships, LSTM-based temporal modeling, and meta-learning for ensemble optimization, the system generates probabilistically-informed number combinations. This is not a guarantee of winning, but rather a sophisticated probabilistic measure based on multi-dimensional patterns learned from historical data.

The project is structured as a command-line application with seven main functionalities:
1.  **Train:** Trains the CVAE-based generative model with graph neural networks and temporal context on historical data.
2.  **Generate:** Uses the trained CVAE and meta-learned ensemble to generate high-quality number combinations.
3.  **Evaluate:** Measures the performance of the trained model against a validation set of historical data.
4.  **Optimize:** Automatically finds the best hyperparameters for optimal model performance.
5.  **System Info:** Displays current system configuration and model status.
6.  **Advanced Options:** Configuration management, diagnostics, and file operations.
7.  **Exit:** Graceful application termination.

---

## 🆕 Major Architecture Overhaul (Latest Release)

### CVAE-Based Generative Model
- **Conditional Variational Autoencoder**: Advanced generative modeling for pattern learning
- **Latent Space Learning**: Compressed representations of number combination patterns
- **Multi-Component Loss**: Reconstruction + KL divergence + contrastive learning
- **Conservative Training**: Stability-focused parameters for reliable convergence

### Graph Neural Networks & Temporal Modeling
- **Graph Attention Networks (GAT)**: Models complex number co-occurrence relationships
- **Temporal Context LSTM**: Processes historical sequence information with attention
- **Dynamic Pattern Recognition**: Learns temporal dependencies in lottery draws

### Meta-Learning & Ensemble Optimization
- **Attention-Based Meta-Learner**: Adapts ensemble weights based on input patterns
- **Confidence Estimation**: Provides uncertainty quantification for predictions
- **Hard Negative Mining**: Sophisticated contrastive learning with carefully selected negatives

### Enhanced Training Pipeline
- **Mixed Precision Handling**: Overflow detection and CPU fallback mechanisms
- **Gradient Management**: Aggressive clipping (0.5) and numerical stability checks
- **Error Recovery**: Graceful handling of failed batches with detailed logging
- **Memory Optimization**: CUDA cache clearing and resource management

---

## System Architecture

The project is built around a sophisticated generative architecture with multiple neural network components.

#### 1. CVAE Core Model (`src/cvae_model.py`)
-   **Architecture:** Conditional Variational Autoencoder with encoder-decoder structure
-   **Latent Dimensions:** 64-dimensional compressed representation space (reduced for stability)
-   **Function:** Learns to reconstruct number combinations while discovering latent patterns
-   **Training:** Multi-component loss combining reconstruction, KL divergence, and contrastive learning

#### 2. Graph Neural Network (`src/graph_encoder.py`)
-   **Architecture:** Graph Attention Network (GAT) for modeling number relationships
-   **Function:** Captures complex co-occurrence patterns between lottery numbers
-   **Features:** Multi-head attention mechanism for relationship learning

#### 3. Temporal Context Module (`src/temporal_context.py`)
-   **Architecture:** LSTM with attention mechanism for sequence modeling
-   **Function:** Processes historical lottery draw sequences to learn temporal patterns
-   **Features:** Bidirectional LSTM with attention-based context aggregation

#### 4. Meta-Learning Component (`src/meta_learner.py`)
-   **Architecture:** Attention-based neural network for ensemble weight optimization
-   **Function:** Dynamically adapts scoring weights based on input patterns
-   **Features:** Confidence estimation and uncertainty quantification

#### 5. Enhanced Feature Engineering (`src/feature_engineering.py`)
-   **Function:** Converts number combinations into rich multi-dimensional feature vectors
-   **Enhanced Features:**
    -   Temporal sequences and graph embeddings
    -   Statistical properties (sum, mean, variance, odd/even ratios)
    -   Historical frequencies and pair analysis
    -   Delta features and number group distributions
    -   Graph-based relationship features

#### 6. Advanced Training Engine (`src/cvae_engine.py`)
-   **Function:** Orchestrates the complete CVAE training process
-   **Features:** Conservative training with stability checks, error recovery, and comprehensive logging

#### 7. Advanced Inference Pipeline (`src/inference_pipeline.py`)
-   **Function:** Sophisticated number generation using CVAE sampling and ensemble scoring
-   **Features:** Meta-learned ensemble weights, confidence-based selection, and iterative refinement
-   **Search Algorithm:** Local search with CVAE-guided exploration for high-scoring combinations

#### 8. Hyperparameter Optimization System (`src/hyperparameter_optimizer.py`)
-   **Function:** Automatically finds optimal CVAE and training parameters
-   **Algorithms:** Grid Search, Random Search, and Bayesian Optimization
-   **Features:** Early stopping, progress tracking, result persistence, and performance analysis

#### 9. Specialized Data Loading (`src/cvae_data_loader.py`)
-   **Function:** Custom data loading for CVAE training with negative sampling
-   **Features:** Hard negative mining, temporal sequence preparation, and graph adjacency matrix generation

---

## Getting Started

### Prerequisites
- Python 3.10+
- Conda package manager
- An NVIDIA GPU with CUDA is highly recommended for training and optimization.

### Quick Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd MarkSix-Probabilistic-Forecasting
    ```

2.  **Run the automated setup (recommended):**
    ```bash
    python setup.py
    ```
    This will create the conda environment, check dependencies, and verify your system setup.

3.  **Manual setup (alternative):**
    ```bash
    conda env create -f environment.yml
    conda activate marksix_ai
    ```

4.  **Test your installation:**
    ```bash
    # Test hyperparameter optimization functionality
    python test_hyperparameter_optimization.py
    
    # Debug and validate model architecture
    python test_model_debug.py
    ```

### Usage

The project is controlled via the main command-line hub with an enhanced interactive menu:

```bash
python main.py
```

This will launch the upgraded interactive menu:

```
🎯 MARK SIX AI PROJECT HUB 🎯
Advanced Lottery Analysis System
===============================
1. 🧠 Train New Model
2. 🎲 Generate Number Sets (Inference)
3. 📊 Evaluate Trained Model
4. ⚙️  Optimize Hyperparameters (NEW!)
5. 💻 System Information
6. 🔧 Advanced Options
7. 🚪 Exit
===============================
```

#### Option 1: Train New Model
- Runs the full training pipeline using the current configuration in `src/config.py`
- Fits the FeatureEngineer and trains the ScoringModel using contrastive loss
- Saves the final model and feature engineer artifacts to the `models/` directory
- **🆕 Now automatically detects and offers to use optimized hyperparameters**

#### Option 2: Generate Number Sets (Inference)
- Loads the trained model and scorers
- Prompts the user for the number of sets to generate
- Asks whether to include the optional I-Ching scorer in the ensemble
- Runs the local search algorithm to find and display the highest-scoring number combinations

#### Option 3: Evaluate Trained Model
- Loads the trained model and scorers
- Tests the model's performance against the validation portion of the dataset
- The performance is measured by a "Win Rate": the percentage of times the model successfully scores a real winning combination higher than a randomly generated one
- A rate above 50% indicates the model has predictive power

#### 🆕 Option 4: Optimize Hyperparameters
- **Automated Parameter Tuning:** Systematically finds the best configuration for your model
- **Multiple Methods:**
  - **Random Search:** Fast and effective (15-30 minutes)
  - **Grid Search:** Thorough exploration (30-60 minutes)
  - **Bayesian Optimization:** Intelligent search (20-40 minutes)
  - **Quick Search:** For testing (5-10 minutes)
- **Automatic Application:** Can immediately train with optimized parameters
- **Result Persistence:** Saves best configurations for future use

#### 🆕 Option 5: System Information
- Displays current system configuration and GPU status
- Shows current hyperparameter settings
- Indicates if optimized parameters are available

#### 🆕 Option 6: Advanced Options
- **Configuration Manager:** Create, edit, and apply parameter presets
- **Optimization History:** View past optimization results
- **File Management:** Clean up generated files and results
- **System Diagnostics:** Advanced troubleshooting and configuration

---

## 🆕 Hyperparameter Optimization Guide

### Quick Start
1. Select option 4 from the main menu
2. Choose **Random Search** for your first optimization
3. Use default settings (20 trials, 3-5 epochs per trial)
4. Let it run for 15-30 minutes
5. Apply the optimized parameters when prompted

### Optimization Methods

| Method | Best For | Time | Thoroughness |
|--------|----------|------|--------------|
| **Random Search** | Beginners, quick results | 15-30 min | Good |
| **Grid Search** | Thorough exploration | 30-60 min | Excellent |
| **Bayesian Optimization** | Balanced approach | 20-40 min | Very Good |
| **Quick Search** | Testing, demos | 5-10 min | Basic |

### Parameters Optimized
- **Learning Rate:** Model training speed and stability
- **Model Architecture:** Hidden size, number of layers, dropout
- **Training Settings:** Batch size, epochs, loss margin
- **Optimizer:** SAM vs AdamW with tuned parameters

### Expected Improvements
With proper optimization, expect:
- **15-30% better model performance**
- **More stable training**
- **Better number generation quality**
- **Reduced overfitting**

---

## Project Structure

```
├── data/
│   ├── raw/
│   │   └── Mark_Six.csv           # Historical lottery data
│   └── processed/                 # Processed data (generated)
├── models/                        # Saved model artifacts (generated)
├── notebooks/                     # Jupyter notebooks for analysis
├── outputs/                       # Output logs and plots (generated)
├── hyperparameter_results/        # Optimization results (generated)
├── configurations/                # Configuration presets (generated)
├── src/                           # Source code
│   ├── config.py                  # All project configurations (enhanced)
│   ├── 🆕 cvae_model.py           # CVAE architecture with graph/temporal encoders
│   ├── 🆕 cvae_engine.py          # CVAE training and evaluation engines
│   ├── 🆕 cvae_data_loader.py     # Specialized data loading with negative sampling
│   ├── 🆕 graph_encoder.py        # Graph Attention Network for number relationships
│   ├── 🆕 temporal_context.py     # LSTM-based temporal pattern modeling
│   ├── 🆕 meta_learner.py         # Attention-based ensemble weight optimization
│   ├── 🆕 debug_utils.py          # Comprehensive debugging utilities
│   ├── 🆕 model_analysis.py       # Advanced model analysis and visualization
│   ├── 🆕 visualization.py        # Enhanced plotting and visualization tools
│   ├── evaluation_pipeline.py    # Logic for model evaluation (enhanced)
│   ├── feature_engineering.py    # Feature creation class (enhanced)
│   ├── inference_pipeline.py     # Logic for generating numbers (enhanced)
│   ├── training_pipeline.py      # Main training orchestration (enhanced)
│   └── hyperparameter_optimizer.py  # Automated parameter optimization
├── .gitattributes                 # Git LFS configuration for large files
├── .gitignore
├── environment.yml                # Conda environment (updated dependencies)
├── main.py                        # Main project hub (completely redesigned)
├── setup.py                       # Automated setup script
├── test_hyperparameter_optimization.py  # Testing script
├── 🆕 test_model_debug.py         # Model architecture validation and debugging
├── best_hyperparameters.json     # Best found parameters (generated)
└── README.md
```

## Performance and Optimization

### System Requirements
- **Minimum:** Python 3.10+, 8GB RAM, CPU-only training
- **Recommended:** Python 3.10+, 16GB+ RAM, NVIDIA GPU with 6GB+ VRAM
- **Optimal:** Python 3.10+, 32GB+ RAM, NVIDIA GPU with 10GB+ VRAM

### Performance Tips
1. **Use GPU when available** - Significantly faster training and optimization
2. **Start with hyperparameter optimization** - Can improve performance by 15-30%
3. **Use configuration presets** - Pre-tuned settings for different hardware
4. **Monitor system resources** - Adjust batch size if memory errors occur

### Troubleshooting
- **CUDA out of memory:** Reduce batch size in configuration
- **Slow training:** Use CPU presets or reduce model size
- **Poor performance:** Run hyperparameter optimization
- **Import errors:** Check conda environment activation

## Advanced Features

### Configuration Presets
Access via **Advanced Options → Configuration Manager**:
- **fast_training:** Quick results, lower quality
- **balanced:** Good balance of speed and quality (default)
- **high_quality:** Best results, longer training time
- **experimental:** Cutting-edge parameters for research

### Optimization History
Track and compare optimization runs:
- View past optimization results
- Compare different methods
- Load previously found best parameters
- Analyze parameter trends and performance

### File Management
Automated cleanup and organization:
- Remove old optimization results
- Clean model artifacts
- Manage configuration files
- System diagnostic reports

## Contributing

When contributing to this project:
1. Test your changes with `python test_hyperparameter_optimization.py`
2. Update configuration presets if adding new parameters
3. Add optimization tests for new features
4. Update documentation for any new functionality

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with local regulations regarding lottery systems and predictions.

---

## 🚀 Getting the Best Results

1. **First Run:** Use Random Search hyperparameter optimization
2. **Training:** Train with optimized parameters
3. **Evaluation:** Check model performance with evaluation pipeline
4. **Generation:** Use the trained model to generate number sets
5. **Iteration:** Re-optimize if needed for better results

**Remember:** This system finds patterns in historical data but cannot guarantee future lottery outcomes. Use responsibly and within your means.