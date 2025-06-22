# MarkSix AI: A Probabilistic Forecasting System

This project implements a sophisticated, multi-stage pipeline to analyze historical Mark Six lottery data and generate probabilistically-informed number combinations. It leverages a deep learning model, heuristic scorers, intelligent search algorithms, and **automated hyperparameter optimization** to identify high-scoring number sets.

## Project Overview

The core of the system is a **Scoring Ensemble** that evaluates the "quality" of a given 6-number combination. This score is not a guarantee of winning, but rather a probabilistic measure based on patterns and trends learned from historical data. The final output is a list of recommended number sets that the system rates most highly.

The project is structured as a command-line application with four main functionalities:
1.  **Train:** Trains the core deep learning model on historical data.
2.  **Generate:** Uses the trained model and ensemble scorers to search for and recommend high-quality number sets.
3.  **Evaluate:** Measures the performance of the trained model against a validation set of historical data.
4.  **🆕 Optimize:** Automatically finds the best hyperparameters for optimal model performance.

---

## 🆕 New Features

### Hyperparameter Optimization
- **Automated Parameter Tuning**: Finds optimal learning rates, model sizes, and training settings
- **Multiple Algorithms**: Grid Search, Random Search, and Bayesian Optimization
- **Intelligent Configuration Management**: Save, load, and compare parameter presets
- **Performance Tracking**: Comprehensive evaluation metrics and optimization history

### Enhanced User Experience
- **Interactive Menus**: Intuitive command-line interface with visual feedback
- **Configuration Presets**: Pre-defined settings for different use cases (fast, balanced, high-quality)
- **System Monitoring**: GPU detection, memory management, and performance optimization
- **Advanced Options**: File management, configuration editing, and system diagnostics

---

## System Architecture

The project is built around several key components that work together in an ensemble.

#### 1. The Scoring Model (`src/model.py`)
-   **Core:** A Transformer-based deep learning model architected with residual blocks.
-   **Function:** Takes a feature vector representing a number combination and outputs a single score.
-   **Training:** Trained using a contrastive learning approach with a ranking loss. It learns to assign higher scores to real historical winning combinations than to random (negative) ones. The training process is enhanced with the Sharpness-Aware Minimization (SAM) optimizer to improve model generalization.

#### 2. The Feature Engineer (`src/feature_engineering.py`)
-   **Function:** Converts any 6-number set into a rich feature vector.
-   **Features Include:**
    -   Basic properties (sum, mean, odd/even count).
    -   Historical frequencies of individual numbers and pairs.
    -   Delta features (differences between consecutive numbers).
    -   Distribution across number groups (e.g., 1-10, 11-20).

#### 3. Heuristic Scorers
-   **Temporal Scorer (`src/temporal_scorer.py`):** Scores sets based on the recency of their numbers. It rewards combinations containing numbers that have appeared more recently.
-   **I-Ching Scorer (`src/i_ching_scorer.py`):** An optional, deterministic scorer that assigns a pre-defined "luck" value to each number.

#### 4. The Scorer Ensemble (`src/inference_pipeline.py`)
-   **Function:** Orchestrates the final scoring process. It normalizes the scores from the deep learning model and the heuristic scorers and combines them using a weighted average defined in `src/config.py`.

#### 5. Local Search Algorithm (`src/inference_pipeline.py`)
-   **Function:** Instead of just generating random sets, this algorithm actively searches for high-scoring combinations. It starts with a random set and iteratively makes small changes (swapping one number), always moving towards a higher ensemble score.

#### 6. 🆕 Hyperparameter Optimization System (`src/hyperparameter_optimizer.py`)
-   **Function:** Automatically finds the best model parameters through systematic search.
-   **Algorithms:** Grid Search, Random Search, and Bayesian Optimization.
-   **Features:** Early stopping, progress tracking, result persistence, and performance analysis.

#### 7. 🆕 Configuration Manager (`src/config_manager.py`)
-   **Function:** Manages configuration presets and provides interactive parameter editing.
-   **Features:** Preset management, parameter validation, configuration comparison, and interactive editing.

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
    python test_hyperparameter_optimization.py
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
├── 🆕 hyperparameter_results/     # Optimization results (generated)
├── 🆕 configurations/             # Configuration presets (generated)
├── src/                           # Source code
│   ├── config.py                  # All project configurations
│   ├── data_loader.py            # PyTorch dataset/loader classes
│   ├── engine.py                 # Core training and evaluation loops
│   ├── evaluation_pipeline.py    # Logic for model evaluation
│   ├── feature_engineering.py    # Feature creation class
│   ├── i_ching_scorer.py         # I-Ching heuristic scorer
│   ├── inference_pipeline.py     # Logic for generating numbers
│   ├── model.py                  # The deep learning model architecture
│   ├── sam.py                    # SAM Optimizer implementation
│   ├── temporal_scorer.py        # Temporal heuristic scorer
│   ├── training_pipeline.py      # Main training orchestration
│   ├── 🆕 hyperparameter_optimizer.py  # Automated parameter optimization
│   └── 🆕 config_manager.py       # Configuration management utility
├── .gitignore
├── environment.yml
├── main.py                        # Main project hub (entry point)
├── 🆕 setup.py                    # Automated setup script
├── 🆕 test_hyperparameter_optimization.py  # Testing script
├── 🆕 best_hyperparameters.json   # Best found parameters (generated)
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