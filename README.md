# MarkSix AI: A Probabilistic Forecasting System

This project implements a sophisticated, multi-stage pipeline to analyze historical Mark Six lottery data and generate probabilistically-informed number combinations. It leverages a deep learning model, heuristic scorers, and an intelligent search algorithm to identify high-scoring sets.

## Project Overview

The core of the system is a **Scoring Ensemble** that evaluates the "quality" of a given 6-number combination. This score is not a guarantee of winning, but rather a probabilistic measure based on patterns and trends learned from historical data. The final output is a list of recommended number sets that the system rates most highly.

The project is structured as a command-line application with three main functionalities:
1.  **Train:** Trains the core deep learning model on historical data.
2.  **Generate:** Uses the trained model and ensemble scorers to search for and recommend high-quality number sets.
3.  **Evaluate:** Measures the performance of the trained model against a validation set of historical data.

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

---

## Getting Started

### Prerequisites
- Python 3.10+
- Conda package manager
- An NVIDIA GPU with CUDA is highly recommended for training.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd MarkSix-Probabilistic-Forecasting
    ```

2.  **Create and activate the Conda environment:**
    The `environment.yml` file contains all the necessary dependencies.
    ```bash
    conda env create -f environment.yml
    conda activate marksix_ai
    ```

### Usage

The project is controlled via the main command-line hub.

```bash
python main.py
This will launch an interactive menu:--- Mark Six AI Project Hub ---
===============================
1. Train New Model
2. Generate Number Sets (Inference)
3. Evaluate Trained Model
4. Exit
===============================
Enter your choice (1-4):
Option 1: Train New ModelThis will run the full training pipeline using the settings in src/config.py.It fits the FeatureEngineer, trains the ScoringModel using contrastive loss, and saves the final model and feature engineer artifacts to the models/ directory.Option 2: Generate Number Sets (Inference)Loads the trained model and scorers.Prompts the user for the number of sets to generate.Asks whether to include the optional I-Ching scorer in the ensemble.Runs the local search algorithm to find and display the highest-scoring number combinations.Option 3: Evaluate Trained ModelLoads the trained model and scorers.Tests the model's performance against the validation portion of the dataset (draws it has never seen).The performance is measured by a "Win Rate": the percentage of times the model successfully scores a real winning combination higher than a randomly generated one. A rate above 50% indicates the model has predictive power.Project Structure.
├── data/
│   └── raw/
│       └── Mark_Six.csv      # Historical lottery data
├── models/                     # Saved model artifacts (generated)
├── notebooks/                  # Jupyter notebooks for analysis and demonstration
├── outputs/                    # Output logs and plots (generated)
├── src/                        # Source code
│   ├── config.py             # All project configurations
│   ├── data_loader.py        # Pytorch dataset/loader classes
│   ├── engine.py             # Core training and evaluation loops
│   ├── evaluation_pipeline.py# Logic for model evaluation
│   ├── feature_engineering.py# Feature creation class
│   ├── i_ching_scorer.py     # I-Ching heuristic scorer
│   ├── inference_pipeline.py # Logic for generating numbers
│   ├── model.py              # The deep learning model architecture
│   ├── sam.py                # SAM Optimizer implementation
│   ├── temporal_scorer.py    # Temporal heuristic scorer
│   └── training_pipeline.py  # Main training orchestration
├── .gitignore
├── environment.yml
├── main.py                     # Main project hub (entry point)
└── README.md
