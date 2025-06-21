# src/config.py
import torch

CONFIG = {
    # Data and Model Parameters
    "data_path": "data/raw/Mark_Six.csv",
    "num_lotto_numbers": 49,
    "d_features": 16,  # This will be dynamically updated in the pipeline
    "hidden_size": 256,
    "num_layers": 6,
    "dropout": 0.15,

    # Training Parameters
    "use_sam_optimizer": True,
    "learning_rate": 1e-4,
    "epochs": 15,
    "batch_size": 64,
    "margin": 0.5,
    "negative_samples": 32,
    "rho": 0.05,  # SAM optimizer sharpness parameter

    # Paths
    "model_save_path": "models/scoring_model.pth",
    "feature_engineer_path": "models/feature_engineer.pkl",
    
    # Ensemble & Inference Parameters
    "ensemble_weights": {
        "model": 0.6,      # Weight for the main deep learning model
        "temporal": 0.4,   # Weight for the recency scorer
        "i_ching": 0.15    # Weight for the I-Ching scorer (only if used)
    },
    "search_iterations": 250, # Number of iterations for local search per set
    "search_neighbors": 20,   # Number of neighbors to explore at each step of the search

    # --- New for Step 6 ---
    # Evaluation Parameters
    "evaluation_neg_samples": 99 # For each real draw, compare it against 99 random sets
}