# src/config.py
import torch

CONFIG = {
    # Data and Basic Parameters
    "data_path": "data/raw/Mark_Six.csv",
    "num_lotto_numbers": 49,
    
    # --- CVAE Architecture Parameters (REDUCED FOR STABILITY) ---
    
    # Graph Neural Network Encoder - REDUCED
    "node_embedding_dim": 32,        # Reduced from 64
    "graph_hidden_dim": 64,          # Reduced from 128
    "num_gat_layers": 2,             # Reduced from 3
    "graph_projection_dim": 128,     # Reduced from 256
    
    # Temporal Context Encoder - REDUCED
    "temporal_sequence_length": 10,  # Reduced from 20
    "temporal_embedding_dim": 16,    # Reduced from 32
    "temporal_hidden_dim": 64,       # Reduced from 128
    "temporal_lstm_layers": 1,       # Reduced from 2
    "temporal_attention_heads": 4,   # Reduced from 8
    "temporal_context_dim": 64,      # Reduced from 128
    "temporal_projection_dim": 128,  # Reduced from 256
    "trend_hidden_dim": 32,          # Reduced from 64
    "trend_features": 16,            # Reduced from 32
    
    # VAE Core Parameters - REDUCED
    "latent_dim": 64,                # Reduced from 128
    "prior_hidden_dim": 128,         # Reduced from 256
    
    # Generative Decoder - REDUCED
    "decoder_hidden_dim": 128,       # Reduced from 256
    "decoder_layers": 2,             # Reduced from 3
    "decoder_projection_dim": 256,   # Reduced from 512
    "number_generation_hidden": 64,  # Reduced from 128
    "constraint_hidden_dim": 128,    # Reduced from 256
    
    # Meta-Learner Parameters - REDUCED
    "scorer_types": ["generative", "temporal", "i_ching"],
    "combo_embedding_dim": 16,       # Reduced from 32
    "pattern_hidden_dim": 64,        # Reduced from 128
    "pattern_features": 32,          # Reduced from 64
    "statistical_features": 13,      # Number of hand-crafted statistical features
    "stat_hidden_dim": 32,           # Reduced from 64
    "stat_features": 16,             # Reduced from 32
    "meta_hidden_dim": 128,          # Reduced from 256
    "integrated_features": 64,       # Reduced from 128
    "meta_attention_heads": 4,       # Reduced from 8
    "weight_hidden_dim": 32,         # Reduced from 64
    "confidence_hidden_dim": 32,     # Reduced from 64
    
    # Training Parameters - CONSERVATIVE
    "dropout": 0.1,                  # Reduced from 0.15
    "learning_rate": 5e-5,           # Reduced from 1e-4 (HALF)
    "epochs": 10,                    # Reduced from 25 for testing
    "batch_size": 8,                 # Reduced from 32 for stability
    "gradient_clip_norm": 0.5,       # Reduced from 1.0 (MORE AGGRESSIVE)
    
    # Loss Function Weights - CONSERVATIVE
    "reconstruction_weight": 1.0,    # Keep main loss
    "kl_weight": 0.01,               # Much reduced from 0.1
    "contrastive_weight": 0.1,       # Much reduced from 0.5
    "meta_learning_weight": 0.05,    # Much reduced from 0.3
    
    # Hard Negative Mining - REDUCED
    "negative_samples": 8,           # Reduced from 16
    "hard_negative_ratio": 0.5,      # Reduced from 0.7
    "negative_pool_size": 5000,      # Much reduced from 25000
    
    # Contrastive Learning - CONSERVATIVE
    "contrastive_margin": 0.3,       # Reduced from 0.5
    "contrastive_temperature": 0.2,  # Increased from 0.1 (less aggressive)
    
    # Inference Parameters
    "generation_temperature": 1.0,   # Increased from 0.8 (less confident)
    "num_generation_samples": 10,    # Reduced from 50
    "top_k_sampling": 5,             # Reduced from 10
    "ensemble_selection_method": "fixed_weights",  # Changed from "meta_learned" for stability
    
    # Evaluation Parameters
    "evaluation_neg_samples": 49,    # Reduced from 99
    "validation_generation_samples": 5,  # Reduced from 20
    
    # File Paths
    "model_save_path": "models/conservative_cvae_model.pth",
    "meta_learner_save_path": "models/conservative_meta_learner.pth", 
    "feature_engineer_path": "models/conservative_feature_engineer.pkl",
    
    # Legacy Heuristic Scorers (still used in ensemble)
    "ensemble_weights": {
        "generative": 0.7,    # Increased generative weight for stability
        "temporal": 0.2,      # Reduced temporal weight
        "i_ching": 0.1        # Reduced i_ching weight
    },
    "search_iterations": 0,    # Disabled - replaced by generation
    "search_neighbors": 0,     # Disabled - replaced by generation
    
    # Device Configuration - STABILITY FOCUSED
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_mixed_precision": False,    # DISABLED - was causing overflow
    
    # Advanced Training Options - CONSERVATIVE
    "use_scheduler": False,          # DISABLED for stability
    "scheduler_type": "none",        # Changed from "cosine"
    "warmup_epochs": 0,              # Disabled
    "weight_decay": 1e-6,            # Much reduced from 1e-5
    "ema_decay": 0,                  # DISABLED - was 0.999
    
    # Data Augmentation - DISABLED FOR STABILITY
    "augment_temporal_noise": 0.0,   # Disabled - was 0.1
    "augment_dropout_prob": 0.0,     # Disabled - was 0.1
    
    # Debugging and Monitoring - INCREASED FREQUENCY
    "log_interval": 10,              # More frequent logging (was 50)
    "save_interval": 2,              # Save more frequently (was 5)
    "plot_latent_space": False,      # DISABLED to reduce overhead
    "save_generation_samples": False, # DISABLED to reduce overhead
    
    # ADDITIONAL CONSERVATIVE SETTINGS
    "dropout_rate": 0.1,             # Alias for dropout (some modules use this name)
    "meta_learner_frequency": 20,    # Train meta-learner less frequently
    "validation_frequency": 2,       # Validate every 2 epochs instead of every epoch
    "early_stopping": True,          # Enable early stopping
    "early_stopping_patience": 3,    # Stop if no improvement for 3 epochs
    "min_improvement": 0.001,        # Minimum improvement to consider
    "max_grad_norm": 0.5,           # Additional gradient clipping parameter
    "numerical_stability_eps": 1e-8, # Small epsilon for numerical stability
    
    # MEMORY MANAGEMENT
    "clear_cache_frequency": 5,      # Clear CUDA cache every 5 batches
    "max_memory_fraction": 0.8,      # Use max 80% of GPU memory
    
    # FALLBACK SETTINGS
    "fallback_to_cpu": True,         # Fallback to CPU if CUDA issues
    "checkpoint_on_error": True,     # Save checkpoint if training fails
    "continue_on_batch_fail": True,  # Continue training even if some batches fail
    "max_failed_batches": 10,        # Stop epoch if more than 10 batches fail
}