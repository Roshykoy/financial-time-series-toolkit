# src/config.py
"""
Unified Configuration for MarkSix Probabilistic Forecasting System
This file contains all configuration parameters organized by functional area.
"""
import torch

# =============================================================================
# CORE DATA AND BASIC PARAMETERS
# =============================================================================

CONFIG = {
    # Data Configuration
    "data_path": "data/raw/Mark_Six.csv",
    "num_lotto_numbers": 49,
    "scorer_types": ["generative", "temporal", "i_ching"],
    "num_scorers": 3,
    
    # =============================================================================
    # NEURAL NETWORK ARCHITECTURE PARAMETERS (REDUCED FOR STABILITY)
    # =============================================================================
    
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
    "hidden_dim": 128,               # Hidden dimension for CVAE
    "prior_hidden_dim": 128,         # Reduced from 256
    
    # Generative Decoder - REDUCED
    "decoder_hidden_dim": 128,       # Reduced from 256
    "decoder_layers": 2,             # Reduced from 3
    "decoder_projection_dim": 256,   # Reduced from 512
    "number_generation_hidden": 64,  # Reduced from 128
    "constraint_hidden_dim": 128,    # Reduced from 256
    
    # Meta-Learner Parameters - REDUCED
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
    
    # =============================================================================
    # TRAINING PARAMETERS - CONSERVATIVE
    # =============================================================================
    
    # Basic Training Settings
    "epochs": 10,                    # Reduced from 25 for testing
    "batch_size": 8,                 # Reduced from 32 for stability
    "learning_rate": 5e-5,           # Reduced from 1e-4 (HALF)
    "dropout": 0.1,                  # Reduced from 0.15
    "dropout_rate": 0.1,             # Alias for dropout (some modules use this name)
    "gradient_clip_norm": 0.5,       # Reduced from 1.0 (MORE AGGRESSIVE)
    "max_grad_norm": 0.5,           # Additional gradient clipping parameter
    "weight_decay": 1e-6,            # Much reduced from 1e-5
    
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
    
    # KL Annealing and Collapse Prevention
    "kl_annealing_epochs": 5,        # Number of epochs for KL annealing
    "kl_min_beta": 0.0,              # Starting KL weight (0 = no KL loss initially)
    "kl_max_beta": 1.0,              # Final KL weight (1 = full KL loss)
    "kl_min_value": 1e-6,            # Minimum KL value to prevent complete collapse
    
    # Advanced Training Options - CONSERVATIVE
    "use_mixed_precision": True,    # DISABLED - was causing overflow issues
    "use_scheduler": False,          # DISABLED for stability
    "scheduler_type": "none",        # Changed from "cosine"
    "warmup_epochs": 0,              # Disabled
    "ema_decay": 0,                  # DISABLED - was 0.999
    
    # Data Augmentation - DISABLED FOR STABILITY
    "augment_temporal_noise": 0.0,   # Disabled - was 0.1
    "augment_dropout_prob": 0.0,     # Disabled - was 0.1
    
    # Early Stopping and Validation
    "early_stopping": True,          # Enable early stopping
    "early_stopping_patience": 3,    # Stop if no improvement for 3 epochs
    "min_improvement": 0.001,        # Minimum improvement to consider
    "meta_learner_frequency": 20,    # Train meta-learner less frequently
    "validation_frequency": 2,       # Validate every 2 epochs instead of every epoch
    
    # =============================================================================
    # INFERENCE PARAMETERS
    # =============================================================================
    
    # Generation Settings
    "generation_temperature": 1.0,   # Increased from 0.8 (less confident)
    "num_generation_samples": 10,    # Reduced from 50
    "top_k_sampling": 5,             # Reduced from 10
    "ensemble_selection_method": "fixed_weights",  # Changed from "meta_learned" for stability
    
    # Evaluation Parameters
    "evaluation_neg_samples": 49,    # Reduced from 99
    "validation_generation_samples": 5,  # Reduced from 20
    
    # Legacy Heuristic Scorers (still used in ensemble)
    "ensemble_weights": {
        "generative": 0.7,    # Increased generative weight for stability
        "temporal": 0.2,      # Reduced temporal weight
        "i_ching": 0.1        # Reduced i_ching weight
    },
    "search_iterations": 0,    # Disabled - replaced by generation
    "search_neighbors": 0,     # Disabled - replaced by generation
    
    # =============================================================================
    # PHASE 1 PERFORMANCE OPTIMIZATIONS (APPROVED BY EXPERT PANEL)
    # =============================================================================
    
    # Master Switch
    "enable_performance_optimizations": True,  # Master switch for all optimizations
    
    # Asynchronous Data Pipeline Enhancement
    "num_workers": "auto",              # Auto-detect optimal worker count
    "pin_memory": "auto",               # Smart pin_memory based on CUDA + memory
    "persistent_workers": True,         # Keep workers alive between epochs
    "prefetch_factor": 4,              # Pre-load 4 batches ahead
    
    # Batch Size Optimization for VRAM Utilization
    "optimized_batch_size": "auto",     # Auto-calculate based on available VRAM
    "max_batch_size": 32,              # Safety limit for batch size scaling
    "vram_utilization_target": 0.80,   # Target 80% VRAM utilization
    
    # Production Configuration Settings
    "enable_mixed_precision": True,     # Re-enable with proper overflow handling
    "enable_torch_compile": True,       # PyTorch 2.0 model compilation
    "channels_last_memory": True,       # Memory layout optimization
    "gradient_checkpointing": False,    # Disabled for speed (vs memory trade-off)
    "cpu_offload": False,              # Keep on GPU for speed
    "memory_efficient_attention": True, # Optimize attention computation
    
    # =============================================================================
    # PHASE 2 MEDIUM-TERM IMPROVEMENTS (EXPERT PANEL APPROVED)
    # =============================================================================
    
    # Feature Engineering Parallelization
    "enable_parallel_features": True,          # Master switch for parallel feature processing
    "feature_parallel_workers": "auto",        # Auto-detect optimal workers for feature computation
    "feature_batch_threshold": 16,             # Minimum batch size for parallel processing
    "feature_vectorization": True,             # Enable vectorized feature computation
    "use_feature_threading": True,             # Use threading vs multiprocessing
    
    # Memory Pool Management System
    "enable_memory_pools": True,               # Master switch for memory pool management
    "tensor_pool_size_gb": 4.0,               # GPU tensor memory pool size
    "batch_cache_size_gb": 8.0,               # Batch data cache size
    "feature_cache_size_gb": 6.0,             # Feature vector cache size
    "enable_cache_compression": True,          # LZ4 compression for cached data
    "memory_pressure_threshold": 0.85,        # Memory usage threshold for cleanup
    
    # Dynamic Batch Size Enhancement  
    "enable_dynamic_batching": True,           # Dynamic batch size based on memory pressure
    "batch_size_scaling_factor": 3.5,         # Scaling factor for optimal batch sizes
    "max_dynamic_batch_size": 64,             # Safety limit for dynamic batch sizing
    "memory_aware_batching": True,            # Adjust batch size based on available memory
    
    # =============================================================================
    # DEVICE AND SYSTEM CONFIGURATION
    # =============================================================================
    
    # Device Configuration - STABILITY FOCUSED
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    
    # Memory Management
    "max_memory_fraction": 0.8,      # Use max 80% of GPU memory
    "clear_cache_frequency": 5,      # Clear CUDA cache every 5 batches
    
    # System Stability
    "numerical_stability_eps": 1e-8, # Small epsilon for numerical stability
    "fallback_to_cpu": True,         # Fallback to CPU if CUDA issues
    "checkpoint_on_error": True,     # Save checkpoint if training fails
    "continue_on_batch_fail": True,  # Continue training even if some batches fail
    "max_failed_batches": 10,        # Stop epoch if more than 10 batches fail
    
    # =============================================================================
    # FILE PATHS CONFIGURATION
    # =============================================================================
    
    # Model Paths
    "model_save_path": "models/conservative_cvae_model.pth",
    "meta_learner_save_path": "models/conservative_meta_learner.pth", 
    "feature_engineer_path": "models/conservative_feature_engineer.pkl",
    
    # New Path Structure (for future migration compatibility)
    "model_save_path_new": "models/active/marksix_model.pth",
    "meta_learner_save_path_new": "models/active/meta_learner.pth",
    "feature_engineer_path_new": "models/active/feature_engineer.pkl",
    "pareto_results_dir": "models/pareto_optimized",
    
    # =============================================================================
    # MONITORING AND DEBUGGING
    # =============================================================================
    
    # Logging and Monitoring
    "log_interval": 10,              # More frequent logging (was 50)
    "save_interval": 2,              # Save more frequently (was 5)
    "plot_latent_space": False,      # DISABLED to reduce overhead
    "save_generation_samples": False, # DISABLED to reduce overhead
    
    # Loss Monitoring
    "save_loss_plots": True,         # Generate detailed loss plots
    "loss_monitoring_enabled": True, # Enable comprehensive loss monitoring
}
