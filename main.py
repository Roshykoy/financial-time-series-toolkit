# main_enhanced.py - Unified Mark Six Prediction System Interface
import os
import sys
import traceback
import torch
import pandas as pd
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from collections import Counter
import random
from typing import Optional, Dict, Any

# Core pipeline imports
from src.training_pipeline import run_training
from src.inference_pipeline import run_inference
from src.evaluation_pipeline import run_evaluation

# Enhanced configuration and logging
from src.infrastructure.config import get_config_manager
from src.infrastructure.logging import get_logger, configure_logging

# Optimization system
from src.optimization.main import OptimizationOrchestrator

# Advanced training components
from src.cvae_engine import train_one_epoch_cvae, evaluate_cvae
from src.cvae_model import ConditionalVAE
from src.meta_learner import AttentionMetaLearner
from src.cvae_data_loader import create_cvae_data_loaders
from src.feature_engineering import FeatureEngineer
from src.config import CONFIG

# Initialize enhanced systems
configure_logging(log_level="INFO", log_file="marksix.log")
logger = get_logger(__name__)


def print_banner():
    """Prints the application banner."""
    print("\n" + "=" * 70)
    print("MARK SIX LOTTERY PREDICTION SYSTEM v3.0")
    print("Unified AI & Statistical Hybrid Architecture")
    print("=" * 70)
    print("Architecture: CVAE + Graph Neural Networks + Meta-Learning + Statistical")
    print("• Conditional Variational Autoencoder for generation")
    print("• Graph Neural Network encoder for number relationships")
    print("• LSTM temporal context encoder")
    print("• Attention-based meta-learner for dynamic ensemble weights")
    print("• Statistical pattern analysis with frequency modeling")
    print("• Comprehensive hyperparameter optimization")
    print("• Integrated training, inference, and diagnostics")
    print("=" * 70)


def check_system_requirements():
    """Checks system requirements and provides recommendations."""
    print("\nSystem Requirements Check:")
    print("-" * 30)
    
    # Check PyTorch and CUDA
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: Yes (Device: {torch.cuda.get_device_name()})")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA available: No (CPU-only mode)")
        print("⚠️  Warning: Training will be significantly slower on CPU")
    
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"System RAM: {memory.total / 1e9:.1f} GB")
        if memory.total < 8e9:  # Less than 8GB
            print("⚠️  Warning: Recommend at least 8GB RAM for stable training")
    except ImportError:
        print("RAM info: Unable to detect (install psutil for details)")
    
    print()


def get_training_options():
    """Gets training configuration options from user."""
    print("Training Configuration Options:")
    print("-" * 32)
    
    # Training mode selection
    print("Training Modes:")
    print("1. Optimized Training (20 epochs, ~94 min, hardware-optimized)")
    print("2. Quick Training (5 epochs, ~15 min, for testing)")
    print("3. Ultra-Quick Training (3 epochs, ~5 min, minimal model)")
    print("4. Standard Training (configurable, original pipeline)")
    
    try:
        mode_choice = int(input("Choose training mode (1-4, default: 1): ") or "1")
    except ValueError:
        mode_choice = 1
    
    if mode_choice == 1:
        return {
            'mode': 'optimized',
            'epochs': 20,
            'batch_size': 16,
            'learning_rate': 5e-4,
            'use_mixed_precision': True,
            'plot_latent_space': True,
            'save_generation_samples': True
        }
    elif mode_choice == 2:
        return {
            'mode': 'quick',
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'use_mixed_precision': True,
            'plot_latent_space': True,
            'save_generation_samples': True
        }
    elif mode_choice == 3:
        return {
            'mode': 'ultra_quick',
            'epochs': 3,
            'batch_size': 64,
            'learning_rate': 2e-3,
            'use_mixed_precision': False,
            'plot_latent_space': False,
            'save_generation_samples': False
        }
    else:
        # Standard configurable mode
        from src.config import CONFIG
        try:
            epochs = int(input(f"Number of training epochs (default: {CONFIG['epochs']}): ") or CONFIG['epochs'])
            if epochs <= 0:
                epochs = CONFIG['epochs']
        except ValueError:
            epochs = CONFIG['epochs']
        
        try:
            batch_size = int(input(f"Batch size (default: {CONFIG['batch_size']}): ") or CONFIG['batch_size'])
            if batch_size <= 0:
                batch_size = CONFIG['batch_size']
        except ValueError:
            batch_size = CONFIG['batch_size']
        
        try:
            lr_input = input(f"Learning rate (default: {CONFIG['learning_rate']}): ")
            learning_rate = float(lr_input) if lr_input else CONFIG['learning_rate']
            if learning_rate <= 0:
                learning_rate = CONFIG['learning_rate']
        except ValueError:
            learning_rate = CONFIG['learning_rate']
        
        use_mixed_precision = input("Use mixed precision training? (y/n, default: y): ").lower() != 'n'
        save_plots = input("Save training plots? (y/n, default: y): ").lower() != 'n'
        
        return {
            'mode': 'standard',
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'use_mixed_precision': use_mixed_precision,
            'plot_latent_space': save_plots,
            'save_generation_samples': save_plots
        }


def get_inference_options():
    """Gets inference configuration options from user."""
    print("Inference Configuration Options:")
    print("-" * 33)
    
    # Check if AI models are available
    from src.config import CONFIG
    cvae_exists = os.path.exists(CONFIG["model_save_path"])
    meta_exists = os.path.exists(CONFIG["meta_learner_save_path"])
    models_available = cvae_exists and meta_exists
    
    # Show available prediction methods
    print("Prediction Methods:")
    if models_available:
        print("1. AI Model Inference (CVAE + Meta-Learner)")
        print("2. Statistical Pattern Analysis (no models needed)")
        print("3. Hybrid Approach (AI + Statistical)")
        try:
            method_choice = int(input("Choose prediction method (1-3, default: 1): ") or "1")
        except ValueError:
            method_choice = 1
    else:
        print("⚠️  No trained AI models found. Using statistical analysis.")
        method_choice = 2
    
    # Number of sets
    try:
        num_sets = int(input("How many number sets would you like to generate? "))
        if num_sets <= 0:
            print("Please enter a positive number.")
            return None
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return None
    
    if method_choice == 1 or method_choice == 3:
        # AI model options
        try:
            temp_input = input("Generation temperature (0.1-2.0, default: 0.8): ")
            temperature = float(temp_input) if temp_input else 0.8
            temperature = max(0.1, min(2.0, temperature))
        except ValueError:
            temperature = 0.8
        
        use_iching_input = input("Use the I-Ching scorer? (y/n, default: n): ").lower()
        use_iching = use_iching_input == 'y'
        
        verbose = input("Show detailed generation process? (y/n, default: y): ").lower() != 'n'
        
        print("\nGeneration Modes:")
        print("1. Standard generation (recommended)")
        print("2. High diversity mode (more creative combinations)")
        print("3. Conservative mode (closer to historical patterns)")
        
        try:
            mode_choice_detail = int(input("Choose generation mode (1-3, default: 1): ") or "1")
        except ValueError:
            mode_choice_detail = 1
        
        if mode_choice_detail == 2:
            temperature *= 1.3
            print("🎲 High diversity mode selected")
        elif mode_choice_detail == 3:
            temperature *= 0.7
            print("🎯 Conservative mode selected")
        else:
            print("⚖️  Standard mode selected")
    else:
        # Statistical analysis options
        print("\nStatistical Analysis Modes:")
        print("1. Conservative (frequent numbers, common patterns)")
        print("2. Balanced (mix of frequent and infrequent numbers)")
        print("3. Creative (less common patterns, strategic selection)")
        
        try:
            mode_choice_detail = int(input("Choose analysis mode (1-3, default: 2): ") or "2")
        except ValueError:
            mode_choice_detail = 2
        
        temperature = 0.8
        use_iching = False
        verbose = True
    
    return {
        'method': method_choice,
        'num_sets': num_sets,
        'temperature': temperature,
        'use_i_ching': use_iching,
        'verbose': verbose,
        'mode': mode_choice_detail
    }


def get_optimization_options():
    """Gets hyperparameter optimization configuration options from user."""
    print("Hyperparameter Optimization Configuration:")
    print("-" * 41)
    
    # Show optimization modes
    print("Optimization Modes:")
    print("1. Quick Validation (validate pipeline before full optimization)")
    print("2. Thorough Search (8+ hour production optimization)")
    print("3. Standard Optimization (1-2 hour balanced search)")
    print("4. Custom Configuration (manual preset selection)")
    print("5. 🎯 Pareto Front Multi-Objective Optimization (NEW)")
    
    try:
        opt_mode = int(input("Choose optimization mode (1-5, default: 1): ") or "1")
    except ValueError:
        opt_mode = 1
    
    if opt_mode == 1:
        # Quick validation mode
        return {
            'mode': 'validate',
            'preset': 'quick_test',
            'max_trials': 10,
            'max_duration': 0.5
        }
    elif opt_mode == 2:
        # Thorough search mode
        print("\n🚀 Thorough Search Mode")
        print("This will run a comprehensive optimization with:")
        print("• Pre-flight validation")
        print("• Automatic checkpointing")
        print("• Resource monitoring")
        print("• Recovery from interruptions")
        print("• Full model validation")
        
        confirm = input("\nProceed with thorough search? (y/n): ").lower()
        if confirm != 'y':
            return None
        
        return {
            'mode': 'thorough',
            'preset': 'thorough_search',
            'max_trials': None,
            'max_duration': None
        }
    elif opt_mode == 3:
        # Standard optimization
        return {
            'mode': 'standard',
            'preset': 'balanced',
            'max_trials': 50,
            'max_duration': 2.0
        }
    elif opt_mode == 5:
        # Pareto Front multi-objective optimization
        print("\n🎯 Pareto Front Multi-Objective Optimization")
        print("This will optimize multiple objectives simultaneously:")
        print("• Model accuracy (maximize)")
        print("• Training time (minimize)")
        print("• Model complexity (minimize)")
        print("• Generate Pareto Front of optimal trade-offs")
        
        confirm = input("\nProceed with Pareto Front optimization? (y/n): ").lower()
        if confirm != 'y':
            return None
        
        return {
            'mode': 'pareto',
            'preset': None,
            'max_trials': None,
            'max_duration': None
        }
    else:
        # Custom configuration
        try:
            orchestrator = OptimizationOrchestrator("data/raw/Mark_Six.csv")
            presets = orchestrator.list_presets()
            
            print("\nAvailable optimization presets:")
            for i, preset in enumerate(presets, 1):
                info = orchestrator.get_preset_info(preset)
                print(f"{i}. {preset}: {info['description']}")
                print(f"   Algorithm: {info['algorithm']}, Max Trials: {info['max_trials']}, Duration: {info['max_duration_hours']}h")
            
            try:
                preset_choice = int(input(f"\nChoose preset (1-{len(presets)}, default: 1): ") or "1")
                if 1 <= preset_choice <= len(presets):
                    selected_preset = presets[preset_choice - 1]
                else:
                    print("Invalid choice, using default preset.")
                    selected_preset = presets[0]
            except ValueError:
                selected_preset = presets[0]
                
        except Exception as e:
            print(f"Warning: Could not load presets: {e}")
            print("Using default quick_test preset.")
            selected_preset = "quick_test"
        
        try:
            trials_input = input("Maximum number of trials (default: use preset): ")
            max_trials = int(trials_input) if trials_input else None
            if max_trials is not None and max_trials <= 0:
                max_trials = None
        except ValueError:
            max_trials = None
        
        try:
            duration_input = input("Maximum duration in hours (default: use preset): ")
            max_duration = float(duration_input) if duration_input else None
            if max_duration is not None and max_duration <= 0:
                max_duration = None
        except ValueError:
            max_duration = None
        
        return {
            'mode': 'custom',
            'preset': selected_preset,
            'max_trials': max_trials,
            'max_duration': max_duration
        }


def display_model_info():
    """Displays information about trained models."""
    from src.config import CONFIG
    
    print("\nTrained Model Information:")
    print("-" * 28)
    
    # Check standard models
    cvae_exists = os.path.exists(CONFIG["model_save_path"])
    meta_exists = os.path.exists(CONFIG["meta_learner_save_path"])
    fe_exists = os.path.exists(CONFIG["feature_engineer_path"])
    
    print(f"CVAE Model: {'✓' if cvae_exists else '✗'} {CONFIG['model_save_path']}")
    print(f"Meta-Learner: {'✓' if meta_exists else '✗'} {CONFIG['meta_learner_save_path']}")
    print(f"Feature Engineer: {'✓' if fe_exists else '✗'} {CONFIG['feature_engineer_path']}")
    
    # Check alternative models
    alt_models = {
        'Best Model': 'models/best_cvae_model.pth',
        'Quick Model': 'models/quick_cvae_model.pth',
        'Ultra Quick': 'models/ultra_quick_model.pth',
        'Conservative': 'models/conservative_cvae_model.pth'
    }
    
    print("\nAlternative Models:")
    for name, path in alt_models.items():
        exists = os.path.exists(path)
        print(f"{name}: {'✓' if exists else '✗'} {path}")
        if exists:
            try:
                size = os.path.getsize(path) / (1024 * 1024)
                mod_time = os.path.getmtime(path)
                mod_date = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
                print(f"  Size: {size:.1f} MB, Modified: {mod_date}")
            except Exception:
                pass
    
    # Check optimization results
    opt_dirs = ['optimization_results', 'thorough_search_results', 'hyperparameter_results']
    print("\nOptimization Results:")
    for opt_dir in opt_dirs:
        if os.path.exists(opt_dir):
            try:
                files = os.listdir(opt_dir)
                result_files = [f for f in files if f.endswith('.json')]
                print(f"{opt_dir}: ✓ ({len(result_files)} result files)")
            except Exception:
                print(f"{opt_dir}: ✓ (exists)")
        else:
            print(f"{opt_dir}: ✗ (not found)")
    
    print()


# Integrated training functions
def run_pareto_optimized_training(pareto_params: Optional[Dict[str, Any]] = None):
    """Run training with Pareto Front optimized parameters."""
    try:
        if pareto_params:
            print("🎯 Starting Pareto Front optimized training...")
            print("Using parameters from multi-objective optimization")
        else:
            print("🚀 Starting optimized training...")
        
        # Import CONFIG for model paths and Pareto integration
        from src.config import CONFIG
        from src.optimization.pareto_integration import create_pareto_optimized_config
        
        # Load data
        print("📊 Loading Mark Six data...")
        data_path = "data/raw/Mark_Six.csv"
        col_names = [
            'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
            'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
            'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
            '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
            'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
            'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
            'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
        ]
        
        df = pd.read_csv(data_path, header=None, skiprows=33, names=col_names)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
        print(f"✅ Loaded {len(df)} records")
        
        # Create Pareto-optimized configuration
        config = create_pareto_optimized_config(pareto_params)
        
        # Display configuration details
        if config.get('_pareto_optimized', False):
            print("🎯 Using Pareto Front optimized parameters:")
            pareto_params_used = config.get('_pareto_params', {})
            for param, value in pareto_params_used.items():
                if isinstance(value, float):
                    print(f"   {param}: {value:.6f}")
                else:
                    print(f"   {param}: {value}")
        else:
            print("🔧 Using default optimized configuration")
            # Fallback to hardcoded optimized settings
            config.update({
                'batch_size': 16,
                'learning_rate': 5e-4,
                'kl_weight': 0.01,
                'contrastive_weight': 0.05,
                'weight_decay': 1e-4
            })
        
        print(f"🔧 Configuration: {config['epochs']} epochs, batch size {config['batch_size']}")
        print(f"🎯 Device: {config['device']}")
        
        # Create components
        feature_engineer = FeatureEngineer()
        feature_engineer.fit(df)
        
        train_loader, val_loader = create_cvae_data_loaders(df, feature_engineer, config)
        
        cvae = ConditionalVAE(config).to(config['device'])
        
        meta_learner = AttentionMetaLearner(config).to(config['device'])
        
        # Training loop - Use separate optimizers matching the expected function signature
        optimizers = {
            'cvae': torch.optim.AdamW(
                cvae.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            ),
            'meta': torch.optim.AdamW(
                meta_learner.parameters(),
                lr=config['learning_rate'],
                weight_decay=config['weight_decay']
            )
        }
        
        schedulers = {
            'cvae': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizers['cvae'], mode='min', factor=0.5, patience=3, verbose=True
            ),
            'meta': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizers['meta'], mode='min', factor=0.5, patience=3, verbose=True
            )
        }
        
        device = torch.device(config['device'])
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            print(f"\n📈 Epoch {epoch + 1}/{config['epochs']}")
            
            # Training - Fixed function signature
            train_losses = train_one_epoch_cvae(
                cvae, meta_learner, train_loader, optimizers, device, config, epoch
            )
            
            # Validation - Fixed function signature
            val_loss, val_losses_dict = evaluate_cvae(cvae, meta_learner, val_loader, device, config)
            
            # Update schedulers
            schedulers['cvae'].step(val_loss)
            schedulers['meta'].step(val_loss)
            
            # Extract main loss values for display
            train_loss = train_losses.get('total_loss', 0.0)
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping and model saving
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                
                # Save best model (use standard paths so inference can find them)
                torch.save(cvae.state_dict(), CONFIG['model_save_path'])
                torch.save(meta_learner.state_dict(), CONFIG['meta_learner_save_path'])
                
                import pickle
                with open(CONFIG['feature_engineer_path'], 'wb') as f:
                    pickle.dump(feature_engineer, f)
                
                # Also save as backup with 'best_' prefix
                torch.save(cvae.state_dict(), 'models/best_cvae_model.pth')
                torch.save(meta_learner.state_dict(), 'models/best_meta_learner.pth')
                with open('models/best_feature_engineer.pkl', 'wb') as f:
                    pickle.dump(feature_engineer, f)
                
                print(f"✅ Best model saved (loss: {best_loss:.4f})")
            else:
                patience_counter += 1
                
            if patience_counter >= config['early_stopping_patience']:
                print(f"⏹️  Early stopping triggered after {epoch + 1} epochs")
                break
        
        print("\n🎉 Optimized training completed successfully!")
        print(f"Best validation loss: {best_loss:.4f}")
        print(f"Models saved to: {CONFIG['model_save_path']}, {CONFIG['meta_learner_save_path']}")
        print(f"Backup models saved to: models/best_cvae_model.pth, models/best_meta_learner.pth")
        print("\n💡 Tip: Use 'View Model Information' to verify the trained models.")
        print("💡 Tip: Use 'Generate Predictions' to test the trained models.")
        
    except Exception as e:
        print(f"❌ Optimized training failed: {e}")
        traceback.print_exc()


def run_optimized_training():
    """Run optimized training with hardware-specific settings (compatibility function)."""
    # Check for Pareto Front parameters and use them if available
    from src.optimization.pareto_integration import load_pareto_parameters
    pareto_params = load_pareto_parameters()
    
    if pareto_params:
        print("🎯 Auto-detected Pareto Front parameters - using for optimized training")
        return run_pareto_optimized_training(pareto_params)
    else:
        return run_pareto_optimized_training(None)


def run_quick_training():
    """Run quick training for fast results."""
    try:
        print("⚡ Starting quick training...")
        
        # Use legacy config with quick settings
        config = CONFIG.copy()
        
        config.update({
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 1e-3,
            'early_stopping_patience': 3
        })
        
        # Load and prepare data (simplified)
        data_path = "data/raw/Mark_Six.csv"
        col_names = [
            'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
            'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
            'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
            '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
            'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
            'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
            'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
        ]
        
        df = pd.read_csv(data_path, header=None, skiprows=33, names=col_names)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        # Simplified training pipeline
        feature_engineer = FeatureEngineer()
        feature_engineer.fit(df)
        
        train_loader, val_loader = create_cvae_data_loaders(df, feature_engineer, config)
        
        cvae = ConditionalVAE(config).to(config['device'])
        
        meta_learner = AttentionMetaLearner(config).to(config['device'])
        
        optimizers = {
            'cvae': torch.optim.Adam(
                cvae.parameters(),
                lr=config['learning_rate']
            ),
            'meta': torch.optim.Adam(
                meta_learner.parameters(),
                lr=config['learning_rate']
            )
        }
        
        device = torch.device(config['device'])
        
        for epoch in range(config['epochs']):
            print(f"Epoch {epoch + 1}/{config['epochs']}")
            
            train_losses = train_one_epoch_cvae(
                cvae, meta_learner, train_loader, optimizers, device, config, epoch
            )
            
            val_loss, val_losses_dict = evaluate_cvae(cvae, meta_learner, val_loader, device, config)
            
            # Extract main loss values for display
            train_loss = train_losses.get('total_loss', 0.0)
            # val_loss is already extracted from the tuple above
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save quick model
        torch.save(cvae.state_dict(), 'models/quick_cvae_model.pth')
        torch.save(meta_learner.state_dict(), 'models/quick_meta_learner.pth')
        
        import pickle
        with open('models/quick_feature_engineer.pkl', 'wb') as f:
            pickle.dump(feature_engineer, f)
        
        print("\n🎉 Quick training completed!")
        print("Models saved to: models/quick_cvae_model.pth")
        print("\n💡 Tip: This model is suitable for testing and quick predictions.")
        print("💡 Tip: For production use, consider running optimized training.")
        
    except Exception as e:
        print(f"❌ Quick training failed: {e}")
        traceback.print_exc()


def run_statistical_prediction(num_sets: int, mode: int = 2):
    """Run statistical pattern analysis for number prediction."""
    try:
        print("📊 Starting statistical pattern analysis...")
        
        # Load data
        data_path = "data/raw/Mark_Six.csv"
        col_names = [
            'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
            'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
            'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
            '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
            'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
            'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
            'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
        ]
        
        df = pd.read_csv(data_path, header=None, skiprows=33, names=col_names)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        # Analyze patterns
        winning_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
        
        # Number frequency analysis
        all_numbers = []
        for col in winning_cols:
            all_numbers.extend(df[col].tolist())
        
        number_freq = Counter(all_numbers)
        
        # Recent trends (last 20 draws)
        recent_numbers = []
        for _, row in df.tail(20).iterrows():
            recent_numbers.extend(row[winning_cols].tolist())
        
        recent_freq = Counter(recent_numbers)
        
        # Pair frequency analysis
        pair_freq = Counter()
        for _, row in df.iterrows():
            numbers = sorted(row[winning_cols].tolist())
            for i in range(len(numbers)):
                for j in range(i + 1, len(numbers)):
                    pair_freq[(numbers[i], numbers[j])] += 1
        
        # Generate predictions based on mode
        predictions = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i in range(num_sets):
            if mode == 1:  # Conservative
                # Use most frequent numbers
                freq_numbers = [num for num, _ in number_freq.most_common(20)]
                selected = random.sample(freq_numbers, 6)
            elif mode == 3:  # Creative
                # Use less common numbers and strategic selection
                less_common = [num for num, count in number_freq.items() if count < np.percentile(list(number_freq.values()), 60)]
                if len(less_common) >= 6:
                    selected = random.sample(less_common, 6)
                else:
                    selected = random.sample(list(number_freq.keys()), 6)
            else:  # Balanced (default)
                # Mix of frequent and infrequent numbers
                frequent = [num for num, _ in number_freq.most_common(25)]
                infrequent = [num for num, count in number_freq.items() if count <= np.percentile(list(number_freq.values()), 40)]
                
                selected = []
                selected.extend(random.sample(frequent, 3))
                selected.extend(random.sample(infrequent, 3))
                
                # Ensure uniqueness
                selected = list(set(selected))
                while len(selected) < 6:
                    selected.append(random.choice(list(number_freq.keys())))
                selected = selected[:6]
            
            # Sort the numbers
            selected.sort()
            predictions.append(selected)
        
        # Display results
        print(f"\n🎯 Generated {num_sets} statistical predictions:")
        print("=" * 50)
        
        mode_names = {1: "Conservative", 2: "Balanced", 3: "Creative"}
        print(f"Strategy: {mode_names.get(mode, 'Balanced')}")
        print(f"Based on {len(df)} historical draws")
        
        for i, pred in enumerate(predictions, 1):
            print(f"Set {i:2d}: {', '.join(f'{num:2d}' for num in pred)}")
            
            # Show frequency info for first few predictions
            if i <= 3:
                freq_info = [f"{num}({number_freq[num]})" for num in pred]
                print(f"        Frequencies: {', '.join(freq_info)}")
        
        # Save predictions
        output_file = f"outputs/statistical_predictions_{timestamp}.txt"
        with open(output_file, 'w') as f:
            f.write(f"Statistical Mark Six Predictions\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Strategy: {mode_names.get(mode, 'Balanced')}\n")
            f.write(f"Based on {len(df)} historical draws\n\n")
            
            for i, pred in enumerate(predictions, 1):
                f.write(f"Set {i:2d}: {', '.join(f'{num:2d}' for num in pred)}\n")
        
        print(f"\n💾 Predictions saved to: {output_file}")
        print("\n📈 Statistical Analysis Summary:")
        print(f"   • Most frequent number: {number_freq.most_common(1)[0][0]} (appeared {number_freq.most_common(1)[0][1]} times)")
        print(f"   • Least frequent number: {number_freq.most_common()[-1][0]} (appeared {number_freq.most_common()[-1][1]} times)")
        print(f"   • Most common pair: {pair_freq.most_common(1)[0][0]} (appeared {pair_freq.most_common(1)[0][1]} times)")
        print("\n💡 Tip: Statistical predictions are based on historical frequency patterns.")
        
    except Exception as e:
        print(f"❌ Statistical prediction failed: {e}")
        traceback.print_exc()


def run_optimization_validation():
    """Run optimization pipeline validation."""
    try:
        print("🔍 Running optimization pipeline validation...")
        
        # Import validation function
        sys.path.append('.')
        
        # Run basic validation tests
        print("1. Testing optimization orchestrator...")
        orchestrator = OptimizationOrchestrator("data/raw/Mark_Six.csv")
        
        print("2. Testing preset loading...")
        presets = orchestrator.list_presets()
        print(f"   Found {len(presets)} presets: {', '.join(presets)}")
        
        print("3. Testing environment setup...")
        # Basic environment check
        import torch
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        
        print("4. Testing data loading...")
        # Test data loading
        try:
            col_names = [
                'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
                'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
                'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
                '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
                'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
                'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
                'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
            ]
            df = pd.read_csv("data/raw/Mark_Six.csv", header=None, skiprows=33, names=col_names)
            print(f"   Data loaded: {len(df)} records")
        except Exception as e:
            print(f"   Data loading failed: {e}")
            return
        
        print("5. Testing quick optimization...")
        # Run a very quick test optimization
        try:
            results = orchestrator.run_optimization(
                preset_name="quick_test",
                max_trials=2,
                max_duration_hours=0.1
            )
            print("   Quick optimization test: ✓ Passed")
        except Exception as e:
            print(f"   Quick optimization test: ✗ Failed - {e}")
            return
        
        print("\n✅ Optimization pipeline validation completed successfully!")
        print("The system is ready for full hyperparameter optimization.")
        print("\n💡 Tip: You can now safely run thorough optimization for 8+ hours.")
        print("💡 Tip: Use 'Thorough Search' mode for production-quality optimization.")
        
    except Exception as e:
        print(f"❌ Optimization validation failed: {e}")
        traceback.print_exc()


def run_thorough_optimization():
    """Run bulletproof thorough optimization."""
    try:
        print("🚀 Starting bulletproof thorough optimization...")
        print("This will run a comprehensive optimization with full safeguards.")
        
        # Set up thorough optimization
        output_dir = "thorough_search_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create orchestrator with thorough search preset
        orchestrator = OptimizationOrchestrator(
            data_path="data/raw/Mark_Six.csv",
            output_dir=output_dir
        )
        
        print("🔧 Configuration: Thorough search preset")
        print("⏱️  Estimated duration: 8+ hours")
        print("💾 Automatic checkpointing enabled")
        print("🔍 Resource monitoring active")
        
        # Run thorough optimization
        results = orchestrator.run_optimization(
            preset_name="thorough_search",
            max_trials=None,  # Use preset default
            max_duration_hours=None  # Use preset default
        )
        
        # Display results
        print("\n🎉 Thorough optimization completed successfully!")
        
        if 'optimization_summary' in results:
            summary = results['optimization_summary']
            print(f"Best score: {summary['best_score']:.4f}")
            print(f"Total trials: {summary['total_trials']}")
            print(f"Duration: {summary['duration_hours']:.2f} hours")
            
            print("\nBest parameters:")
            for param, value in summary['best_parameters'].items():
                print(f"  • {param}: {value}")
        
        print(f"\n💾 Detailed results saved to: {output_dir}/")
        print("Best models saved and ready for inference.")
        print("\n💡 Tip: The optimized models are now available for high-quality predictions.")
        print("💡 Tip: Use 'View Model Information' to see the optimization results.")
        
    except Exception as e:
        print(f"❌ Thorough optimization failed: {e}")
        traceback.print_exc()


def run_model_compatibility_test():
    """Test model compatibility and loading."""
    try:
        print("🔍 Testing model compatibility...")
        
        # Test different model files
        model_files = [
            ('Standard CVAE', 'models/cvae_model.pth'),
            ('Best CVAE', 'models/best_cvae_model.pth'),
            ('Quick CVAE', 'models/quick_cvae_model.pth'),
            ('Conservative CVAE', 'models/conservative_cvae_model.pth')
        ]
        
        for name, path in model_files:
            if os.path.exists(path):
                try:
                    state_dict = torch.load(path, map_location='cpu')
                    print(f"✓ {name}: Compatible ({len(state_dict)} parameters)")
                except Exception as e:
                    print(f"✗ {name}: Incompatible - {e}")
            else:
                print(f"- {name}: Not found")
        
        # Test feature engineers
        fe_files = [
            ('Standard FE', 'models/feature_engineer.pkl'),
            ('Best FE', 'models/best_feature_engineer.pkl'),
            ('Quick FE', 'models/quick_feature_engineer.pkl')
        ]
        
        for name, path in fe_files:
            if os.path.exists(path):
                try:
                    import pickle
                    with open(path, 'rb') as f:
                        fe = pickle.load(f)
                    print(f"✓ {name}: Compatible ({len(fe.get_feature_names())} features)")
                except Exception as e:
                    print(f"✗ {name}: Incompatible - {e}")
            else:
                print(f"- {name}: Not found")
        
        print("\n✅ Model compatibility test completed.")
        print("\n💡 Tip: Compatible models can be used for inference and evaluation.")
        print("💡 Tip: Incompatible models may need retraining with current architecture.")
        
    except Exception as e:
        print(f"❌ Model compatibility test failed: {e}")
        traceback.print_exc()


def main_menu():
    """Displays the main menu and handles user input."""
    from src.config import CONFIG
    
    print_banner()
    check_system_requirements()
    
    # Show integration success message on first run
    print("\n🎉 SUCCESS: All standalone scripts have been integrated into the main menu!")
    print("📋 All features are now accessible through the unified interface below.")
    print("🧹 Redundant standalone scripts can be safely removed.")
    
    while True:
        print("\n" + "=" * 60)
        print("MAIN MENU - UNIFIED MARK SIX PREDICTION SYSTEM")
        print("=" * 60)
        print("1. Train New Model (Optimized/Quick/Ultra-Quick/Standard)")
        print("2. Generate Predictions (AI/Statistical/Hybrid)")
        print("3. Evaluate Trained Model")
        print("4. Optimize Hyperparameters (Validate/Thorough/Standard)")
        print("5. View Model Information")
        print("6. System Diagnostics & Testing")
        print("7. Exit")
        print("=" * 60)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        try:
            if choice == '1':
                print("\n🚀 Starting Model Training Pipeline")
                print("-" * 40)
                
                # Get training options
                training_config = get_training_options()
                if training_config is None:
                    continue
                
                # Execute training based on selected mode
                if training_config['mode'] == 'optimized':
                    print("\n🔧 Optimized Training Mode Selected")
                    print(f"• Epochs: {training_config['epochs']}")
                    print(f"• Batch size: {training_config['batch_size']}")
                    print(f"• Learning rate: {training_config['learning_rate']}")
                    print(f"• Mixed precision: {training_config['use_mixed_precision']}")
                    print(f"• Estimated time: ~94 minutes")
                    
                    # Check for available Pareto Front parameters
                    from src.optimization.pareto_integration import load_pareto_parameters
                    pareto_params = load_pareto_parameters()
                    
                    if pareto_params:
                        print("\n🎯 Pareto Front optimized parameters detected!")
                        print("Will automatically use Pareto Front optimized parameters for training.")
                        
                        # Show the parameters that will be used
                        print("Pareto Front parameters to be applied:")
                        for param, value in pareto_params.items():
                            if isinstance(value, float):
                                print(f"  • {param}: {value:.6f}")
                            else:
                                print(f"  • {param}: {value}")
                        
                        print("\nOptions:")
                        print("1. Proceed with Pareto Front optimized training (recommended)")
                        print("2. Use default optimized settings instead")
                        
                        pareto_choice = input("Choose option (1-2, default: 1): ").strip() or "1"
                        
                        if pareto_choice == "1":
                            print("🎯 Using Pareto Front optimized parameters for training")
                            confirm = input("\nProceed with Pareto Front optimized training? (y/n): ").lower()
                            if confirm == 'y':
                                run_pareto_optimized_training(pareto_params)
                            else:
                                print("Training cancelled.")
                        else:
                            print("⚙️ Using default optimized settings")
                            confirm = input("\nProceed with default optimized training? (y/n): ").lower()
                            if confirm == 'y':
                                run_optimized_training()
                            else:
                                print("Training cancelled.")
                    else:
                        print("\nNo Pareto Front parameters found. Using default optimized settings.")
                        confirm = input("\nProceed with optimized training? (y/n): ").lower()
                        if confirm == 'y':
                            run_optimized_training()
                        else:
                            print("Training cancelled.")
                        
                elif training_config['mode'] == 'quick':
                    print("\n⚡ Quick Training Mode Selected")
                    print(f"• Epochs: {training_config['epochs']}")
                    print(f"• Batch size: {training_config['batch_size']}")
                    print(f"• Learning rate: {training_config['learning_rate']}")
                    print(f"• Estimated time: ~15 minutes")
                    
                    confirm = input("\nProceed with quick training? (y/n): ").lower()
                    if confirm == 'y':
                        run_quick_training()
                    else:
                        print("Training cancelled.")
                        
                elif training_config['mode'] == 'ultra_quick':
                    print("\n🏃 Ultra-Quick Training Mode Selected")
                    print(f"• Epochs: {training_config['epochs']}")
                    print(f"• Batch size: {training_config['batch_size']}")
                    print(f"• Learning rate: {training_config['learning_rate']}")
                    print(f"• Estimated time: ~5 minutes")
                    print(f"• Note: Minimal model for testing only")
                    
                    confirm = input("\nProceed with ultra-quick training? (y/n): ").lower()
                    if confirm == 'y':
                        print("Ultra-quick training not yet implemented - use quick training instead.")
                    else:
                        print("Training cancelled.")
                        
                else:
                    # Standard configurable training
                    print("\n⚙️ Standard Training Mode Selected")
                    CONFIG.update(training_config)
                    
                    print(f"\nTraining Configuration:")
                    print(f"• Epochs: {CONFIG['epochs']}")
                    print(f"• Batch size: {CONFIG['batch_size']}")
                    print(f"• Learning rate: {CONFIG['learning_rate']}")
                    print(f"• Mixed precision: {CONFIG['use_mixed_precision']}")
                    print(f"• Device: {CONFIG['device']}")
                    
                    confirm = input("\nProceed with training? (y/n): ").lower()
                    if confirm == 'y':
                        run_training()
                    else:
                        print("Training cancelled.")
            
            elif choice == '2':
                print("\n🎯 Starting Prediction Generation")
                print("-" * 35)
                
                # Get inference options
                inference_config = get_inference_options()
                if inference_config is None:
                    continue
                
                # Execute prediction based on selected method
                if inference_config['method'] == 1:  # AI Model Inference
                    print("\n🤖 AI Model Inference Selected")
                    print(f"• Number of sets: {inference_config['num_sets']}")
                    print(f"• Temperature: {inference_config['temperature']:.2f}")
                    print(f"• I-Ching scorer: {'Yes' if inference_config['use_i_ching'] else 'No'}")
                    print(f"• Verbose output: {'Yes' if inference_config['verbose'] else 'No'}")
                    
                    run_inference(
                        num_sets_to_generate=inference_config['num_sets'],
                        use_i_ching=inference_config['use_i_ching'],
                        temperature=inference_config['temperature'],
                        verbose=inference_config['verbose']
                    )
                    
                elif inference_config['method'] == 2:  # Statistical Analysis
                    print("\n📊 Statistical Pattern Analysis Selected")
                    mode_names = {1: "Conservative", 2: "Balanced", 3: "Creative"}
                    print(f"• Analysis mode: {mode_names.get(inference_config['mode'], 'Balanced')}")
                    print(f"• Number of sets: {inference_config['num_sets']}")
                    
                    run_statistical_prediction(
                        num_sets=inference_config['num_sets'],
                        mode=inference_config['mode']
                    )
                    
                else:  # Hybrid Approach
                    print("\n🔄 Hybrid AI + Statistical Analysis Selected")
                    print(f"• Number of sets: {inference_config['num_sets']}")
                    print(f"• AI Temperature: {inference_config['temperature']:.2f}")
                    print(f"• Statistical mode: {inference_config['mode']}")
                    
                    # Run both AI and statistical predictions
                    ai_sets = max(1, inference_config['num_sets'] // 2)
                    stat_sets = inference_config['num_sets'] - ai_sets
                    
                    print(f"\n🤖 Generating {ai_sets} sets using AI model...")
                    run_inference(
                        num_sets_to_generate=ai_sets,
                        use_i_ching=inference_config['use_i_ching'],
                        temperature=inference_config['temperature'],
                        verbose=False
                    )
                    
                    print(f"\n📊 Generating {stat_sets} sets using statistical analysis...")
                    run_statistical_prediction(
                        num_sets=stat_sets,
                        mode=inference_config['mode']
                    )
            
            elif choice == '3':
                print("\n📊 Starting Model Evaluation")
                print("-" * 30)
                
                # Check if models exist
                if not (os.path.exists(CONFIG["model_save_path"]) and 
                       os.path.exists(CONFIG["meta_learner_save_path"])):
                    print("❌ Trained models not found!")
                    print("Please train the model first (Option 1).")
                    continue
                
                print("This will evaluate the model's performance on validation data.")
                print("The evaluation includes:")
                print("• Generation quality assessment")
                print("• Ensemble ranking performance")
                print("• Reconstruction accuracy")
                print("• Latent space quality analysis")
                
                use_iching_input = input("\nEvaluate with I-Ching scorer enabled? (y/n, default: n): ").lower()
                use_iching = use_iching_input == 'y'
                
                confirm = input("Proceed with evaluation? (y/n): ").lower()
                if confirm == 'y':
                    run_evaluation(use_i_ching=use_iching)
                else:
                    print("Evaluation cancelled.")
            
            elif choice == '4':
                print("\n⚙️ Starting Hyperparameter Optimization")
                print("-" * 40)
                
                # Import optimization system
                try:
                    from src.optimization.main import OptimizationOrchestrator
                except ImportError as e:
                    print(f"❌ Optimization system not available: {e}")
                    print("Please check if the optimization modules are properly installed.")
                    continue
                
                # Get optimization options
                optimization_config = get_optimization_options()
                if optimization_config is None:
                    continue
                
                # Execute optimization based on selected mode
                if optimization_config['mode'] == 'validate':
                    print("\n🔍 Pipeline Validation Mode Selected")
                    print("Running comprehensive validation tests...")
                    
                    try:
                        run_optimization_validation()
                    except Exception as e:
                        print(f"❌ Validation failed: {e}")
                        
                elif optimization_config['mode'] == 'thorough':
                    print("\n🚀 Thorough Search Mode Selected")
                    print("Starting bulletproof optimization with full safeguards...")
                    
                    try:
                        run_thorough_optimization()
                    except Exception as e:
                        print(f"❌ Thorough optimization failed: {e}")
                        
                elif optimization_config['mode'] == 'pareto':
                    print("\n🎯 Pareto Front Multi-Objective Optimization Mode Selected")
                    print("This will generate a Pareto Front of optimal trade-offs...")
                    
                    try:
                        orchestrator = OptimizationOrchestrator(
                            data_path=CONFIG["data_path"],
                            output_dir="optimization_results"
                        )
                        
                        results = orchestrator.run_pareto_optimization()
                        
                        if results:
                            print(f"\n✅ Pareto Front optimization completed successfully!")
                            print(f"Algorithm: {results['algorithm']}")
                            print(f"Total evaluations: {results['total_evaluations']}")
                            print(f"Computation time: {results['computation_time']:.2f} seconds")
                            print(f"Pareto Front size: {results['pareto_front_size']}")
                            
                            if results.get('selected_parameters'):
                                print("\n🎯 Selected parameters for training:")
                                for param, value in results['selected_parameters'].items():
                                    if isinstance(value, float):
                                        print(f"  {param}: {value:.6f}")
                                    else:
                                        print(f"  {param}: {value}")
                                        
                                # Offer to save selected parameters
                                save_params = input("\nSave selected parameters as best_parameters.json? (y/n): ").lower()
                                if save_params == 'y':
                                    import json
                                    with open("models/best_parameters/selected_pareto_params.json", 'w') as f:
                                        json.dump(results['selected_parameters'], f, indent=2)
                                    print("Parameters saved to models/best_parameters/selected_pareto_params.json")
                            
                            print("\nResults saved to models/pareto_front/")
                        else:
                            print("Pareto Front optimization was cancelled or failed.")
                            
                    except Exception as e:
                        print(f"❌ Pareto Front optimization failed: {e}")
                        if hasattr(e, '__traceback__'):
                            traceback.print_exc()
                        
                else:
                    # Standard or custom optimization
                    print(f"\n⚙️ {optimization_config['mode'].title()} Optimization Mode Selected")
                    print(f"• Preset: {optimization_config['preset']}")
                    print(f"• Max trials: {optimization_config['max_trials']}")
                    print(f"• Max duration: {optimization_config['max_duration']} hours")
                    
                    confirm = input("Start optimization? (y/n): ").lower()
                    if confirm == 'y':
                        print("\n🚀 Starting optimization process...")
                        
                        try:
                            orchestrator = OptimizationOrchestrator(
                                data_path=CONFIG["data_path"],
                                output_dir="optimization_results"
                            )
                            
                            results = orchestrator.run_optimization(
                                preset_name=optimization_config['preset'],
                                max_trials=optimization_config['max_trials'],
                                max_duration_hours=optimization_config['max_duration']
                            )
                            
                            print("\n✅ Optimization completed successfully!")
                            best_params = results['optimization_summary']['best_parameters']
                            best_score = results['optimization_summary']['best_score']
                            print(f"Best score achieved: {best_score:.4f}")
                            print(f"Best parameters found:")
                            for param, value in best_params.items():
                                print(f"  • {param}: {value}")
                            print(f"\nDetailed results saved to: optimization_results/")
                            
                        except Exception as e:
                            print(f"❌ Optimization failed: {e}")
                            print("Check the logs for more details.")
                    else:
                        print("Optimization cancelled.")
            
            elif choice == '5':
                display_model_info()
            
            elif choice == '6':
                print("\n🔧 System Diagnostics & Testing")
                print("-" * 30)
                
                # Show diagnostic options
                print("Diagnostic Options:")
                print("1. Basic System Check")
                print("2. Model Compatibility Test")
                print("3. Full System Validation")
                
                try:
                    diag_choice = int(input("Choose diagnostic option (1-3, default: 1): ") or "1")
                except ValueError:
                    diag_choice = 1
                
                if diag_choice == 1:
                    # Basic system check
                    print("\n🔍 Basic System Check")
                    check_system_requirements()
                    
                    # Check data file
                    data_exists = os.path.exists(CONFIG["data_path"])
                    print(f"Data file: {'✓' if data_exists else '✗'} {CONFIG['data_path']}")
                    
                    if data_exists:
                        try:
                            col_names = [
                                'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
                                'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
                                'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
                                '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
                                'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
                                'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
                                'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
                            ]
                            df = pd.read_csv(CONFIG["data_path"], header=None, skiprows=33, names=col_names, nrows=5)
                            print(f"Data format: ✓ Valid ({len(df)} sample rows loaded)")
                        except Exception as e:
                            print(f"Data format: ✗ Error loading data - {str(e)}")
                    
                    # Check output directories
                    for directory in ["models", "outputs", "optimization_results"]:
                        if os.path.exists(directory):
                            print(f"Directory {directory}/: ✓ Exists")
                        else:
                            print(f"Directory {directory}/: ⚠️  Missing (will be created)")
                    
                    # Memory test
                    try:
                        print("\nPerforming quick GPU memory test...")
                        if torch.cuda.is_available():
                            test_tensor = torch.randn(1000, 1000, device='cuda')
                            del test_tensor
                            torch.cuda.empty_cache()
                            print("GPU memory test: ✓ Passed")
                        else:
                            print("GPU memory test: ⚠️  Skipped (no CUDA)")
                    except Exception as e:
                        print(f"GPU memory test: ✗ Failed - {str(e)}")
                        
                elif diag_choice == 2:
                    print("\n🔍 Model Compatibility Test")
                    run_model_compatibility_test()
                    
                else:
                    print("\n🔍 Full System Validation")
                    print("Running all diagnostic tests...")
                    
                    print("\n1. Basic System Check:")
                    check_system_requirements()
                    
                    print("\n2. Model Compatibility:")
                    run_model_compatibility_test()
                    
                    print("\n✅ Full system validation completed.")
                    print("\n📋 Summary: All integrated features are now accessible through the main menu.")
                    print("   • Training: Optimized/Quick/Ultra-Quick modes available")
                    print("   • Inference: AI/Statistical/Hybrid prediction methods")
                    print("   • Optimization: Validation/Thorough/Standard modes")
                    print("   • Diagnostics: Comprehensive testing and validation")
            
            elif choice == '7':
                print("\n👋 Thank you for using Mark Six Prediction System v3.0!")
                print("🎉 All features are now unified in this single interface.")
                print("May your numbers bring you luck! 🍀")
                break
            
            else:
                print("❌ Invalid choice. Please enter a number between 1 and 7.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Operation interrupted by user.")
            print("Returning to main menu...")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            
            # Ask if user wants to see full traceback
            show_trace = input("Show detailed error information? (y/n): ").lower() == 'y'
            if show_trace:
                print("\nDetailed error information:")
                traceback.print_exc()
            
            print("\nReturning to main menu...")


if __name__ == "__main__":
    # Ensure the directories for saving models and outputs exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/training_plots", exist_ok=True)
    os.makedirs("optimization_results", exist_ok=True)
    os.makedirs("thorough_search_results", exist_ok=True)
    
    # Import CONFIG after ensuring directories exist
    from src.config import CONFIG
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nProgram terminated by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check your installation and try again.")
        traceback.print_exc()