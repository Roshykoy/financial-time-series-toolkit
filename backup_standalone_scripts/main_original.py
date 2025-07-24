# main.py
import os
import traceback
import torch
from src.training_pipeline import run_training
from src.inference_pipeline import run_inference
from src.evaluation_pipeline import run_evaluation

# Enhanced configuration and logging
from src.infrastructure.config import get_config_manager
from src.infrastructure.logging import get_logger, configure_logging

# Initialize enhanced systems
configure_logging(log_level="INFO", log_file="marksix.log")
logger = get_logger(__name__)


def print_banner():
    """Prints the application banner."""
    print("\n" + "=" * 70)
    print("MARK SIX LOTTERY PREDICTION SYSTEM v2.0")
    print("Hybrid Generative-Ensemble Architecture")
    print("=" * 70)
    print("Architecture: CVAE + Graph Neural Networks + Meta-Learning")
    print("• Conditional Variational Autoencoder for generation")
    print("• Graph Neural Network encoder for number relationships")
    print("• LSTM temporal context encoder")
    print("• Attention-based meta-learner for dynamic ensemble weights")
    print("• Hard negative mining with contrastive learning")
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
    
    # Epochs
    try:
        epochs = int(input(f"Number of training epochs (default: {CONFIG['epochs']}): ") or CONFIG['epochs'])
        if epochs <= 0:
            epochs = CONFIG['epochs']
    except ValueError:
        epochs = CONFIG['epochs']
    
    # Batch size
    try:
        batch_size = int(input(f"Batch size (default: {CONFIG['batch_size']}): ") or CONFIG['batch_size'])
        if batch_size <= 0:
            batch_size = CONFIG['batch_size']
    except ValueError:
        batch_size = CONFIG['batch_size']
    
    # Learning rate
    try:
        lr_input = input(f"Learning rate (default: {CONFIG['learning_rate']}): ")
        learning_rate = float(lr_input) if lr_input else CONFIG['learning_rate']
        if learning_rate <= 0:
            learning_rate = CONFIG['learning_rate']
    except ValueError:
        learning_rate = CONFIG['learning_rate']
    
    # Advanced options
    print("\nAdvanced Options:")
    use_mixed_precision = input("Use mixed precision training? (y/n, default: y): ").lower() != 'n'
    save_plots = input("Save training plots? (y/n, default: y): ").lower() != 'n'
    
    return {
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
    
    # Number of sets
    try:
        num_sets = int(input("How many number sets would you like to generate? "))
        if num_sets <= 0:
            print("Please enter a positive number.")
            return None
    except ValueError:
        print("Invalid input. Please enter a valid number.")
        return None
    
    # Temperature
    try:
        temp_input = input("Generation temperature (0.1-2.0, default: 0.8): ")
        temperature = float(temp_input) if temp_input else 0.8
        temperature = max(0.1, min(2.0, temperature))  # Clamp between 0.1 and 2.0
    except ValueError:
        temperature = 0.8
    
    # I-Ching scorer
    use_iching_input = input("Use the I-Ching scorer? (y/n, default: n): ").lower()
    use_iching = use_iching_input == 'y'
    
    # Verbose output
    verbose = input("Show detailed generation process? (y/n, default: y): ").lower() != 'n'
    
    # Advanced options
    print("\nAdvanced Options:")
    print("1. Standard generation (recommended)")
    print("2. High diversity mode (more creative combinations)")
    print("3. Conservative mode (closer to historical patterns)")
    
    try:
        mode_choice = int(input("Choose generation mode (1-3, default: 1): ") or "1")
    except ValueError:
        mode_choice = 1
    
    # Adjust temperature based on mode
    if mode_choice == 2:  # High diversity
        temperature *= 1.3
        print("🎲 High diversity mode selected - more creative combinations")
    elif mode_choice == 3:  # Conservative
        temperature *= 0.7
        print("🎯 Conservative mode selected - closer to historical patterns")
    else:
        print("⚖️  Standard mode selected - balanced generation")
    
    return {
        'num_sets': num_sets,
        'temperature': temperature,
        'use_i_ching': use_iching,
        'verbose': verbose,
        'mode': mode_choice
    }


def get_optimization_options():
    """Gets hyperparameter optimization configuration options from user."""
    print("Hyperparameter Optimization Configuration:")
    print("-" * 41)
    
    # Show available presets
    try:
        from src.optimization.main import OptimizationOrchestrator
        orchestrator = OptimizationOrchestrator("data/raw/Mark_Six.csv")
        presets = orchestrator.list_presets()
        
        print("Available optimization presets:")
        for i, preset in enumerate(presets, 1):
            info = orchestrator.get_preset_info(preset)
            print(f"{i}. {preset}: {info['description']}")
            print(f"   Algorithm: {info['algorithm']}, Max Trials: {info['max_trials']}, Duration: {info['max_duration_hours']}h")
        
        # Get preset choice
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
    
    # Get custom parameters
    print(f"\nSelected preset: {selected_preset}")
    print("You can customize the following parameters:")
    
    # Max trials
    try:
        trials_input = input("Maximum number of trials (default: use preset): ")
        max_trials = int(trials_input) if trials_input else None
        if max_trials is not None and max_trials <= 0:
            print("Invalid number of trials, using preset default.")
            max_trials = None
    except ValueError:
        print("Invalid input for trials, using preset default.")
        max_trials = None
    
    # Max duration
    try:
        duration_input = input("Maximum duration in hours (default: use preset): ")
        max_duration = float(duration_input) if duration_input else None
        if max_duration is not None and max_duration <= 0:
            print("Invalid duration, using preset default.")
            max_duration = None
    except ValueError:
        print("Invalid input for duration, using preset default.")
        max_duration = None
    
    return {
        'preset': selected_preset,
        'max_trials': max_trials,
        'max_duration': max_duration
    }


def display_model_info():
    """Displays information about trained models."""
    from src.config import CONFIG
    
    print("\nTrained Model Information:")
    print("-" * 28)
    
    # Check if models exist
    cvae_exists = os.path.exists(CONFIG["model_save_path"])
    meta_exists = os.path.exists(CONFIG["meta_learner_save_path"])
    fe_exists = os.path.exists(CONFIG["feature_engineer_path"])
    
    print(f"CVAE Model: {'✓' if cvae_exists else '✗'} {CONFIG['model_save_path']}")
    print(f"Meta-Learner: {'✓' if meta_exists else '✗'} {CONFIG['meta_learner_save_path']}")
    print(f"Feature Engineer: {'✓' if fe_exists else '✗'} {CONFIG['feature_engineer_path']}")
    
    if cvae_exists:
        try:
            # Get model file size
            model_size = os.path.getsize(CONFIG["model_save_path"]) / (1024 * 1024)  # MB
            print(f"CVAE Model size: {model_size:.1f} MB")
            
            # Get modification time
            import datetime
            mod_time = os.path.getmtime(CONFIG["model_save_path"])
            mod_date = datetime.datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M")
            print(f"Last trained: {mod_date}")
        except Exception:
            pass
    
    print()


def main_menu():
    """Displays the main menu and handles user input."""
    from src.config import CONFIG
    
    print_banner()
    check_system_requirements()
    
    while True:
        print("\n" + "=" * 50)
        print("MAIN MENU")
        print("=" * 50)
        print("1. Train New CVAE Model")
        print("2. Generate Number Combinations (Inference)")
        print("3. Evaluate Trained Model")
        print("4. Optimize Hyperparameters")
        print("5. View Model Information")
        print("6. System Diagnostics")
        print("7. Exit")
        print("=" * 50)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        try:
            if choice == '1':
                print("\n🚀 Starting CVAE Training Pipeline")
                print("-" * 40)
                
                # Get training options
                training_config = get_training_options()
                
                # Update CONFIG with user choices
                CONFIG.update(training_config)
                
                # Confirm training
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
                print("\n🎯 Starting Generative Inference")
                print("-" * 35)
                
                # Check if models exist
                if not (os.path.exists(CONFIG["model_save_path"]) and 
                       os.path.exists(CONFIG["meta_learner_save_path"])):
                    print("❌ Trained models not found!")
                    print("Please train the model first (Option 1).")
                    continue
                
                # Get inference options
                inference_config = get_inference_options()
                if inference_config is None:
                    continue
                
                print(f"\nInference Configuration:")
                print(f"• Number of sets: {inference_config['num_sets']}")
                print(f"• Temperature: {inference_config['temperature']:.2f}")
                print(f"• I-Ching scorer: {'Yes' if inference_config['use_i_ching'] else 'No'}")
                print(f"• Verbose output: {'Yes' if inference_config['verbose'] else 'No'}")
                
                # Run inference
                run_inference(
                    num_sets_to_generate=inference_config['num_sets'],
                    use_i_ching=inference_config['use_i_ching'],
                    temperature=inference_config['temperature'],
                    verbose=inference_config['verbose']
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
                
                print(f"\nOptimization Configuration:")
                print(f"• Preset: {optimization_config['preset']}")
                print(f"• Max trials: {optimization_config['max_trials']}")
                print(f"• Max duration: {optimization_config['max_duration']} hours")
                
                confirm = input("Start hyperparameter optimization? (y/n): ").lower()
                if confirm == 'y':
                    print("\n🚀 Starting optimization process...")
                    print("This may take a while depending on your configuration.")
                    
                    try:
                        # Create orchestrator and run optimization
                        orchestrator = OptimizationOrchestrator(
                            data_path=CONFIG["data_path"],
                            output_dir="optimization_results"
                        )
                        
                        results = orchestrator.run_optimization(
                            preset_name=optimization_config['preset'],
                            max_trials=optimization_config['max_trials'],
                            max_duration_hours=optimization_config['max_duration']
                        )
                        
                        # Display results summary
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
                print("\n🔧 System Diagnostics")
                print("-" * 20)
                
                # Extended system check
                check_system_requirements()
                
                # Check data file
                data_exists = os.path.exists(CONFIG["data_path"])
                print(f"Data file: {'✓' if data_exists else '✗'} {CONFIG['data_path']}")
                
                if data_exists:
                    try:
                        import pandas as pd
                        # Try to load a few rows
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
                for directory in ["models", "outputs"]:
                    if os.path.exists(directory):
                        print(f"Directory {directory}/: ✓ Exists")
                    else:
                        print(f"Directory {directory}/: ⚠️  Missing (will be created)")
                
                # Memory test
                try:
                    print("\nPerforming quick GPU memory test...")
                    if torch.cuda.is_available():
                        # Try to allocate a small tensor
                        test_tensor = torch.randn(1000, 1000, device='cuda')
                        del test_tensor
                        torch.cuda.empty_cache()
                        print("GPU memory test: ✓ Passed")
                    else:
                        print("GPU memory test: ⚠️  Skipped (no CUDA)")
                except Exception as e:
                    print(f"GPU memory test: ✗ Failed - {str(e)}")
            
            elif choice == '7':
                print("\n👋 Thank you for using Mark Six Prediction System!")
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