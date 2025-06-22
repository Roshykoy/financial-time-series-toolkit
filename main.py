# main.py
import os
import traceback
import json
from src.training_pipeline import run_training
from src.inference_pipeline import run_inference
from src.evaluation_pipeline import run_evaluation
from src.hyperparameter_optimizer import run_hyperparameter_optimization
from src.config_manager import run_config_manager
from src.config import CONFIG

def display_system_info():
    """Display system information and current configuration."""
    import torch
    print("\n--- System Information ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print("\n--- Current Configuration ---")
    key_params = ['learning_rate', 'hidden_size', 'num_layers', 'dropout', 'batch_size', 'epochs']
    for param in key_params:
        if param in CONFIG:
            print(f"{param}: {CONFIG[param]}")
    
    # Check if optimized parameters exist
    if os.path.exists('best_hyperparameters.json'):
        print("\n💡 Optimized hyperparameters available! Check 'best_hyperparameters.json'")

def load_optimized_config():
    """Load and apply optimized hyperparameters if available."""
    if os.path.exists('best_hyperparameters.json'):
        try:
            with open('best_hyperparameters.json', 'r') as f:
                optimized_config = json.load(f)
            
            print("\n--- Found Optimized Hyperparameters ---")
            print("Current vs Optimized:")
            for key, optimized_value in optimized_config.items():
                if key in CONFIG:
                    current_value = CONFIG[key]
                    print(f"  {key}: {current_value} → {optimized_value}")
            
            if input("\nApply optimized hyperparameters for this session? (y/n): ").lower() == 'y':
                CONFIG.update(optimized_config)
                print("✅ Optimized hyperparameters applied!")
                return True
        except Exception as e:
            print(f"Error loading optimized config: {e}")
    
    return False

def main_menu():
    """Displays the main menu and handles user input."""
    
    # Display welcome message and system info
    print("\n" + "="*60)
    print("           🎯 MARK SIX AI PROJECT HUB 🎯")
    print("         Advanced Lottery Analysis System")
    print("="*60)
    
    # Check for optimized hyperparameters on startup
    load_optimized_config()
    
    while True:
        print("\n" + "-"*50)
        print("               MAIN MENU")
        print("-"*50)
        print("1. 🧠 Train New Model")
        print("2. 🎲 Generate Number Sets (Inference)")
        print("3. 📊 Evaluate Trained Model")
        print("4. ⚙️  Optimize Hyperparameters (NEW!)")
        print("5. 💻 System Information")
        print("6. 🔧 Advanced Options")
        print("7. 🚪 Exit")
        print("-"*50)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        try:
            if choice == '1':
                print("\n🧠 Starting Model Training...")
                run_training()
            
            elif choice == '2':
                print("\n🎲 Number Generation Mode")
                try:
                    num_sets = int(input("How many number sets would you like to generate? "))
                    if num_sets <= 0:
                        print("❌ Please enter a positive number.")
                        continue
                    if num_sets > 50:
                        print("⚠️  Generating more than 50 sets may take a while...")
                        if input("Continue? (y/n): ").lower() != 'y':
                            continue
                    
                    use_iching_input = input("Use the optional I-Ching scorer? (y/n): ").lower()
                    use_iching = use_iching_input == 'y'
                    if use_iching:
                        print("✨ I-Ching scorer has been enabled.")

                    run_inference(num_sets, use_i_ching=use_iching)
                except ValueError:
                    print("❌ Invalid input. Please enter a valid number.")
                    continue

            elif choice == '3':
                print("\n📊 Model Evaluation Mode")
                print("This will evaluate the model's performance on validation data.")
                use_iching_input = input("Evaluate with the I-Ching scorer enabled? (y/n): ").lower()
                use_iching = use_iching_input == 'y'
                if use_iching:
                    print("✨ Evaluating with I-Ching scorer enabled.")
                
                run_evaluation(use_i_ching=use_iching)

            elif choice == '4':
                print("\n⚙️ Hyperparameter Optimization Mode")
                print("This will automatically find the best parameters for your model.")
                print("⏱️  This process can take 15-60 minutes depending on the method chosen.")
                
                if input("Continue with hyperparameter optimization? (y/n): ").lower() == 'y':
                    run_hyperparameter_optimization()
                else:
                    print("Optimization cancelled.")

            elif choice == '5':
                display_system_info()

            elif choice == '6':
                advanced_options_menu()

            elif choice == '7':
                print("\n👋 Thank you for using Mark Six AI!")
                print("🍀 Good luck with your number selections!")
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1 and 7.")

        except ValueError:
            print("\n❌ Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n\n⏸️  Operation cancelled by user.")
        except Exception as e:
            print(f"\n❌ An unexpected error occurred:")
            print(f"Error: {str(e)}")
            if input("Show detailed error information? (y/n): ").lower() == 'y':
                traceback.print_exc()

def advanced_options_menu():
    """Handle advanced options."""
    while True:
        print("\n" + "-"*40)
        print("           ADVANCED OPTIONS")
        print("-"*40)
        print("1. 📝 View Current Configuration")
        print("2. 🔄 Reset to Default Configuration")
        print("3. 📥 Load Optimized Hyperparameters")
        print("4. ⚙️  Configuration Manager")
        print("5. 🗂️  View Optimization History")
        print("6. 🧹 Clean Up Generated Files")
        print("7. ⬅️  Back to Main Menu")
        print("-"*40)
        
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            view_current_config()
        elif choice == '2':
            reset_to_default_config()
        elif choice == '3':
            load_optimized_config()
        elif choice == '4':
            run_config_manager()
        elif choice == '5':
            view_optimization_history()
        elif choice == '6':
            cleanup_files()
        elif choice == '7':
            break
        else:
            print("❌ Invalid choice. Please enter a number between 1 and 7.")

def view_current_config():
    """Display the current configuration."""
    print("\n--- Current Configuration ---")
    for key, value in sorted(CONFIG.items()):
        print(f"  {key}: {value}")

def reset_to_default_config():
    """Reset configuration to defaults."""
    if input("⚠️  Reset all parameters to default values? (y/n): ").lower() == 'y':
        # Reload the default config
        from src.config import CONFIG as DEFAULT_CONFIG
        CONFIG.clear()
        CONFIG.update(DEFAULT_CONFIG)
        print("✅ Configuration reset to default values.")

def view_optimization_history():
    """View hyperparameter optimization history."""
    results_dir = "hyperparameter_results"
    if not os.path.exists(results_dir):
        print("❌ No optimization history found.")
        return
    
    import glob
    summary_files = glob.glob(f"{results_dir}/optimization_summary_*.json")
    
    if not summary_files:
        print("❌ No optimization summary files found.")
        return
    
    print(f"\n--- Found {len(summary_files)} Optimization Run(s) ---")
    
    for i, file_path in enumerate(sorted(summary_files, key=os.path.getctime, reverse=True)):
        try:
            with open(file_path, 'r') as f:
                summary = json.load(f)
            
            filename = os.path.basename(file_path)
            print(f"\n{i+1}. {filename}")
            print(f"   Method: {summary['method']}")
            print(f"   Best Score: {summary['best_score']:.4f}")
            print(f"   Total Trials: {summary['total_trials']}")
            print(f"   Date: {summary['timestamp'][:19]}")
            
        except Exception as e:
            print(f"   Error reading {file_path}: {e}")

def cleanup_files():
    """Clean up generated files."""
    print("\n--- File Cleanup Options ---")
    print("1. 🗑️  Delete hyperparameter results")
    print("2. 🗑️  Delete model files")
    print("3. 🗑️  Delete output logs")
    print("4. 🗑️  Delete configuration files")
    print("5. 🗑️  Delete all generated files")
    print("6. ⬅️  Cancel")
    
    choice = input("Choose what to clean up (1-6): ").strip()
    
    if choice == '1':
        cleanup_directory("hyperparameter_results", "hyperparameter optimization results")
    elif choice == '2':
        cleanup_directory("models", "trained model files")
    elif choice == '3':
        cleanup_directory("outputs", "output logs")
    elif choice == '4':
        cleanup_directory("configurations", "configuration files")
        if os.path.exists("best_hyperparameters.json"):
            if input("Delete best_hyperparameters.json? (y/n): ").lower() == 'y':
                os.remove("best_hyperparameters.json")
                print("✅ Deleted best_hyperparameters.json")
    elif choice == '5':
        if input("⚠️  Delete ALL generated files? This cannot be undone! (yes/no): ").lower() == 'yes':
            cleanup_directory("hyperparameter_results", "hyperparameter results")
            cleanup_directory("models", "model files")
            cleanup_directory("outputs", "output logs")
            cleanup_directory("configurations", "configuration files")
            if os.path.exists("best_hyperparameters.json"):
                os.remove("best_hyperparameters.json")
                print("✅ Deleted best_hyperparameters.json")
    elif choice == '6':
        print("Cleanup cancelled.")
    else:
        print("❌ Invalid choice.")

def cleanup_directory(directory, description):
    """Clean up a specific directory."""
    if os.path.exists(directory):
        import shutil
        if input(f"⚠️  Delete all {description}? (y/n): ").lower() == 'y':
            shutil.rmtree(directory)
            print(f"✅ Deleted {description}")
        else:
            print("Cleanup cancelled.")
    else:
        print(f"ℹ️  No {description} found to delete.")

if __name__ == "__main__":
    # Ensure the directories for saving models and outputs exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("hyperparameter_results", exist_ok=True)
    os.makedirs("configurations", exist_ok=True)
    
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()