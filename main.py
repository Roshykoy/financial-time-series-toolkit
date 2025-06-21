# main.py
import os
import traceback
from src.training_pipeline import run_training
from src.inference_pipeline import run_inference
from src.evaluation_pipeline import run_evaluation

def main_menu():
    """Displays the main menu and handles user input."""
    while True:
        print("\n--- Mark Six AI Project Hub ---")
        print("===============================")
        print("1. Train New Model")
        print("2. Generate Number Sets (Inference)")
        print("3. Evaluate Trained Model")
        print("4. Exit")
        print("===============================")
        
        choice = input("Enter your choice (1-4): ")
        
        try:
            if choice == '1':
                run_training()
            
            elif choice == '2':
                num_sets = int(input("How many number sets would you like to generate? "))
                if num_sets <= 0:
                    print("Please enter a positive number.")
                    continue
                
                use_iching_input = input("Use the optional I-Ching scorer? (y/n): ").lower()
                use_iching = use_iching_input == 'y'
                if use_iching:
                    print("I-Ching scorer has been enabled.")

                run_inference(num_sets, use_i_ching=use_iching)

            elif choice == '3':
                print("\nThis will evaluate the model's performance on the validation data.")
                use_iching_input = input("Evaluate with the I-Ching scorer enabled? (y/n): ").lower()
                use_iching = use_iching_input == 'y'
                if use_iching:
                    print("Evaluating with I-Ching scorer enabled.")
                
                run_evaluation(use_i_ching=use_iching)

            elif choice == '4':
                print("Exiting project hub. Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1 and 4.")

        except ValueError:
            print("\n[ERROR] Invalid input. Please enter a valid number.")
        except Exception:
            print("\n[ERROR] An unexpected error occurred:")
            traceback.print_exc()

if __name__ == "__main__":
    # Ensure the directories for saving models and outputs exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    main_menu()