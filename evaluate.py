import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Import our custom modules and configurations
from src.model import TransformerMDN, MODEL_PARAMS
from train import create_sequences, CONFIG as TRAIN_CONFIG

def evaluate_model():
    """
    Loads the trained model, evaluates its performance on the validation set,
    compares it to a baseline, and plots the results.
    """
    try:
        # --- 1. Load Artifacts ---
        print("Loading model, scalers, and data...")
        feature_scaler = joblib.load(TRAIN_CONFIG["feature_scaler_path"])
        target_scaler = joblib.load(TRAIN_CONFIG["target_scaler_path"])
        feature_columns = joblib.load('outputs/models/feature_columns.pkl')
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL_PARAMS['d_features'] = len(feature_columns)
        
        model = TransformerMDN(MODEL_PARAMS).to(device)
        model.load_state_dict(torch.load(TRAIN_CONFIG["model_save_path"]))
        model.eval()
        print("Artifacts loaded successfully.")

        # --- 2. Prepare Validation Data (Corrected Logic) ---
        df = pd.read_csv(TRAIN_CONFIG["data_path"])
        target_column = 'Sum'
        
        features = df[feature_columns].values
        target = df[target_column].values.reshape(-1, 1)

        # Split data to get the raw training and validation sets
        X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
            features, target, test_size=TRAIN_CONFIG["test_split_size"], random_state=TRAIN_CONFIG["random_state"], shuffle=False
        )
        
        # Scale the validation features
        X_val_scaled = feature_scaler.transform(X_val_raw)
        
        # Create sequences ONLY from the validation set
        X_val_sequences, _ = create_sequences(X_val_scaled, y_val_raw.flatten(), TRAIN_CONFIG["window_size"])
        
        # The true values are the validation set, starting from the end of the first window
        actual_values = y_val_raw[TRAIN_CONFIG["window_size"]:]
        
        # --- 3. Get Model Predictions ---
        print("Generating model predictions on the validation set...")
        with torch.no_grad():
            input_tensor = torch.tensor(X_val_sequences, dtype=torch.float32).to(device)
            pi, mu, sigma = model(input_tensor)

        # --- 4. Process and Un-scale Predictions ---
        target_mean_scaler = target_scaler.mean_[0]
        target_scale_scaler = target_scaler.scale_[0]

        mu_unscaled = (mu.cpu().numpy() * target_scale_scaler) + target_mean_scaler
        sigma_unscaled = sigma.cpu().numpy() * target_scale_scaler
        pi_unscaled = pi.cpu().numpy()
        
        point_predictions = np.sum(pi_unscaled * mu_unscaled, axis=1)
        
        gmm_variance = np.sum(pi_unscaled * (mu_unscaled**2 + sigma_unscaled**2), axis=1) - point_predictions**2
        gmm_std_dev = np.sqrt(np.maximum(0, gmm_variance)) # Add maximum(0,...) for stability
        
        # --- 5. Quantitative Evaluation & Baseline Comparison ---
        model_mae = mean_absolute_error(actual_values, point_predictions)
        
        baseline_prediction = np.full_like(actual_values, y_train_raw.mean())
        baseline_mae = mean_absolute_error(actual_values, baseline_prediction)
        
        print("\n--- Model Performance Evaluation ---")
        print(f"Model Mean Absolute Error (MAE): {model_mae:.2f}")
        print(f"Baseline (Mean) MAE:           {baseline_mae:.2f}")

        if model_mae < baseline_mae:
            print("\nResult: The model performs better than the simple baseline. It has found some predictive patterns.")
        else:
            print("\nResult: The model does not perform better than the simple baseline. The patterns found may not be predictive.")

        # --- 6. Plotting ---
        print("\nGenerating plot...")
        plot_save_path = 'outputs/plots/predictions_vs_actuals.png'
        
        lower_bound = point_predictions - 1.645 * gmm_std_dev
        upper_bound = point_predictions + 1.645 * gmm_std_dev
        
        plt.figure(figsize=(20, 8))
        plt.plot(actual_values.flatten(), color='blue', label='Actual Sum')
        plt.plot(point_predictions, color='red', linestyle='--', label='Predicted Sum (Model)')
        plt.fill_between(range(len(point_predictions)), lower_bound.flatten(), upper_bound.flatten(), color='red', alpha=0.2, label='90% Prediction Interval')
        
        plt.title('Model Predictions vs. Actual Values on Validation Set')
        plt.xlabel('Draw Index (in Validation Set)')
        plt.ylabel('Sum of Winning Numbers')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(plot_save_path)
        print(f"Evaluation plot saved to '{plot_save_path}'")
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please ensure you have run train.py successfully first.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    evaluate_model()