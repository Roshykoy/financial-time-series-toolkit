import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Import our custom modules
from src.model import TransformerMDN, MODEL_PARAMS
from train import create_sequences, CONFIG as TRAIN_CONFIG

def plot_loss_curve():
    """Plots the training and validation loss curve from the saved history."""
    loss_history_path = 'outputs/plots/loss_history.csv'
    plot_save_path = 'outputs/plots/loss_curve.png'
    
    try:
        df = pd.read_csv(loss_history_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_loss'], label='Training Loss')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Negative Log-Likelihood)')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(plot_save_path)
        print(f"Loss curve plot saved to '{plot_save_path}'")
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find '{loss_history_path}'. Please run train.py first.")

def plot_predictions():
    """Loads the model to plot its predictions against actual values on the validation set."""
    plot_save_path = 'outputs/plots/predictions_vs_actuals.png'
    
    try:
        # --- 1. Load all necessary artifacts ---
        feature_scaler = joblib.load(TRAIN_CONFIG["feature_scaler_path"])
        target_scaler = joblib.load(TRAIN_CONFIG["target_scaler_path"])
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_features = feature_scaler.n_features_in_
        MODEL_PARAMS['d_features'] = num_features
        
        model = TransformerMDN(MODEL_PARAMS).to(device)
        model.load_state_dict(torch.load(TRAIN_CONFIG["model_path"]))
        model.eval()

        # --- 2. Prepare the validation data exactly as in training ---
        df = pd.read_csv(TRAIN_CONFIG["data_path"])
        target_column = 'Sum'
        feature_columns = [col for col in df.columns if col not in ['Draw', 'Date', 'Sum'] and col in feature_scaler.feature_names_in_]
        
        features = df[feature_columns].values
        target = df[target_column].values.reshape(-1, 1)

        # We need the original, unscaled validation target for plotting
        _, _, _, y_val_raw = train_test_split(
            features, target, test_size=TRAIN_CONFIG["test_split_size"], random_state=TRAIN_CONFIG["random_state"], shuffle=False
        )

        # We create the validation set to feed into the model
        features_scaled = feature_scaler.transform(features)
        target_scaled = target_scaler.transform(target)
        _, X_val, _, y_val = train_test_split(
            features_scaled, target_scaled.flatten(), test_size=TRAIN_CONFIG["test_split_size"], random_state=TRAIN_CONFIG["random_state"], shuffle=False
        )
        X_val_sequences, _ = create_sequences(X_val, y_val, TRAIN_CONFIG["window_size"])
        
        # --- 3. Get model predictions for the entire validation set ---
        all_predictions_unscaled = []
        with torch.no_grad():
            input_tensor = torch.tensor(X_val_sequences, dtype=torch.float32).to(device)
            pi, mu, sigma = model(input_tensor)
            
            # Calculate the expected value (point forecast) for each prediction
            predicted_scaled = torch.sum(pi * mu, dim=1).cpu().numpy().reshape(-1, 1)
            
            # Inverse transform to get back to the original 'Sum' scale
            all_predictions_unscaled = target_scaler.inverse_transform(predicted_scaled)

        # The actual values need to be sliced to match the prediction length
        actual_values = y_val_raw[TRAIN_CONFIG["window_size"]:]

        # --- 4. Create the plot ---
        plt.figure(figsize=(15, 7))
        plt.plot(actual_values, label='Actual Sum', color='blue', alpha=0.7)
        plt.plot(all_predictions_unscaled, label='Predicted Sum', color='red', linestyle='--')
        plt.title('Model Predictions vs. Actual Values on Validation Set')
        plt.xlabel('Draw Index (in Validation Set)')
        plt.ylabel('Sum of Winning Numbers')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(plot_save_path)
        print(f"Predictions plot saved to '{plot_save_path}'")
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: Could not find a required file. {e}")
        print("Please ensure you have run train.py successfully first.")

if __name__ == "__main__":
    print("--- Generating Loss Curve Plot ---")
    plot_loss_curve()
    
    print("\n--- Generating Predictions Plot ---")
    plot_predictions()