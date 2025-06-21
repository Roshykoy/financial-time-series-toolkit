import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib

# Import our updated modules
from src.model import TransformerMultiTarget, MODEL_PARAMS
from src.engine import train_one_epoch, evaluate

# --- Configuration ---
CONFIG = {
    "data_path": "data/processed/processed_mark_six.csv",
    "model_save_path": "outputs/models/mark_six_multitarget_predictor.pth",
    "feature_scaler_path": "outputs/models/feature_scaler_multitarget.pkl",
    "target_scaler_path": "outputs/models/target_scaler_multitarget.pkl",
    "feature_columns_path": "outputs/models/feature_columns_multitarget.pkl",
    "window_size": 10,
    "batch_size": 32,
    "epochs": 100, # Increased epochs for a more complex task
    "learning_rate": 1e-4,
    "test_split_size": 0.2,
    "random_state": 42
}

def create_sequences(features, targets, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:(i + window_size)])
        y.append(targets[i + window_size])
    return np.array(X), np.array(y)

def main():
    print("Loading processed data for multi-target prediction...")
    df = pd.read_csv(CONFIG["data_path"])
    
    # --- NEW: Define multiple target columns ---
    target_columns = ['Sum', 'Odd', 'Low', 'Bin_1_10', 'Bin_11_20', 'Bin_21_30', 'Bin_31_40', 'Bin_41_50']
    
    # Features are all columns except identifiers and targets
    feature_columns = df.drop(columns=['Draw', 'Date'] + target_columns).columns
    
    print(f"Using {len(feature_columns)} features and {len(target_columns)} targets.")
    
    features = df[feature_columns].values
    targets = df[target_columns].values

    print("Splitting and scaling data...")
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        features, targets, test_size=CONFIG["test_split_size"], random_state=CONFIG["random_state"], shuffle=False
    )

    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train_raw)
    X_val_scaled = feature_scaler.transform(X_val_raw)

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_raw)
    y_val_scaled = target_scaler.transform(y_val_raw)
    
    print("Creating sequences...")
    X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, CONFIG["window_size"])
    X_val, y_val = create_sequences(X_val_scaled, y_val_scaled, CONFIG["window_size"])

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Dynamically set model parameters
    MODEL_PARAMS['d_features'] = X_train.shape[2]
    MODEL_PARAMS['d_targets'] = y_train.shape[1]
    
    model = TransformerMultiTarget(MODEL_PARAMS).to(device)
    
    # Use Mean Squared Error loss for regression
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    print("\nStarting multi-target training...")
    for epoch in range(CONFIG["epochs"]):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        avg_val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train MSE Loss: {avg_train_loss:.4f} | Validation MSE Loss: {avg_val_loss:.4f}")

    print("\nTraining complete. Saving model and artifacts...")
    os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)
    torch.save(model.state_dict(), CONFIG["model_save_path"])
    joblib.dump(feature_scaler, CONFIG["feature_scaler_path"])
    joblib.dump(target_scaler, CONFIG["target_scaler_path"])
    joblib.dump(feature_columns.tolist(), CONFIG["feature_columns_path"])
    print("Model, scalers, and feature list saved successfully.")

if __name__ == "__main__":
    main()