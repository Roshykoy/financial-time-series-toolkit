import torch
import pandas as pd
import numpy as np
import joblib
import math

from src.model import TransformerMDN, MODEL_PARAMS
from train import CONFIG as TRAIN_CONFIG

def get_prediction():
    """
    Loads the trained model and data to generate a probabilistic forecast for the next draw's Sum.
    Returns the point forecast and the full GMM distribution.
    """
    # Load all artifacts
    feature_scaler = joblib.load(TRAIN_CONFIG["feature_scaler_path"])
    target_scaler = joblib.load(TRAIN_CONFIG["target_scaler_path"])
    feature_columns = joblib.load(TRAIN_CONFIG["feature_columns_path"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PARAMS['d_features'] = len(feature_columns)
    
    model = TransformerMDN(MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(TRAIN_CONFIG["model_save_path"], map_location=device))
    model.eval()

    # Prepare the most recent data sequence
    df = pd.read_csv(TRAIN_CONFIG["data_path"])
    recent_draws = df.head(TRAIN_CONFIG["window_size"])
    input_features_raw = recent_draws[feature_columns].values[::-1]
    input_features_scaled = feature_scaler.transform(input_features_raw)
    input_tensor = torch.tensor(input_features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # Get model output
    with torch.no_grad():
        pi, mu, sigma = model(input_tensor)

    # Un-scale the results to be human-readable
    target_mean_scaler = target_scaler.mean_[0]
    target_scale_scaler = target_scaler.scale_[0]
    mu_unscaled = (mu.cpu().numpy() * target_scale_scaler) + target_mean_scaler
    sigma_unscaled = sigma.cpu().numpy() * target_scale_scaler
    pi_unscaled = pi.cpu().numpy()
    
    # Calculate final point forecast
    point_forecast = np.sum(pi_unscaled * mu_unscaled, axis=1)[0]
    
    return point_forecast, pi_unscaled, mu_unscaled, sigma_unscaled

if __name__ == "__main__":
    print("Running prediction for the next draw...")
    
    point_forecast, pi, mu, sigma = get_prediction()

    print("\n--- Prediction for Next Draw's Sum ---")
    print(f"\nPoint Forecast (Expected Sum): {point_forecast:.2f}")
    
    print("\nProbabilistic Forecast (GMM Components):")
    print("The model predicts the outcome as a mixture of these possibilities:")
    for i in range(MODEL_PARAMS['n_gmm_components']):
        weight = pi[0, i] * 100
        mean = mu[0, i]
        std_dev = sigma[0, i]
        print(f"  - Possibility {i+1}: Mean Sum of {mean:.2f} (StdDev: {std_dev:.2f}), with a weight of {weight:.1f}%")