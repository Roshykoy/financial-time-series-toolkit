import torch
import pandas as pd
import numpy as np
import joblib
import time

# Import our custom model and configurations
from src.model import TransformerMultiTarget, MODEL_PARAMS
from train import CONFIG as TRAIN_CONFIG

# --- Generation Parameters ---
NUM_SETS_TO_GENERATE = 5      # The 'n' sets required by the user
SUM_TOLERANCE = 10            # How close the sum can be (+/-)
CHARACTERISTIC_TOLERANCE = 1  # How close the other counts can be (+/-)
MAX_TRIES = 2000000           # Safety limit to prevent a very long run

def get_multi_target_prediction():
    """
    Loads the trained multi-target model and data to generate a forecast vector.
    """
    print("Loading multi-target model and artifacts...")
    # Load all necessary files
    feature_scaler = joblib.load(TRAIN_CONFIG["feature_scaler_path"])
    target_scaler = joblib.load(TRAIN_CONFIG["target_scaler_path"])
    feature_columns = joblib.load(TRAIN_CONFIG["feature_columns_path"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Configure model parameters dynamically
    MODEL_PARAMS['d_features'] = len(feature_columns)
    MODEL_PARAMS['d_targets'] = len(target_scaler.mean_) # Get number of targets from scaler
    
    model = TransformerMultiTarget(MODEL_PARAMS).to(device)
    model.load_state_dict(torch.load(TRAIN_CONFIG["model_save_path"], map_location=device))
    model.eval()

    # Prepare the input sequence from the latest data
    df = pd.read_csv(TRAIN_CONFIG["data_path"])
    recent_draws = df.head(TRAIN_CONFIG["window_size"])
    input_features_raw = recent_draws[feature_columns].values[::-1]
    input_features_scaled = feature_scaler.transform(input_features_raw)
    input_tensor = torch.tensor(input_features_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    # Get model prediction (scaled)
    with torch.no_grad():
        prediction_scaled = model(input_tensor)

    # Inverse transform to get human-readable predictions
    prediction_unscaled = target_scaler.inverse_transform(prediction_scaled.cpu().numpy())
    
    # The target columns in the order they were trained on
    target_columns = ['Sum', 'Odd', 'Low', 'Bin_1_10', 'Bin_11_20', 'Bin_21_30', 'Bin_31_40', 'Bin_41_50']
    
    # Return a dictionary of predictions
    return dict(zip(target_columns, prediction_unscaled[0]))

def generate_number_sets(num_sets, predicted_chars, tolerances):
    """
    Generates sets of Mark Six numbers that match the predicted characteristics.
    """
    print(f"\nGenerating {num_sets} number sets based on model's predicted characteristics...")
    
    generated_sets = []
    attempts = 0
    start_time = time.time()
    
    while len(generated_sets) < num_sets and attempts < MAX_TRIES:
        attempts += 1
        
        # 1. Generate a random candidate set
        candidate_set = np.random.choice(np.arange(1, 50), size=7, replace=False)
        
        # 2. Calculate its characteristics
        candidate_chars = {
            "Sum": np.sum(candidate_set),
            "Odd": np.sum(candidate_set % 2 != 0),
            "Low": np.sum(candidate_set <= 24),
            "Bin_1_10": np.sum((candidate_set >= 1) & (candidate_set <= 10)),
            "Bin_11_20": np.sum((candidate_set >= 11) & (candidate_set <= 20)),
            "Bin_21_30": np.sum((candidate_set >= 21) & (candidate_set <= 30)),
            "Bin_31_40": np.sum((candidate_set >= 31) & (candidate_set <= 40)),
            "Bin_41_50": np.sum((candidate_set >= 41) & (candidate_set <= 49)),
        }

        # 3. Check if it matches ALL predicted characteristics within tolerance
        is_match = True
        for key, pred_val in predicted_chars.items():
            actual_val = candidate_chars[key]
            tolerance = tolerances.get(key, CHARACTERISTIC_TOLERANCE)
            # Round predicted value for comparison with integer counts
            if abs(actual_val - round(pred_val)) > tolerance:
                is_match = False
                break
        
        if is_match:
            candidate_set.sort()
            if not any(np.array_equal(candidate_set, s) for s in generated_sets):
                generated_sets.append(candidate_set)
                print(f"Found set {len(generated_sets)}/{num_sets}: {candidate_set} (Sum: {candidate_chars['Sum']}, Odd: {candidate_chars['Odd']})")

    end_time = time.time()
    print(f"\nGeneration complete in {end_time - start_time:.2f} seconds ({attempts} attempts).")
    return generated_sets

if __name__ == "__main__":
    # 1. Get the multi-target prediction from our model
    predicted_characteristics = get_multi_target_prediction()
    
    print("\n--- Model's Forecast for Next Draw's Characteristics ---")
    for key, value in predicted_characteristics.items():
        print(f"{key:>12}: {value:.2f}")

    # 2. Define tolerances for the generator
    tolerances = {
        "Sum": SUM_TOLERANCE,
        "Odd": CHARACTERISTIC_TOLERANCE,
        "Low": CHARACTERISTIC_TOLERANCE,
        "Bin_1_10": CHARACTERISTIC_TOLERANCE,
        "Bin_11_20": CHARACTERISTIC_TOLERANCE,
        "Bin_21_30": CHARACTERISTIC_TOLERANCE,
        "Bin_31_40": CHARACTERISTIC_TOLERANCE,
        "Bin_41_50": CHARACTERISTIC_TOLERANCE,
    }

    # 3. Generate number sets that match the forecast
    final_sets = generate_number_sets(NUM_SETS_TO_GENERATE, predicted_characteristics, tolerances)
    
    print("\n--- Recommended Number Sets Based on Model Forecast ---")
    if final_sets:
        for i, num_set in enumerate(final_sets):
            main_numbers = ' '.join(map(str, num_set[:6]))
            extra_number = num_set[6]
            print(f"Set {i+1}: {main_numbers}  |  Extra: {extra_number}")
    else:
        print(f"Could not generate {NUM_SETS_TO_GENERATE} sets within {MAX_TRIES} attempts.")
        print("This indicates the model's predicted combination of characteristics is rare.")
        print("Consider increasing MAX_TRIES or loosening the tolerances at the top of the script.")