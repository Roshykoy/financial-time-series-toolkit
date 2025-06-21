# src/evaluation_pipeline.py
import torch
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import os
import random

from src.config import CONFIG
from src.model import ScoringModel
from src.feature_engineering import FeatureEngineer
from src.temporal_scorer import TemporalScorer
from src.i_ching_scorer import IChingScorer
from src.inference_pipeline import ScorerEnsemble

def run_evaluation(use_i_ching=False):
    """
    Evaluates the trained ensemble model on the validation set.

    This function measures the model's ability to rank historical winning
    combinations higher than randomly generated "negative" combinations.
    A high "win rate" indicates a successful model.
    """
    print("\n--- Starting Model Evaluation Pipeline ---")

    if not os.path.exists(CONFIG["model_save_path"]):
        print(f"\n[ERROR] Model file not found at '{CONFIG['model_save_path']}'.")
        print("Please train a model first (Main Menu -> Option 1).")
        return

    # --- Load Artifacts ---
    print("Loading trained model and feature engineer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG['device'] = device

    fe = joblib.load(CONFIG["feature_engineer_path"])
    CONFIG['d_features'] = len(fe.transform([1,2,3,4,5,6], 0))

    model = ScoringModel(CONFIG).to(device)
    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device))
    model.eval()

    # --- Load Data & Scorers ---
    col_names = [
        'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
        'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
        'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
        '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
        'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
        'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
        'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
    ]
    df = pd.read_csv(CONFIG["data_path"], header=None, skiprows=33, names=col_names)
    df = df.sort_values(by='Date').reset_index(drop=True)

    temporal_scorer = TemporalScorer(CONFIG)
    temporal_scorer.fit(df)
    i_ching_scorer = IChingScorer(CONFIG)
    
    scorer_ensemble = ScorerEnsemble(model, fe, temporal_scorer, i_ching_scorer, df, CONFIG, use_i_ching)

    # --- Split Data ---
    # We evaluate on the validation set, which the model has not been trained on.
    train_size = int(len(df) * 0.85)
    val_df = df.iloc[train_size:].reset_index()

    print(f"Evaluating the model on {len(val_df)} historical draws...")

    # --- Evaluation Logic ---
    wins = 0
    total_comparisons = 0

    winning_num_cols = [f'Winning_Num_{i}' for i in range(1, 7)]

    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating Draws"):
        # Get the actual historical winning set and its score
        positive_set = row[winning_num_cols].astype(int).tolist()
        positive_score = scorer_ensemble.score(positive_set)

        # Generate N random negative samples to compare against
        for _ in range(CONFIG['evaluation_neg_samples']):
            # Ensure the negative sample is not a known historical winner
            while True:
                negative_set = sorted(random.sample(range(1, CONFIG['num_lotto_numbers'] + 1), 6))
                if tuple(negative_set) not in fe.historical_sets:
                    break
            
            negative_score = scorer_ensemble.score(negative_set)
            
            if positive_score > negative_score:
                wins += 1
            total_comparisons += 1
    
    win_rate = (wins / total_comparisons) * 100 if total_comparisons > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Ensemble Configuration: {'DL + Temporal + I-Ching' if use_i_ching else 'DL + Temporal'}")
    print(f"Positive (Winning) sets ranked higher: {wins}/{total_comparisons}")
    print(f"Model Win Rate: {win_rate:.2f}%")
    print("----------------------------")
    print("(A win rate > 50% suggests the model has learned to distinguish real winning patterns from random noise.)")