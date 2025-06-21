# src/inference_pipeline.py
import torch
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
import random
import os

from src.config import CONFIG
from src.model import ScoringModel
from src.feature_engineering import FeatureEngineer
from src.temporal_scorer import TemporalScorer
from src.i_ching_scorer import IChingScorer

class ScorerEnsemble:
    """Orchestrates scoring from multiple sources: the main model and heuristic scorers."""
    def __init__(self, model, fe, temporal_scorer, i_ching_scorer, df_all_draws, config, use_i_ching=False):
        self.model = model
        self.fe = fe
        self.temporal_scorer = temporal_scorer
        self.i_ching_scorer = i_ching_scorer
        self.config = config
        self.use_i_ching = use_i_ching
        self.device = config['device']
        self.latest_draw_index = len(df_all_draws) - 1

        self.min_max_scores = {
            'model': [float('inf'), float('-inf')],
            'temporal': [float('inf'), float('-inf')],
            'i_ching': [float('inf'), float('-inf')]
        }

    def _normalize(self, score, key):
        """Normalizes a score to a 0-1 range based on min/max values seen so far."""
        min_s, max_s = self.min_max_scores[key]
        self.min_max_scores[key][0] = min(score, min_s)
        self.min_max_scores[key][1] = max(score, max_s)
        
        min_s, max_s = self.min_max_scores[key]
        return 0.5 if max_s == min_s else (score - min_s) / (max_s - min_s)

    def score(self, number_set):
        """Calculates a final weighted score for a number set from the ensemble."""
        features = self.fe.transform(number_set, self.latest_draw_index)
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            model_score = self.model(features_tensor).item()
        
        temporal_score = self.temporal_scorer.score(number_set, self.latest_draw_index)
        
        norm_model_score = self._normalize(model_score, 'model')
        norm_temporal_score = self._normalize(temporal_score, 'temporal')
        
        weights = self.config['ensemble_weights']
        ensemble_score = (weights['model'] * norm_model_score + 
                          weights['temporal'] * norm_temporal_score)
        
        if self.use_i_ching:
            i_ching_score = self.i_ching_scorer.score(number_set)
            norm_i_ching_score = self._normalize(i_ching_score, 'i_ching')
            ensemble_score += weights['i_ching'] * norm_i_ching_score
            
        return ensemble_score

def local_search(initial_set, scorer_ensemble, max_iterations, num_neighbors):
    """Performs a local search to find a high-scoring number combination."""
    current_set = list(initial_set)
    current_score = scorer_ensemble.score(current_set)
    
    for _ in range(max_iterations):
        best_neighbor, best_neighbor_score = None, -1
        for _ in range(num_neighbors):
            neighbor = current_set[:]
            idx_to_swap = random.randint(0, len(neighbor) - 1)
            
            while True:
                num_to_swap_in = random.randint(1, CONFIG['num_lotto_numbers'])
                if num_to_swap_in not in neighbor:
                    break
            
            neighbor[idx_to_swap] = num_to_swap_in
            neighbor.sort()

            neighbor_score = scorer_ensemble.score(neighbor)
            if neighbor_score > best_neighbor_score:
                best_neighbor_score, best_neighbor = neighbor_score, neighbor

        if best_neighbor_score > current_score:
            current_set, current_score = best_neighbor, best_neighbor_score
        else:
            break
    return current_set, current_score

def run_inference(num_sets_to_generate, use_i_ching=False):
    """Main pipeline for generating number sets."""
    print("\n--- Starting Number Generation Pipeline ---")
    
    if not os.path.exists(CONFIG["model_save_path"]):
        print(f"\n[ERROR] Model file not found at '{CONFIG['model_save_path']}'.")
        print("Please train a model first (Main Menu -> Option 1).")
        return

    print("Loading trained model and feature engineer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    CONFIG['device'] = device
    
    fe = joblib.load(CONFIG["feature_engineer_path"])
    CONFIG['d_features'] = len(fe.transform([1,2,3,4,5,6], 0))
    
    model = ScoringModel(CONFIG).to(device)
    model.load_state_dict(torch.load(CONFIG["model_save_path"], map_location=device))
    model.eval()

    # --- CORRECTED DATA LOADING ---
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
    # --- END OF CORRECTION ---
    
    temporal_scorer = TemporalScorer(CONFIG)
    temporal_scorer.fit(df)
    i_ching_scorer = IChingScorer(CONFIG)
    scorer_ensemble = ScorerEnsemble(model, fe, temporal_scorer, i_ching_scorer, df, CONFIG, use_i_ching)

    print(f"Generating {num_sets_to_generate} high-scoring sets using local search on {device}...")
    generated_sets = []
    
    with tqdm(total=num_sets_to_generate, desc="Generating Sets") as pbar:
        while len(generated_sets) < num_sets_to_generate:
            initial_set = sorted(random.sample(range(1, CONFIG['num_lotto_numbers'] + 1), 6))
            best_set, best_score = local_search(
                initial_set, scorer_ensemble, 
                max_iterations=CONFIG['search_iterations'],
                num_neighbors=CONFIG['search_neighbors']
            )
            
            if best_set not in generated_sets:
                generated_sets.append(best_set)
                pbar.set_postfix({"Best Score": f"{best_score:.4f}"})
                pbar.update(1)

    print("\n--- Recommended Number Sets ---")
    for i, num_set in enumerate(generated_sets):
        print(f"Set {i+1}: {num_set}")
    print("---------------------------------")