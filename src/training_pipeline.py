# src/training_pipeline.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import os
import joblib
from tqdm import tqdm
import random

# Import from other src files
from src.config import CONFIG
from src.model import ScoringModel
from src.engine import train_one_epoch, evaluate
from src.feature_engineering import FeatureEngineer
from src.sam import SAM

def build_negative_pool(df_all_draws, pool_size, config):
    print(f"Building a pool of {pool_size} negative samples...")
    winning_num_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
    historical_sets = {tuple(sorted(draw)) for draw in df_all_draws[winning_num_cols].itertuples(index=False)}

    negative_pool = []
    with tqdm(total=pool_size, desc="Generating Negative Pool") as pbar:
        while len(negative_pool) < pool_size:
            candidate = tuple(sorted(random.sample(range(1, config['num_lotto_numbers'] + 1), 6)))
            if candidate not in historical_sets:
                negative_pool.append(list(candidate))
                pbar.update(1)
    return negative_pool

class ContrastiveDataset(Dataset):
    def __init__(self, df, fe, config, negative_pool):
        self.df, self.fe, self.config, self.negative_pool = df, fe, config, negative_pool
        self.winning_num_cols = [f'Winning_Num_{i}' for i in range(1, 7)]

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        positive_row = self.df.iloc[idx]
        positive_set = positive_row[self.winning_num_cols].astype(int).tolist()
        positive_features = self.fe.transform(positive_set, idx)
        neg_indices = np.random.choice(len(self.negative_pool), self.config['negative_samples'], replace=False)
        negative_features = np.array([self.fe.transform(self.negative_pool[i], idx) for i in neg_indices])

        return {"positive_features": torch.tensor(positive_features, dtype=torch.float32),
                "negative_features": torch.tensor(negative_features, dtype=torch.float32)}

def run_training():
    """The main pipeline for contrastive training of the scoring model."""
    print("--- Starting Scoring Model Training Pipeline (v3) ---")

    # --- CORRECTED DATA LOADING ---
    # Define column names manually to handle the malformed CSV header
    col_names = [
        'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
        'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
        'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
        '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
        'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
        'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
        'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
    ]
    # Skip the first 33 rows which contain the broken header
    df = pd.read_csv(CONFIG["data_path"], header=None, skiprows=33, names=col_names)
    df = df.sort_values(by='Date').reset_index(drop=True)
    # --- END OF CORRECTION ---

    feature_engineer = FeatureEngineer()
    feature_engineer.fit(df)

    negative_pool = build_negative_pool(df, pool_size=50000, config=CONFIG)

    train_size = int(len(df) * 0.85)
    train_df, val_df = df.iloc[:train_size], df.iloc[train_size:]
    train_dataset = ContrastiveDataset(train_df, feature_engineer, CONFIG, negative_pool)
    val_dataset = ContrastiveDataset(val_df, feature_engineer, CONFIG, negative_pool)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get feature dimension from the engineer
    sample_features = feature_engineer.transform([1,2,3,4,5,6], 0)
    CONFIG['d_features'] = len(sample_features)

    model = ScoringModel(CONFIG).to(device)

    if CONFIG['use_sam_optimizer']:
        print("Using Sharpness-Aware Minimization (SAM) optimizer.")
        base_optimizer = optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, lr=CONFIG["learning_rate"], rho=CONFIG['rho'])
    else:
        print("Using standard AdamW optimizer.")
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])

    print(f"\nStarting contrastive training on {device}...")
    for epoch in range(CONFIG["epochs"]):
        avg_train_loss = train_one_epoch(model, train_loader, optimizer, device, CONFIG)
        avg_val_loss = evaluate(model, val_loader, device, CONFIG)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Ranking Loss: {avg_train_loss:.4f} | Validation Ranking Loss: {avg_val_loss:.4f}")

    print("\nTraining complete. Saving model and feature engineer...")
    os.makedirs(os.path.dirname(CONFIG["model_save_path"]), exist_ok=True)
    torch.save(model.state_dict(), CONFIG["model_save_path"])
    joblib.dump(feature_engineer, CONFIG["feature_engineer_path"])
    print("Model and feature engineer saved successfully.")