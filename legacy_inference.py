#!/usr/bin/env python3
"""
Legacy inference script that can load old LSTM-based models.
Use this temporarily until you retrain with the new architecture.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Import legacy temporal encoder
from src.temporal_context_legacy import TemporalContextEncoder as LegacyTemporalContextEncoder
from src.graph_encoder import GraphContextEncoder
from src.meta_learner import AttentionMetaLearner
from src.feature_engineering import FeatureEngineer
from src.config_legacy import CONFIG

class LegacyCVAEModel(nn.Module):
    """
    CVAE model with legacy LSTM-based temporal encoder for loading old models.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_numbers = config['num_lotto_numbers']
        self.latent_dim = config['latent_dim']
        self.hidden_size = config['hidden_size']
        self.context_dim = config['temporal_context_dim']
        
        # Core components with legacy temporal encoder
        self.temporal_encoder = LegacyTemporalContextEncoder(config)
        self.graph_encoder = GraphContextEncoder(config)
        
        # Rest of the architecture remains the same...
        self.pair_embedding = nn.Embedding(
            num_embeddings=config['num_lotto_numbers'] * config['num_lotto_numbers'],
            embedding_dim=config['pair_embedding_dim']
        )
        
        # Encoder network
        self.encoder_input_dim = (
            6 * config['number_embedding_dim'] +  # 6 numbers
            config['pair_embedding_dim'] * 15 +   # 15 pairs
            config['graph_context_dim']            # Graph features
        )
        
        self.encoder = nn.Sequential(
            nn.Linear(self.encoder_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.hidden_size, 2 * self.latent_dim)  # mu and logvar
        )
        
        # Prior network (context-dependent)
        self.prior_network = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.hidden_size, 2 * self.latent_dim)  # mu_prior and logvar_prior
        )
        
        # Decoder
        self.decoder_input_dim = self.latent_dim + self.context_dim
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_input_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.hidden_size, 6 * self.num_numbers)  # 6 positions × num_numbers
        )
        
        # Number embeddings
        self.number_embeddings = nn.Embedding(self.num_numbers + 1, config['number_embedding_dim'])

def load_legacy_model(model_path, device='cuda'):
    """Load a legacy CVAE model with LSTM architecture."""
    
    print(f"🔄 Loading legacy model from {model_path}")
    
    # Create legacy model
    model = LegacyCVAEModel(CONFIG)
    
    # Load state dict
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("✅ Legacy model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading legacy model: {e}")
        return None

def generate_predictions_legacy():
    """Generate predictions using legacy model."""
    
    print("🎯 Legacy Inference with LSTM Model")
    print("="*50)
    
    # Load data
    print("📊 Loading data...")
    data_path = "data/raw/Mark_Six.csv"
    col_names = [
        'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
        'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
        'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
        '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Div_2_Winners',
        'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize', 'Div_4_Winners',
        'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
        'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
    ]
    
    df = pd.read_csv(data_path, header=None, skiprows=33, names=col_names)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)
    print(f"✅ Loaded {len(df)} records")
    
    # Load feature engineer
    feature_path = "models/feature_engineer.pkl"
    if Path(feature_path).exists():
        with open(feature_path, 'rb') as f:
            feature_engineer = pickle.load(f)
        print("✅ Feature engineer loaded")
    else:
        print("⚠️ Creating new feature engineer...")
        feature_engineer = FeatureEngineer()
        feature_engineer.fit(df)
    
    # Try to load legacy model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_paths = [
        "models/cvae_model.pth",
        "models/conservative_cvae_model.pth"
    ]
    
    model = None
    for path in model_paths:
        if Path(path).exists():
            print(f"🔄 Trying to load {path}...")
            model = load_legacy_model(path, device)
            if model is not None:
                break
    
    if model is None:
        print("❌ Could not load any model. Please retrain with:")
        print("   python train_optimized.py")
        return
    
    # Generate predictions
    print("\n🎲 Generating predictions...")
    
    # Get recent temporal context
    recent_draws = df.tail(CONFIG['temporal_sequence_length'])
    winning_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
    
    # Simple prediction generation (without full inference pipeline)
    with torch.no_grad():
        # Create dummy temporal sequence
        sequence = torch.zeros(1, CONFIG['temporal_sequence_length'], 6, dtype=torch.long, device=device)
        for i, (_, row) in enumerate(recent_draws.iterrows()):
            if i < CONFIG['temporal_sequence_length']:
                numbers = row[winning_cols].values
                sequence[0, i] = torch.tensor(numbers, dtype=torch.long)
        
        # Get temporal context
        temporal_context, _ = model.temporal_encoder(sequence)
        
        # Sample from prior
        mu_prior, logvar_prior = torch.chunk(model.prior_network(temporal_context), 2, dim=-1)
        z = torch.randn(1, model.latent_dim, device=device) * torch.exp(0.5 * logvar_prior) + mu_prior
        
        # Decode
        decoder_input = torch.cat([z, temporal_context], dim=-1)
        logits = model.decoder(decoder_input)
        logits = logits.view(1, 6, model.num_numbers)
        
        # Generate multiple sets
        predictions = []
        for _ in range(9):  # Generate 9 sets
            # Sample from categorical distribution
            probs = torch.softmax(logits, dim=-1)
            samples = torch.multinomial(probs.view(-1, model.num_numbers), 1).view(6) + 1
            
            # Ensure uniqueness
            numbers = samples.cpu().numpy().tolist()
            numbers = list(dict.fromkeys(numbers))  # Remove duplicates
            while len(numbers) < 6:
                new_num = torch.randint(1, model.num_numbers + 1, (1,)).item()
                if new_num not in numbers:
                    numbers.append(new_num)
            
            predictions.append(sorted(numbers[:6]))
    
    # Display results
    print("\n🎯 Generated Predictions:")
    print("="*40)
    for i, pred in enumerate(predictions, 1):
        print(f"{i:2d}. {' '.join(f'{n:2d}' for n in pred)}")
    
    print(f"\n📊 Model Info:")
    print(f"   Architecture: Legacy LSTM-based CVAE")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n⚠️  Note: This is using the legacy LSTM model.")
    print(f"   For best results, retrain with: python train_optimized.py")

if __name__ == "__main__":
    generate_predictions_legacy()