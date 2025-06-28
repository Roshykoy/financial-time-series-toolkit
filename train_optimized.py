#!/usr/bin/env python3
"""
Train CVAE model with optimized settings based on hyperparameter optimization insights.
"""

import torch
from src.cvae_engine import train_one_epoch_cvae, evaluate_cvae
from src.cvae_model import ConditionalVAE
from src.meta_learner import AttentionMetaLearner
from src.cvae_data_loader import create_cvae_data_loaders
from src.feature_engineering import FeatureEngineer
from src.config_legacy import CONFIG
import pandas as pd
import numpy as np
import time
from pathlib import Path

def train_optimized_model():
    """Train CVAE model with settings optimized for your hardware."""
    
    print("🚀 Training CVAE Model with Optimized Settings")
    print("=" * 60)
    
    # Load data
    print("📊 Loading Mark Six data...")
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
    
    # Create optimized configuration based on your hardware (RTX 3080, 16GB RAM)
    config = CONFIG.copy()
    
    # Hardware-optimized settings
    config.update({
        'device': 'cuda',
        'batch_size': 16,              # Optimal for RTX 3080
        'epochs': 20,                  # Reasonable training duration
        'learning_rate': 5e-4,         # Conservative learning rate
        'hidden_size': 256,            # Balanced model size
        'latent_dim': 128,             # Good representation capacity
        'dropout': 0.2,                # Moderate regularization
        
        # Training stability
        'gradient_clip_norm': 1.0,     # Prevent gradient explosion
        'early_stopping_patience': 5, # Stop if no improvement
        'save_frequency': 5,           # Save every 5 epochs
        
        # Loss weights (conservative)
        'reconstruction_weight': 1.0,
        'kl_weight': 0.1,              # Reduced from default
        'contrastive_weight': 0.1,     # Reduced from default
        'meta_learning_weight': 0.01,
        
        # Logging
        'log_interval': 50,
    })
    
    print(f"🔧 Configuration:")
    print(f"   Device: {config['device']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Learning Rate: {config['learning_rate']}")
    print(f"   Hidden Size: {config['hidden_size']}")
    print()
    
    # Prepare data
    print("🔄 Preparing data and features...")
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(df)
    
    train_loader, val_loader = create_cvae_data_loaders(df, feature_engineer, config)
    print(f"✅ Created data loaders: {len(train_loader)} train, {len(val_loader)} validation batches")
    
    # Create models
    print("🏗️ Creating models...")
    device = torch.device(config['device'])
    cvae_model = ConditionalVAE(config).to(device)
    meta_learner = AttentionMetaLearner(config).to(device)
    
    # Create optimizers
    optimizers = {
        'cvae': torch.optim.AdamW(cvae_model.parameters(), lr=config['learning_rate'], weight_decay=1e-5),
        'meta': torch.optim.AdamW(meta_learner.parameters(), lr=config['learning_rate'], weight_decay=1e-5)
    }
    
    print(f"✅ Models created on {device}")
    cvae_params = sum(p.numel() for p in cvae_model.parameters())
    meta_params = sum(p.numel() for p in meta_learner.parameters())
    print(f"   CVAE parameters: {cvae_params:,}")
    print(f"   Meta-learner parameters: {meta_params:,}")
    print()
    
    # Training loop
    print("🚀 Starting training...")
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train for one epoch
        train_losses = train_one_epoch_cvae(
            cvae_model, meta_learner, train_loader, 
            optimizers, device, config, epoch
        )
        
        # Validate
        val_loss, val_metrics = evaluate_cvae(
            cvae_model, meta_learner, val_loader, device, config
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        print(f"Epoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s):")
        print(f"  Train Loss: {train_losses.get('total_cvae_loss', 0):.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Components: Recon={train_losses.get('reconstruction_loss', 0):.3f}, "
              f"KL={train_losses.get('kl_loss', 0):.3f}, "
              f"Contrast={train_losses.get('contrastive_loss', 0):.3f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            torch.save({
                'cvae_state_dict': cvae_model.state_dict(),
                'meta_learner_state_dict': meta_learner.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_metrics': val_metrics
            }, model_dir / "best_cvae_model.pth")
            
            print(f"  ✅ New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"  ⏹️ Early stopping at epoch {epoch+1}")
                break
        
        print()
        
        # Memory cleanup
        if (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    
    print("🎉 Training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: models/best_cvae_model.pth")
    print()
    print("🎯 Next steps:")
    print("   1. Use main.py → Option 2 to generate predictions")
    print("   2. Use main.py → Option 3 to evaluate the model")
    print("   3. Check models/ directory for saved model files")

if __name__ == "__main__":
    train_optimized_model()