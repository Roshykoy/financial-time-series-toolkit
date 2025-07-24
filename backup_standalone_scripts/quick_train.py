#!/usr/bin/env python3
"""
QUICK 10-15 minute training for immediate use.
Minimal epochs with fast convergence settings.
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

def quick_train():
    """Train CVAE model with FAST settings for immediate use."""
    
    print("⚡ QUICK TRAINING - 10-15 Minutes")
    print("=" * 50)
    print("🎯 Minimal training for immediate functionality")
    print()
    
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
    
    # ULTRA-FAST configuration for 10-15 minute training
    config = CONFIG.copy()
    config.update({
        'device': 'cuda',
        'batch_size': 32,              # Larger batches = faster training
        'epochs': 5,                   # Just 5 epochs for quick results
        'learning_rate': 1e-3,         # Higher LR for faster convergence
        'hidden_size': 128,            # Smaller model = faster training
        'latent_dim': 64,              # Reduced latent space
        'dropout': 0.1,                # Minimal dropout
        
        # Speed optimizations
        'temporal_sequence_length': 10, # Shorter sequences
        'negative_samples': 8,          # Fewer negative samples
        'temporal_lstm_layers': 1,      # Single layer LSTM
        'temporal_hidden_dim': 64,      # Smaller temporal model
        'temporal_attention_heads': 2,  # Fewer attention heads
        
        # Fast convergence settings
        'gradient_clip_norm': 2.0,      # Allow larger gradients
        'early_stopping_patience': 2,  # Quick early stopping
        
        # Aggressive loss weights for fast learning
        'reconstruction_weight': 1.0,
        'kl_weight': 0.05,              # Very low KL weight
        'contrastive_weight': 0.05,     # Very low contrastive weight
        'meta_learning_weight': 0.001,  # Minimal meta-learning
        
        # Logging
        'log_interval': 10,             # Frequent progress updates
    })
    
    print(f"⚡ FAST Configuration:")
    print(f"   Epochs: {config['epochs']} (ultra-fast)")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Learning Rate: {config['learning_rate']} (aggressive)")
    print(f"   Model Size: {config['hidden_size']} hidden units")
    print(f"   Estimated Time: 10-15 minutes")
    print()
    
    # Prepare data
    print("🔄 Preparing data...")
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(df)
    
    train_loader, val_loader = create_cvae_data_loaders(df, feature_engineer, config)
    print(f"✅ Data ready: {len(train_loader)} train, {len(val_loader)} validation batches")
    
    # Create smaller, faster models
    print("🏗️ Creating fast models...")
    device = torch.device(config['device'])
    cvae_model = ConditionalVAE(config).to(device)
    meta_learner = AttentionMetaLearner(config).to(device)
    
    # Aggressive optimizers for fast training
    optimizers = {
        'cvae': torch.optim.Adam(cvae_model.parameters(), lr=config['learning_rate'], weight_decay=1e-6),
        'meta': torch.optim.Adam(meta_learner.parameters(), lr=config['learning_rate'], weight_decay=1e-6)
    }
    
    cvae_params = sum(p.numel() for p in cvae_model.parameters())
    meta_params = sum(p.numel() for p in meta_learner.parameters())
    print(f"✅ Fast models created:")
    print(f"   CVAE: {cvae_params:,} parameters")
    print(f"   Meta-learner: {meta_params:,} parameters")
    print(f"   Total: {cvae_params + meta_params:,} parameters")
    print()
    
    # Quick training loop
    print("⚡ Starting FAST training...")
    start_time = time.time()
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{config['epochs']}:")
        
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
        elapsed_total = time.time() - start_time
        
        # Print progress
        print(f"  ⏱️  Time: {epoch_time:.1f}s (Total: {elapsed_total/60:.1f}min)")
        print(f"  📊 Train Loss: {train_losses.get('total_cvae_loss', 0):.4f}")
        print(f"  📈 Val Loss: {val_loss:.4f}")
        print(f"  🔧 Recon: {train_losses.get('reconstruction_loss', 0):.3f}, "
              f"KL: {train_losses.get('kl_loss', 0):.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save quick model
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)
            
            # Save with clear naming
            quick_model_path = model_dir / "quick_cvae_model.pth"
            torch.save({
                'cvae_state_dict': cvae_model.state_dict(),
                'meta_learner_state_dict': meta_learner.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_metrics': val_metrics,
                'training_time_minutes': elapsed_total / 60,
                'model_type': 'quick_training'
            }, quick_model_path)
            
            print(f"  ✅ Best model saved: {quick_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"  ⏹️ Early stopping (no improvement)")
                break
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 QUICK TRAINING COMPLETED!")
    print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"   📈 Best validation loss: {best_val_loss:.4f}")
    print(f"   💾 Model saved: models/quick_cvae_model.pth")
    print()
    print(f"🎯 Ready for immediate use:")
    print(f"   python main.py → Option 2 (Generate Predictions)")
    print(f"   python main.py → Option 3 (Evaluate Model)")
    print()
    print(f"⚠️  Note: This is a QUICK model for immediate use.")
    print(f"   For best results later, run full training with:")
    print(f"   python train_optimized.py")

if __name__ == "__main__":
    quick_train()