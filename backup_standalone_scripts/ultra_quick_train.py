#!/usr/bin/env python3
"""
ULTRA-QUICK 5-minute training for IMMEDIATE use.
Absolutely minimal setup just to get a working model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from src.config_legacy import CONFIG

class UltraMinimalCVAE(nn.Module):
    """Ultra-simple CVAE for quick training."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = 64  # Very small
        self.latent_dim = 32   # Very small
        
        # Simple encoder: 6 numbers -> latent space
        self.encoder = nn.Sequential(
            nn.Linear(6, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2 * self.latent_dim)  # mu, logvar
        )
        
        # Simple decoder: latent -> 6 numbers
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 6 * 49)  # 6 positions × 49 numbers
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # x: [batch, 6] normalized winning numbers
        
        # Encode
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        logits = self.decoder(z)
        logits = logits.view(-1, 6, 49)  # [batch, 6, 49]
        
        return logits, mu, logvar
    
    def generate(self, num_samples=1, device='cuda'):
        """Generate new combinations."""
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim, device=device)
            
            # Decode
            logits = self.decoder(z)
            logits = logits.view(num_samples, 6, 49)
            
            # Sample from categorical
            probs = torch.softmax(logits, dim=-1)
            samples = torch.multinomial(probs.view(-1, 49), 1).view(num_samples, 6)
            
            return samples + 1  # Convert to 1-49 range

def ultra_quick_train():
    """Ultra-fast training in 5 minutes."""
    
    print("🚀 ULTRA-QUICK TRAINING - 5 Minutes")
    print("=" * 50)
    print("⚡ Minimal model for immediate use")
    print()
    
    # Load data quickly
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
    
    # Extract winning numbers and normalize
    winning_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
    data = df[winning_cols].values.astype(np.float32)
    data = (data - 1) / 48.0  # Normalize to [0,1]
    
    print(f"✅ Processed {len(data)} combinations")
    
    # Create simple train/val split
    train_size = int(0.8 * len(data))
    train_data = torch.tensor(data[:train_size])
    val_data = torch.tensor(data[train_size:])
    
    # Ultra-minimal model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UltraMinimalCVAE(CONFIG).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"✅ Ultra-minimal model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"   Device: {device}")
    print()
    
    # Ultra-fast training
    print("⚡ Training (3 epochs only)...")
    model.train()
    
    batch_size = 64
    epochs = 3
    
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Simple batching
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, mu, logvar = model(batch)
            
            # Simple reconstruction loss
            target = ((batch * 48) + 1).long() - 1  # Back to 0-48 range
            recon_loss = 0
            for pos in range(6):
                recon_loss += nn.functional.cross_entropy(logits[:, pos], target[:, pos])
            recon_loss /= 6
            
            # Simple KL loss
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch.size(0)
            
            # Total loss
            loss = recon_loss + 0.01 * kl_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch+1}/3: Loss = {avg_loss:.4f}")
    
    print("✅ Training completed!")
    
    # Save ultra-minimal model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    ultra_model_path = models_dir / "ultra_quick_model.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': CONFIG,
        'model_type': 'ultra_minimal',
        'training_epochs': epochs,
        'final_loss': avg_loss
    }, ultra_model_path)
    
    print(f"💾 Model saved: {ultra_model_path}")
    
    # Test generation
    print("\n🎲 Testing generation...")
    model.eval()
    samples = model.generate(5, device)
    
    print("Sample predictions:")
    for i, sample in enumerate(samples):
        numbers = sample.cpu().numpy()
        # Ensure uniqueness
        unique_numbers = []
        for num in numbers:
            if num not in unique_numbers and 1 <= num <= 49:
                unique_numbers.append(int(num))
        
        # Fill to 6 if needed
        while len(unique_numbers) < 6:
            new_num = np.random.randint(1, 50)
            if new_num not in unique_numbers:
                unique_numbers.append(new_num)
        
        print(f"  {i+1}. {' '.join(f'{n:2d}' for n in sorted(unique_numbers[:6]))}")
    
    print(f"\n🎯 Ultra-quick model ready!")
    print(f"   Time: ~5 minutes")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Use with: python use_ultra_model.py")

if __name__ == "__main__":
    ultra_quick_train()