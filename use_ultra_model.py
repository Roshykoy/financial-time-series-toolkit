#!/usr/bin/env python3
"""
Use the ultra-quick trained model for immediate predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

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
    
    def generate(self, num_samples=1, device='cuda', temperature=1.0):
        """Generate new combinations."""
        with torch.no_grad():
            # Sample from prior
            z = torch.randn(num_samples, self.latent_dim, device=device)
            
            # Decode
            logits = self.decoder(z)
            logits = logits.view(num_samples, 6, 49) / temperature
            
            # Sample from categorical
            probs = torch.softmax(logits, dim=-1)
            samples = torch.multinomial(probs.view(-1, 49), 1).view(num_samples, 6)
            
            return samples + 1  # Convert to 1-49 range

def generate_predictions():
    """Generate predictions using ultra-quick model."""
    
    print("🎯 ULTRA-QUICK AI PREDICTIONS")
    print("=" * 50)
    
    # Load model
    model_path = Path("models/ultra_quick_model.pth")
    if not model_path.exists():
        print("❌ Ultra-quick model not found!")
        print("   Please run: python ultra_quick_train.py first")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model data
    model_data = torch.load(model_path, map_location=device)
    config = model_data['config']
    
    # Create and load model
    model = UltraMinimalCVAE(config)
    model.load_state_dict(model_data['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Ultra-quick AI model loaded")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Training epochs: {model_data['training_epochs']}")
    print(f"   Device: {device}")
    print()
    
    # Generate predictions with different temperatures
    temperatures = [0.8, 1.0, 1.2]  # Conservative, balanced, creative
    temp_names = ["Conservative", "Balanced", "Creative"]
    
    all_predictions = []
    
    for temp, name in zip(temperatures, temp_names):
        print(f"{name} AI Predictions (temp={temp}):")
        
        # Generate 3 combinations per temperature
        samples = model.generate(3, device, temperature=temp)
        
        for i, sample in enumerate(samples):
            numbers = sample.cpu().numpy()
            
            # Ensure uniqueness and valid range
            unique_numbers = []
            for num in numbers:
                if num not in unique_numbers and 1 <= num <= 49:
                    unique_numbers.append(int(num))
            
            # Fill missing numbers if needed
            while len(unique_numbers) < 6:
                new_num = np.random.randint(1, 50)
                if new_num not in unique_numbers:
                    unique_numbers.append(new_num)
            
            # Sort and add to all predictions
            combination = sorted(unique_numbers[:6])
            all_predictions.append(combination)
            
            print(f"  {len(all_predictions):2d}. {' '.join(f'{n:2d}' for n in combination)}")
        
        print()
    
    print("📋 SUMMARY - 9 AI-Generated Combinations:")
    print("=" * 45)
    for i, pred in enumerate(all_predictions, 1):
        strategy = "Cons" if i <= 3 else "Bal" if i <= 6 else "Crea"
        print(f"{i:2d}. {' '.join(f'{n:2d}' for n in pred)} [{strategy}]")
    
    print(f"\n🤖 AI Model Info:")
    print(f"   Type: Ultra-minimal CVAE")
    print(f"   Training: {model_data['training_epochs']} epochs (~5 min)")
    print(f"   Architecture: {model.hidden_size}→{model.latent_dim}→{model.hidden_size}")
    print(f"   Status: Ready for immediate use")
    
    print(f"\n💡 Temperature Settings:")
    print(f"   • Conservative (0.8): More likely numbers")
    print(f"   • Balanced (1.0): Standard AI prediction") 
    print(f"   • Creative (1.2): More diverse combinations")
    
    print(f"\n⚠️  Note: This is an ultra-quick model for immediate use.")
    print(f"   For better quality, use the full training later.")

if __name__ == "__main__":
    generate_predictions()