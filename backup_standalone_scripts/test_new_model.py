#!/usr/bin/env python3
"""
Test the newly trained model for compatibility.
"""

import sys
import torch
from pathlib import Path

def test_model_compatibility():
    """Test if the new model can be loaded with current architecture."""
    
    print("🔍 Testing New Model Compatibility")
    print("=" * 50)
    
    # Check which models exist
    models_dir = Path("models")
    model_files = [
        "best_cvae_model.pth",
        "cvae_model.pth", 
        "ultra_quick_model.pth"
    ]
    
    print("📁 Available models:")
    for model_file in model_files:
        model_path = models_dir / model_file
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024*1024)
            print(f"   ✅ {model_file} ({size_mb:.1f} MB)")
        else:
            print(f"   ❌ {model_file} (missing)")
    
    print()
    
    # Test loading the new model
    try:
        from src.cvae_model import ConditionalVAE
        from src.config_legacy import CONFIG
        
        print("🔄 Testing new model loading...")
        
        # Create model with current architecture
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ConditionalVAE(CONFIG).to(device)
        
        # Try loading the new model
        model_path = models_dir / "best_cvae_model.pth"
        if model_path.exists():
            model_data = torch.load(model_path, map_location=device)
            
            print(f"📊 New model structure:")
            print(f"   Keys: {list(model_data.keys())}")
            
            if 'cvae_state_dict' in model_data:
                # Load state dict
                model.load_state_dict(model_data['cvae_state_dict'])
                print("   ✅ Successfully loaded new model!")
                
                # Test inference
                print("🎲 Testing inference...")
                model.eval()
                
                # Create dummy inputs
                dummy_numbers = [[1, 2, 3, 4, 5, 6]]
                dummy_temporal = torch.zeros(1, CONFIG['temporal_sequence_length'], 6, device=device)
                dummy_indices = [0]
                dummy_pairs = torch.zeros(CONFIG['num_lotto_numbers'], CONFIG['num_lotto_numbers'])
                
                with torch.no_grad():
                    try:
                        output = model(dummy_numbers, dummy_pairs, dummy_temporal, dummy_indices)
                        print("   ✅ Inference test passed!")
                        print(f"   📈 Output shapes: {[x.shape if hasattr(x, 'shape') else type(x) for x in output]}")
                        return True
                    except Exception as e:
                        print(f"   ❌ Inference failed: {e}")
                        return False
            else:
                print("   ❌ No 'cvae_state_dict' key found")
                return False
        else:
            print("   ❌ Model file not found")
            return False
            
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def suggest_solution():
    """Suggest solution based on test results."""
    
    print("\n🎯 SOLUTION:")
    print("The newly trained model uses the updated architecture (sequence_processor)")
    print("but the inference pipeline may be using old configuration.")
    print()
    print("✅ Quick fixes:")
    print("1. Use ultra-quick model: python use_ultra_model.py")
    print("2. Use statistical predictions: python quick_predict.py") 
    print("3. Update inference pipeline to use new architecture")
    print()
    print("🔧 The model training was successful!")
    print("   Training time: 94.1 minutes")
    print("   Validation loss: 3.2686")
    print("   Model saved with new architecture")

if __name__ == "__main__":
    success = test_model_compatibility()
    suggest_solution()
    
    if success:
        print("\n🎉 New model is compatible and working!")
    else:
        print("\n⚠️  Model architecture mismatch detected.")
        print("   Use the ultra-quick model for immediate predictions.")