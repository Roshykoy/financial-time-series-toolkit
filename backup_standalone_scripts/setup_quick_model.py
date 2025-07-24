#!/usr/bin/env python3
"""
Setup script to configure the system to use the quick-trained model.
"""

import shutil
from pathlib import Path
import torch

def setup_quick_model():
    """Setup the quick model for use with main.py"""
    
    print("🔧 Setting up quick model for main.py...")
    
    models_dir = Path("models")
    quick_model = models_dir / "quick_cvae_model.pth"
    target_model = models_dir / "cvae_model.pth"
    
    if not quick_model.exists():
        print("❌ Quick model not found. Please run: python quick_train.py first")
        return False
    
    # Copy quick model to expected location
    shutil.copy2(quick_model, target_model)
    print(f"✅ Quick model copied to {target_model}")
    
    # Verify the model can be loaded
    try:
        model_data = torch.load(quick_model, map_location='cpu')
        print(f"✅ Model verification:")
        print(f"   Epochs trained: {model_data['epoch'] + 1}")
        print(f"   Training time: {model_data['training_time_minutes']:.1f} minutes")
        print(f"   Validation loss: {model_data['best_val_loss']:.4f}")
        print(f"   Model type: {model_data['model_type']}")
        
        # Also copy meta-learner if exists
        if 'meta_learner_state_dict' in model_data:
            torch.save(model_data['meta_learner_state_dict'], models_dir / "meta_learner.pth")
            print(f"✅ Meta-learner also configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Model verification failed: {e}")
        return False

if __name__ == "__main__":
    success = setup_quick_model()
    if success:
        print(f"\n🎯 Quick model ready!")
        print(f"   Now you can use: python main.py → Option 2")
    else:
        print(f"\n❌ Setup failed. Run quick training first:")
        print(f"   python quick_train.py")