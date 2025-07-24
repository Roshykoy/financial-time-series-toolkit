#!/usr/bin/env python3
"""
Verify that the configuration fix resolves the model size mismatch.
"""

import torch
from pathlib import Path

def verify_model_config():
    """Verify the saved model configuration."""
    
    print("🔍 VERIFYING MODEL CONFIGURATION FIX")
    print("=" * 50)
    
    # Load the saved model
    model_path = "models/best_cvae_model.pth"
    if not Path(model_path).exists():
        print("❌ best_cvae_model.pth not found")
        return False
    
    try:
        model_data = torch.load(model_path, map_location='cpu')
        print(f"✅ Loaded model data from {model_path}")
        
        # Check structure
        print(f"📊 Model data keys: {list(model_data.keys())}")
        
        if 'config' in model_data:
            config = model_data['config']
            print(f"✅ Found saved configuration:")
            print(f"   hidden_size: {config.get('hidden_size', 'not found')}")
            print(f"   latent_dim: {config.get('latent_dim', 'not found')}")
            print(f"   device: {config.get('device', 'not found')}")
            print(f"   learning_rate: {config.get('learning_rate', 'not found')}")
            
            # Now test if we can create models with this config
            try:
                from src.cvae_model import ConditionalVAE
                from src.meta_learner import AttentionMetaLearner
                
                print(f"\n🔧 Testing model creation with saved config...")
                
                # Create models with saved config
                device = torch.device('cpu')
                cvae_model = ConditionalVAE(config).to(device)
                meta_learner = AttentionMetaLearner(config).to(device)
                
                print(f"✅ Models created successfully with saved config")
                
                # Test loading state dicts
                if 'cvae_state_dict' in model_data:
                    cvae_model.load_state_dict(model_data['cvae_state_dict'])
                    print(f"✅ CVAE state dict loaded successfully")
                
                if 'meta_learner_state_dict' in model_data:
                    meta_learner.load_state_dict(model_data['meta_learner_state_dict'])
                    print(f"✅ Meta-learner state dict loaded successfully")
                
                print(f"\n🎉 CONFIGURATION FIX VERIFIED!")
                print(f"   The updated inference pipeline should now work")
                print(f"   Model dimensions match between saved and loaded models")
                
                return True
                
            except Exception as e:
                print(f"❌ Model creation/loading failed: {e}")
                return False
        else:
            print("❌ No 'config' key found in saved model")
            print("   The model was saved without configuration")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def show_next_steps(success):
    """Show next steps based on verification result."""
    
    print(f"\n{'='*50}")
    if success:
        print("🎯 VERIFICATION SUCCESSFUL!")
        print("   ✅ Model configuration fix is working")
        print("   ✅ Dimension mismatch resolved")
        print("   ✅ Ready to use with main.py")
        print()
        print("🚀 Next steps:")
        print("   conda activate marksix_ai")
        print("   python main.py → Option 2")
        print("   Your 94.1-minute trained model will load correctly!")
    else:
        print("⚠️ VERIFICATION INCOMPLETE")
        print("   ℹ️  Some issues may remain")
        print("   ℹ️  Use fallback options for now")
        print()
        print("🔄 Fallback options:")
        print("   python use_ultra_model.py  # Ultra-quick AI model")
        print("   python quick_predict.py    # Statistical analysis")
    print("="*50)

if __name__ == "__main__":
    success = verify_model_config()
    show_next_steps(success)