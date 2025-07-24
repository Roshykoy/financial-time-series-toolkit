#!/usr/bin/env python3
"""
Test the updated main inference pipeline with the best_cvae_model.pth
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_inference():
    """Test the updated inference pipeline."""
    
    print("🔍 TESTING UPDATED INFERENCE PIPELINE")
    print("=" * 60)
    
    try:
        from src.inference_pipeline import run_inference
        from src.config_legacy import CONFIG
        
        print("✅ Imports successful")
        print(f"📂 Looking for model: models/best_cvae_model.pth")
        print(f"📂 Model exists: {os.path.exists('models/best_cvae_model.pth')}")
        print()
        
        # Test inference configuration
        inference_config = {
            'num_combinations': 3,
            'temperature': 0.618,
            'use_i_ching': True,
            'generation_mode': 'standard',
            'verbose': True
        }
        
        print("🚀 Testing inference with best model...")
        print("Configuration:")
        for key, value in inference_config.items():
            print(f"   {key}: {value}")
        print()
        
        # Run inference
        results = run_inference(inference_config)
        
        print("🎉 INFERENCE SUCCESSFUL!")
        print("✅ Updated pipeline working with best_cvae_model.pth")
        print()
        print("Generated combinations:")
        for i, combo in enumerate(results.get('generated_combinations', []), 1):
            print(f"   {i}. {' '.join(f'{n:2d}' for n in combo)}")
        
        print(f"\n📊 Results summary:")
        print(f"   Combinations: {len(results.get('generated_combinations', []))}")
        print(f"   Model used: {results.get('model_info', {}).get('model_type', 'unknown')}")
        print(f"   Device: {results.get('device_info', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ INFERENCE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    
    success = test_inference()
    
    print(f"\n{'='*60}")
    if success:
        print("🎯 RESULT: Updated inference pipeline is working!")
        print("   Your best_cvae_model.pth can now be used with main.py")
        print("   Run: python main.py → Option 2")
    else:
        print("⚠️ RESULT: Inference pipeline needs more updates")
        print("   Use fallback: python use_ultra_model.py")
    print("="*60)

if __name__ == "__main__":
    main()