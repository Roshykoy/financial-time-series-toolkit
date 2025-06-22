# test_hyperparameter_optimization.py
"""
Test script to verify hyperparameter optimization is working correctly.
Run this before using the full optimization to catch any issues early.
"""

import os
import sys
import traceback
import torch
import tempfile
import shutil

# Add src to path
sys.path.append('src')

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from src.hyperparameter_optimizer import HyperparameterOptimizer
        from src.config import CONFIG
        print("✅ Hyperparameter optimizer imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_creation():
    """Test configuration creation and validation."""
    print("\n🔧 Testing configuration creation...")
    
    try:
        from src.hyperparameter_optimizer import HyperparameterOptimizer
        from src.config import CONFIG
        
        optimizer = HyperparameterOptimizer(CONFIG)
        
        # Test random config generation
        random_config = optimizer._generate_random_config()
        print(f"✅ Random config generated: {len(random_config)} parameters")
        
        # Validate config has required parameters
        required_params = ['learning_rate', 'hidden_size', 'num_layers', 'dropout']
        for param in required_params:
            if param not in random_config:
                print(f"❌ Missing required parameter: {param}")
                return False
        
        print("✅ Configuration validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading functionality."""
    print("\n📊 Testing data loading...")
    
    try:
        from src.hyperparameter_optimizer import HyperparameterOptimizer
        from src.config import CONFIG
        
        optimizer = HyperparameterOptimizer(CONFIG)
        
        # Check if data file exists
        if not os.path.exists(CONFIG["data_path"]):
            print(f"⚠️  Data file not found: {CONFIG['data_path']}")
            print("This test requires the Mark Six CSV file to be in place.")
            return False
        
        # Test data loading
        df = optimizer._load_data()
        print(f"✅ Data loaded: {len(df)} rows")
        
        if len(df) < 100:
            print("⚠️  Warning: Very small dataset, results may not be reliable")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation with different configurations."""
    print("\n🧠 Testing model creation...")
    
    try:
        from src.model import ScoringModel
        from src.feature_engineering import FeatureEngineer
        from src.hyperparameter_optimizer import HyperparameterOptimizer
        from src.config import CONFIG
        
        optimizer = HyperparameterOptimizer(CONFIG)
        
        # Create feature engineer (needed for d_features)
        if os.path.exists(CONFIG["data_path"]):
            df = optimizer._load_data()
            fe = FeatureEngineer()
            fe.fit(df)
            sample_features = fe.transform([1,2,3,4,5,6], 0)
            d_features = len(sample_features)
        else:
            print("⚠️  Using default d_features (data file not found)")
            d_features = 16
        
        # Test different model configurations
        test_configs = [
            {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.1, 'd_features': d_features},
            {'hidden_size': 256, 'num_layers': 6, 'dropout': 0.2, 'd_features': d_features},
            {'hidden_size': 512, 'num_layers': 4, 'dropout': 0.15, 'd_features': d_features}
        ]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Testing on device: {device}")
        
        for i, config in enumerate(test_configs):
            model = ScoringModel(config).to(device)
            
            # Test forward pass
            test_input = torch.randn(1, d_features).to(device)
            output = model(test_input)
            
            print(f"✅ Model {i+1} created and tested: output shape {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        traceback.print_exc()
        return False

def test_quick_optimization():
    """Test a very quick optimization run."""
    print("\n⚡ Testing quick optimization (this may take a few minutes)...")
    
    try:
        from src.hyperparameter_optimizer import HyperparameterOptimizer
        from src.config import CONFIG
        
        # Check if data file exists
        if not os.path.exists(CONFIG["data_path"]):
            print("⚠️  Skipping optimization test - data file not found")
            return True
        
        # Create temporary results directory
        temp_dir = tempfile.mkdtemp()
        original_dir = None
        
        try:
            optimizer = HyperparameterOptimizer(CONFIG)
            
            # Temporarily change results directory
            original_dir = optimizer.results_dir
            optimizer.results_dir = temp_dir
            
            print("Running 2 quick trials with 1 epoch each...")
            
            # Run very quick optimization
            best_config, best_score = optimizer.random_search(num_trials=2, epochs_per_trial=1)
            
            print(f"✅ Quick optimization completed!")
            print(f"Best score: {best_score:.4f}")
            print(f"Optimization history: {len(optimizer.optimization_history)} trials")
            
            return True
            
        finally:
            # Cleanup temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Restore original directory
            if original_dir:
                optimizer.results_dir = original_dir
        
    except Exception as e:
        print(f"❌ Quick optimization test failed: {e}")
        traceback.print_exc()
        return False

def test_file_operations():
    """Test file saving and loading operations."""
    print("\n💾 Testing file operations...")
    
    try:
        import json
        import tempfile
        
        # Test JSON operations
        test_data = {
            'config': {'learning_rate': 0.001, 'batch_size': 64},
            'score': 0.75,
            'trial_num': 1
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f, indent=2)
            temp_file = f.name
        
        # Test loading
        with open(temp_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Cleanup
        os.unlink(temp_file)
        
        if loaded_data == test_data:
            print("✅ File operations test passed")
            return True
        else:
            print("❌ File operations test failed - data mismatch")
            return False
            
    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        traceback.print_exc()
        return False

def test_gpu_functionality():
    """Test GPU functionality if available."""
    print("\n🎮 Testing GPU functionality...")
    
    try:
        if not torch.cuda.is_available():
            print("ℹ️  No GPU available, skipping GPU tests")
            return True
        
        # Test basic GPU operations
        device = torch.device('cuda')
        test_tensor = torch.randn(100, 100).to(device)
        result = torch.matmul(test_tensor, test_tensor.T)
        
        print(f"✅ GPU test passed on {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test model on GPU
        from src.model import ScoringModel
        config = {'hidden_size': 128, 'num_layers': 3, 'dropout': 0.1, 'd_features': 16}
        model = ScoringModel(config).to(device)
        
        test_input = torch.randn(1, 16).to(device)
        output = model(test_input)
        
        print("✅ Model GPU test passed")
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 HYPERPARAMETER OPTIMIZATION TEST SUITE")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config_creation),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("File Operations", test_file_operations),
        ("GPU Functionality", test_gpu_functionality),
        ("Quick Optimization", test_quick_optimization),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name.upper()} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("Your hyperparameter optimization setup is working correctly.")
        print("You can now run the full optimization from the main menu.")
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("Please check the error messages above and fix any issues.")
        print("You may still be able to use basic functionality.")
    
    print("\n💡 NEXT STEPS:")
    print("- If all tests passed: Run 'python main.py' and select option 4")
    print("- If tests failed: Check your environment and data file setup")
    print("- For quick testing: Use Custom Quick Search (5 trials)")

if __name__ == "__main__":
    main()