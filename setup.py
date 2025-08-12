# setup.py
"""
Setup and installation script for Mark Six AI with Hyperparameter Optimization
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible.")
    return True

def check_conda():
    """Check if conda is available."""
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Conda found: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Conda not found. Please install Miniconda or Anaconda.")
    print("Download from: https://docs.conda.io/en/latest/miniconda.html")
    return False

def create_environment():
    """Create the conda environment."""
    print("\n📦 Creating conda environment...")
    
    if not os.path.exists('environment.yml'):
        print("❌ environment.yml not found. Please ensure you're in the project root directory.")
        return False
    
    try:
        # Check if environment already exists
        result = subprocess.run(['conda', 'env', 'list'], capture_output=True, text=True)
        if 'marksix_ai' in result.stdout:
            print("⚠️  Environment 'marksix_ai' already exists.")
            recreate = input("Recreate environment? (y/n): ").lower() == 'y'
            if recreate:
                print("🗑️  Removing existing environment...")
                subprocess.run(['conda', 'env', 'remove', '-n', 'marksix_ai', '-y'], check=True)
            else:
                print("✅ Using existing environment.")
                return True
        
        # Create new environment
        print("🔨 Creating new environment (this may take several minutes)...")
        result = subprocess.run(['conda', 'env', 'create', '-f', 'environment.yml'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Environment created successfully!")
            return True
        else:
            print("❌ Failed to create environment:")
            print(result.stderr)
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating environment: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating project directories...")
    
    directories = [
        'models',
        'outputs',
        'hyperparameter_results',
        'config',
        'data/raw',
        'data/processed'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_data_file():
    """Check if the data file exists."""
    data_path = 'data/raw/Mark_Six.csv'
    if os.path.exists(data_path):
        print(f"✅ Data file found: {data_path}")
        return True
    else:
        print(f"⚠️  Data file not found: {data_path}")
        print("Please download the Mark Six CSV file and place it in data/raw/")
        return False

def test_imports():
    """Test if all required modules can be imported."""
    print("\n🧪 Testing module imports...")
    
    test_modules = [
        'torch',
        'pandas', 
        'numpy',
        'sklearn',
        'tqdm',
        'matplotlib'
    ]
    
    failed_imports = []
    
    for module in test_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please check your conda environment installation.")
        return False
    
    print("✅ All required modules imported successfully!")
    return True

def check_gpu():
    """Check GPU availability."""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU available: {gpu_name}")
            print(f"✅ GPU memory: {gpu_memory:.1f} GB")
            
            # Test GPU functionality
            try:
                test_tensor = torch.randn(100, 100).cuda()
                test_result = test_tensor @ test_tensor.T
                print("✅ GPU computation test passed")
                return True
            except Exception as e:
                print(f"⚠️  GPU computation test failed: {e}")
                print("GPU detected but may not be working properly")
                return False
        else:
            print("⚠️  No GPU detected. Training will use CPU (slower).")
            return False
    except ImportError:
        print("❌ Cannot check GPU - PyTorch not available")
        return False

def create_sample_config():
    """Create a sample configuration file."""
    print("\n⚙️  Creating sample configuration...")
    
    sample_config = {
        "system_info": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.platform(),
            "gpu_available": False
        },
        "recommended_settings": {
            "cpu_only": {
                "batch_size": 32,
                "hidden_size": 256,
                "epochs": 10,
                "note": "Optimized for CPU training"
            },
            "gpu_available": {
                "batch_size": 64,
                "hidden_size": 512, 
                "epochs": 15,
                "note": "Optimized for GPU training"
            }
        }
    }
    
    try:
        import torch
        sample_config["system_info"]["gpu_available"] = torch.cuda.is_available()
    except ImportError:
        pass
    
    import json
    config_path = os.path.join('config', 'system_config.json')
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print(f"✅ System configuration saved to {config_path}")

def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("1. Activate the environment:")
    print("   conda activate marksix_ai")
    
    print("\n2. Ensure your data file is in place:")
    print("   data/raw/Mark_Six.csv")
    
    print("\n3. Start the application:")
    print("   python main.py")
    
    print("\n4. Try the new hyperparameter optimization:")
    print("   - Select option 4 from the main menu")
    print("   - Start with Random Search for best results")
    
    print("\n💡 TIPS:")
    print("- Read HYPERPARAMETER_OPTIMIZATION_GUIDE.md for detailed instructions")
    print("- Use Configuration Manager (Advanced Options) to manage settings")
    print("- Start with Quick Search if you want to test the feature first")
    
    print("\n🆘 IF YOU ENCOUNTER ISSUES:")
    print("- Check config/system_config.json for your system recommendations")
    print("- Use smaller batch_size if you get memory errors")
    print("- Reduce epochs_per_trial in hyperparameter optimization if training is slow")

def main():
    """Main setup function."""
    print("🚀 Mark Six AI Setup with Hyperparameter Optimization")
    print("="*60)
    
    # Check requirements
    if not check_python_version():
        sys.exit(1)
    
    if not check_conda():
        sys.exit(1)
    
    # Setup environment
    if not create_environment():
        print("\n❌ Environment setup failed. Please check the error messages above.")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Check data
    check_data_file()
    
    # Create sample config
    create_sample_config()
    
    # Print completion message
    print_next_steps()
    
    print(f"\n{'='*60}")
    print("Setup completed! You can now run: python main.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()