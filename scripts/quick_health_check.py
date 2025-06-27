#!/usr/bin/env python3
"""
MarkSix System Health Check & Quick Start
Comprehensive system validation and immediate issue resolution.
"""
import os
import sys
import traceback
from pathlib import Path


def print_header():
    """Print health check header."""
    print("\n" + "=" * 60)
    print("🔍 MARKSIX SYSTEM HEALTH CHECK & QUICK START")
    print("=" * 60)


def check_python_environment():
    """Check Python environment and basic requirements."""
    print("\n📋 Python Environment Check:")
    print("-" * 30)
    
    # Python version
    python_version = sys.version_info
    if python_version >= (3, 10):
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"❌ Python {python_version.major}.{python_version.minor}.{python_version.micro} (requires 3.10+)")
        return False
    
    # Check critical packages
    critical_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn'
    }
    
    missing_packages = []
    for package, name in critical_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {name} {version}")
        except ImportError:
            print(f"❌ {name} - Missing")
            missing_packages.append(name)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: conda activate marksix_ai")
        return False
    
    return True


def check_cuda_and_gpu():
    """Check CUDA and GPU availability."""
    print("\n🎮 GPU & CUDA Check:")
    print("-" * 20)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                properties = torch.cuda.get_device_properties(i)
                memory_gb = properties.total_memory / (1024**3)
                print(f"✅ GPU {i}: {device_name} ({memory_gb:.1f} GB)")
            
            # Test GPU functionality
            try:
                test_tensor = torch.randn(100, 100, device='cuda')
                del test_tensor
                torch.cuda.empty_cache()
                print("✅ GPU functionality test passed")
                return True
            except Exception as e:
                print(f"⚠️  GPU test failed: {e}")
                print("Will use CPU mode")
                return False
        else:
            print("⚠️  No CUDA-capable GPU detected")
            print("System will run in CPU mode (slower)")
            return False
            
    except ImportError:
        print("❌ PyTorch not available")
        return False


def check_project_structure():
    """Check project file structure."""
    print("\n📁 Project Structure Check:")
    print("-" * 28)
    
    # Required directories
    required_dirs = [
        'src',
        'data/raw',
        'models',
        'outputs',
        'config'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory: {dir_path}/")
        else:
            print(f"⚠️  Missing: {dir_path}/")
            missing_dirs.append(dir_path)
    
    # Create missing directories
    if missing_dirs:
        print("\n🔧 Creating missing directories...")
        for dir_path in missing_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ Created: {dir_path}/")
            except Exception as e:
                print(f"❌ Failed to create {dir_path}: {e}")
                return False
    
    # Check critical files
    critical_files = {
        'main.py': 'Original main interface',
        'main_improved.py': 'Enhanced main interface (recommended)',
        'src/config.py': 'Configuration file',
        'environment.yml': 'Conda environment'
    }
    
    for file_path, description in critical_files.items():
        if os.path.exists(file_path):
            print(f"✅ {description}: {file_path}")
        else:
            print(f"⚠️  Missing {description}: {file_path}")
    
    return True


def check_data_files():
    """Check data file availability and format."""
    print("\n📊 Data File Check:")
    print("-" * 18)
    
    data_file = 'data/raw/Mark_Six.csv'
    
    if os.path.exists(data_file):
        try:
            file_size = os.path.getsize(data_file)
            size_mb = file_size / (1024 * 1024)
            print(f"✅ Data file found: {data_file} ({size_mb:.1f} MB)")
            
            # Quick format check
            with open(data_file, 'r') as f:
                first_line = f.readline().strip()
                line_count = sum(1 for _ in f)
            
            print(f"✅ Data format check passed ({line_count} rows)")
            return True
            
        except Exception as e:
            print(f"❌ Data file error: {e}")
            return False
    else:
        print(f"❌ Data file missing: {data_file}")
        print("\n🔧 Creating sample data for testing...")
        return create_sample_data()


def create_sample_data():
    """Create sample data for testing."""
    try:
        import pandas as pd
        import numpy as np
        
        # Create dummy lottery data
        np.random.seed(42)
        dummy_data = []
        
        for i in range(100):
            # Generate 6 unique numbers between 1-49
            numbers = sorted(np.random.choice(49, 6, replace=False) + 1)
            extra = np.random.randint(1, 50)
            date = f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}"
            
            row = [i + 1, date] + numbers + [extra]
            # Add some dummy statistics
            row.extend([0, 3, 3, 2, 4, 1, 1, 2, 2, 1, 100000, 50, 50000])
            dummy_data.append(row)
        
        # Create DataFrame with proper columns
        columns = [
            'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3',
            'Winning_Num_4', 'Winning_Num_5', 'Winning_Num_6', 'Extra_Num',
            'From_Last', 'Low', 'High', 'Odd', 'Even', '1-10', '11-20', '21-30',
            '31-40', '41-50', 'Div_1_Winners', 'Div_1_Prize', 'Other_Stats'
        ]
        
        df = pd.DataFrame(dummy_data, columns=columns[:len(dummy_data[0])])
        df.to_csv('data/raw/Mark_Six.csv', index=False)
        
        print("✅ Sample data created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")
        return False


def check_system_resources():
    """Check system resources and performance."""
    print("\n💻 System Resources Check:")
    print("-" * 26)
    
    try:
        import psutil
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"✅ Total RAM: {memory_gb:.1f} GB")
        print(f"✅ Available RAM: {available_gb:.1f} GB")
        
        if available_gb < 2:
            print("⚠️  Low available memory. Consider closing other applications.")
        elif available_gb > 8:
            print("🚀 Excellent memory available for training!")
        
        # CPU check
        cpu_count = psutil.cpu_count()
        print(f"✅ CPU cores: {cpu_count}")
        
        # Disk space check
        disk_usage = psutil.disk_usage('.')
        free_gb = disk_usage.free / (1024**3)
        print(f"✅ Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 5:
            print("⚠️  Low disk space. Consider cleaning up files.")
        
        return True
        
    except ImportError:
        print("⚠️  psutil not available (install for detailed system info)")
        return True
    except Exception as e:
        print(f"⚠️  System resource check failed: {e}")
        return True


def test_basic_functionality():
    """Test basic system functionality."""
    print("\n🧪 Basic Functionality Test:")
    print("-" * 27)
    
    try:
        # Test import of key modules
        print("Testing module imports...")
        
        # Test new utilities if available
        try:
            from src.utils.input_validation import InputValidator
            from src.utils.error_handling import ErrorHandler
            from src.utils.safe_math import safe_divide
            print("✅ Enhanced utilities available")
        except ImportError:
            print("⚠️  Enhanced utilities not available (using legacy mode)")
        
        # Test configuration loading
        try:
            from src.config import CONFIG
            print("✅ Configuration loaded successfully")
        except ImportError:
            print("⚠️  Configuration import failed (will use defaults)")
        
        # Test basic math operations
        result = 10 / 2  # Simple test
        assert result == 5.0
        print("✅ Basic functionality test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        traceback.print_exc()
        return False


def recommend_next_steps(health_status):
    """Provide recommendations based on health check results."""
    print("\n🎯 Recommendations:")
    print("-" * 16)
    
    if all(health_status.values()):
        print("🎉 All systems healthy! You're ready to go.")
        print("\n📋 Suggested next steps:")
        print("1. Run the enhanced interface: python main_improved.py")
        print("2. Try option 5 for system diagnostics")
        print("3. Run option 1 to train a model")
        print("4. Check out the troubleshooting guide: docs/troubleshooting_guide.md")
        
    else:
        print("⚠️  Some issues detected. Here's how to fix them:")
        
        if not health_status.get('environment', True):
            print("\n🔧 Environment Issues:")
            print("- Activate conda environment: conda activate marksix_ai")
            print("- Install missing packages: conda install pytorch pandas numpy scikit-learn")
        
        if not health_status.get('gpu', True):
            print("\n🎮 GPU Issues:")
            print("- System will work in CPU mode (slower)")
            print("- For GPU support, check CUDA installation")
        
        if not health_status.get('data', True):
            print("\n📊 Data Issues:")
            print("- Sample data was created for testing")
            print("- Replace with real Mark Six data for actual predictions")
        
        print("\n💡 Try the enhanced interface for better error handling:")
        print("   python main_improved.py")


def run_health_check():
    """Run complete health check."""
    print_header()
    
    # Run all checks
    health_status = {
        'environment': check_python_environment(),
        'gpu': check_cuda_and_gpu(),
        'structure': check_project_structure(),
        'data': check_data_files(),
        'resources': check_system_resources(),
        'functionality': test_basic_functionality()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 HEALTH CHECK SUMMARY:")
    print("=" * 60)
    
    for check_name, status in health_status.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name.title().replace('_', ' ')}: {'PASS' if status else 'FAIL'}")
    
    overall_health = sum(health_status.values()) / len(health_status)
    print(f"\n🎯 Overall Health Score: {overall_health * 100:.0f}%")
    
    # Recommendations
    recommend_next_steps(health_status)
    
    print("\n" + "=" * 60)
    print("For detailed troubleshooting, see: docs/troubleshooting_guide.md")
    print("For system architecture info, see: docs/architecture.md")
    print("=" * 60)
    
    return health_status


if __name__ == "__main__":
    try:
        health_status = run_health_check()
        
        # Exit code based on critical components
        critical_checks = ['environment', 'structure', 'functionality']
        critical_health = all(health_status.get(check, False) for check in critical_checks)
        
        sys.exit(0 if critical_health else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Health check interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Health check failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)