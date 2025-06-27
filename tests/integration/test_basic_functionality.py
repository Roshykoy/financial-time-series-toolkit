#!/usr/bin/env python3
"""
Basic functionality test for MarkSix system.
Works in any Python environment and provides helpful guidance.
"""
import os
import sys
import warnings
from pathlib import Path


def test_python_environment():
    """Test Python environment setup."""
    print("🐍 Python Environment Test")
    print("-" * 30)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 10):
        print("❌ Python 3.10+ required")
        return False
    else:
        print("✅ Python version OK")
    
    # Check if we're in the right directory
    if not os.path.exists('src'):
        print("❌ Not in MarkSix project directory")
        print("Please run from: /home/rheuks/projects/MarkSix-Probabilistic-Forecasting")
        return False
    else:
        print("✅ In MarkSix project directory")
    
    return True


def test_conda_environment():
    """Test conda environment setup."""
    print("\n🐍 Conda Environment Test")
    print("-" * 28)
    
    # Check if conda is available
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    print(f"Current conda environment: {conda_env}")
    
    if conda_env == 'marksix_ai':
        print("✅ In marksix_ai environment")
        return True
    else:
        print("⚠️  Not in marksix_ai environment")
        print("To activate: conda activate marksix_ai")
        return False


def test_package_imports():
    """Test critical package imports."""
    print("\n📦 Package Import Test")
    print("-" * 22)
    
    packages_status = {}
    
    # Test critical packages
    critical_packages = {
        'torch': 'PyTorch (deep learning)',
        'numpy': 'NumPy (numerical computing)',
        'pandas': 'Pandas (data analysis)', 
        'sklearn': 'Scikit-learn (machine learning)',
        'yaml': 'PyYAML (configuration)',
        'psutil': 'psutil (system monitoring)'
    }
    
    for package, description in critical_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                module = sklearn
            else:
                module = __import__(package)
            
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {description}: {version}")
            packages_status[package] = True
        except ImportError:
            print(f"❌ {description}: Not installed")
            packages_status[package] = False
    
    return packages_status


def test_project_structure():
    """Test project file structure."""
    print("\n📁 Project Structure Test")
    print("-" * 25)
    
    # Check critical files and directories
    critical_paths = {
        'src/': 'Source code directory',
        'src/config.py': 'Configuration file',
        'main.py': 'Original main interface',
        'main_improved.py': 'Enhanced main interface',
        'data/': 'Data directory',
        'models/': 'Models directory',
        'outputs/': 'Outputs directory'
    }
    
    all_exist = True
    for path, description in critical_paths.items():
        if os.path.exists(path):
            print(f"✅ {description}: {path}")
        else:
            print(f"❌ {description}: {path} (missing)")
            all_exist = False
    
    return all_exist


def test_enhanced_utilities():
    """Test enhanced utility modules."""
    print("\n🛠️  Enhanced Utilities Test")
    print("-" * 26)
    
    # Add src to path
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    utilities = {
        'utils.input_validation': 'Input validation system',
        'utils.error_handling': 'Error handling system',
        'utils.safe_math': 'Safe mathematical operations',
        'utils.progress_feedback': 'User feedback system',
        'utils.performance_monitor': 'Performance monitoring'
    }
    
    utilities_status = {}
    for module_name, description in utilities.items():
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"✅ {description}: Available")
            utilities_status[module_name] = True
        except ImportError as e:
            print(f"❌ {description}: {e}")
            utilities_status[module_name] = False
    
    return utilities_status


def test_configuration_loading():
    """Test configuration loading."""
    print("\n⚙️  Configuration Test")
    print("-" * 20)
    
    # Add src to path if not already
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    config_methods = [
        ('Enhanced config system', 'infrastructure.config', 'get_flat_config'),
        ('Legacy config', 'config', 'CONFIG'),
        ('Backup config', 'config_legacy', 'CONFIG')
    ]
    
    for description, module_name, attr_name in config_methods:
        try:
            module = __import__(module_name, fromlist=[attr_name])
            if hasattr(module, attr_name):
                config_obj = getattr(module, attr_name)
                if callable(config_obj):
                    config = config_obj()
                else:
                    config = config_obj
                
                print(f"✅ {description}: Available ({len(config)} parameters)")
                return True
            else:
                print(f"❌ {description}: Missing {attr_name}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
    
    print("⚠️  No configuration system available")
    return False


def run_basic_functionality_test():
    """Run all basic functionality tests."""
    print("\n" + "=" * 60)
    print("🧪 MARKSIX BASIC FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Run all tests
    test_results = {
        'Python Environment': test_python_environment(),
        'Conda Environment': test_conda_environment(),
        'Package Imports': test_package_imports(),
        'Project Structure': test_project_structure(),
        'Enhanced Utilities': test_enhanced_utilities(),
        'Configuration': test_configuration_loading()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        # Handle dict results (from package imports)
        if isinstance(result, dict):
            passed = sum(1 for v in result.values() if v)
            total = len(result)
            status = "PARTIAL" if 0 < passed < total else ("PASS" if passed == total else "FAIL")
            status_icon = "🟡" if status == "PARTIAL" else ("✅" if status == "PASS" else "❌")
            print(f"{status_icon} {test_name}: {status} ({passed}/{total})")
            if passed > 0:
                passed_tests += 0.5 if status == "PARTIAL" else 1
        else:
            status_icon = "✅" if result else "❌"
            status = "PASS" if result else "FAIL"
            print(f"{status_icon} {test_name}: {status}")
            if result:
                passed_tests += 1
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.0f}%")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 16)
    
    if success_rate >= 80:
        print("🎉 System is ready! Try these next steps:")
        print("1. Run the enhanced interface: python main_improved.py")
        print("2. Run system health check: python quick_health_check.py")
    elif success_rate >= 50:
        print("⚠️  Partial setup detected. Recommended actions:")
        if not test_results['Conda Environment']:
            print("• Activate conda environment: conda activate marksix_ai")
        if isinstance(test_results['Package Imports'], dict):
            missing = [k for k, v in test_results['Package Imports'].items() if not v]
            if missing:
                print(f"• Install missing packages: {', '.join(missing)}")
    else:
        print("❌ Setup issues detected. Please:")
        print("1. Ensure you're in the correct directory")
        print("2. Activate conda environment: conda activate marksix_ai")
        print("3. Install dependencies: conda env create -f environment.yml")
    
    print("\n📚 For detailed help, see:")
    print("• docs/troubleshooting_guide.md")
    print("• DEBUGGING_AUDIT_REPORT.md")
    
    return success_rate >= 50


def test_simple_math():
    """Test basic mathematical operations."""
    print("\n🧮 Simple Math Test")
    print("-" * 17)
    
    try:
        # Test basic operations
        assert 2 + 2 == 4
        assert 10 / 2 == 5
        assert 3 ** 2 == 9
        print("✅ Basic math operations work")
        
        # Test imports work
        import json
        import os
        import sys
        print("✅ Standard library imports work")
        
        return True
    except Exception as e:
        print(f"❌ Basic functionality error: {e}")
        return False


if __name__ == "__main__":
    try:
        # First test simple functionality
        if not test_simple_math():
            print("❌ Basic Python functionality failed")
            sys.exit(1)
        
        # Run comprehensive test
        success = run_basic_functionality_test()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)