#!/usr/bin/env python3
"""
Dependency checker and environment validator for MarkSix project.
Run this to check if the environment is properly set up.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_conda_environment():
    """Check if we're in the correct conda environment."""
    print("\n🌍 Checking conda environment...")
    try:
        result = subprocess.run(['conda', 'info', '--json'], 
                              capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)
        active_env = info.get('active_prefix_name', 'base')
        
        if active_env == 'marksix_ai':
            print("✅ marksix_ai environment is active")
            return True
        else:
            print(f"⚠️  Current environment: {active_env}")
            print("💡 Activate with: conda activate marksix_ai")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
        print("❌ conda not available or error checking environment")
        return False

def check_core_dependencies():
    """Check if core dependencies are available."""
    print("\n📦 Checking core dependencies...")
    dependencies = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('yaml', 'PyYAML'),
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - missing")
            missing.append(name)
    
    return len(missing) == 0

def check_test_dependencies():
    """Check if testing dependencies are available."""
    print("\n🧪 Checking test dependencies...")
    test_deps = [
        ('pytest', 'pytest'),
        ('coverage', 'pytest-cov'),
    ]
    
    missing = []
    for module, name in test_deps:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - missing")
            missing.append(name)
    
    return len(missing) == 0

def check_project_structure():
    """Check if project structure is correct."""
    print("\n📁 Checking project structure...")
    required_paths = [
        'src/',
        'tests/',
        'docs/',
        'notebooks/',
        'config/',
        'main.py',
        'run_tests.py',
        'environment.yml'
    ]
    
    project_root = Path(__file__).parent
    missing = []
    
    for path in required_paths:
        full_path = project_root / path
        if full_path.exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - missing")
            missing.append(path)
    
    return len(missing) == 0

def check_config_files():
    """Check if configuration is working."""
    print("\n⚙️ Checking configuration...")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from config import CONFIG
        print("✅ Configuration loading works")
        print(f"   Data path: {CONFIG.get('data_path', 'Not set')}")
        print(f"   Device: {CONFIG.get('device', 'Not set')}")
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def provide_solutions():
    """Provide solutions for common issues."""
    print("\n💡 SOLUTIONS FOR COMMON ISSUES")
    print("=" * 60)
    
    print("\n🔧 If environment is wrong:")
    print("   conda activate marksix_ai")
    
    print("\n🔧 If environment doesn't exist:")
    print("   conda env create -f environment.yml")
    print("   conda activate marksix_ai")
    
    print("\n🔧 If core dependencies missing:")
    print("   conda activate marksix_ai")
    print("   pip install -r requirements/base.txt")
    
    print("\n🔧 If test dependencies missing:")
    print("   pip install pytest pytest-cov")
    print("   # OR for full dev setup:")
    print("   pip install -r requirements/dev.txt")
    
    print("\n🔧 If configuration broken:")
    print("   Check if src/config_legacy.py exists")
    print("   Verify src/config.py imports")
    
    print("\n🚀 Quick start command:")
    print("   conda activate marksix_ai && python main.py")

def main():
    """Run all checks and provide summary."""
    print("🔍 MarkSix Environment Dependency Checker")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Conda Environment", check_conda_environment),
        ("Core Dependencies", check_core_dependencies),
        ("Test Dependencies", check_test_dependencies),
        ("Project Structure", check_project_structure),
        ("Configuration", check_config_files),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n📊 SUMMARY")
    print("=" * 30)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! The environment is ready.")
        print("Try running: python main.py")
    else:
        print(f"\n⚠️  {total - passed} issues found. See solutions below:")
        provide_solutions()
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())