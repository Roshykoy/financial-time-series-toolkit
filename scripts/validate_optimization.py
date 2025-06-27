#!/usr/bin/env python3
"""
Validation script for the hyperparameter optimization module.
This script performs basic validation without requiring full model training.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def validate_imports():
    """Validate that all optimization modules can be imported."""
    print("🔍 Validating optimization module imports...")
    
    try:
        # Test core utilities
        from utils.error_handling import safe_execute
        print("✅ Error handling utilities")
        
        # Test optimization utilities
        from optimization.utils import OptimizationUtils
        print("✅ Optimization utilities")
        
        # Test configuration management
        from optimization.config_manager import OptimizationConfigManager
        print("✅ Configuration management")
        
        # Test algorithms
        from optimization.algorithms import SearchSpaceHandler
        print("✅ Optimization algorithms")
        
        # Test hardware management
        from optimization.hardware_manager import HardwareResourceManager
        print("✅ Hardware resource management")
        
        # Test main interface
        from optimization.main import OptimizationOrchestrator
        print("✅ Main optimization interface")
        
        return True
        
    except Exception as e:
        print(f"❌ Import validation failed: {e}")
        return False

def main():
    """Run all validations."""
    print("🚀 Starting hyperparameter optimization module validation...\n")
    
    success = validate_imports()
    
    if success:
        print("\n🎉 All validations passed! The optimization module is ready to use.")
        return 0
    else:
        print("\n⚠️  Some validations failed. Check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())