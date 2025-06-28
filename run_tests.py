#!/usr/bin/env python3
"""
Unified test runner for MarkSix Probabilistic Forecasting project.
Provides menu-driven interface for running different test categories.
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class TestRunner:
    """Unified test runner with menu interface."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
        self.test_framework = self.detect_test_framework()
    
    def detect_test_framework(self) -> str:
        """Detect available test framework and return preference."""
        try:
            import pytest
            return 'pytest'
        except ImportError:
            print("⚠️  pytest not found, falling back to unittest")
            return 'unittest'
        
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status."""
        print(f"\n🏃 {description}")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 60)
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root, check=True)
            print(f"✅ {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed with exit code {e.returncode}")
            return False
        except FileNotFoundError:
            print(f"❌ Command not found: {cmd[0]}")
            if 'pytest' in cmd:
                print("💡 pytest not available. Installing dependencies...")
                return self.handle_missing_pytest(description)
            return False
    
    def handle_missing_pytest(self, description: str) -> bool:
        """Handle missing pytest by providing installation instructions and fallback."""
        print("\n🔧 DEPENDENCY ISSUE DETECTED")
        print("=" * 50)
        print("pytest is required for the test runner.")
        print("\n📦 To install dependencies:")
        print("Option 1 (Recommended): conda activate marksix_ai")
        print("Option 2 (Quick fix): pip install pytest pytest-cov")
        print("Option 3 (Full setup): pip install -r requirements/dev.txt")
        
        user_choice = input("\n❓ Try automatic installation? (y/N): ").strip().lower()
        if user_choice == 'y':
            try:
                print("\n📥 Installing pytest and pytest-cov...")
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"
                ], check=True, capture_output=True, text=True)
                print("✅ Installation successful!")
                return True
            except subprocess.CalledProcessError as e:
                print(f"❌ Installation failed: {e}")
                print("Please install manually or activate the marksix_ai environment")
        
        return False
    
    def run_unit_tests(self) -> bool:
        """Run unit tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/unit/", "-v"],
            "Running unit tests"
        )
    
    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/integration/", "-v"],
            "Running integration tests"
        )
    
    def run_optimization_tests(self) -> bool:
        """Run optimization module tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/optimization/", "-v"],
            "Running optimization module tests"
        )
    
    def run_all_tests(self) -> bool:
        """Run all tests."""
        return self.run_command(
            ["python", "-m", "pytest", "tests/", "-v"],
            "Running all tests"
        )
    
    def run_with_coverage(self) -> bool:
        """Run tests with coverage report."""
        success = self.run_command(
            ["python", "-m", "pytest", "tests/", "--cov=src", "--cov-report=html", "--cov-report=term"],
            "Running tests with coverage analysis"
        )
        
        if success:
            print("\n📊 Coverage report generated in htmlcov/index.html")
        
        return success
    
    def run_specific_test(self, test_file: str) -> bool:
        """Run a specific test file."""
        test_path = self.tests_dir / test_file
        if not test_path.exists():
            # Try finding the file in subdirectories
            matching_files = list(self.tests_dir.rglob(f"*{test_file}*"))
            if matching_files:
                test_path = matching_files[0]
            else:
                print(f"❌ Test file not found: {test_file}")
                return False
        
        return self.run_command(
            ["python", "-m", "pytest", str(test_path), "-v"],
            f"Running specific test: {test_path.name}"
        )
    
    def validate_optimization_module(self) -> bool:
        """Run optimization module validation."""
        validation_script = self.project_root / "validate_optimization.py"
        if validation_script.exists():
            return self.run_command(
                ["python", "validate_optimization.py"],
                "Validating optimization module"
            )
        else:
            print("❌ Validation script not found")
            return False
    
    def list_available_tests(self) -> None:
        """List all available test files."""
        print("\n📋 Available test files:")
        print("=" * 40)
        
        for category in ["unit", "integration", "optimization"]:
            category_dir = self.tests_dir / category
            if category_dir.exists():
                print(f"\n{category.upper()} TESTS:")
                test_files = list(category_dir.glob("test_*.py"))
                if test_files:
                    for test_file in sorted(test_files):
                        print(f"  - {test_file.name}")
                else:
                    print(f"  No test files found in {category}/")
    
    def show_menu(self) -> None:
        """Display interactive menu."""
        print("\n🧪 MarkSix Test Runner")
        print("=" * 50)
        print("1. Run all tests")
        print("2. Run unit tests")
        print("3. Run integration tests")
        print("4. Run optimization module tests")
        print("5. Run tests with coverage report")
        print("6. Run specific test file")
        print("7. Validate optimization module")
        print("8. List available tests")
        print("9. Environment setup help")
        print("10. Exit")
        print("=" * 50)
    
    def show_environment_help(self) -> None:
        """Show environment setup instructions."""
        print("\n🔧 ENVIRONMENT SETUP HELP")
        print("=" * 60)
        print("This project requires the marksix_ai conda environment.")
        print("\n📦 Setup Instructions:")
        print("1. Create environment: conda env create -f environment.yml")
        print("2. Activate environment: conda activate marksix_ai")
        print("3. Install dev dependencies: pip install -r requirements/dev.txt")
        print("\n🚀 Quick Test:")
        print("conda activate marksix_ai && python run_tests.py --list")
        print("\n⚡ Quick Fix (current environment):")
        print("pip install pytest pytest-cov")
    
    def interactive_mode(self) -> None:
        """Run in interactive menu mode."""
        # Check environment on startup
        if self.test_framework == 'unittest':
            print("\n⚠️  WARNING: pytest not available, functionality may be limited")
            print("Run option 10 for setup help")
        
        while True:
            self.show_menu()
            choice = input("\nEnter your choice (1-10): ").strip()
            
            if choice == "1":
                self.run_all_tests()
            elif choice == "2":
                self.run_unit_tests()
            elif choice == "3":
                self.run_integration_tests()
            elif choice == "4":
                self.run_optimization_tests()
            elif choice == "5":
                self.run_with_coverage()
            elif choice == "6":
                test_file = input("Enter test file name: ").strip()
                if test_file:
                    self.run_specific_test(test_file)
            elif choice == "7":
                self.validate_optimization_module()
            elif choice == "8":
                self.list_available_tests()
            elif choice == "9":
                self.show_environment_help()
            elif choice == "10":
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-10.")
            
            input("\nPress Enter to continue...")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MarkSix Test Runner")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--optimization", action="store_true", help="Run optimization tests")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage")
    parser.add_argument("--validate", action="store_true", help="Validate optimization module")
    parser.add_argument("--test", type=str, help="Run specific test file")
    parser.add_argument("--list", action="store_true", help="List available tests")
    
    args = parser.parse_args()
    runner = TestRunner()
    
    # Non-interactive mode
    if any(vars(args).values()):
        success = True
        
        if args.all:
            success &= runner.run_all_tests()
        if args.unit:
            success &= runner.run_unit_tests()
        if args.integration:
            success &= runner.run_integration_tests()
        if args.optimization:
            success &= runner.run_optimization_tests()
        if args.coverage:
            success &= runner.run_with_coverage()
        if args.validate:
            success &= runner.validate_optimization_module()
        if args.test:
            success &= runner.run_specific_test(args.test)
        if args.list:
            runner.list_available_tests()
            success = True
        
        sys.exit(0 if success else 1)
    
    # Interactive mode
    else:
        runner.interactive_mode()

if __name__ == "__main__":
    main()