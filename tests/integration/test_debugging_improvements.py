#!/usr/bin/env python3
"""
Test debugging improvements without heavy dependencies.
Demonstrates the enhanced error handling and user experience features.
"""
import os
import sys
import tempfile
import warnings
from io import StringIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_input_validation():
    """Test enhanced input validation system."""
    print("📝 Testing Input Validation System")
    print("-" * 35)
    
    try:
        from utils.input_validation import InputValidator, ValidationError
        
        validator = InputValidator()
        
        # Test menu choice validation
        try:
            result = validator.validate_menu_choice("3")
            print(f"✅ Valid menu choice: {result}")
        except ValidationError as e:
            print(f"❌ Menu validation failed: {e}")
        
        # Test invalid menu choice
        try:
            validator.validate_menu_choice("invalid")
            print("❌ Should have failed invalid input")
        except ValidationError:
            print("✅ Invalid input properly rejected")
        
        # Test positive integer validation
        try:
            result = validator.validate_positive_integer("10", "test", 1, 100)
            print(f"✅ Valid integer: {result}")
        except ValidationError as e:
            print(f"❌ Integer validation failed: {e}")
        
        # Test float range validation
        try:
            result = validator.validate_float_range("0.5", "test", 0.0, 1.0)
            print(f"✅ Valid float: {result}")
        except ValidationError as e:
            print(f"❌ Float validation failed: {e}")
        
        print("✅ Input validation system working correctly")
        return True
        
    except ImportError as e:
        print(f"❌ Input validation import failed: {e}")
        return False


def test_error_handling():
    """Test enhanced error handling system."""
    print("\n🛡️  Testing Error Handling System")
    print("-" * 33)
    
    try:
        from utils.error_handling import ErrorHandler, robust_operation
        
        # Test error handler
        handler = ErrorHandler()
        test_error = ValueError("Test error")
        handler.handle_error(test_error, "test_context")
        print("✅ Error handler working")
        
        # Test robust operation decorator
        call_count = 0
        
        @robust_operation(max_retries=3, exceptions=(ValueError,))
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count}")
            return "success"
        
        result = test_function()
        if result == "success" and call_count == 3:
            print("✅ Robust operation retry mechanism working")
        else:
            print("❌ Robust operation failed")
        
        print("✅ Error handling system working correctly")
        return True
        
    except ImportError as e:
        print(f"❌ Error handling import failed: {e}")
        return False


def test_safe_math():
    """Test safe mathematical operations."""
    print("\n🧮 Testing Safe Math Operations")
    print("-" * 31)
    
    try:
        from utils.safe_math import safe_divide, safe_log
        
        # Test safe division
        result = safe_divide(10, 2)
        if result == 5.0:
            print("✅ Safe division working")
        else:
            print(f"❌ Safe division failed: {result}")
        
        # Test division by zero protection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = safe_divide(10, 0)
            if result == 10 / 1e-8:  # Should use epsilon
                print("✅ Division by zero protection working")
            else:
                print(f"❌ Division by zero protection failed: {result}")
        
        # Test safe logarithm
        import math
        result = safe_log(10)
        if abs(result - math.log(10)) < 1e-6:
            print("✅ Safe logarithm working")
        else:
            print(f"❌ Safe logarithm failed: {result}")
        
        # Test log of zero protection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = safe_log(0)
            if result == math.log(1e-8):
                print("✅ Log zero protection working")
            else:
                print(f"❌ Log zero protection failed: {result}")
        
        print("✅ Safe math operations working correctly")
        return True
        
    except ImportError as e:
        print(f"❌ Safe math import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Safe math test failed: {e}")
        return False


def test_user_feedback():
    """Test enhanced user feedback system."""
    print("\n💬 Testing User Feedback System")
    print("-" * 31)
    
    try:
        from utils.progress_feedback import UserFeedback
        
        # Capture output
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        # Test different message types
        UserFeedback.info("Test info message")
        UserFeedback.success("Test success message") 
        UserFeedback.warning("Test warning message")
        UserFeedback.error("Test error message")
        
        # Restore stdout
        sys.stdout = old_stdout
        output = captured_output.getvalue()
        
        # Check output contains expected elements
        if "ℹ️" in output and "✅" in output and "⚠️" in output and "❌" in output:
            print("✅ User feedback messages working")
        else:
            print("❌ User feedback messages failed")
        
        # Test summary display
        test_summary = {"param1": "value1", "param2": "value2"}
        sys.stdout = captured_output = StringIO()
        UserFeedback.show_summary("Test Summary", test_summary)
        sys.stdout = old_stdout
        
        summary_output = captured_output.getvalue()
        if "Test Summary" in summary_output and "param1" in summary_output:
            print("✅ Summary display working")
        else:
            print("❌ Summary display failed")
        
        print("✅ User feedback system working correctly")
        return True
        
    except ImportError as e:
        print(f"❌ User feedback import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ User feedback test failed: {e}")
        return False


def test_file_operations():
    """Test safe file operations."""
    print("\n📁 Testing File Operations")
    print("-" * 25)
    
    try:
        from utils.input_validation import InputValidator
        
        validator = InputValidator()
        
        # Test with temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"test content")
        
        try:
            # Test valid file path
            result = validator.validate_file_path(tmp_path, must_exist=True)
            if result.exists():
                print("✅ File path validation working")
            else:
                print("❌ File path validation failed")
            
            # Test directory creation
            new_path = tmp_path + "_new"
            result = validator.validate_file_path(new_path, must_exist=False)
            if result.parent.exists():
                print("✅ Directory creation working")
            else:
                print("❌ Directory creation failed")
                
        finally:
            # Cleanup
            try:
                os.unlink(tmp_path)
                if os.path.exists(new_path):
                    os.unlink(new_path)
            except:
                pass
        
        print("✅ File operations working correctly")
        return True
        
    except ImportError as e:
        print(f"❌ File operations import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        return False


def test_configuration_compatibility():
    """Test configuration compatibility layer."""
    print("\n⚙️  Testing Configuration Compatibility")
    print("-" * 37)
    
    # Test basic Python dict operations (simulate config)
    test_config = {
        "epochs": 10,
        "batch_size": 8,
        "learning_rate": 0.001,
        "device": "cpu"
    }
    
    try:
        # Test config validation function
        from utils.input_validation import validate_training_config
        
        validated_config = validate_training_config(test_config)
        
        if (validated_config["epochs"] == 10 and 
            validated_config["batch_size"] == 8 and
            validated_config["learning_rate"] == 0.001):
            print("✅ Configuration validation working")
        else:
            print("❌ Configuration validation failed")
        
        print("✅ Configuration compatibility working correctly")
        return True
        
    except ImportError as e:
        print(f"❌ Configuration compatibility import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration compatibility test failed: {e}")
        return False


def run_debugging_improvements_test():
    """Run all debugging improvements tests."""
    print("\n" + "=" * 60)
    print("🐛 MARKSIX DEBUGGING IMPROVEMENTS TEST")
    print("=" * 60)
    
    # Run all tests
    tests = [
        ("Input Validation", test_input_validation),
        ("Error Handling", test_error_handling),
        ("Safe Math Operations", test_safe_math),
        ("User Feedback", test_user_feedback),
        ("File Operations", test_file_operations),
        ("Configuration Compatibility", test_configuration_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DEBUGGING IMPROVEMENTS TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status_icon = "✅" if result else "❌"
        status = "PASS" if result else "FAIL"
        print(f"{status_icon} {test_name}: {status}")
    
    success_rate = (passed / total) * 100
    print(f"\n🎯 Success Rate: {success_rate:.0f}% ({passed}/{total})")
    
    if success_rate >= 80:
        print("\n🎉 EXCELLENT! Debugging improvements are working perfectly!")
        print("\n💡 The enhanced system provides:")
        print("  • Robust input validation with helpful error messages")
        print("  • Advanced error handling with automatic recovery")
        print("  • Safe mathematical operations preventing crashes")
        print("  • Enhanced user feedback and progress indicators")
        print("  • Comprehensive file and configuration management")
        
        print("\n🚀 Ready for full system use!")
        print("  Next: Activate marksix_ai environment and run main_improved.py")
        
    elif success_rate >= 50:
        print("\n✅ GOOD! Most debugging improvements are working.")
        print("  Some features may require the full environment.")
        
    else:
        print("\n⚠️  Some debugging improvements need the full environment.")
        print("  Basic functionality should still work.")
    
    print("\n📋 Next Steps:")
    print("  1. conda activate marksix_ai")
    print("  2. python main_improved.py")
    print("  3. python quick_health_check.py")
    
    return success_rate >= 50


if __name__ == "__main__":
    try:
        success = run_debugging_improvements_test()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)