#!/bin/bash
# Script to activate the correct environment and test the system

echo "🔄 Activating MarkSix environment and testing system..."
echo "=" * 60

# Source conda initialization
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the marksix_ai environment
echo "Activating conda environment..."
conda activate marksix_ai

# Check environment is active
echo "Current environment: $CONDA_DEFAULT_ENV"

# Test PyTorch availability
echo ""
echo "🧪 Testing PyTorch..."
python -c "import torch; print(f'✅ PyTorch {torch.__version__} available')" || echo "❌ PyTorch not available"

# Test basic functionality
echo ""
echo "🧪 Running basic functionality test..."
python test_basic_functionality.py

# Test enhanced main interface if packages are available
echo ""
echo "🚀 Testing enhanced main interface..."
python -c "
try:
    import torch
    print('✅ Dependencies available - you can run:')
    print('   python main_improved.py')
    print('   python quick_health_check.py')
except ImportError:
    print('⚠️  Some dependencies missing. Run in marksix_ai environment.')
"

echo ""
echo "📋 Quick Commands:"
echo "  conda activate marksix_ai      # Activate environment"
echo "  python main_improved.py        # Enhanced interface"
echo "  python quick_health_check.py   # System health check"
echo "  python test_basic_functionality.py  # Test system"