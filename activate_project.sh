#!/bin/bash
# Custom activation script for Financial Time Series Toolkit

echo "🚀 Activating Financial Time Series Toolkit Environment..."

# Activate virtual environment
source .venv/Scripts/activate

# Set aliases for convenience
alias python='.venv/Scripts/python.exe'
alias pip='.venv/Scripts/pip.exe'

# Test that everything works
echo "✅ Environment activated!"
echo "🐍 Python: $(.venv/Scripts/python.exe --version)"
echo "📦 Pip: $(.venv/Scripts/pip.exe --version | head -n1)"
echo ""
echo "💡 Quick commands:"
echo "   python main.py        - Test the setup"
echo "   tree -a              - View project structure"
echo "   python --help        - Python help"
echo ""
