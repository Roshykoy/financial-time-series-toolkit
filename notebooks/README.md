# MarkSix Notebooks

This directory contains Jupyter notebooks for various aspects of the MarkSix project.

## Directory Structure

### 📊 Analysis (`analysis/`)
- `1_Data_Analysis_and_Feature_Engineering.ipynb` - Comprehensive data exploration and feature analysis

### 🧪 Experiments (`experiments/`)
- `2_Model_Training.ipynb` - Interactive model training and experimentation
- `3_Inference_and_Evaluation.ipynb` - Model evaluation and performance analysis

### 📚 Tutorials (`tutorials/`)
- `4_Hyperparameter_Optimization_Demo.ipynb` - Guide to using the hyperparameter optimization system including new Pareto Front multi-objective optimization with JSD Alignment Fidelity

## Getting Started

1. **Environment Setup**:
   ```bash
   conda activate marksix_ai
   jupyter lab
   ```

2. **Dependencies**: All notebooks are designed to work with the main project environment. No additional requirements needed.

3. **Execution Order**: For new users, we recommend following this order:
   1. Analysis notebooks (understand the data)
   2. Tutorial notebooks (learn the system)
   3. Experiment notebooks (run experiments)

## Usage Notes

- All notebooks assume you're running from the project root directory
- Outputs have been cleared for clean repository state
- Make sure the marksix_ai conda environment is activated before running
- Each notebook includes its own dependencies and setup cells

## Recent Updates (July 27, 2025)

### 🎯 New Features Covered in Notebooks:
- **JSD Alignment Fidelity**: Statistical metric for ensuring model-generated numbers match historical lottery data distributions
- **Updated Pareto Front Optimization**: Four-objective optimization now prioritizes:
  1. **Model Complexity** (minimize) - Weight 1.0 (HIGH)
  2. **JSD Alignment Fidelity** (maximize) - Weight 1.0 (HIGH) 
  3. **Training Time** (minimize) - Weight 0.8 (MEDIUM-HIGH)
  4. **Accuracy** (maximize) - Weight 0.6 (MEDIUM)
- **Option 4.5 Integration**: Pareto Front multi-objective optimization accessible via main menu

## Troubleshooting

If you encounter import errors:
1. Ensure you're in the correct conda environment
2. Check that you're running jupyter from the project root
3. Verify that `src/` is in your Python path (handled automatically in notebooks)

For more help, see the main project README.md or the troubleshooting guide in `/docs/`.