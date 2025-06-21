# Project: Probabilistic Forecasting for Mark Six Characteristics

## Description

This project uses a Transformer-based Mixture Density Network (Transformer-MDN) to analyze and forecast characteristics of the Hong Kong Mark Six lottery. The primary goal is to explore the existence of time-dependent patterns in the lottery's historical data.

This implementation focuses on predicting the **Sum** of the winning numbers as a probabilistic distribution.

## Directory Structure
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── init.py
│   ├── data_loader.py
│   ├── model.py
│   └── engine.py
├── outputs/
│   ├── models/
│   └── plots/
├── .gitignore
├── README.md
├── environment.yml
├── train.py
└── predict.py

## Setup

1.  Ensure you have Conda installed.
2.  Create the environment using the provided file:
    ```bash
    conda env create -f environment.yml
    ```
3.  Activate the environment:
    ```bash
    conda activate marksix_ai
    ```

## Usage

1.  **Place Data:** Place the raw `Mark_Six.csv` file inside the `data/raw/` directory.

2.  **Process Data:** Run the data loader to clean the raw data and generate the processed file.
    ```bash
    python src/data_loader.py
    ```

3.  **Train the Model:** Run the main training script. This will train the model and save the final weights and data scalers into the `outputs/models/` directory.
    ```bash
    python train.py
    ```

4.  **Make a Prediction:** Use the trained model to predict the next draw's characteristic.
    ```bash
    python predict.py
    ```