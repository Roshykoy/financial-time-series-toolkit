import pandas as pd
import re

def load_and_process_data(raw_filepath, processed_filepath):
    """
    Loads the raw Mark Six CSV, robustly finds the data start, cleans the data, 
    calculates the sum of winning numbers, and saves the result to a new CSV file.
    
    Args:
        raw_filepath (str): The path to the raw Mark_Six.csv file.
        processed_filepath (str): The path where the cleaned data will be saved.
    """
    try:
        # --- Step 1: Robustly find the first line of actual data ---
        data_start_row = -1
        draw_pattern = re.compile(r'^\d{2}/\d{3},')

        with open(raw_filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if draw_pattern.match(line):
                    data_start_row = i
                    break
        
        if data_start_row == -1:
            raise ValueError("Could not find any data rows matching the expected format (e.g., '25/067,...').")

        print(f"Found first data row at line: {data_start_row + 1}")

        # --- Step 2: Manually define clean column names ---
        column_names = [
            'Draw', 'Date', 'Winning_Num_1', 'Winning_Num_2', 'Winning_Num_3', 'Winning_Num_4',
            'Winning_Num_5', 'Winning_Num_6', 'Extra_Num', 'From_Last', 'Low', 'High', 'Odd',
            'Even', 'Bin_1_10', 'Bin_11_20', 'Bin_21_30', 'Bin_31_40', 'Bin_41_50', 'Div_1_Winners',
            'Div_1_Prize', 'Div_2_Winners', 'Div_2_Prize', 'Div_3_Winners', 'Div_3_Prize',
            'Div_4_Winners', 'Div_4_Prize', 'Div_5_Winners', 'Div_5_Prize', 'Div_6_Winners',
            'Div_6_Prize', 'Div_7_Winners', 'Div_7_Prize', 'Turnover'
        ]

        # --- Step 3: Load data ---
        df = pd.read_csv(raw_filepath, header=None, skiprows=data_start_row, names=column_names, encoding='utf-8')
        print(f"Successfully loaded {len(df)} rows from the raw CSV file.")
        
        # --- Step 4: Data Cleaning and Feature Engineering ---
        for col in df.columns:
            if col not in ['Draw', 'Date']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Date'] = pd.to_datetime(df['Date'])

        # --- MODIFIED: Drop ANY row that contains ANY NaN value ---
        initial_rows = len(df)
        df.dropna(inplace=True)
        final_rows = len(df)
        print(f"Dropped {initial_rows - final_rows} rows with any missing data.")
        
        winning_num_cols = [f'Winning_Num_{i}' for i in range(1, 7)] + ['Extra_Num']
        for col in winning_num_cols:
            df[col] = df[col].astype(int)

        df['Sum'] = df[winning_num_cols].sum(axis=1)

        df.sort_values(by='Date', inplace=True, ascending=False)
        df.reset_index(drop=True, inplace=True)

        # --- Step 5: Save the processed data ---
        df.to_csv(processed_filepath, index=False)
        
        print(f"\nData successfully processed and saved to '{processed_filepath}'")
        print(f"Final dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
        print("\nFirst 5 rows of processed data (most recent draws):")
        print(df.head())

    except FileNotFoundError:
        print(f"Error: The file was not found at '{raw_filepath}'.")
        print("Please make sure 'Mark_Six.csv' is in the 'data/raw/' directory.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    raw_file = 'data/raw/Mark_Six.csv'
    processed_file = 'data/processed/processed_mark_six.csv'
    load_and_process_data(raw_file, processed_file)