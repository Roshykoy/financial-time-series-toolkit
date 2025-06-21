# src/feature_engineering.py
import numpy as np
import pandas as pd
from collections import Counter, deque

class FeatureEngineer:
    """
    Handles all feature creation for a given set of lottery numbers.
    This class is fitted on historical data to learn patterns, and then
    can transform any new combination into a feature vector.
    """
    def __init__(self, recency_window=50):
        # Basic stats
        self.number_counts = Counter()
        self.pair_counts = Counter()
        
        # This set will store all historical winning combinations
        self.historical_sets = set()
        
        self.total_draws = 0

    def fit(self, df: pd.DataFrame):
        """
        Fits the feature engineer on the entire historical dataset.
        It calculates frequencies and stores all historical combinations.
        """
        print("Fitting FeatureEngineer with historical data...")
        winning_num_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
        self.total_draws = len(df)

        for _, row in df.iterrows():
            # Convert row to a sorted list of integers
            draw_numbers = sorted(row[winning_num_cols].astype(int).tolist())

            # --- THIS IS THE FIX ---
            # Add the sorted tuple of numbers to our set of historical combinations
            self.historical_sets.add(tuple(draw_numbers))
            # --- END OF FIX ---

            # Update counts
            self.number_counts.update(draw_numbers)

            # Update pair counts
            for i in range(len(draw_numbers)):
                for j in range(i + 1, len(draw_numbers)):
                    self.pair_counts[(draw_numbers[i], draw_numbers[j])] += 1
        
        print("FeatureEngineer fitting complete.")

    def transform(self, number_set: list[int], current_index: int) -> np.ndarray:
        """
        Transforms a single number set into a feature vector.
        """
        number_set = sorted(number_set)
        features = []

        # 1. Basic properties
        features.append(sum(number_set))
        features.append(np.mean(number_set))
        features.append(len([n for n in number_set if n % 2 == 0]))
        features.append(len([n for n in number_set if n < 25]))

        # 2. Historical Frequency Features
        features.append(np.mean([self.number_counts.get(n, 0) / self.total_draws for n in number_set]))
        features.append(np.min([self.number_counts.get(n, 0) / self.total_draws for n in number_set]))
        features.append(np.max([self.number_counts.get(n, 0) / self.total_draws for n in number_set]))

        # 3. Pair Frequency Features
        pair_freq_sum = 0
        num_pairs = 0
        for i in range(len(number_set)):
            for j in range(i + 1, len(number_set)):
                pair_freq_sum += self.pair_counts.get((number_set[i], number_set[j]), 0)
                num_pairs +=1
        features.append(pair_freq_sum / (self.total_draws * num_pairs if num_pairs > 0 else 1))

        # 4. Delta Features (differences between consecutive numbers)
        deltas = [number_set[i+1] - number_set[i] for i in range(len(number_set)-1)]
        features.append(np.mean(deltas))
        features.append(np.std(deltas))
        features.append(np.min(deltas))
        features.append(np.max(deltas))

        # 5. Decade/Group Features
        tens = sum(1 for n in number_set if 10 <= n < 20)
        twenties = sum(1 for n in number_set if 20 <= n < 30)
        thirties = sum(1 for n in number_set if 30 <= n < 40)
        forties = sum(1 for n in number_set if 40 <= n < 50)
        features.extend([tens, twenties, thirties, forties])

        return np.array(features, dtype=np.float32)