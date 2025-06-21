# src/temporal_scorer.py
import pandas as pd
import numpy as np

class TemporalScorer:
    """
    Scores number sets based on the recency of their constituent numbers.
    The core idea is that numbers that have appeared more recently might have
    a different probability of appearing again compared to numbers that haven't
    appeared in a long time. This scorer rewards sets containing more
    recent numbers.
    """
    def __init__(self, config):
        self.config = config
        # This dictionary will store the last draw index for each number, e.g., {number: index}
        self.last_drawn_index = {}

    def fit(self, df: pd.DataFrame):
        """
        Calculates the last drawn index for every number in the historical data.
        """
        print("Fitting TemporalScorer with historical data...")
        num_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
        # We iterate through the dataset to find the most recent draw for each number
        for index, row in df.iterrows():
            for num in row[num_cols]:
                self.last_drawn_index[num] = index
        print("TemporalScorer fitting complete.")

    def score(self, number_set: list[int], current_index: int) -> float:
        """
        Scores a single number set.

        The score is based on the average recency of the numbers. To reward
        more recent appearances (smaller recency value), we use the inverse.
        A higher final score indicates a more "temporally active" number set.
        """
        if not self.last_drawn_index:
            raise RuntimeError("TemporalScorer has not been fitted. Call fit() first.")

        recencies = []
        for num in number_set:
            last_idx = self.last_drawn_index.get(num, -1)
            # If a number has never been drawn, we treat its recency as the total number of draws
            recency = current_index - last_idx if last_idx != -1 else current_index
            recencies.append(recency)

        # We calculate the mean of the inverse recencies. Adding 1 to avoid division by zero.
        # This results in a higher score for sets with numbers that appeared more recently.
        avg_recency_score = np.mean([1 / (r + 1) for r in recencies])
        return avg_recency_score