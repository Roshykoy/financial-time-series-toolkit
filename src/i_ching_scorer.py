# src/i_ching_scorer.py
import numpy as np

class IChingScorer:
    """
    An optional heuristic scorer inspired by I-Ching.

    This implementation assigns a static "luck" value to each possible lottery
    number. The score for a set is the average luck of its numbers.
    A fixed seed ensures the luck values are the same every time the program runs.
    """
    def __init__(self, config):
        self.config = config
        # Use a fixed seed for reproducible "luck" values
        np.random.seed(42)
        # Assign a random "luck" value between 0 and 1 for each lottery number
        self.luck_values = {
            i: np.random.rand() for i in range(1, config['num_lotto_numbers'] + 1)
        }
        print("IChingScorer initialized with pre-defined luck values.")

    def score(self, number_set: list[int]) -> float:
        """
        Scores a number set based on its average assigned "luck" value.
        """
        if not all(num in self.luck_values for num in number_set):
            raise ValueError("Number set contains values outside the valid lottery number range.")

        total_luck = np.mean([self.luck_values[num] for num in number_set])
        return total_luck