# src/temporal_context.py
import torch
import torch.nn as nn
import numpy as np

class TemporalContextEncoder(nn.Module):
    """
    LSTM-based encoder that processes the historical sequence of lottery draws
    to capture temporal patterns and trends.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sequence_length = config['temporal_sequence_length']
        self.num_numbers = config['num_lotto_numbers']
        self.temporal_embedding_dim = config['temporal_embedding_dim']
        self.temporal_hidden_dim = config['temporal_hidden_dim']
        self.context_dim = config['temporal_context_dim']
        
        # Embeddings for lottery numbers in temporal context
        self.number_embeddings = nn.Embedding(self.num_numbers + 1, self.temporal_embedding_dim)
        
        # Bidirectional LSTM for processing sequences
        self.lstm = nn.LSTM(
            input_size=6 * self.temporal_embedding_dim,  # 6 numbers per draw
            hidden_size=self.temporal_hidden_dim,
            num_layers=config['temporal_lstm_layers'],
            dropout=config['dropout'] if config['temporal_lstm_layers'] > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism for sequence summarization
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * self.temporal_hidden_dim,  # Bidirectional
            num_heads=config['temporal_attention_heads'],
            dropout=config['dropout'],
            batch_first=True
        )
        
        # Context projection
        self.context_projection = nn.Sequential(
            nn.Linear(2 * self.temporal_hidden_dim, config['temporal_projection_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['temporal_projection_dim'], self.context_dim)
        )
        
        # Trend analysis components
        self.trend_analyzer = nn.Sequential(
            nn.Linear(self.context_dim, config['trend_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['trend_hidden_dim'], config['trend_features'])
        )
        
    def prepare_sequence_data(self, df, current_index):
        """
        Prepares the sequence of recent draws for temporal analysis.
        
        Args:
            df: DataFrame with historical draws
            current_index: Current position in the dataset
        
        Returns:
            sequence_tensor: Tensor of shape [1, sequence_length, 6] ON CORRECT DEVICE
        """
        winning_cols = [f'Winning_Num_{i}' for i in range(1, 7)]
        
        # Get the last 'sequence_length' draws before current_index
        start_idx = max(0, current_index - self.sequence_length)
        end_idx = current_index
        
        if start_idx == end_idx:
            # If we're at the beginning, create a dummy sequence
            sequence = np.zeros((self.sequence_length, 6), dtype=int)
        else:
            # Extract the sequence
            sequence_data = df.iloc[start_idx:end_idx][winning_cols].values
            
            # Pad if necessary
            if len(sequence_data) < self.sequence_length:
                padding = np.zeros((self.sequence_length - len(sequence_data), 6), dtype=int)
                sequence = np.vstack([padding, sequence_data])
            else:
                sequence = sequence_data[-self.sequence_length:]
        
        # FIXED: Move tensor to the same device as model parameters
        device = next(self.parameters()).device
        return torch.tensor(sequence, dtype=torch.long, device=device).unsqueeze(0)
    
    def forward(self, sequence_tensor):
        """
        Processes temporal sequence to generate context representation.
        
        Args:
            sequence_tensor: [batch_size, sequence_length, 6]
        
        Returns:
            context_vector: [batch_size, context_dim]
            trend_features: [batch_size, trend_features]
        """
        batch_size, seq_len, num_picks = sequence_tensor.shape
        device = sequence_tensor.device
        
        # FIXED: Ensure input tensor is on correct device
        if sequence_tensor.device != next(self.parameters()).device:
            sequence_tensor = sequence_tensor.to(next(self.parameters()).device)
        
        # Embed numbers and flatten for LSTM input
        embedded = self.number_embeddings(sequence_tensor)  # [batch, seq_len, 6, embed_dim]
        lstm_input = embedded.view(batch_size, seq_len, -1)  # [batch, seq_len, 6*embed_dim]
        
        # Process through LSTM
        lstm_output, (hidden, cell) = self.lstm(lstm_input)  # [batch, seq_len, 2*hidden_dim]
        
        # Apply attention to get sequence summary
        attended_output, attention_weights = self.attention(
            lstm_output, lstm_output, lstm_output
        )  # [batch, seq_len, 2*hidden_dim]
        
        # Global average pooling over sequence dimension
        context_raw = attended_output.mean(dim=1)  # [batch, 2*hidden_dim]
        
        # Project to final context dimension
        context_vector = self.context_projection(context_raw)  # [batch, context_dim]
        
        # Extract trend features
        trend_features = self.trend_analyzer(context_vector)  # [batch, trend_features]
        
        return context_vector, trend_features
    
    def get_recent_patterns(self, sequence_tensor):
        """
        Analyzes recent patterns in the sequence for additional insights.
        
        Returns:
            pattern_dict: Dictionary with various pattern statistics
        """
        batch_size, seq_len, _ = sequence_tensor.shape
        patterns = {}
        
        # Move to CPU for numpy operations
        sequence_cpu = sequence_tensor.cpu().numpy()
        
        for b in range(batch_size):
            sequence = sequence_cpu[b]
            
            # Calculate recent frequency distributions
            recent_numbers = sequence.flatten()
            unique, counts = np.unique(recent_numbers[recent_numbers > 0], return_counts=True)
            freq_dist = dict(zip(unique, counts))
            
            # Calculate consecutive number patterns
            consecutive_patterns = []
            for draw in sequence:
                if np.any(draw > 0):  # Skip padding
                    sorted_draw = np.sort(draw[draw > 0])
                    diffs = np.diff(sorted_draw)
                    consecutive_patterns.extend(diffs.tolist())
            
            patterns[b] = {
                'frequency_distribution': freq_dist,
                'consecutive_patterns': consecutive_patterns,
                'recent_range': {
                    'min': int(np.min(recent_numbers[recent_numbers > 0])) if np.any(recent_numbers > 0) else 1,
                    'max': int(np.max(recent_numbers[recent_numbers > 0])) if np.any(recent_numbers > 0) else 49
                }
            }
        
        return patterns