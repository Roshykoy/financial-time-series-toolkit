import torch
import torch.nn as nn
import math

# --- Model Hyperparameters ---
# We will now define d_features and d_targets dynamically in the training script
MODEL_PARAMS = {
    'd_model': 128,
    'nhead': 8,
    'd_hid': 256,
    'nlayers': 4,
    'dropout': 0.1,
}

class PositionalEncoding(nn.Module):
    """Standard Positional Encoding for Transformer models."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerMultiTarget(nn.Module):
    """
    A Transformer model for predicting a vector of multiple regression targets.
    """
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        d_model = params['d_model']
        
        self.embedding = nn.Linear(params['d_features'], d_model)
        self.pos_encoder = PositionalEncoding(d_model, params['dropout'])
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, params['nhead'], params['d_hid'], params['dropout'], batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, params['nlayers'])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # New Regression Head: outputs a vector of size d_targets
        self.regression_head = nn.Linear(d_model, params['d_targets'])

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        embedded_src = self.embedding(src) * math.sqrt(self.params['d_model'])
        pos_encoded_src = self.pos_encoder(embedded_src)
        transformer_output = self.transformer_encoder(pos_encoded_src)
        
        last_step_output = transformer_output[:, -1, :]
        normalized_output = self.final_norm(last_step_output)
        
        # Get direct value predictions from the regression head
        predictions = self.regression_head(normalized_output)
        
        return predictions