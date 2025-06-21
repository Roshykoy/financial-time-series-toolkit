# src/model.py
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config['hidden_size']
        dropout = config['dropout']
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.linear1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + shortcut

class ScoringModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.embedding = nn.Linear(config['d_features'], config['hidden_size'])
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config['num_layers'])])
        self.output_head = nn.Linear(config['hidden_size'], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        score = self.output_head(x).squeeze(-1)
        return score