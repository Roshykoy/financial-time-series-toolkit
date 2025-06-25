# src/graph_encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import combinations

class GraphAttentionLayer(nn.Module):
    """Single Graph Attention layer for processing number relationships."""
    
    def __init__(self, in_features, out_features, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, h, adj):
        """
        Args:
            h: Node features [batch_size, num_nodes, in_features]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        """
        batch_size, num_nodes = h.size(0), h.size(1)
        
        # Transform node features
        Wh = self.W(h)  # [batch_size, num_nodes, out_features]
        
        # Compute attention coefficients
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(self.a(a_input).squeeze(-1))  # [batch_size, num_nodes, num_nodes]
        
        # Mask attention coefficients where no edge exists
        e = e.masked_fill(adj == 0, -1e9)
        
        # Softmax to get attention weights
        attention = F.softmax(e, dim=-1)
        attention = self.dropout_layer(attention)
        
        # Apply attention to features
        h_prime = torch.bmm(attention, Wh)  # [batch_size, num_nodes, out_features]
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        batch_size, num_nodes, out_features = Wh.size()
        
        # Create all pairs for attention computation
        Wh1 = Wh.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [batch, nodes, nodes, features]
        Wh2 = Wh.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [batch, nodes, nodes, features]
        
        # Concatenate pairs
        all_combinations = torch.cat([Wh1, Wh2], dim=-1)  # [batch, nodes, nodes, 2*features]
        
        return all_combinations

class NumberGraphEncoder(nn.Module):
    """
    Graph Neural Network encoder that treats lottery numbers as nodes
    and their relationships (co-occurrence, proximity) as edges.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_numbers = config['num_lotto_numbers']
        self.node_embedding_dim = config['node_embedding_dim']
        self.graph_hidden_dim = config['graph_hidden_dim']
        self.latent_dim = config['latent_dim']
        
        # Node embeddings for each possible lottery number
        self.number_embeddings = nn.Embedding(self.num_numbers + 1, self.node_embedding_dim)
        
        # Graph attention layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(self.node_embedding_dim if i == 0 else self.graph_hidden_dim, 
                              self.graph_hidden_dim, 
                              config['dropout'])
            for i in range(config['num_gat_layers'])
        ])
        
        # Output projection to latent space
        self.graph_to_latent = nn.Sequential(
            nn.Linear(6 * self.graph_hidden_dim, config['graph_projection_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['graph_projection_dim'], self.latent_dim * 2)  # For mean and logvar
        )
        
    def create_adjacency_matrix(self, number_sets, pair_counts):
        """
        Creates adjacency matrices for batches of number combinations.
        Edge weights based on historical co-occurrence frequencies.
        """
        batch_size = len(number_sets)
        adj_matrices = torch.zeros(batch_size, 6, 6, device=next(self.parameters()).device)
        
        for batch_idx, number_set in enumerate(number_sets):
            for i in range(6):
                for j in range(6):
                    if i != j:
                        num1, num2 = sorted([number_set[i], number_set[j]])
                        pair_freq = pair_counts.get((num1, num2), 0)
                        # Normalize by maximum possible frequency
                        max_freq = max(pair_counts.values()) if pair_counts else 1
                        adj_matrices[batch_idx, i, j] = pair_freq / max_freq
            
            # Add self-loops
            adj_matrices[batch_idx].fill_diagonal_(1.0)
        
        return adj_matrices
    
    def forward(self, number_sets, pair_counts):
        """
        Encodes number combinations into latent representations.
        
        Args:
            number_sets: List of 6-number combinations [[1,2,3,4,5,6], ...]
            pair_counts: Dictionary of pair frequencies from feature engineer
        
        Returns:
            mu, logvar: Mean and log-variance for VAE latent space
        """
        batch_size = len(number_sets)
        device = next(self.parameters()).device
        
        # Convert to tensor and get embeddings
        number_tensor = torch.tensor(number_sets, device=device)  # [batch_size, 6]
        node_features = self.number_embeddings(number_tensor)  # [batch_size, 6, node_embedding_dim]
        
        # Create adjacency matrices
        adj_matrices = self.create_adjacency_matrix(number_sets, pair_counts)
        
        # Apply graph attention layers
        h = node_features
        for gat_layer in self.gat_layers:
            h = gat_layer(h, adj_matrices)
            h = F.relu(h)
        
        # Global pooling: flatten the graph representation
        graph_repr = h.view(batch_size, -1)  # [batch_size, 6 * graph_hidden_dim]
        
        # Project to latent space parameters
        latent_params = self.graph_to_latent(graph_repr)  # [batch_size, latent_dim * 2]
        
        # Split into mean and log-variance
        mu, logvar = torch.chunk(latent_params, 2, dim=-1)
        
        return mu, logvar