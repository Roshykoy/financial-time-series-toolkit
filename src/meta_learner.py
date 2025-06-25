# src/meta_learner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CombinationAnalyzer(nn.Module):
    """
    Analyzes the characteristics of a number combination to understand
    which scoring methods might be most relevant.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_numbers = config['num_lotto_numbers']
        
        # Feature extraction layers
        self.combination_embedding = nn.Embedding(self.num_numbers + 1, config['combo_embedding_dim'])
        
        # Pattern analysis network
        self.pattern_analyzer = nn.Sequential(
            nn.Linear(6 * config['combo_embedding_dim'], config['pattern_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['pattern_hidden_dim'], config['pattern_features'])
        )
        
        # Statistical feature extractor
        self.stat_extractor = nn.Sequential(
            nn.Linear(config['statistical_features'], config['stat_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['stat_hidden_dim'], config['stat_features'])
        )
        
    def extract_statistical_features(self, number_sets):
        """
        Extracts hand-crafted statistical features from number combinations.
        
        Args:
            number_sets: List of number combinations
        
        Returns:
            statistical_features: Tensor of statistical features
        """
        features = []
        
        for number_set in number_sets:
            nums = np.array(sorted(number_set))
            
            # Basic statistics
            combination_sum = nums.sum()
            combination_mean = nums.mean()
            combination_std = nums.std()
            combination_range = nums.max() - nums.min()
            
            # Distribution features
            odd_count = np.sum(nums % 2 == 1)
            even_count = 6 - odd_count
            low_count = np.sum(nums <= 24)  # Lower half
            high_count = 6 - low_count
            
            # Consecutive patterns
            diffs = np.diff(nums)
            consecutive_pairs = np.sum(diffs == 1)
            max_gap = diffs.max()
            min_gap = diffs.min()
            
            # Decade distribution
            decades = nums // 10
            unique_decades = len(np.unique(decades))
            decade_spread = decades.max() - decades.min()
            
            # Combine all features
            stat_features = [
                combination_sum, combination_mean, combination_std, combination_range,
                odd_count, even_count, low_count, high_count,
                consecutive_pairs, max_gap, min_gap,
                unique_decades, decade_spread
            ]
            
            features.append(stat_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def forward(self, number_sets):
        """
        Analyzes number combinations to extract features for meta-learning.
        
        Args:
            number_sets: List of 6-number combinations
        
        Returns:
            analysis_features: Combined feature representation
        """
        device = next(self.parameters()).device
        batch_size = len(number_sets)
        
        # Convert to tensor and embed
        number_tensor = torch.tensor(number_sets, device=device)
        embedded = self.combination_embedding(number_tensor)  # [batch, 6, embed_dim]
        
        # Pattern analysis
        flattened = embedded.view(batch_size, -1)  # [batch, 6 * embed_dim]
        pattern_features = self.pattern_analyzer(flattened)
        
        # Statistical features
        stat_features_raw = self.extract_statistical_features(number_sets).to(device)
        stat_features = self.stat_extractor(stat_features_raw)
        
        # Combine features
        analysis_features = torch.cat([pattern_features, stat_features], dim=-1)
        
        return analysis_features

class AttentionMetaLearner(nn.Module):
    """
    Meta-learner that dynamically determines optimal weights for different
    scoring methods based on the characteristics of each number combination.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_scorers = len(config['scorer_types'])  # generative, temporal, i_ching
        
        # Combination analyzer
        self.combination_analyzer = CombinationAnalyzer(config)
        
        # Context integration
        total_features = config['pattern_features'] + config['stat_features'] + config['temporal_context_dim']
        
        self.context_integrator = nn.Sequential(
            nn.Linear(total_features, config['meta_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['meta_hidden_dim'], config['integrated_features'])
        )
        
        # Multi-head attention for scorer weight determination
        self.scorer_attention = nn.MultiheadAttention(
            embed_dim=config['integrated_features'],
            num_heads=config['meta_attention_heads'],
            dropout=config['dropout'],
            batch_first=True
        )
        
        # Weight generation networks for each scorer type
        self.weight_generators = nn.ModuleDict({
            scorer_type: nn.Sequential(
                nn.Linear(config['integrated_features'], config['weight_hidden_dim']),
                nn.ReLU(),
                nn.Dropout(config['dropout']),
                nn.Linear(config['weight_hidden_dim'], 1)
            )
            for scorer_type in config['scorer_types']
        })
        
        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config['integrated_features'], config['confidence_hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['confidence_hidden_dim'], 1),
            nn.Sigmoid()
        )
        
        # Score normalization parameters (learnable)
        self.score_normalizers = nn.ParameterDict({
            scorer_type: nn.Parameter(torch.tensor([1.0, 0.0]))  # scale, shift
            for scorer_type in config['scorer_types']
        })
        
    def forward(self, number_sets, temporal_context, scorer_scores):
        """
        Determines optimal ensemble weights for each combination.
        
        Args:
            number_sets: List of number combinations
            temporal_context: Temporal context features [batch_size, context_dim]
            scorer_scores: Dict of scores from different methods
                          {scorer_type: [batch_size] tensor}
        
        Returns:
            ensemble_weights: [batch_size, num_scorers]
            final_scores: [batch_size]
            confidence: [batch_size]
        """
        batch_size = len(number_sets)
        device = temporal_context.device
        
        # Analyze combinations
        combo_features = self.combination_analyzer(number_sets)
        
        # Integrate with temporal context
        integrated_input = torch.cat([combo_features, temporal_context], dim=-1)
        integrated_features = self.context_integrator(integrated_input)
        
        # Apply self-attention to refine features
        attended_features, _ = self.scorer_attention(
            integrated_features.unsqueeze(1),
            integrated_features.unsqueeze(1),
            integrated_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Generate weights for each scorer
        raw_weights = {}
        for scorer_type in self.config['scorer_types']:
            weight = self.weight_generators[scorer_type](attended_features).squeeze(-1)
            raw_weights[scorer_type] = weight
        
        # Convert to tensor and apply softmax
        weight_tensor = torch.stack([raw_weights[st] for st in self.config['scorer_types']], dim=-1)
        ensemble_weights = F.softmax(weight_tensor, dim=-1)
        
        # Normalize scores using learnable parameters
        normalized_scores = {}
        for scorer_type in self.config['scorer_types']:
            if scorer_type in scorer_scores:
                scale, shift = self.score_normalizers[scorer_type]
                normalized_scores[scorer_type] = scorer_scores[scorer_type] * scale + shift
            else:
                normalized_scores[scorer_type] = torch.zeros(batch_size, device=device)
        
        # Compute final ensemble scores
        score_tensor = torch.stack([normalized_scores[st] for st in self.config['scorer_types']], dim=-1)
        final_scores = (ensemble_weights * score_tensor).sum(dim=-1)
        
        # Estimate confidence
        confidence = self.confidence_estimator(attended_features).squeeze(-1)
        
        return ensemble_weights, final_scores, confidence
    
    def get_weight_explanations(self, number_sets, temporal_context, scorer_scores):
        """
        Provides explanations for why certain weights were assigned.
        
        Returns:
            explanations: List of dictionaries with weight reasoning
        """
        self.eval()
        with torch.no_grad():
            weights, scores, confidence = self.forward(number_sets, temporal_context, scorer_scores)
            
            explanations = []
            for i, number_set in enumerate(number_sets):
                explanation = {
                    'combination': number_set,
                    'weights': {
                        scorer_type: weights[i, j].item()
                        for j, scorer_type in enumerate(self.config['scorer_types'])
                    },
                    'confidence': confidence[i].item(),
                    'final_score': scores[i].item(),
                    'reasoning': self._generate_reasoning(number_set, weights[i], confidence[i])
                }
                explanations.append(explanation)
            
            return explanations
    
    def _generate_reasoning(self, number_set, weights, confidence):
        """Generates human-readable reasoning for weight assignment."""
        reasoning = []
        weight_dict = {
            scorer_type: weights[j].item()
            for j, scorer_type in enumerate(self.config['scorer_types'])
        }
        
        # Find dominant scorer
        dominant_scorer = max(weight_dict.keys(), key=lambda k: weight_dict[k])
        
        # Analyze combination characteristics
        nums = np.array(sorted(number_set))
        consecutive_pairs = np.sum(np.diff(nums) == 1)
        spread = nums.max() - nums.min()
        
        if dominant_scorer == 'generative':
            reasoning.append("High weight on generative model suggests this combination follows learned patterns")
        elif dominant_scorer == 'temporal':
            reasoning.append("High weight on temporal scorer indicates recent number trends are important")
        elif dominant_scorer == 'i_ching':
            reasoning.append("High weight on I-Ching suggests numerological factors are significant")
        
        if consecutive_pairs > 2:
            reasoning.append(f"Contains {consecutive_pairs} consecutive pairs")
        if spread < 20:
            reasoning.append("Numbers are clustered in a narrow range")
        elif spread > 35:
            reasoning.append("Numbers are spread across the full range")
        
        if confidence.item() > 0.8:
            reasoning.append("High confidence in this weighting")
        elif confidence.item() < 0.4:
            reasoning.append("Low confidence - combination has unusual characteristics")
        
        return "; ".join(reasoning)