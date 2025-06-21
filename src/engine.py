# src/engine.py

import torch

def ranking_loss(positive_scores, negative_scores, margin):
    loss = torch.clamp(margin - positive_scores.unsqueeze(1) + negative_scores, min=0)
    return loss.mean()

def train_one_epoch(model, train_loader, optimizer, device, config):
    """Performs one full epoch of contrastive training with SAM."""
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        pos_features = batch['positive_features'].to(device)
        neg_features = batch['negative_features'].to(device)

        if config['use_sam_optimizer']:
            # --- Corrected SAM Logic ---
            # 1. First forward/backward pass to compute gradients for the first step
            optimizer.zero_grad()
            pos_scores = model(pos_features)
            neg_scores = model(neg_features.view(-1, neg_features.size(-1))).view(pos_features.size(0), -1)
            loss = ranking_loss(pos_scores, neg_scores, config['margin'])
            loss.backward()
            
            # 2. SAM step. The optimizer will use the gradients from the pass above for its first step,
            #    and then use the closure to perform its second step.
            def closure():
                # This closure re-evaluates the loss on the perturbed weights.
                pos_scores = model(pos_features)
                neg_scores = model(neg_features.view(-1, neg_features.size(-1))).view(pos_features.size(0), -1)
                inner_loss = ranking_loss(pos_scores, neg_scores, config['margin'])
                inner_loss.backward() # Compute gradients for the second step
                return inner_loss
            
            optimizer.step(closure)
            
            # Track the loss from the first, non-perturbed step
            total_loss += loss.item()

        else:
            # --- Corrected Standard Optimizer Logic ---
            optimizer.zero_grad()
            pos_scores = model(pos_features)
            neg_scores = model(neg_features.view(-1, neg_features.size(-1))).view(pos_features.size(0), -1)
            loss = ranking_loss(pos_scores, neg_scores, config['margin'])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device, config):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            pos_features = batch['positive_features'].to(device)
            neg_features = batch['negative_features'].to(device)

            pos_scores = model(pos_features)
            neg_scores = model(neg_features.view(-1, neg_features.size(-1))).view(pos_features.size(0), -1)
            loss = ranking_loss(pos_scores, neg_scores, config['margin'])
            total_loss += loss.item()
            
    return total_loss / len(val_loader)