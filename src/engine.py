import torch
import torch.nn as nn

def train_one_epoch(model, data_loader, optimizer, loss_fn, device):
    """Performs one full epoch of training for the multi-target model."""
    model.train()
    total_loss = 0
    for X_batch, y_batch in data_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Model now returns a single prediction tensor
        predictions = model(X_batch)
        loss = loss_fn(predictions, y_batch)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(data_loader)

def evaluate(model, data_loader, loss_fn, device):
    """Performs one full epoch of validation for the multi-target model."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Model now returns a single prediction tensor
            predictions = model(X_batch)
            loss = loss_fn(predictions, y_batch)
            total_loss += loss.item()
            
    return total_loss / len(data_loader)