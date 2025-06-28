"""
Safe mathematical operations for MarkSix forecasting system.
Provides numerical stability and prevents common mathematical errors.
"""
import torch
import numpy as np
import warnings
from typing import Union, Optional


def safe_divide(
    numerator: Union[torch.Tensor, float], 
    denominator: Union[torch.Tensor, float], 
    eps: float = 1e-8,
    default_value: Union[torch.Tensor, float] = 0.0
) -> Union[torch.Tensor, float]:
    """
    Perform safe division with protection against division by zero.
    
    Args:
        numerator: The numerator
        denominator: The denominator
        eps: Small epsilon to add to denominator if it's zero
        default_value: Value to return if division is invalid
    
    Returns:
        Result of safe division
    """
    if isinstance(denominator, torch.Tensor):
        # Check for zero or very small values
        safe_denominator = torch.where(
            torch.abs(denominator) < eps,
            torch.full_like(denominator, eps),
            denominator
        )
        return numerator / safe_denominator
    else:
        # Scalar case
        if abs(denominator) < eps:
            if denominator == 0:
                warnings.warn("Division by zero detected, using epsilon", RuntimeWarning)
            return numerator / eps if eps != 0 else default_value
        return numerator / denominator


def safe_log(
    x: Union[torch.Tensor, float], 
    eps: float = 1e-8,
    base: Optional[float] = None
) -> Union[torch.Tensor, float]:
    """
    Perform safe logarithm with protection against log(0) and log(negative).
    
    Args:
        x: Input value(s)
        eps: Small epsilon to add if x is zero or negative
        base: Logarithm base (None for natural log)
    
    Returns:
        Safe logarithm result
    """
    if isinstance(x, torch.Tensor):
        safe_x = torch.clamp(x, min=eps)
        if base is None:
            return torch.log(safe_x)
        else:
            return torch.log(safe_x) / np.log(base)
    else:
        safe_x = max(x, eps)
        if safe_x != x and x <= 0:
            warnings.warn(f"log of non-positive number ({x}) clamped to eps", RuntimeWarning)
        
        if base is None:
            return np.log(safe_x)
        else:
            return np.log(safe_x) / np.log(base)


def safe_exp(
    x: Union[torch.Tensor, float], 
    max_exp: float = 80.0
) -> Union[torch.Tensor, float]:
    """
    Perform safe exponential with protection against overflow.
    
    Args:
        x: Input value(s)
        max_exp: Maximum exponent value to prevent overflow
    
    Returns:
        Safe exponential result
    """
    if isinstance(x, torch.Tensor):
        clamped_x = torch.clamp(x, max=max_exp)
        if torch.any(x > max_exp):
            warnings.warn("Exponential overflow protection applied", RuntimeWarning)
        return torch.exp(clamped_x)
    else:
        clamped_x = min(x, max_exp)
        if x > max_exp:
            warnings.warn(f"Exponential overflow protection: {x} clamped to {max_exp}", RuntimeWarning)
        return np.exp(clamped_x)


def safe_softmax(
    logits: torch.Tensor, 
    dim: int = -1, 
    temperature: float = 1.0,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Perform numerically stable softmax.
    
    Args:
        logits: Input logits
        dim: Dimension to apply softmax
        temperature: Temperature scaling parameter
        eps: Small epsilon for numerical stability
    
    Returns:
        Stable softmax probabilities
    """
    # Scale by temperature
    scaled_logits = logits / max(temperature, eps)
    
    # Subtract max for numerical stability
    max_logits = torch.max(scaled_logits, dim=dim, keepdim=True)[0]
    stable_logits = scaled_logits - max_logits
    
    # Compute softmax
    exp_logits = torch.exp(stable_logits)
    sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
    
    # Protect against zero sum
    safe_sum = torch.clamp(sum_exp, min=eps)
    
    return exp_logits / safe_sum


def safe_normalize(
    tensor: torch.Tensor, 
    dim: int = -1, 
    eps: float = 1e-8,
    p: float = 2.0
) -> torch.Tensor:
    """
    Perform safe L-p normalization with protection against zero norm.
    
    Args:
        tensor: Input tensor
        dim: Dimension to normalize
        eps: Small epsilon to add to norm if it's zero
        p: Norm type (2 for L2 norm)
    
    Returns:
        Normalized tensor
    """
    norm = torch.norm(tensor, p=p, dim=dim, keepdim=True)
    safe_norm = torch.clamp(norm, min=eps)
    return tensor / safe_norm


def check_tensor_health(
    tensor: torch.Tensor, 
    name: str = "tensor",
    check_finite: bool = True,
    check_range: Optional[tuple] = None,
    warn_threshold: float = 1e6
) -> bool:
    """
    Check tensor for numerical health issues.
    
    Args:
        tensor: Tensor to check
        name: Name for error messages
        check_finite: Whether to check for NaN/Inf
        check_range: Optional (min, max) range to check
        warn_threshold: Threshold for large value warnings
    
    Returns:
        True if tensor is healthy, False otherwise
    """
    if not isinstance(tensor, torch.Tensor):
        return True
    
    # Check for NaN
    if torch.isnan(tensor).any():
        warnings.warn(f"{name} contains NaN values", RuntimeWarning)
        return False
    
    # Check for Inf
    if check_finite and torch.isinf(tensor).any():
        warnings.warn(f"{name} contains infinite values", RuntimeWarning)
        return False
    
    # Check for very large values
    max_val = tensor.abs().max().item()
    if max_val > warn_threshold:
        warnings.warn(f"{name} has very large values (max: {max_val})", RuntimeWarning)
    
    # Check for FP16 overflow
    if tensor.dtype == torch.float16 and max_val > 65504:
        warnings.warn(f"{name} may have FP16 overflow (max: {max_val})", RuntimeWarning)
        return False
    
    # Check custom range
    if check_range is not None:
        min_val, max_val = check_range
        tensor_min = tensor.min().item()
        tensor_max = tensor.max().item()
        
        if tensor_min < min_val or tensor_max > max_val:
            warnings.warn(
                f"{name} values outside expected range [{min_val}, {max_val}]. "
                f"Actual range: [{tensor_min}, {tensor_max}]",
                RuntimeWarning
            )
            return False
    
    return True


def safe_cosine_similarity(
    x1: torch.Tensor, 
    x2: torch.Tensor, 
    dim: int = -1, 
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute cosine similarity with protection against zero norms.
    
    Args:
        x1: First tensor
        x2: Second tensor
        dim: Dimension for similarity computation
        eps: Small epsilon for numerical stability
    
    Returns:
        Safe cosine similarity
    """
    # Normalize both tensors safely
    x1_norm = safe_normalize(x1, dim=dim, eps=eps)
    x2_norm = safe_normalize(x2, dim=dim, eps=eps)
    
    # Compute similarity
    similarity = torch.sum(x1_norm * x2_norm, dim=dim)
    
    # Clamp to valid range [-1, 1]
    return torch.clamp(similarity, min=-1.0, max=1.0)


def stable_kl_divergence(
    mu1: torch.Tensor, 
    logvar1: torch.Tensor, 
    mu2: torch.Tensor, 
    logvar2: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute KL divergence between two Gaussian distributions with numerical stability.
    
    Args:
        mu1: Mean of first distribution
        logvar1: Log variance of first distribution
        mu2: Mean of second distribution
        logvar2: Log variance of second distribution
        eps: Small epsilon for numerical stability
    
    Returns:
        Stable KL divergence
    """
    # Ensure log variances are not too negative (prevent underflow)
    logvar1 = torch.clamp(logvar1, min=-20, max=20)
    logvar2 = torch.clamp(logvar2, min=-20, max=20)
    
    # Compute variance ratio with stability
    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)
    var_ratio = safe_divide(var1, var2, eps=eps)
    
    # Compute mean difference squared term
    mean_diff_sq = (mu1 - mu2).pow(2)
    mean_term = safe_divide(mean_diff_sq, var2, eps=eps)
    
    # Compute log term
    log_term = logvar2 - logvar1
    
    # KL divergence formula: 0.5 * (var_ratio + mean_term + log_term - 1)
    kl_div = 0.5 * (var_ratio + mean_term + log_term - 1)
    
    # Sum over latent dimensions and take mean over batch
    return kl_div.sum(dim=-1).mean()


def gradient_clipping_with_health_check(
    parameters, 
    max_norm: float, 
    norm_type: float = 2.0,
    error_if_nonfinite: bool = True
) -> float:
    """
    Clip gradients with health check for NaN/Inf gradients.
    
    Args:
        parameters: Model parameters
        max_norm: Maximum gradient norm
        norm_type: Type of norm to use
        error_if_nonfinite: Whether to raise error on non-finite gradients
    
    Returns:
        Total gradient norm before clipping
    """
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    if len(parameters) == 0:
        return 0.0
    
    # Check for NaN/Inf gradients
    has_nan_or_inf = False
    for p in parameters:
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            has_nan_or_inf = True
            break
    
    if has_nan_or_inf:
        if error_if_nonfinite:
            raise RuntimeError("NaN or Inf gradients detected during training")
        else:
            warnings.warn("NaN or Inf gradients detected, zeroing gradients", RuntimeWarning)
            for p in parameters:
                if p.grad is not None:
                    p.grad.zero_()
            return 0.0
    
    # Compute total norm
    device = parameters[0].grad.device
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
        norm_type
    )
    
    # Clip gradients
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef.to(p.grad.device))
    
    return total_norm.item()


def safe_tensor_operation(operation_name: str = "tensor operation"):
    """
    Decorator for safe tensor operations with automatic health checking.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                
                # Check result health if it's a tensor
                if isinstance(result, torch.Tensor):
                    check_tensor_health(result, f"{operation_name}_result")
                elif isinstance(result, (list, tuple)):
                    for i, item in enumerate(result):
                        if isinstance(item, torch.Tensor):
                            check_tensor_health(item, f"{operation_name}_result_{i}")
                
                return result
                
            except Exception as e:
                warnings.warn(f"Error in {operation_name}: {e}", RuntimeWarning)
                raise
        
        return wrapper
    return decorator