import torch
import torch.nn.functional as F
from torch import Tensor

def focal_loss(input: Tensor, target: Tensor, alpha: float = 0.8, gamma: float = 2.0, epsilon: float = 1e-6):
    """
    Compute the focal loss for binary segmentation tasks.

    Focal loss helps address class imbalance by focusing more on hard-to-classify examples.

    Args:
        input (Tensor): Predicted probabilities for the positive class. Shape: (B, H, W).
        target (Tensor): Ground truth binary mask. Shape: (B, H, W).
        alpha (float): Balancing factor for positive and negative classes (default: 0.8).
        gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted (default: 2.0).
        epsilon (float): Small constant to avoid numerical instability (default: 1e-6).

    Returns:
        Tensor: The computed focal loss.
    """
    assert input.size() == target.size(), "Input and target tensors must have the same shape."

    # Clip probabilities to avoid log(0) or log(1)
    input_clamped = torch.clamp(input, min=epsilon, max=1.0 - epsilon)

    # Calculate the probability for the correct class
    pt = torch.where(target == 1, input_clamped, 1 - input_clamped)

    # Compute the focal loss
    focal = -alpha * (1 - pt)**gamma * pt.log()
    return focal.mean()


def multiclass_focal_loss(input: Tensor, target: Tensor, alpha: float = 0.8, gamma: float = 2.0, epsilon: float = 1e-6):
    """
    Compute the focal loss for multi-class segmentation tasks.

    Args:
        input (Tensor): Predicted probabilities for each class. Shape: (B, C, H, W).
        target (Tensor): One-hot encoded ground truth. Shape: (B, C, H, W).
        alpha (float): Balancing factor for positive and negative classes (default: 0.8).
        gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted (default: 2.0).
        epsilon (float): Small constant to avoid numerical instability (default: 1e-6).

    Returns:
        Tensor: The computed focal loss.
    """
    assert input.size() == target.size(), "Input and target tensors must have the same shape."

    # Clip probabilities to avoid log(0) or log(1)
    input_clamped = torch.clamp(input, min=epsilon, max=1.0 - epsilon)

    # Calculate the probability for the correct class
    pt = torch.where(target == 1, input_clamped, 1 - input_clamped)

    # Compute the focal loss
    focal = -alpha * (1 - pt)**gamma * pt.log()
    return focal.mean()


def focal_loss_wrapper(input: Tensor, target: Tensor, multiclass: bool = False, alpha: float = 0.8, gamma: float = 2.0):
    """
    Wrapper function to compute focal loss for both binary and multi-class segmentation tasks.

    Args:
        input (Tensor): 
            - For binary: Predicted probabilities for the positive class. Shape: (B, H, W).
            - For multi-class: Predicted probabilities for each class. Shape: (B, C, H, W).
        target (Tensor): 
            - For binary: Ground truth binary mask. Shape: (B, H, W).
            - For multi-class: One-hot encoded ground truth. Shape: (B, C, H, W).
        multiclass (bool): Flag to indicate if the task is multi-class (default: False).
        alpha (float): Balancing factor for positive and negative classes (default: 0.8).
        gamma (float): Focusing parameter to adjust the rate at which easy examples are down-weighted (default: 2.0).

    Returns:
        Tensor: The computed focal loss.
    """
    if multiclass:
        return multiclass_focal_loss(input, target, alpha=alpha, gamma=gamma)
    else:
        return focal_loss(input, target, alpha=alpha, gamma=gamma)
