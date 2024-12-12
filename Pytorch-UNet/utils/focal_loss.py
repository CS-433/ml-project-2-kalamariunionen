import torch
import torch.nn.functional as F
from torch import Tensor

def focal_loss(input: Tensor, target: Tensor, alpha: float = 0.8, gamma: float = 2.0, epsilon: float = 1e-6):
    """
    Compute focal loss for binary case, assuming `input` and `target` are probabilities for the positive class.
    `input`: (B, H, W) probabilities for the positive class
    `target`: (B, H, W) binary ground truth
    """
    assert input.size() == target.size()
    # Clip probabilities to avoid numerical issues
    input_clamped = torch.clamp(input, min=epsilon, max=1.0 - epsilon)

    # Focal loss calculation
    pt = torch.where(target == 1, input_clamped, 1 - input_clamped)
    focal = -alpha * (1 - pt)**gamma * pt.log()
    return focal.mean()

def multiclass_focal_loss(input: Tensor, target: Tensor, alpha: float = 0.8, gamma: float = 2.0, epsilon: float = 1e-6):
    """
    Compute focal loss for multi-class case, assuming `input` and `target` are one-hot encoded probabilities.
    `input`: (B, C, H, W) probabilities for each class
    `target`: (B, C, H, W) one-hot encoded ground truth
    """
    assert input.size() == target.size()
    # Clip probabilities
    input_clamped = torch.clamp(input, min=epsilon, max=1.0 - epsilon)

    # Compute focal loss for each class separately and average
    pt = torch.where(target == 1, input_clamped, 1 - input_clamped)
    focal = -alpha * (1 - pt)**gamma * pt.log()
    return focal.mean()

def focal_loss_wrapper(input: Tensor, target: Tensor, multiclass: bool = False, alpha: float = 0.8, gamma: float = 2.0):
    """
    A wrapper to mimic the dice_loss interface.
    For binary: Expect (B, H, W) input probabilities and (B, H, W) target.
    For multiclass: Expect (B, C, H, W) input probabilities and (B, C, H, W) one-hot target.
    """
    if multiclass:
        return multiclass_focal_loss(input, target, alpha=alpha, gamma=gamma)
    else:
        return focal_loss(input, target, alpha=alpha, gamma=gamma)
