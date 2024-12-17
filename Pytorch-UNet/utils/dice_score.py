import torch
from torch import Tensor

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Compute the Dice coefficient for binary segmentation tasks.

    Args:
        input (Tensor): The predicted mask tensor. Shape: (B, H, W) or (H, W).
        target (Tensor): The ground truth mask tensor. Must have the same shape as `input`.
        reduce_batch_first (bool): Whether to compute the Dice coefficient across the batch dimension.
        epsilon (float): A small constant to avoid division by zero.

    Returns:
        Tensor: The mean Dice coefficient.
    """
    # Ensure input and target have the same shape
    assert input.size() == target.size(), "Input and target tensors must have the same shape."
    assert input.dim() == 3 or not reduce_batch_first,  "If reduce_batch_first is True, input must be 3D (B, H, W)."

    # Determine the dimensions to sum over
    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    # Calculate intersection and union (sum of sets)
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    # Avoid division by zero by using `epsilon`
    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Compute the average Dice coefficient for multi-class segmentation tasks.

    Args:
        input (Tensor): The predicted mask tensor. Shape: (B, C, H, W).
        target (Tensor): The ground truth mask tensor. Shape: (B, C, H, W).
        reduce_batch_first (bool): Whether to compute the Dice coefficient across the batch dimension.
        epsilon (float): A small constant to avoid division by zero.

    Returns:
        Tensor: The mean Dice coefficient across all classes.
    """
    # Flatten the class dimension and compute the Dice coefficient
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    """
    Compute the Dice loss, which is 1 - Dice coefficient.

    Args:
        input (Tensor): The predicted mask tensor.
        target (Tensor): The ground truth mask tensor.
        multiclass (bool): Whether the input and target are multi-class masks.

    Returns:
        Tensor: The Dice loss.
    """
    # Select the appropriate function for binary or multi-class segmentation
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
