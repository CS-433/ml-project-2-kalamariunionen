import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff

# The background class is hard-coded here with a default value of 4.
# This was derived from the unique classes in the dataset being enumerated as 0, 3, 4, 5, 6.
# In this case, the background value 6 was mapped to 4 (when enumerated as 0, 1, 2, 3, 4).
# If the dataset changes or the class enumeration changes, you need to identify the new background value
# and pass it to the `evaluate` function using the `background_class` parameter.

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, background_class=4):
    """
    Evaluate the performance of a segmentation model using the Dice coefficient.

    This function evaluates the model on a given validation dataset and computes
    the Dice coefficient to measure segmentation performance. For multi-class
    segmentation, it excludes the background class when computing the Dice score.

    Args:
        net: The neural network model (should have `n_classes` attribute).
        dataloader: DataLoader providing batches of images and corresponding masks.
        device: The device (CPU or GPU) on which to perform the evaluation.
        amp: Boolean indicating whether to use Automatic Mixed Precision (AMP).
        background_class: The index of the background class to exclude when calculating the Dice score for multi-class segmentation.

    Returns:
        The average Dice score over all validation batches.
    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # Move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # Predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes)'

                # Convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()

                # Exclude the background class dynamically
                mask_true_foreground = torch.cat([mask_true[:, i:i+1] for i in range(net.n_classes) if i != background_class], dim=1)
                mask_pred_foreground = torch.cat([mask_pred[:, i:i+1] for i in range(net.n_classes) if i != background_class], dim=1)

                # Compute the Dice score for foreground classes only
                dice_score += multiclass_dice_coeff(mask_pred_foreground, mask_true_foreground, reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
