import argparse
import logging
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

from unet import UNet
from utils.utils import plot_img_and_mask

"""
Class Enumeration and Color Mapping:
------------------------------------
The original unique values for the dataset were 0, 3, 4, 5, and 6, corresponding to different body parts.
For the purpose of generating masks and visualizing them in grayscale, these values have been mapped as follows:

- **Head (original value 0)** -> Grayscale value 0 (dark gray)
- **Thorax (original value 3)** -> Grayscale value 100 (medium gray)
- **Abdomen (original value 4)** -> Grayscale value 150 (lighter gray)
- **Eye (original value 5)** -> Grayscale value 200 (light gray)
- **Background (original value 6)** -> Grayscale value 255 (white)

When predicting masks with the model, ensure that these mappings are consistent to visualize the correct colors.
"""

def predict_img(net, full_img, device, out_threshold=0.5):
    """
    Predict a segmentation mask for a single input image.

    Args:
        net (torch.nn.Module): The trained U-Net model.
        full_img (PIL.Image): The input image.
        device (torch.device): The device to run the prediction on (CPU or GPU).
        out_threshold (float): Threshold for converting probabilities to binary mask (default: 0.5).

    Returns:
        np.ndarray: The predicted segmentation mask.
    """
    net.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img = transform(full_img).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        mask = output.argmax(dim=1).squeeze().numpy()

    return mask


def get_args():
    """Parse command-line arguments for the script."""
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input-dir', '-i', required=True, metavar='INPUT_DIR',
                        help='Directory containing input images')
    parser.add_argument('--output-dir', '-o', required=True, metavar='OUTPUT_DIR',
                        help='Directory to save predicted masks and visualizations')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
    
    return parser.parse_args()


def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))


def save_img_and_mask(img, mask, out_path):
    """
    Save the input image and predicted mask side-by-side as a single visualization.

    Args:
        img (PIL.Image): The input image.
        mask (np.ndarray): The predicted segmentation mask.
        out_path (str or Path): Path to save the visualization.
    """
    # Convert mask to grayscale image
    mask_img = mask_to_image(mask)

    # Resize mask to match the original image size for visualization
    mask_img = mask_img.resize(img.size, Image.NEAREST)

    # Create a side-by-side visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display the original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display the predicted mask
    axes[1].imshow(mask_img, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    # Save the visualization
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def mask_to_image(mask: np.ndarray):
    """
    Convert a predicted multiclass mask to a grayscale PIL Image using class-to-grayscale mappings.

    Args:
        mask (np.ndarray): The predicted segmentation mask. Shape: (H, W).

    Returns:
        PIL.Image: The grayscale mask image.
    """
    # Define grayscale values for each class based on the class enumeration
    class_to_value = {
        0: 0,    # Head (dark gray)
        1: 100,   # Thorax (medium gray)
        2: 150,   # Abdomen (lighter gray)
        3: 200,   # Eye (light gray)
        4: 255    # Background (white)
    }

    # Create an empty array with the same shape as the mask
    out = np.zeros_like(mask, dtype=np.uint8)

    for class_idx, value in class_to_value.items():
        out[mask == class_idx] = value

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Load the model
    net = UNet(n_channels=3, n_classes=args.classes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)

    logging.info(f'Loading model {args.model}')
    state_dict = torch.load(args.model, map_location=device)
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files in the input directory
    input_dir = Path(args.input_dir)
    image_files = list(input_dir.glob('*.png')) + list(input_dir.glob('*.jpg'))

    logging.info(f'Found {len(image_files)} images in {input_dir}')

    for img_path in image_files:
        try:
            logging.info(f'Processing {img_path}')
            img = Image.open(img_path).convert('RGB')

            # Predict the mask
            mask = predict_img(net=net, full_img=img, device=device)

            # Save the predicted mask
            mask_out_path = output_dir / f'{img_path.stem}_mask.png'
            mask_img = mask_to_image(mask)
            mask_img.save(mask_out_path)
            logging.info(f'Saved mask to {mask_out_path}')

            # Optionally visualize the image and mask
            if args.viz:
                logging.info(f'Visualizing results for image {img_path}, close the window to continue...')
                plot_img_and_mask(img, mask)

        except UnidentifiedImageError:
            logging.error(f"UnidentifiedImageError: Cannot identify image file {img_path}")
        except Exception as e:
            logging.error(f"Error processing {img_path}: {e}")

    logging.info("Processing complete.")