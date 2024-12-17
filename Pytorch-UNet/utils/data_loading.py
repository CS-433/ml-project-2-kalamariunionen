import logging
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path

from functools import partial
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

def load_image(filename):
    """
    Load an image from a file, supporting multiple formats (.npy, .pt, .pth, image files).

    Args:
        filename (str): Path to the image file.

    Returns:
        PIL.Image: The loaded image.
    """
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir):
    """
    Retrieve unique values from a mask file associated with a given ID.

    Args:
        idx (str): The ID of the mask file.
        mask_dir (Path): Directory containing the mask files.

    Returns:
        np.ndarray: Unique values present in the mask.

    Raises:
        RuntimeError: If no mask or multiple masks are found for the given ID.
        ValueError: If the mask does not have 2 or 3 dimensions.
    """
    mask_files = list(mask_dir.glob(f'{idx}_p_msk.*'))
    if len(mask_files) != 1:
        raise RuntimeError(f'Either no mask or multiple masks found for the ID {idx}: {mask_files}')
    
    mask_file = mask_files[0]
    mask = np.asarray(load_image(mask_file))
    
    unique_values = np.unique(mask)
    
    if mask.ndim == 2:
        return unique_values
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    """
    Basic dataset class for loading images and masks for segmentation tasks.

    Args:
        images_dir (str): Directory containing the image files.
        mask_dir (str): Directory containing the mask files.
        scale (float): Scale factor for resizing images and masks.
        binary_class (int): Optional class label for binary segmentation.
        train (bool): Flag indicating whether the dataset is for training (True) or validation (False).
    """
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, binary_class: int = None, train: bool = True):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.binary_class = binary_class  # Optional parameter for binary class separation
        self.train = train

        # Gather IDs by extracting the base name before '_p_' from files
        self.ids = [file.split('_p_')[0] for file in listdir(images_dir) if isfile(join(images_dir, file))]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        # Check unique mask values for each ID
        with Pool() as p:
            unique_values_per_id = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir), self.ids),
                total=len(self.ids)
            ))

        # Filter out IDs that do not have exactly 5 unique mask values
        valid_ids = []
        valid_unique_values = []
        for idx, uniq_vals in zip(self.ids, unique_values_per_id):
            # We expect exactly 5 unique values per mask
            if len(uniq_vals) == 5:
                valid_ids.append(idx)
                valid_unique_values.append(uniq_vals)

        self.ids = valid_ids

        if not self.ids:
            raise RuntimeError('No valid masks found with exactly 5 unique values.')

        logging.info(f'Final dataset size: {len(self.ids)} examples')

        # Set mask values for multi-class or binary segmentation
        if self.binary_class is not None:
            logging.info(f'Using binary segmentation mode for class: {self.binary_class}')
        else:
            logging.info('Using multi-class segmentation mode')

        if self.binary_class is None:
            # Determine global mask values only for multi-class mode
            all_uniques = np.unique(np.concatenate(valid_unique_values), axis=0)
            if all_uniques.ndim == 1:
                # 1D unique values
                self.mask_values = sorted(all_uniques.tolist())
            else:
                # For RGB or multi-channel masks, convert each row to a tuple for sorting
                all_uniques_as_tuples = [tuple(row) for row in all_uniques]
                self.mask_values = sorted(all_uniques_as_tuples)
        else:
            # For binary mode, only two classes: 0 (background), 1 (target class)
            self.mask_values = [0, 1]


    def __len__(self):
        return len(self.ids)


    @staticmethod
    def preprocess(mask_values, pil_img, target_size=(256,256), is_mask=False, binary_class=None):
        """
        Preprocess an image or mask: resize and convert to the appropriate format.

        Args:
            mask_values (list): List of unique mask values.
            pil_img (PIL.Image): The image or mask to preprocess.
            target_size (tuple): Desired (width, height) for resizing.
            is_mask (bool): Whether the input is a mask.
            binary_class (int): Class label for binary segmentation.

        Returns:
            np.ndarray: Preprocessed image or mask.
        """
        target_width, target_height = target_size
        assert target_width > 0 and target_height > 0, 'Target size must be positive.'

        if pil_img.size == target_size:
            # Skip resizing if the image is already the target size
            img = np.asarray(pil_img)
        else:
            # Resize the input image
            pil_img = pil_img.resize(target_size, resample=Image.NEAREST if is_mask else Image.BICUBIC)
            img = np.asarray(pil_img)

        if is_mask:
            if binary_class is not None:
                # Convert mask to binary: 1 for the specified class, 0 for everything else
                mask = (img == binary_class).astype(np.int64)
                return mask
            else:
                # Map the values in img to indices based on mask_values for multi-class case
                mask = np.zeros((target_height, target_width), dtype=np.int64)
                for i, v in enumerate(mask_values):
                    if img.ndim == 2:
                        mask[img == v] = i
                    else:
                        mask[(img == v).all(axis=-1)] = i
                return mask
        else:
            # Preprocess image: (H, W, C) or (H, W) -> (C, H, W)
            if img.ndim == 2:
                img = img[np.newaxis, ...]  # Add channel dimension if grayscale
            else:
                img = img.transpose((2, 0, 1))
            if (img > 1).any():
                img = img / 255.0
            return img


    def apply_transforms(self, img, mask):
        # Convert NumPy arrays to tensors
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        # Random rotation by 0, 90, 180, or 270 degrees
        k = int(torch.randint(0, 4, (1,)))
        img = torch.rot90(img, k=k, dims=(-2, -1))
        mask = torch.rot90(mask, k=k, dims=(-2, -1))

        # Random transpose (equivalent to vertical flip after rotation)
        if torch.rand(1) > 0.5:
            img = img.transpose(-2, -1)
            mask = mask.transpose(-2, -1)

        # Apply color jitter (brightness, contrast, saturation, hue)
        color_jitter = T.ColorJitter(
            brightness=0.2,  # Randomly change brightness by ±20%
            contrast=0.2,    # Randomly change contrast by ±20%
            saturation=0.2,  # Randomly change saturation by ±20%
            hue=0.1          # Randomly change hue by ±0.1
        )

        # Ensure img has 3 channels (C, H, W) for ColorJitter
        if img.ndim == 3:
            if torch.rand(1) > 0.5:
                img = color_jitter(img)

        return img, mask


    def __getitem__(self, idx):
        """
        Get a single data sample (image and mask) by index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with 'image' and 'mask' tensors.
        """
        name = self.ids[idx]

        # Construct file paths for image and mask
        img_file = list(self.images_dir.glob(f'{name}_p_1.*'))
        mask_file = list(self.mask_dir.glob(f'{name}_p_msk.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        # Load and preprocess image and mask
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        # Convert to tensors
        img = self.preprocess(self.mask_values, img, target_size=(256,256), is_mask=False)
        mask = self.preprocess(self.mask_values, mask, target_size=(256,256), is_mask=True, binary_class=self.binary_class)

        if self.train:
            # Apply augmentations only on the training dataset
            img, mask = self.apply_transforms(img, mask)
        
        # Convert to tensors for both train and validation
        img = torch.from_numpy(np.array(img)).float()
        mask = torch.from_numpy(np.array(mask)).long()
        
        return {
            'image': img.clone().contiguous(),
            'mask': mask.clone().contiguous()
        }


class AntDataset(BasicDataset):
    """
    AntDataset class for loading image and mask datasets, extending BasicDataset.

    Args:
        images_dir (str): Directory containing the image files.
        mask_dir (str): Directory containing the mask files.
        scale (float): Scale factor for resizing images and masks.
        binary_class (int): Optional class label for binary segmentation.
        train (bool): Flag indicating whether the dataset is for training (True) or validation (False).
    """
    def __init__(self, images_dir, mask_dir, scale=1, binary_class=None, train=True):
        super().__init__(images_dir, mask_dir, scale, binary_class, train)