import os
import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

###############################
# RETRIEVE THE VALID MASKS
# Here we remove the training data, but it can be kept
###############################

image_dir = "./original"
mask_dir = "./inference_ant_masks"
trained_images_dir = "ml-project-2-kalamariunionen/Pytorch-UNet/data/images_folder_5"

# Load trained image IDs to exclude
trained_images = set(
    [f.replace(".png", "").replace("_mask", "") for f in os.listdir(trained_images_dir) if f.endswith(".png")]
)

# Get all masks and exclude trained masks
all_masks = [f for f in os.listdir(mask_dir) if f.endswith("_mask.png")]
original_mask_count = len(all_masks)

# Exclude masks that are part of the training set
all_masks = [f for f in all_masks if f.replace("_mask.png", "") not in trained_images]
excluded_mask_count = original_mask_count - len(all_masks)

print(f"Original number of masks: {original_mask_count}")
print(f"Number of masks excluded (training set): {excluded_mask_count}")
print(f"Number of masks after excluding training set: {len(all_masks)}")

# Function to check mask conditions
def is_valid_mask(mask_path):
    try:
        mask = np.array(Image.open(mask_path))
        unique_values = np.unique(mask)
        
        # Check if there are 5 unique values
        if len(unique_values) == 5:
            return True
        
        # Check if there are exactly 4 unique values and 200 is excluded
        if len(unique_values) == 4:
            return 200 not in unique_values
        
        return False
    except (UnidentifiedImageError, IOError):
        return False

# Filter masks
valid_ids = []
invalid_ids = []

for mask_name in tqdm(all_masks, desc="Filtering masks"):
    mask_path = os.path.join(mask_dir, mask_name)
    image_id = mask_name.replace("_mask.png", "")
    if is_valid_mask(mask_path):
        valid_ids.append(image_id)
    else:
        invalid_ids.append(image_id)

# Display count comparison
print(f"Valid masks (4+ unique values, meeting criteria): {len(valid_ids)}")
print(f"Invalid masks: {len(invalid_ids)}")


###############################
#  MAGENTIFY ANT SECTIONS
###############################

# Create output directories for head, thorax, and abdomen
output_base_dir = "magenta_ants"

output_dirs = {
    "head": os.path.join(output_base_dir, "head"),
    "thorax": os.path.join(output_base_dir, "thorax"),
    "abdomen": os.path.join(output_base_dir, "abdomen")
}

for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Class to value mapping
class_to_value = {
    "head": 0,      # Head (dark gray)
    "thorax": 100,  # Thorax (medium gray)
    "abdomen": 150  # Abdomen (lighter gray)
}

# Magenta background (R=255, G=0, B=255)
magenta_bg = [255, 0, 255]

# Function to process each image and mask
def process_image_and_mask(image_path, mask_path, output_dirs):
    try:
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).resize(image.size, resample=Image.NEAREST)
        
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # For each body part, extract the masked area and place it on a magenta background
        for part, value in class_to_value.items():
            # Create a magenta background
            result = np.full_like(image_np, magenta_bg)
            
            # Create a binary mask for the current part
            part_mask = (mask_np == value)
            
            # Copy the original image where the mask matches the current part
            result[part_mask] = image_np[part_mask]
            
            # Save the result with the appropriate suffix
            image_name = os.path.basename(image_path).replace(".jpg", f"_{part}.png")
            output_path = os.path.join(output_dirs[part], image_name)
            Image.fromarray(result).save(output_path)
    
    except (UnidentifiedImageError, IOError) as e:
        print(f"Error processing {image_path}: {e}")

# Get all valid image-mask pairs
image_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
mask_files = [f.replace(".jpg", "_mask.png") for f in image_files]

# Process each pair with tqdm tracking
for image_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files), desc="Processing images and masks"):
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, mask_file)
    
    if os.path.exists(mask_path):
        process_image_and_mask(image_path, mask_path, output_dirs)
    else:
        print(f"Mask not found for {image_file}")

print("Processing complete. Masked images saved to the 'magenta_ants' directory.")