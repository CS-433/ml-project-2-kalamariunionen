import os
from skimage.io import imread


def read_image_file_names(folder_path):
    # List all files in the folder and sort them
    all_files = sorted(os.listdir(folder_path))

    # Filter only the image files (e.g., with extensions .jpg, .png)
    image_file_names = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    return image_file_names


def read_images(file_names, folder_path_img):

    # Reading in the 100 images
    images = []
    for file in file_names:
        image_path = os.path.join(folder_path_img, file)
        
        if file.startswith('._'):
            continue
        try:
            img = imread(image_path)
            images.append(img)
        except Exception as e:
            print(f"Error reading {file}: {e}")

    return images

