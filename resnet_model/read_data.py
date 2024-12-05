import os
from skimage.io import imread
import numpy as np
import pickle


def read_image_file_names(folder_path):
    # List all files in the folder and sort them
    all_files = sorted(os.listdir(folder_path))

    # Filter only the image files (e.g., with extensions .jpg, .png)
    image_file_names = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    return image_file_names

def clean_file_name_jpg(file_names):
    """
    Extracting specimen name from jpg files
    """
    cleaned_file_names = [file.replace('_p_1.jpg', '') for file in file_names]

    return cleaned_file_names

def get_matching_specimen(df,file_names):
    """
    This function filters df by specimen so that only data in folder is contained in the rows.
    """
    
    specimen_set = set(df['specimen'])  # Convert to a set for faster lookup

    clean_file_names = clean_file_name_jpg(file_names)

    image_file_specimen = clean_file_name_jpg(clean_file_names)

    #Here we want to filter rows in df and not file names!
    filtered_clean_file_names = [file for file in image_file_specimen if file in specimen_set]


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


def save_data(file_name,data):
    try:
        data_np = [tensor.cpu().numpy() for tensor in data]  # Convert to numpy
        data_np = np.stack(data_np)  # Stack into a single array
        np.save(file_name, data_np)
    
    except ValueError as e:
        print(f"Error: {e}. Ensure that all arrays have the same shape.")
        pickle_file_name = file_name.replace('.npy', '.pkl')
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(data, f)

