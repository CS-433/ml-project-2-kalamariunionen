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


def save_data(file_name,data):
    """
    Function to save data as .npy or .pkl format
    """
    try:
        data_np = [tensor.cpu().numpy() for tensor in data]  # Convert to numpy
        data_np = np.stack(data_np)  # Stack into a single array
        np.save(file_name, data_np)
    
    except ValueError as e:
        print(f"Error: {e}. Ensure that all arrays have the same shape.")
        pickle_file_name = file_name.replace('.npy', '.pkl')
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(data, f)


def get_bodypart_rgb_values(df, body_part,column_name = 'original_file'):
    """
    Create a copy of the DataFrame and return only the scaled RGB values 
    for the specified body part.

    Args:
        df (pd.DataFrame): Input DataFrame with RGB columns for body parts.
        body_part (str): The body part to extract RGB values for (e.g., 'head', 'thorax', 'abdomen').

    Returns:
        pd.DataFrame: A new DataFrame containing scaled RGB values (0-1) for the specified body part.
    """
    # Validate the body part
    valid_body_parts = ['head', 'thorax', 'abdomen']
    if body_part not in valid_body_parts:
        raise ValueError(f"Invalid body part. Choose from {valid_body_parts}.")
    
    # Column names for bodypart
    r_col = f"r_{body_part}"
    g_col = f"g_{body_part}"
    b_col = f"b_{body_part}"
    
    # Copy of DataFrame with only relevant columns
    df_color = df[[column_name,r_col, g_col, b_col]].copy()
    
    # Scale the RGB values to the range [0, 1]
    df_color[r_col] = df_color[r_col] / 255.0
    df_color[g_col] = df_color[g_col] / 255.0
    df_color[b_col] = df_color[b_col] / 255.0
    
    df_color.columns = ['original_file','r', 'g', 'b']
    
    return df_color

