import os
import numpy as np
import pandas as pd
from PIL import Image


def read_image_data(image_dir, image_size=(256, 256)):
    # List to hold flattened images
    data = []
    errors = []
    label_file_names = []
    
    for file in os.listdir(image_dir):
        if file.endswith((".jpg")):
            img_path = os.path.join(image_dir, file)
            
            # Check if the file is empty
            if os.path.getsize(img_path) == 0:
                print(f"Skipping empty file: {file}")
                errors.append(file)
                continue
            
            try:
                # Open and resize the image
                img = Image.open(img_path).resize(image_size)
                # Convert image to RGB
                img = img.convert("RGB")
                # Flatten the image
                flattened_image = np.array(img).flatten()
                data.append(flattened_image)
                label_file_names.append(file)

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                errors.append(file)
                continue
    
    return np.array(data),label_file_names


def read_label_data(file_path, label_file_names):
    df = pd.read_csv(file_path)

    labels = df.loc[df["original_file"].isin(label_file_names), ["original_file", "r_thorax", "g_thorax", "b_thorax"]]
    labels["original_file"] = pd.Categorical(labels["original_file"], categories=label_file_names, ordered=True)
    labels = labels.sort_values("original_file").drop(columns="original_file")

    return labels