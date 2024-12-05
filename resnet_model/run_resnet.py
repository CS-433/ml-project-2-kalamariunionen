
import pandas as pd
import numpy as np

import torchvision.transforms as transforms
from torchvision import models
from torch import nn

from read_data import *
from data_loading import *
from plot import *
from train_model import *
import pickle


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Cuda is availible using device '{device}'")

    path_df = '../colour_ants.csv'
    images_dir = '../../data/AntProject/original'

    #Filter out names which are in both sets 

    df = pd.read_csv(path_df)

    specimen_set = set(df['specimen'])  # Convert to a set for faster lookup

    image_file_names = read_image_file_names(images_dir)

    image_file_set = set(image_file_names)

    # Filter the DataFrame by checking if 'original_file' is in image_file_set
    filtered_df = df[df['original_file'].isin(image_file_set)]

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images and masks to a fixed size
        transforms.ToTensor()          # Convert images to tensors
    ])

    train_dataset = ImageLabelDataset(images_dir,filtered_df,transform, split='train')
    val_dataset = ImageLabelDataset(images_dir,filtered_df,transform, split='val')

    print(f"Train dataset length {len(train_dataset)}")
    print(f"Validation dataset length {len(val_dataset)}")

    output_colors,target_colors,val_loss = train_resnet18(train_dataset, val_dataset)

 
    with open('output_colors.pkl', 'wb') as f:
        pickle.dump(output_colors, f)
    
    with open('target_colors.pkl', 'wb') as f:
        pickle.dump(target_colors, f)