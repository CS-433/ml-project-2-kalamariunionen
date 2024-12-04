
import pandas as pd
import numpy as np

import torchvision.transforms as transforms
from torchvision import models
from torch import nn

from read_data import *
from data_loading import *
from plot import *
from train_model import *


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

    #No distinct clusters of colors! 
    #It would be interesting to just sample the background of images as well to see if we could separate 
    #the ant from the background.
    
    #plot_ant_colors(df)

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

    data_results = {
    'output_colors_r': [color[0] for color in output_colors],
    'output_colors_g': [color[1] for color in output_colors],
    'output_colors_b': [color[2] for color in output_colors],
    'target_colors_r': [color[0] for color in target_colors],
    'target_colors_g': [color[1] for color in target_colors],
    'target_colors_b': [color[2] for color in target_colors],
    'val_loss': val_loss
    }

    df_results = pd.DataFrame(data_results)

    # Save to CSV
    df_results.to_csv('output_full_data.csv', index=False)

#Träna 1 epoch i interactive mode och se hur lång tid det tar
#Tensor boardlog?? 
#Spara modellen efter du har tränat, checkpoint directory
#Spara loss, validation loss, mean square error for validation.
#Sedan analysera exempel lokalt, hur ser färgerna ut? 
