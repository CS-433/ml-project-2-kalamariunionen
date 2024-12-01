
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

    path_df = '/Volumes/T7 Shield/AntProject/colour_ants.csv'
    images_dir = '/Volumes/T7 Shield/AntProject/original'

    df = pd.read_csv(path_df)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images and masks to a fixed size
        transforms.ToTensor()          # Convert images to tensors
    ])

    train_dataset = ImageLabelDataset(images_dir,df,transform, split='train')
    val_dataset = ImageLabelDataset(images_dir,df,transform, split='val')

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
    df_results.to_csv('output.csv', index=False)
        

#Träna 1 epoch i interactive mode och se hur lång tid det tar
#Tensor boardlog?? 
#Spara modellen efter du har tränat, checkpoint directory
#Spara loss, validation loss, mean square error for validation.
#Sedan analysera exempel lokalt, hur ser färgerna ut? 