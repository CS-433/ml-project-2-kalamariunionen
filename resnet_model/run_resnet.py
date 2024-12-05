
import pandas as pd

import torchvision.transforms as transforms

from read_data import *
from data_loading import *
from plot import *
from train_model import *


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Cuda is availible using device '{device}'")

    #path_df = '/Volumes/T7 Shield/AntProject/colour_ants.csv'
    #images_dir = '/Volumes/T7 Shield/AntProject/original'

    path_df = '../colour_ants.csv'
    images_dir = '../../data/AntProject/original'

    #Filter out names which are in both sets 

    mean_value = [0.485, 0.456, 0.406]
    std_value= [0.229, 0.224, 0.225]

    df = pd.read_csv(path_df)

    # Standardize the columns
    df['r_thorax'] = df['r_thorax'] / 255.0
    df['g_thorax'] = df['g_thorax'] / 255.0
    df['b_thorax'] = df['b_thorax'] / 255.0

    specimen_set = set(df['specimen'])  # Convert to a set for faster lookup

    image_file_names = read_image_file_names(images_dir)

    image_file_set = set(image_file_names)

    # Filter the DataFrame by checking if 'original_file' is in image_file_set
    filtered_df = df[df['original_file'].isin(image_file_set)]

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),         
        transforms.Normalize(mean=mean_value, std=std_value)  
    ])

    train_dataset = ImageLabelDataset(images_dir,filtered_df,transform, split='train')
    val_dataset = ImageLabelDataset(images_dir,filtered_df,transform, split='val')

    print(f"Train dataset length {len(train_dataset)}")
    print(f"Validation dataset length {len(val_dataset)}")

    output_colors,target_colors,val_loss = train_resnet18(train_dataset, val_dataset)

    output_colors_np = [tensor.cpu().numpy() for tensor in output_colors]
    target_colors_np = [tensor.cpu().numpy() for tensor in target_colors]

    save_data('output_colors.npy',output_colors)
    save_data('target_colors.npy',target_colors)




    

    

	
