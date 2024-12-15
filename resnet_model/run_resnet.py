
import pandas as pd

import torchvision.transforms as transforms
from itertools import product
from datetime import datetime


from read_data import *
from data_loading import *
from plot import *
from train_model import *

"""
This file runs and tunes the hyperparameters of the resnet model
"""

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Cuda is availible using device '{device}'")

    path_df = '/Volumes/T7 Shield/AntProject/colour_ants.csv'
    images_dir = '/Volumes/T7 Shield/AntProject/original'

    #path_df = '../colour_ants.csv'
    #images_dir = '../../data/AntProject/original'

    df = pd.read_csv(path_df)

    # Standardize the RGB columns
    df['r_thorax'] = df['r_thorax'] / 255.0
    df['g_thorax'] = df['g_thorax'] / 255.0
    df['b_thorax'] = df['b_thorax'] / 255.0

    #Read image file names
    image_file_names = read_image_file_names(images_dir)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor()
    ])

    #Get training and validation data
    train_ratio = 0.8
    val_ratio = 0.1
    
    train_dataset = ImageLabelDataset(images_dir,df,train_ratio,val_ratio,transform, split='train')
    val_dataset = ImageLabelDataset(images_dir,df,train_ratio,val_ratio,transform, split='val')

    #Baseline model hyperparameters
    hparams_baseline = {'lr': 0.0001, 'batch_size': 512}
    baseline_model = nn.Sequential(
            nn.Linear(512, 3),    
            nn.Sigmoid())
    
    # Set the log directory
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    #Creating directories for output
    output_dirs = [
        f'output_training/run_{current_time}/models',
        f'output_training/run_{current_time}/target_colors',
        f'output_training/run_{current_time}/output_colors'
    ]

    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)
    
    print('Running baseline model')
    output_colors,target_colors,val_loss = train_resnet18(train_dataset, val_dataset,baseline_model,
                                                          hparams_baseline,f'output_training/run_{current_time}/models/baseline_model_epochs_5.pth',num_epochs = 5)

    save_data(f'output_training/run_{current_time}/output_colors/output_colors_baseline.npy',output_colors)
    save_data(f'output_training/run_{current_time}/target_colors/target_colors_baseline.npy',target_colors)

    #Adding layer to model and tuning more complex model
    hparam_space = {
        "lr": [0.001, 0.0001],
        "batch_size": [256, 512],
        "nodes": [128, 256, 512]
    }

    # Generate combinations of hyperparameters
    keys, values = zip(*hparam_space.items())
    hparam_combinations = [dict(zip(keys, v)) for v in product(*values)]
    
    #Tuning hyperparameters
    for i, hparams in enumerate(hparam_combinations):
        print(f"Testing hyperparameters: {hparams}")
        
        layers_model = nn.Sequential(
            nn.Linear(512, hparams['nodes']),  
            nn.ReLU(),            
            nn.Linear(hparams['nodes'], 3),    
            nn.Sigmoid())  
        
        output_colors,target_colors,val_loss = train_resnet18(train_dataset, val_dataset,layers_model,hparams,
                                                              f'output_training/run_{current_time}/models/model{i}_epochs_5.pth', num_epochs = 5)

        output_colors_np = [tensor.cpu().numpy() for tensor in output_colors]
        target_colors_np = [tensor.cpu().numpy() for tensor in target_colors]

        save_data(f'output_training/run_{current_time}/output_colors/output_colors_model{i}.npy',output_colors)
        save_data(f'output_training/run_{current_time}/target_colors/target_colors_model{i}.npy',target_colors)





    

    

	
