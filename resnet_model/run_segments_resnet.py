
import pandas as pd
import torchvision.transforms as transforms
from itertools import product
from datetime import datetime


from helpers_run_resnet import *
from data_loading import *
from plot import *
from train_model import *


if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Cuda is availible using device '{device}'")

path_df = '../colour_ants.csv'
images_dir = '../../data/AntProject/magenta_ants'

body_parts = ['head','thorax','abdomen']

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor()
])

#Get training and validation data
train_ratio = 0.8
val_ratio = 0.1

# Get and standardize the RGB columns for bodypart
df = pd.read_csv(path_df)

for body_part in body_parts:

    file_column = 'file' + '_' + body_part

    images_dir_bodypart = images_dir + '/' + body_part

    df_rgb = get_bodypart_rgb_values(df, body_part,file_column)

    train_dataset = ImageLabelDataset(images_dir_bodypart,df_rgb,train_ratio,val_ratio,transform , split='train')
    val_dataset = ImageLabelDataset(images_dir_bodypart,df_rgb,train_ratio,val_ratio,transform, split='val')

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

    print("Running model with hyperparameters:")
    print(f"  Body part: {body_part}")

    hparams = {'lr': 0.001, 'batch_size': 256, 'nodes': 512}
    
    layers_model = nn.Sequential(
            nn.Linear(512, hparams['nodes']),  
            nn.ReLU(),            
            nn.Linear(hparams['nodes'], 3),    
            nn.Sigmoid())  

    output_colors,target_colors = train_resnet18(train_dataset, val_dataset,layers_model,
                                                        hparams,f'output_training/run_{current_time}/models/baseline_model_epochs_5.pth',num_epochs = 5)

    save_data(f'output_training/run_{current_time}/output_colors/output_colors_baseline.npy',output_colors)
    save_data(f'output_training/run_{current_time}/target_colors/target_colors_baseline.npy',target_colors)
    print('Run completed')
