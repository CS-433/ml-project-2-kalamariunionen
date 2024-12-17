
import pandas as pd
import torchvision.transforms as transforms
from itertools import product
from datetime import datetime
import argparse


from utils.helpers_train import *
from utils.data_loading import *
from utils.train_model import *

"""
To run the training for resnet run this file
"""

def main():
    args = parse_arguments()
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Cuda is availible using device '{device}'")

    # Get and standardize the RGB columns for bodypart
    df = pd.read_csv(args.path_df)

    if args.file_format == True:
        df_rgb = get_bodypart_rgb_values(df, args.body_part)
    else:
        df_rgb = get_bodypart_rgb_values(df, args.body_part, 'file_'+ args.body_part)

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor()
    ])

    #Get training and validation data
    train_ratio = 0.8
    val_ratio = 0.1
    
    train_dataset = ImageLabelDataset(args.images_dir,df_rgb,train_ratio,val_ratio,transform, split='train')
    val_dataset = ImageLabelDataset(args.images_dir,df_rgb,train_ratio,val_ratio,transform, split='val')

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

    if args.run_model:
        print("Running model with hyperparameters:")
        print(f"  Body part: {args.body_part}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Nodes: {args.nodes}")

        hparams = {'lr': args.learning_rate, 'batch_size': args.batch_size, 'nodes': args.nodes}
        
        layers_model = nn.Sequential(
                nn.Linear(512, hparams['nodes']),  
                nn.ReLU(),            
                nn.Linear(hparams['nodes'], 3),    
                nn.Sigmoid())  

        output_colors,target_colors = train_resnet18(train_dataset, val_dataset,layers_model,
                                                            hparams,f'output_training/run_{current_time}/models/model_{args.body_part}.pth',num_epochs = 5)

        save_data(f'output_training/run_{current_time}/output_colors/output_colors_{args.body_part}.npy',output_colors)
        save_data(f'output_training/run_{current_time}/target_colors/target_colors_{args.body_part}.npy',target_colors)
        print('Run completed')


    elif args.run_baseline:
        #Baseline model hyperparameters
        hparams_baseline = {'lr': 0.0001, 'batch_size': 512}
        baseline_model = nn.Sequential(
                nn.Linear(512, 3),    
                nn.Sigmoid())
        
        print('Running baseline model')
        output_colors,target_colors = train_resnet18(train_dataset, val_dataset,baseline_model,
                                                            hparams_baseline,f'output_training/run_{current_time}/models/baseline_model_epochs_5.pth',num_epochs = 5)

        save_data(f'output_training/run_{current_time}/output_colors/output_colors_baseline.npy',output_colors)
        save_data(f'output_training/run_{current_time}/target_colors/target_colors_baseline.npy',target_colors)
        print('Run completed')
    
    elif args.run_hypertuning:
        print("Running hyperparameter tuning")
        #Adding layer to model and tuning more complex model
        hparam_space = {
            "lr": [0.001, 0.0001],
            "batch_size": [256,512],
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
            
            output_colors,target_colors = train_resnet18(train_dataset, val_dataset,layers_model,hparams,
                                                                f'output_training/run_{current_time}/models/model{i}_epochs_5.pth', num_epochs = 5)

            save_data(f'output_training/run_{current_time}/output_colors/output_colors_model{i}.npy',output_colors)
            save_data(f'output_training/run_{current_time}/target_colors/target_colors_model{i}.npy',target_colors)

        print('Run completed')

    else:
        print("No operation specified. Use --run_model, --run_baseline, or --run_hypertuning.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Ant Project RGB Prediction")

    #Image directory argument
    parser.add_argument('--path_df', type=str, default='../colour_ants.csv', 
                        help="Path to the CSV file containing RGB values")
    parser.add_argument('--images_dir', type=str, default='../../data/AntProject/original', 
                        help="Path to the directory containing images")
    
    # Body Part argument
    parser.add_argument('--body_part', type=str, default='thorax', 
                        help="Body part to extract RGB values for: 'head', 'thorax', or 'abdomen'")
    
    # Image argument
    parser.add_argument('--file_format', type=str, default=True, 
                        help="To run .jpg format set True, to run .png format for a specific body part set False")

    # Training options
    parser.add_argument('--run_baseline', action='store_true', 
                        help="Flag to run the baseline model")
    parser.add_argument('--run_hypertuning', action='store_true', 
                        help="Flag to run hyperparameter tuning")
    parser.add_argument('--run_model', action='store_true', 
                        help="Flag to run the model")

    # Hyperparameter options
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help="Learning rate for the model (default: 0.001)")
    parser.add_argument('--batch_size', type=int, default=256,
                        help="Batch size for training (default: 256)")
    parser.add_argument('--nodes', type=int, default=256,
                        help="Number of nodes in the hidden layer (default: 256)")

    return parser.parse_args()


if __name__ == "__main__":
    main()

    





    

    

	
