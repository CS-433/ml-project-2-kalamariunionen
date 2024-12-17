import json
import os
import pickle
import numpy as np

from torch import nn

def get_loss(sorted_file_paths):
    """
    Get sorted validation loss for runs
    """
    val_loss = []
    for file_path in sorted_file_paths: 
        with open(file_path, "r") as file:
            data_dict = json.load(file)

            validation_loss = [entry[2] for entry in data_dict]

            val_loss.append(validation_loss)
    
    return val_loss

def read_sort_json_file_names(folder_path_run):
    '''
    Read and sort .jsom file names in directory
    '''
    file_paths = []
    for file_name in os.listdir(folder_path_run):
        if file_name.endswith(".json"): 
            file_path = os.path.join(folder_path_run, file_name)
            file_paths.append(str(file_path))

    sorted_file_paths = sorted(file_paths)

    return sorted_file_paths


def open_pickle(file_path):
    '''
    Function to open pickle file
    '''
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


def convert(tensors):
    '''
    Converting tensor to numpy
    '''
    return [t.numpy() for t in tensors] 

def convert_tensor_in_batch_to_numpy(output_tensor):
    '''
    Converting tensor in batch to numpy
    '''

    output_np = []
    for batch in output_tensor:
        for value in batch:
            output_np.append(value.numpy())

    return np.array(output_np)


def calulate_MSE_loss(target_colors,output_colors):
    criterion = nn.MSELoss()

    loss = 0 
    for i in range(0,len(target_colors)):
        loss += criterion(target_colors[i], output_colors[i])

    return loss


