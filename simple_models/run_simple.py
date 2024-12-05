
import numpy as np
import pickle
from xgboost import XGBRegressor
import pandas as pd
from data_loading import *
import torchvision.transforms as transforms
import json

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

def prepare_data(data_set,break_threshold):
    x = []
    y = []

    for i, (images, labels) in enumerate(data_set):
        print(images, labels)
        images_np = images.numpy()
        images_np_flat = images_np.reshape(-1)

        x.append(images_np_flat)
        y.append(labels.numpy())

        if i == break_threshold:
            break

    # Combine batches into single array
    x = np.array(x)
    y = np.array(y)

    return x,y

def save_dict(dictionary, dictionary_file_name):
    with open(dictionary_file_name, 'w') as f:
        json.dump(dictionary, f)


def run_XGBOOST(x_tr, y_tr, x_te, y_te, n_estimators, max_depth,objective='reg:squarederror', model_save_path='xgb_model.pkl'):
    # Initialize model
    xgb_regr = MultiOutputRegressor(XGBRegressor(n_estimators = n_estimators,max_depth = max_depth,objective='reg:squarederror', random_state=1))
    
    # Fit model
    xgb_regr.fit(x_tr, y_tr)
    
    # Save the model
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(xgb_regr, model_file)
    
    # Make predictions
    y_pred = xgb_regr.predict(x_te)
    
    # Evaluate performance
    mse = mean_squared_error(y_te, y_pred, multioutput='raw_values')

    return {
        'model': xgb_regr,
        'mse': mse,
        'predictions': y_pred,
        'test_labels': y_te
    }

def run_RandomForest(x_tr, y_tr, x_te, y_te, max_depth, n_estimators,model_save_path='rf_model.pkl'):
    # Initialize model
    regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=1)
    
    # Fit model
    regr.fit(x_tr, y_tr)
    
    # Save model
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(regr, model_file)
    
    # Make predictions
    y_pred = regr.predict(x_te)
    
    # Evaluate performance
    mse = mean_squared_error(y_te, y_pred, multioutput='raw_values')

    # Get feature importance
    feature_importance = regr.feature_importances_
    
    # Return results as a dictionary
    return {
        'model': regr,
        'mse': mse,
        'predictions': y_pred,
        'test_labels': y_te,
        'feature_importance': feature_importance,
        'feature_importance_save_path': 'feature_importance_rf.csv'
    }

path_df = '/Volumes/T7 Shield/AntProject/colour_ants.csv'
images_dir = '/Volumes/T7 Shield/AntProject/original'

df = pd.read_csv(path_df)

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images and masks to a fixed size
    transforms.ToTensor()          # Convert images to tensors
])

train_dataset = ImageLabelDataset(images_dir,df,transform, split='train')
val_dataset = ImageLabelDataset(images_dir,df,transform, split='val')

for i, (images, labels) in enumerate(val_dataset):
    print(images, labels)
    break
"""
x_tr,y_tr = prepare_data(train_dataset, 100)

x_te,y_te = prepare_data(val_dataset, 10)

max_depth = 2
n_estimators = 200

xgboost_result = run_XGBOOST(x_tr, y_tr, x_te, y_te, n_estimators, max_depth,
                             objective='reg:squarederror', model_save_path='xgb_model.pkl')

save_dict(xgboost_result, 'xgboost_result.json')

random_forest_result = run_RandomForest(x_tr, y_tr, x_te, y_te, max_depth, 
                                        n_estimators,model_save_path='rf_model.pkl')


save_dict(random_forest_result, 'random_forest_result.json')

"""