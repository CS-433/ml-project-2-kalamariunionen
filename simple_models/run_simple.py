
import numpy as np
import pickle
from xgboost import XGBRegressor
import xgboost as xgb
from simple_models.load_data import *
import json

from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split


def save_dict(dictionary, dictionary_file_name):
    with open(dictionary_file_name, 'w') as f:
        json.dump(dictionary, f)


if __name__ == '__main__':

    path_df = '/Volumes/T7 Shield/AntProject/colour_ants.csv'
    image_dir = "/Volumes/T7 Shield/AntProject/original"
    data_images, label_file_names = read_image_data(image_dir)
    labels = read_label_data(path_df, label_file_names)

    X_train, X_test, y_train, y_test = train_test_split(data_images, labels, test_size=0.2, random_state=42)

    # Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set hyperparameters
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        eval_metric='rmse',
        tree_method='gpu_hist',        # Use GPU for training (optional)
        predictor='gpu_predictor',     # Use GPU for prediction (optional)
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100
    )

    # Wrap the XGBoost model with MultiOutputRegressor
    xgb_regr = MultiOutputRegressor(xgb_model)

    # Train the model
    xgb_regr.fit(X_train, y_train)

    # Predict on the test set
    y_pred = xgb_regr.predict(X_test)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results = {'model': xgb_regr,
            'mse': rmse,
            'predictions': y_pred,
            'test_labels': y_test}

    save_dict(results, 'xgb_depth6_estimators100.json')

    model_save_path='xgb_model.pkl'
    with open(model_save_path, 'wb') as model_file:
            pickle.dump(xgb_regr, model_file)