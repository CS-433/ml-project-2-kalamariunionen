# ML4 Science Ant color detection
This project is in collaboration with the Department of Ecology and Evolution at UNIL. The goal of the project is to use Machine Learning to be able to predict the RGB values of 3 different body parts of the ant; head, thorax and abdomen. Two machine learning models have been developed to be used in a pipeline to predict the color of ants using images. The first model is a segmentation model in order to extract the body parts of the ants ...

The second model is a resnet used to make the predictions of the RGB values from the images. This model has been evaluated when just using the images and when using the segments of the ant. Prediction of thorax was also evaluated when cropping the image to only contain the thorax segment. Doing this provided the best preforming estimations. MSE was used as the metric to evaluate the preformance and compare models to each other. 

#### Pipeline overview

#### Best preforming model
For this project several models have been evaluated....
All models and their predicted output stored...
In the file analysis_resnet.ipynb all models are loaded and the results analysed.
Best preforming model weights for thorax prediciton is under ....


### Resnet model
#### Features:
**Pretrained Weights:** The pretrained ResNet18 weights are included in the repository under the 'weights' folder and are automatically loaded during training.

**Three Modes of Operation:**
Baseline Model: Run the baseline model using --run_baseline.
Hyperparameter Tuning: Perform grid search for hyperparameter optimization using --run_hypertuning.
Custom Model Runs: Train or test the model with custom hyperparameters using --run_model.

**Usage:**
1. Running the Model
The script supports several arguments for customization:
- File Format: Use --file_format to specify the image format. Set True for .jpg files or False for .png files.
- Hyperparameters: When using --run_model, you can specify hyperparameters such as:
--learning_rate: Set the learning rate for training.
--hidden_nodes: Define the number of nodes in the second added fully connected layer.
--batch_size: Set the batch size for training.
- Image Path: Provide the path to the image file and any additional image-related information using relevant arguments.
- Body Part: Specify which ant body part (head, thorax, or abdomen) to use as target variable.

2. Results
The run model is saved in a folder with the timestap for the run
Output Files:
The output and the target colors are saved in .pkl
Model weights.

Training and validation losses are logged using TensorBoard.
1. Run the following command to launch TensorBoard
Viewing Results with TensorBoard:

```console  
tensorboard --logdir=<path_to_tensorboard_logs>
```

 3. Open the provided URL (typically http://localhost:6006) in your web browser to view the training metrics and loss curves.


Remember: We need to cite all external libraries used.


[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

