# ML4 Science Ant color detection
This project is in collaboration with the Department of Ecology and Evolution at UNIL. The goal of the project is to use Machine Learning to be able to predict the RGB values of 3 different body parts of the ant; head, thorax and abdomen. Two machine learning models have been developed to be used in a pipeline to predict the color of ants using images. The first model is a segmentation model in order to extract the body parts of the ants. The second model is a resnet used to make the predictions of the RGB values from the images. The ResNet model has been evaluated when just using the original images and when using the images of the segments of the ants. MSE loss was used as the metric to evaluate the preformance and compare models to each other. 

#### Pipeline overview
The pipeline consists of a UNet model trained to predict per pixel classification of bodyparts of ants. The UNet model and further documentation is availible under Pytorch-UNET. Then the next step after predicting the masks is to extract the images for the bodyparts. After the resnet is to be trained to predict the color of the ants. The weights for the additional layers for all of the trained resnet models for each body part can be found under output training. Below is a description on how to run the resnet model.

### Resnet model
#### Features:
**Pretrained Weights:** The pretrained ResNet18 weights are included in the repository under the 'weights' folder and are automatically loaded during training.

**Three Modes of Operation:**
Baseline Model: Run the baseline model using --run_baseline.\
Hyperparameter Tuning: Perform grid search for hyperparameter optimization using --run_hypertuning.\
Custom Model Runs: Train or test the model with custom hyperparameters using --run_model.

**Usage:**
1. Running the Model
The script supports several arguments for customization:
- File Format: Use --file_format to specify the image format. Set True for .jpg files or False for .png files of a specific body part.
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

 2. Open the provided URL (typically http://localhost:6006) in your web browser to view the training metrics and loss curves.

#### Analysis of best preforming models
In the file analysis_resnet.ipynb all models are loaded and the results analysed.
Best preforming model weights for thorax prediciton is under ....


## How to create environment required to run code

```console  
git clone https://github.com/CS-433/ml-project-2-kalamariunionen.git
```

Navigate to the cloned directory

```console
cd ml-project-2-kalamariunionen
```

Create environment

```console
conda create --name <env> --file requirements.txt
```


## Sciting external packages used in code
The project relies on the following key Python packages:
[Torch](https://pytorch.org/)
[Torchvision](https://pytorch.org/vision/stable/index.html)
[TensorBoard](https://www.tensorflow.org/tensorboard/get_started)
[Pillow](https://pillow.readthedocs.io/en/stable/)
[Pandas](https://pandas.pydata.org/)
[Matplotlib](https://matplotlib.org/)
[scikit-image](https://scikit-image.org/)




[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

