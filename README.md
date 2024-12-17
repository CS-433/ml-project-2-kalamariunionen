# ML4 Science Ant color detection
This project is in collaboration with the Department of Ecology and Evolution at UNIL. The goal of the project is to use Machine Learning to be able to predict the RGB values of 3 different body parts of the ant; head, thorax and abdomen. Two machine learning models have been developed to be used in a pipeline to predict the color of ants using images. The first model is a segmentation model in order to extract the body parts of the ants ...

The second model is a resnet used to make the predictions of the RGB values from the images. This model has been evaluated when just using the images and when using the segments of the ant. Prediction of thorax was also evaluated when cropping the image to only contain the thorax segment. Doing this provided the best preforming estimations. MSE was used as the metric to evaluate the preformance and compare models to each other. 

#### Pipeline overview


#### Best preforming model
For this project several models have been evaluated....
All models and their predicted output stored...
In the file analysis_resnet.ipynb all models are loaded and the results analysed.
Best preforming model weights for thorax prediciton is under ....


#### Resnet model
To predict the color of ants ResNET 18 was used with pre-trained weights. The used weights are availible in the repository under ...

The model can be run in 3 different modes. The Baseline model 
--run_baseline
For hypertuning
--run_hypertuning
Run model
--run_model

you can specify during run model what hyper parameters learning rate, number of nodes in 2nd added hidden layer and batch size. 
You can specify path to image file to be run and information about images. 
You can specify body part to run

The results will be stored in a file timestamp of the run, model weights, predicted colors and target color in .pkl format.
The validation and training loss is stored using tensorboard and to view results using tensorboard...





Remember: We need to cite all external libraries used.


[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UDdkOEMs)

