# U-Net: Ant Body Part Segmentation with PyTorch

![Sample Input and Output](/Users/yasminekroknes-gomez/ml-project-2-kalamariunionen/casent0172857_p_1_pair.png)

---

## Overview

This project provides a **U-Net model** implemented in **PyTorch** for the **semantic segmentation of ants** into distinct body parts. The segmentation identifies five key categories:

1. **Head** (including the eye)  
2. **Thorax**  
3. **Abdomen**  
4. **Eye** (separately identifiable within the head)  
5. **Background**  

### Purpose

This code is designed to assist with automated ant body part segmentation for biological research, entomology studies, and computer vision applications. The dataset primarily features **side-profile images of ants**, where the typical body part order (from left to right) is:

- **Head** (with eye) → **Thorax** → **Abdomen**

However, there are exceptions where ants might be flipped, and this segmentation model accounts for such variations.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)  
2. [Dataset Structure](#dataset-structure)  
3. [Installation](#installation)  
4. [Training the Model](#training-the-model)  
5. [Making Predictions](#making-predictions)  
6. [Class Enumeration and Color Mapping](#class-enumeration-and-color-mapping)  
7. [Evaluation](#evaluation)  
8. [Visualization](#visualization)  
9. [Weights & Biases Integration](#weights--biases-integration)  
10. [Credits](#credits)  

---

## Pipeline Overview

This project follows a well-defined pipeline for ant segmentation:

1. **Data Preparation**:  
   - Collect side-profile images of ants.
   - Create corresponding masks with labeled body parts.  

2. **Dataset Loading**:  
   - Use the `BasicDataset` or `AntDataset` classes to load images and masks.  
   - Masks must contain exactly **5 unique values** representing the body parts.  

3. **Model Training**:  
   - Train the U-Net model using the provided `train.py` script.  
   - Supports mixed-precision training and gradient checkpointing for efficiency.  

4. **Evaluation**:  
   - Evaluate model performance using the Dice coefficient with `evaluate.py`.  

5. **Prediction**:  
   - Use the trained model to predict masks for new images using `predict.py`.  

6. **Visualization**:  
   - Visualize the input images and predicted masks to validate segmentation results.  

---

## Dataset Structure

### Image and Mask Files

The dataset should be organized into two directories: one for images and one for masks. Each image should have a corresponding mask with the same filename structure.

`data/ ├── imgs/ │ ├── ant_1_p_1.png │ ├── ant_2_p_1.png │ └── ... └── masks/ ├── ant_1_p_msk.png ├── ant_2_p_msk.png └── ...` 

- **Images**: RGB images of ants, typically in side profile.  
- **Masks**: Grayscale masks with pixel values corresponding to the following categories:  
  - **Head**  
  - **Thorax**  
  - **Abdomen**  
  - **Eye**  
  - **Background**  

### Side-Profile Considerations

The dataset primarily contains ants viewed from a **side profile** with a typical left-to-right order of:

- **Head** (with eye) → **Thorax** → **Abdomen**

However, be aware that **some images may deviate from this setup** (e.g., different angles or partial views). The segmentation model is designed to handle these variations.

---

## Installation

1. **Install CUDA** (for GPU acceleration, optional):  
   [Download CUDA](https://developer.nvidia.com/cuda-downloads)

2. **Install PyTorch** (version 1.13 or later):  
   Follow the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt


## Training the Model

Once you have your dataset organized and dependencies installed, you can train the U-Net model using the `train.py` script. The script supports various configurations for training, including options for mixed precision and checkpoint saving.

### Command to Train the Model

Use the following command to start training:

```bash
python train.py --epochs 50 --batch-size 5 --learning-rate 1e-4 --scale 0.5 --amp
```

### Explanation of Command-Line Arguments

- **`--epochs, -e`**: Number of training epochs (default: 50).  
- **`--batch-size, -b`**: Batch size for training (default: 5).  
- **`--learning-rate, -l`**: Learning rate for the optimizer (default: 1e-4).  
- **`--scale, -s`**: Downscaling factor for input images. A lower value reduces memory usage but may impact segmentation quality (default: 0.5).  
- **`--amp`**: Use Automatic Mixed Precision (AMP) for faster and more memory-efficient training on GPUs.  

### Example Training Output
During training, the script will display progress bars, loss values, and validation Dice scores. A sample output might look like this:

```yaml
INFO: Starting training:
        Epochs:          50
        Batch size:      5
        Learning rate:   0.0001
        Training size:   900
        Validation size: 100
        Checkpoints:     True
        Device:          cuda
        Images scaling:  0.5
        Mixed Precision: True

Epoch 1/50: 100%|██████████| 900/900 [00:45<00:00, 19.68 img/s, loss (batch)=0.342]
Validation Dice score: 0.912
```

### Checkpoints
If `--save-checkpoint` is enabled (default behavior), the model's state will be saved at the end of each epoch to the checkpoints directory. The checkpoint file will be named `checkpoint_augmented.pth`.

### Logging with Weights & Biases
Training progress, including loss curves, validation scores, and model parameters, is logged using Weights & Biases (W&B). The script will automatically initialize a W&B run and provide a link to the dashboard.

If you don’t have a W&B account, it will create an anonymous run that is deleted after 7 days.

# Predicting Masks
Once the model is trained, you can use it to predict segmentation masks on new images using the `predict.py` script.

## Command to Predict Masks
```bash
python predict.py --model MODEL.pth --input-dir ./data/imgs --output-dir ./predictions --viz
```

### Explanation of Command-Line Arguments

- **`--model, -m`**: Path to the trained model file (default: `MODEL.pth`).  
- **`--input-dir, -i`**: Directory containing input images.  
- **`--output-dir, -o`**: Directory to save the predicted masks and visualizations.  
- **`--viz, -v`**: Visualize the images and masks during prediction.  
- **`--no-save, -n`**: Do not save the predicted masks.  
- **`--mask-threshold, -t`**: Minimum probability to consider a mask pixel white (default: 0.5).  
- **`--scale, -s`**: Scale factor for input images (default: 0.5).  

### Example output
```yaml
INFO: Loading model MODEL.pth
INFO: Model loaded!
INFO: Found 10 images in ./data/imgs
INFO: Processing ./data/imgs/ant_1_p_1.png
INFO: Saved mask to ./predictions/ant_1_p_1_mask.png
INFO: Visualizing results for image ./data/imgs/ant_1_p_1.png, close the window to continue...
INFO: Processing complete.
```

# Data structure
Ensure your dataset follows this structure:

```yaml
data/
├── imgs/
│   ├── ant_1_p_1.png
│   ├── ant_2_p_1.png
│   └── ...
└── masks/
    ├── ant_1_p_msk.png
    ├── ant_2_p_msk.png
    └── ...
```

## Image
RGB images of ants, typically in side profile.

## Masks
Grayscale masks with pixel values corresponding to the following categories:

### Masks Table

| **Category** | **Original Value** | **Grayscale Value** |
|--------------|--------------------|--------------------|
| **Head**     | 0                  | 0                  |
| **Thorax**   | 3                  | 100                |
| **Abdomen**  | 4                  | 150                |
| **Eye**      | 5                  | 200                |
| **Background** | 6                | 255                |


# Evaluation
Evaluate the model's performance on the validation dataset using the `evaluate.py` script.

```bash
python evaluate.py
```

The script calculates the Dice coefficient for the predicted masks. The background class is excluded from this calculation to focus on the ant's body parts.

The background class is hard-coded as class 4 (mapped from the original value 6). If the dataset changes or the class enumeration changes, update the background_class parameter in the evaluate function accordingly.

# Credits
This implementation is based on the original U-Net model by **[milesial](https://github.com/milesial/Pytorch-UNet)**.

## Original Paper
**U-Net: Convolutional Networks for Biomedical Image Segmentation**
By Olaf Ronneberger, Philipp Fischer, Thomas Brox.