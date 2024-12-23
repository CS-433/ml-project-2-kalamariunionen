
from torchvision import models
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter

import warnings


def load_model(layers_model,device):
    """
    Function loading pretrained ResNET and adding additional layers
    Nodes not to train are freezed
    """
    model = models.resnet18(pretrained=False)
    #Loading weights from file
    model.load_state_dict(torch.load('weights_resnet/resnet18_weights.pth', map_location=device, weights_only=True))
    model.fc = layers_model
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    return model


def train_resnet18(train_dataset, val_dataset,layers_model,hparams,model_name,num_epochs = 5):
    """
    Function to train the ResNet-18 model.
    
    Parameters:
        train_dataset (DataLoader): Dataset for training model. It should provide batches of images and labels.
        val_dataset (DataLoader): Dataset for validation
        layers_model (torch.nn.Module): Layers to be added to pre-trained resnet
        hparams (dict): A dictionary of hyperparameters learning rate and batch size
        model_name (str): The name used for saving the trained model
        num_epochs (int, default=5): The number of epochs to train model.

    Returns:
        Predicted colors and target colors from model
    """

    # Suppressing warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch.serialization")

    #Setting seed
    torch.manual_seed(3)

    device = ("cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")

    writer = SummaryWriter()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False, num_workers=0)

    model = load_model(layers_model,device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    output_colors = []
    target_colors = []

    for epoch in range(num_epochs):
        # Training Loop
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for X, y in train_loader: 
            X, y = X.to(device), y.to(device) #Add tensors to devide
            optimizer.zero_grad() #Set gradient to zero

            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        
        # Validation Loop
        with torch.no_grad():  # No gradients needed for validation
            model.eval() 
            val_loss = 0.0
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss = criterion(y_pred, y)
                val_loss += loss.item()

                if epoch == (num_epochs-1):
                    output_colors.append(y_pred)
                    target_colors.append(y)

            avg_val_loss = val_loss / len(val_loader)
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)

        writer.flush()
    writer.close()
    torch.save(model.state_dict(), model_name)

    return output_colors,target_colors
