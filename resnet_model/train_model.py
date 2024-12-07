
from torchvision import models
from torch import nn
import torch

def train_resnet18_baseline(train_dataset, val_dataset):

    device = ("cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")

    model = models.resnet18(pretrained=True)

    model.fc = nn.Sequential(nn.Linear(512, 3)
                             ,nn.Sigmoid())
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for X, y in train_loader: 
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # Zero the gradients

            y_pred = model(X)  # Forward pass
            loss = criterion(y_pred, y)  # Calculate loss
            loss.backward()  # Backpropagate

            optimizer.step()  # Update model parameters

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    model.eval()
    val_loss = 0.0

    output_colors = []
    target_colors = []
    with torch.no_grad():  # No gradients needed for evaluation
        for X, y in val_loader:  # Assuming `val_loader` is your validation data loader
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            output_colors.append(y_pred)
            target_colors.append(y)
            loss = criterion(y_pred, y)
            val_loss += loss.item()

    torch.save(model.state_dict(), 'model_epochs_10.pth')
    return output_colors,target_colors,val_loss

def train_resnet18(train_dataset, val_dataset):

    device = ("cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu")

    model = models.resnet18(pretrained=True)  
    
    model.fc = nn.Sequential(
    nn.Linear(512, 256),  
    nn.ReLU(),            
    nn.Linear(256, 3),    
    nn.Sigmoid()          #To ensure outputs are between 0 and 1
)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the final fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        for X, y in train_loader: 
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()  # Zero the gradients

            y_pred = model(X)  # Forward pass
            loss = criterion(y_pred, y)  # Calculate loss
            loss.backward()  # Backpropagate

            optimizer.step()  # Update model parameters

            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

    model.eval()
    val_loss = 0.0

    output_colors = []
    target_colors = []
    with torch.no_grad():  # No gradients needed for evaluation
        for X, y in val_loader:  # Assuming `val_loader` is your validation data loader
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            output_colors.append(y_pred)
            target_colors.append(y)
            loss = criterion(y_pred, y)
            val_loss += loss.item()

    torch.save(model.state_dict(), 'model_epochs_10_layers.pth')
    return output_colors,target_colors,val_loss

