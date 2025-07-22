import sys
import os
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'classes')))
from model_unet96 import UNetColorization96
from colorization_dataset import STL10Colorization

# =================== Detect if CUDA is available ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===================================================================

# Load dataset (use train or test predefined split from STL-10 dataset)
# Create instance train and test instances
train_dataset = STL10Colorization(train=True)
test_dataset = STL10Colorization(train=False)

# Wrap train and testsets in DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define the model, loss function, and optimizer
model = UNetColorization96().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Early stopping parameters
patience = 3 # If validation loss does not improve for this many epochs, stop training
best_val_loss = float("inf") # Keep track of the best validation loss
epochs_without_improvement = 0 # Counter for early stopping


# Number of epochs to train, initialize training loss and validation loss lists 
num_epochs = 30
train_losses = []
val_losses = []

print("Training U-Net96 colorizer...")

# Training loop, iterate through whole dataset on each epoch
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_loss = 0.0 
    for batch_idx, (gray_imgs, color_imgs) in enumerate(train_loader): # Iterate through all training batches
        gray_imgs = gray_imgs.to(device) 
        color_imgs = color_imgs.to(device)

        outputs = model(gray_imgs) # Forward pass through the model
        loss = criterion(outputs, color_imgs) # Calculate loss

        optimizer.zero_grad() # Zero the gradients
        loss.backward() # Backward pass to compute gradients
        optimizer.step() # Update model parameters

        running_loss += loss.item() * gray_imgs.size(0) # Add total loss for the batch (average loss × batch size) to epoch loss

    epoch_loss = running_loss / len(train_loader.dataset)  # Calculate epoch loss by averaging total accumulated loss across all training images
    train_losses.append(epoch_loss)

    # Validation
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad(): # Disable gradient calculation for validation
        for gray_imgs, color_imgs in test_loader:
            gray_imgs = gray_imgs.to(device)
            color_imgs = color_imgs.to(device)

            outputs = model(gray_imgs)
            loss = criterion(outputs, color_imgs)
            val_loss += loss.item() * gray_imgs.size(0)

    val_loss /= len(test_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Early stopping to avoid overfitting based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        # Save the best model so far
        os.makedirs("../model", exist_ok=True)
        torch.save(model.state_dict(), "../model/colorizer_model_unet96_best.pth")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break


print("Model saved to ../model/colorizer_model_unet96_best.pth")

# Save training history
history = {
    "train_loss": train_losses,
    "val_loss": val_losses
}

with open("../model/colorizer_training_history_unet96.pkl", "wb") as f:
    pickle.dump(history, f)

print("Training history saved to ../model/colorizer_training_history_unet96.pkl")
