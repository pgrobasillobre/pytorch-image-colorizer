import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import STL10
import torchvision.transforms.functional as TF
import sys
import os
import pickle
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'utils')))
from model_unet96 import UNetColorization96

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom dataset: grayscale input, RGB output
class STL10Colorization(Dataset):
    def __init__(self, train=True):
        split = "train" if train else "test"
        self.data = STL10(
            root="./data",
            split=split,
            download=True,
            transform=transforms.ToTensor()
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        color_img, _ = self.data[idx]               # (3, 96, 96)
        gray_img = TF.rgb_to_grayscale(color_img)   # (1, 96, 96)
        return gray_img, color_img


# Load dataset
train_dataset = STL10Colorization(train=True)
test_dataset = STL10Colorization(train=False)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Model, loss, optimizer
model = UNetColorization96().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training U-Net96 colorizer...")
num_epochs = 20
train_losses = []
val_losses = []

patience = 3 # Early stopping patience
best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (gray_imgs, color_imgs) in enumerate(train_loader):
        gray_imgs = gray_imgs.to(device)
        color_imgs = color_imgs.to(device)

        outputs = model(gray_imgs)
        loss = criterion(outputs, color_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * gray_imgs.size(0)

        
        # Print batch progress
        #f (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
        #    print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")


    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
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
        os.makedirs("pytorch/models", exist_ok=True)
        torch.save(model.state_dict(), "pytorch/models/colorizer_model_unet96_best.pth")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break


print("Model saved to pytorch/models/colorizer_model_unet96_best.pth")

# Save training history
history = {
    "train_loss": train_losses,
    "val_loss": val_losses
}

with open("pytorch/models/colorizer_training_history_unet96.pkl", "wb") as f:
    pickle.dump(history, f)

print("Training history saved to pytorch/models/colorizer_training_history_unet96.pkl")
