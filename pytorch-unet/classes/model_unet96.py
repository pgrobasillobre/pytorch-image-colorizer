import torch
import torch.nn as nn
import torch.nn.functional as F

# U-Net for 96x96 input
class UNetColorization96(nn.Module):
    """
    U-Net architecture for image colorization with 96x96 grayscale input.

    Encoder:
        - 4 convolutional layers with increasing channels.
        - Max pooling for downsampling.

    Decoder:
        - 3 transposed convolution layers for upsampling.
        - Skip connections from encoder layers.
        - Final convolution outputs 3-channel RGB image.

    Args:
        None

    Inputs:
        x (torch.Tensor): Grayscale input tensor of shape (N, 1, 96, 96).

    Returns:
        torch.Tensor: Colorized output tensor of shape (N, 3, 96, 96).
    """
    def __init__(self):
        # Call the parent class (nn.Module) initializer.
        # This sets up all the internal PyTorch functionality
    
        super().__init__()
        
        # Encoder
        # padding=1 adds a border of zeros around the input so the output size stays the same after convolution.
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2) # Downsampling layer, which reduces the spatial dimensions by half, according to the maximum value in each 2x2 block.

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # Upsampling layer, which doubles the spatial dimensions of the input feature map.
        self.dec1 = nn.Conv2d(512, 256, kernel_size=3, padding=1) # This convolution layer takes the concatenated output from the upsampled feature map and the corresponding encoder feature map, and reduces it to 256 channels.

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Grayscale input tensor of shape (N, 1, 96, 96).

        Returns:
            torch.Tensor: Colorized output tensor of shape (N, 3, 96, 96).
        """
        # Encoder
        x1 = F.relu(self.enc1(x))           # 96x96
        x2 = F.relu(self.enc2(self.pool(x1))) # 48x48
        x3 = F.relu(self.enc3(self.pool(x2))) # 24x24
        x4 = F.relu(self.enc4(self.pool(x3))) # 12x12

        # Decoder with skip connections
        x = self.up1(x4)                    # 12x12 → 24x24
        x = torch.cat([x, x3], dim=1)
        x = F.relu(self.dec1(x))

        x = self.up2(x)                     # 24x24 → 48x48
        x = torch.cat([x, x2], dim=1)
        x = F.relu(self.dec2(x))

        x = self.up3(x)                     # 48x48 → 96x96
        x = torch.cat([x, x1], dim=1)
        x = F.relu(self.dec3(x))

        x = self.sigmoid(self.final(x))     # Takes the concatenated feature maps and reduces them to 3 channels for RGB output
        return x
