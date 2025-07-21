import torch
import torch.nn as nn
import torch.nn.functional as F

# Modified U-Net for 96x96 input
class UNetColorization96(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.final = nn.Conv2d(64, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
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

        x = self.sigmoid(self.final(x))     # RGB output
        return x
