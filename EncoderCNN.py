import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, start_filters=64, num_classes=35):
        super(UNetEncoder, self).__init__()

        self.enc1 = self.conv_block(in_channels, start_filters)
        self.enc2 = self.conv_block(start_filters, start_filters * 2)
        self.enc3 = self.conv_block(start_filters * 2, start_filters * 4)
        self.enc4 = self.conv_block(start_filters * 4, start_filters * 8)
        self.enc5 = self.conv_block(start_filters * 8, start_filters * 16)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global average pooling and fully connected layer for classification
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(start_filters * 16, num_classes)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc5 = self.enc5(self.pool(enc4))

        # Global average pooling
        x = self.global_avg_pool(enc5)
        x = x.view(x.size(0), -1)  # Flatten

        # Fully connected layer
        x = self.fc(x)

        return x

