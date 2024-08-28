import torch  # Import the main PyTorch package
import torch.nn as nn  # Import the neural network module
import torch.nn.functional as F  # Import functional module for additional operations

# Define a class for the U-Net encoder, which inherits from nn.Module
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=1, start_filters=64, num_classes=35, dropOutProbability=0.5):
        super(UNetEncoder, self).__init__()  # Initialize the base class
        self.dropOutProbability = dropOutProbability  # Set the dropout probability

        # Define the first encoding block with convolution layers
        self.enc1 = self.conv_block(in_channels, start_filters)
        # Define the second encoding block with more filters
        self.enc2 = self.conv_block(start_filters, start_filters * 2)
        # Define the third encoding block with even more filters
        self.enc3 = self.conv_block(start_filters * 2, start_filters * 4)
        # Define the fourth encoding block with even more filters
        self.enc4 = self.conv_block(start_filters * 4, start_filters * 8)
        # Define the fifth encoding block with the most filters
        self.enc5 = self.conv_block(start_filters * 8, start_filters * 16)

        # Define a max-pooling layer to reduce the spatial dimensions by half
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Define a global average pooling layer to reduce each feature map to a single value
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # Define a dropout layer with the given probability
        self.dropout = nn.Dropout(self.dropOutProbability)
        # Define a fully connected layer to output the final number of classes
        self.fc = nn.Linear(start_filters * 16, num_classes)

    # Define a method to create a block of convolutional layers
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Convolutional layer with 3x3 kernel
            nn.BatchNorm2d(out_channels),  # Batch normalization layer to stabilize learning
            nn.ReLU(inplace=True),  # ReLU activation function
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),  # Second convolutional layer
            nn.BatchNorm2d(out_channels),  # Another batch normalization layer
            nn.ReLU(inplace=True),  # Another ReLU activation
            nn.Dropout(self.dropOutProbability)  # Dropout layer for regularization
        )

    # Define the forward pass of the network
    def forward(self, x):
        enc1 = self.enc1(x)  # Pass the input through the first convolutional block
        enc2 = self.enc2(self.pool(enc1))  # Pool and pass through the second block
        enc3 = self.enc3(self.pool(enc2))  # Pool and pass through the third block
        enc4 = self.enc4(self.pool(enc3))  # Pool and pass through the fourth block
        enc5 = self.enc5(self.pool(enc4))  # Pool and pass through the fifth block

        # Apply global average pooling
        x = self.global_avg_pool(enc5)
        x = x.view(x.size(0), -1)  # Flatten the output to fit into the fully connected layer

        # Apply dropout
        x = self.dropout(x)

        # Pass through the fully connected layer to get the final output
        x = self.fc(x)

        return x  # Return the output
