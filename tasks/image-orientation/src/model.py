from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F


class OrientationClassifier(nn.Module):

    def __init__(self) -> None:
        super(OrientationClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)  # Reduces to 256x256
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1)
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)  # Reduces to 128x128
        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)  # Reduces to 64x64
        self.conv4 = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)  # Reduces to 32x32

        # Flattening is implied before the first dense layer
        self.fc1 = nn.Linear(128 * 32 * 32, 256)  # Adjust for flattened size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 4)
        self.fc5 = nn.Linear(4, 2)  # Output 2 values for normal and flipped

        # Dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = self.pool3(F.leaky_relu(self.conv3(x)))
        x = self.pool4(F.leaky_relu(self.conv4(x)))

        x = x.view(-1, 128 * 32 * 32)  # Flatten the output for dense layers

        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
