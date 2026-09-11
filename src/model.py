import torch
import torch.nn as nn
import torch.nn.functional as F

class WaferCNN(nn.Module):
    """
    Lightweight CNN tailored for 56x56 Wafer Map Defect Pattern Classification.
    Architecture:
      - 3 Convolutional Blocks with BatchNorm, ReLU, and MaxPool
      - Global Average Pooling (GAP) for spatial invariance & parameter efficiency
      - 2-layer MLP Classifier with Dropout
    """
    def __init__(self, num_classes=9):
        super(WaferCNN, self).__init__()
        
        # Block 1: 56x56 -> 28x28
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: 28x28 -> 14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: 14x14 -> 7x7 (Target layer for Grad-CAM)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier Head
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Hooks for Grad-CAM
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Register hook on the last conv feature map for Grad-CAM
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x
        
        x = self.pool(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block (Channel Attention)
    Selectively emphasizes informative defect features and suppresses noise.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.view(b, c, -1).mean(dim=2) # Global Average Pooling per channel
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    """
    Residual Block with Skip Connection and SE Attention:
      x_out = ReLU(x + F(x))
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + residual)
        return out

class WaferResNet(nn.Module):
    """
    Modern Residual Attention CNN for 56x56 Wafer Map Classification.
    Features:
      - 3 Residual Stages with Skip Connections
      - Channel-wise SE-Attention
      - Preserves fine defect details (like Scratches)
    """
    def __init__(self, num_classes=9):
        super(WaferResNet, self).__init__()
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # Stage 1: 56x56 -> 28x28
        self.res1 = ResidualBlock(32, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Stage 2: 28x28 -> 14x14
        self.res2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Stage 3: 14x14 -> 7x7 (Target for Grad-CAM)
        self.res3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        
        # Grad-CAM hooks
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.res3(x)
        
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x
        
        x = self.pool3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out
