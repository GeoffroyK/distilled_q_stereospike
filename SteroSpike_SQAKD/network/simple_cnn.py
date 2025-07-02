# models/simple_cnn.py
import torch.nn as nn
from .custom_modules import QConv

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)
    
class QuantSimpleCNN(nn.Module):
    def __init__(self, args=None):
        super(QuantSimpleCNN, self).__init__()
        self.model = nn.Sequential(
            QConv(3, 32, 3, padding=1, args=args),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            QConv(32, 64, 3, padding=1, args=args),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.model(x)