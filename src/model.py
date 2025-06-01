import torch
import torch.nn as nn
from torchvision import models

class CustomCNN(nn.Module):
    """Custom CNN architecture"""
    def __init__(self, num_classes):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 26 * 26, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def create_efficient_net(num_classes):
    """Create EfficientNet-based model"""
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    return model

def create_resnet(num_classes):
    """Create ResNet-based model"""
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def create_model(model_name, num_classes):
    """Factory function to create models"""
    models = {
        'efficientnet': create_efficient_net,
        'resnet': create_resnet,
        'custom_cnn': CustomCNN
    }
    return models[model_name](num_classes)

def load_model(model_path):
    """Load trained PyTorch model"""
    return torch.load(model_path, map_location=torch.device('cpu'))

def save_model(model, path):
    """Save PyTorch model"""
    torch.save(model, path)