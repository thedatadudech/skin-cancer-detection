"""
Deep Learning Model Architectures for Skin Cancer Detection

This module provides various neural network architectures for classifying
dermoscopic images into different types of skin lesions. The models are
designed for the HAM10000 dataset with 7 lesion classes.

Classes:
    CustomCNN: A baseline 3-layer convolutional neural network
    
Functions:
    create_efficient_net: Creates EfficientNet-B0 based model (recommended)
    create_resnet: Creates ResNet-50 based model  
    create_model: Factory function for model creation
    load_model: Loads trained model from file
    save_model: Saves model to file

Author: Abdullah Isa Markus
Date: 2025
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Union, Dict, Callable
import logging

# Configure logging
logger = logging.getLogger(__name__)


class CustomCNN(nn.Module):
    """
    Custom Convolutional Neural Network for skin lesion classification.
    
    A baseline 3-layer CNN architecture with batch normalization and dropout
    for regularization. Serves as a comparison baseline against pre-trained models.
    
    Architecture:
        - Conv2D(3→32) → BatchNorm → ReLU → MaxPool
        - Conv2D(32→64) → BatchNorm → ReLU → MaxPool  
        - Conv2D(64→128) → BatchNorm → ReLU → MaxPool
        - Flatten → Linear(128*26*26→512) → ReLU → Dropout → Linear(512→num_classes)
    
    Args:
        num_classes (int): Number of output classes (default: 7 for HAM10000)
        
    Input:
        x (torch.Tensor): Batch of images with shape (N, 3, 224, 224)
        
    Output:
        torch.Tensor: Logits with shape (N, num_classes)
        
    Example:
        >>> model = CustomCNN(num_classes=7)
        >>> x = torch.randn(16, 3, 224, 224)
        >>> logits = model(x)
        >>> print(logits.shape)  # torch.Size([16, 7])
    """
    
    def __init__(self, num_classes: int = 7):
        super(CustomCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block  
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),  # Adjusted for padding
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, 3, 224, 224)
            
        Returns:
            torch.Tensor: Output logits of shape (N, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def create_efficient_net(num_classes: int = 7) -> nn.Module:
    """
    Create EfficientNet-B0 based model with transfer learning.
    
    Uses a pre-trained EfficientNet-B0 as backbone and replaces the final
    classification layer for skin lesion classification. This is the
    recommended model architecture due to its efficiency and accuracy.
    
    Args:
        num_classes (int): Number of output classes (default: 7)
        
    Returns:
        nn.Module: EfficientNet model ready for training/inference
        
    Note:
        The model is initialized with ImageNet pre-trained weights.
        Only the final classification layer needs training for fine-tuning.
        
    Example:
        >>> model = create_efficient_net(num_classes=7)
        >>> print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    """
    logger.info("Creating EfficientNet-B0 model with transfer learning")
    
    # Load pre-trained EfficientNet-B0
    model = models.efficientnet_b0(pretrained=True)
    
    # Freeze feature extraction layers (optional - can be unfrozen for fine-tuning)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    
    # Replace classifier for our number of classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    logger.info(f"EfficientNet created with {num_classes} output classes")
    return model


def create_resnet(num_classes: int = 7) -> nn.Module:
    """
    Create ResNet-50 based model with transfer learning.
    
    Uses a pre-trained ResNet-50 as backbone and replaces the final
    fully connected layer for skin lesion classification. Alternative
    to EfficientNet with proven performance on medical imaging tasks.
    
    Args:
        num_classes (int): Number of output classes (default: 7)
        
    Returns:
        nn.Module: ResNet model ready for training/inference
        
    Note:
        The model uses ImageNet pre-trained weights. ResNet-50 has
        50 layers with residual connections for better gradient flow.
        
    Example:
        >>> model = create_resnet(num_classes=7)
        >>> print(f"Model depth: 50 layers")
    """
    logger.info("Creating ResNet-50 model with transfer learning")
    
    # Load pre-trained ResNet-50
    model = models.resnet50(pretrained=True)
    
    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    logger.info(f"ResNet-50 created with {num_classes} output classes")
    return model


def create_model(model_name: str, num_classes: int = 7) -> nn.Module:
    """
    Factory function to create models by name.
    
    Provides a unified interface for creating different model architectures.
    Supports EfficientNet, ResNet, and custom CNN models.
    
    Args:
        model_name (str): Model architecture name. Options:
            - 'efficientnet': EfficientNet-B0 (recommended)
            - 'resnet': ResNet-50
            - 'custom_cnn': Custom baseline CNN
        num_classes (int): Number of output classes (default: 7)
        
    Returns:
        nn.Module: Instantiated model ready for training/inference
        
    Raises:
        KeyError: If model_name is not supported
        
    Example:
        >>> # Create recommended model
        >>> model = create_model('efficientnet', num_classes=7)
        >>> 
        >>> # Create baseline model for comparison
        >>> baseline = create_model('custom_cnn', num_classes=7)
    """
    model_registry: Dict[str, Callable] = {
        'efficientnet': create_efficient_net,
        'resnet': create_resnet,
        'custom_cnn': CustomCNN
    }
    
    if model_name not in model_registry:
        available_models = list(model_registry.keys())
        raise KeyError(f"Model '{model_name}' not supported. Available: {available_models}")
    
    logger.info(f"Creating {model_name} model with {num_classes} classes")
    return model_registry[model_name](num_classes)


def load_model(model_path: str, device: str = 'cpu') -> nn.Module:
    """
    Load a trained PyTorch model from file.
    
    Loads a complete model (architecture + weights) that was saved using
    torch.save(). The model is automatically moved to the specified device.
    
    Args:
        model_path (str): Path to the saved model file (.pth or .pt)
        device (str): Device to load model on ('cpu', 'cuda', etc.)
        
    Returns:
        nn.Module: Loaded model ready for inference
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        RuntimeError: If model loading fails
        
    Note:
        For security, consider using weights_only=True for production:
        torch.load(model_path, map_location=device, weights_only=True)
        
    Example:
        >>> model = load_model('models/best_model.pth', device='cuda')
        >>> model.eval()  # Set to evaluation mode
        >>> # model is ready for inference
    """
    try:
        logger.info(f"Loading model from {model_path}")
        
        # Load model with explicit device mapping
        device_map = torch.device(device)
        model = torch.load(model_path, map_location=device_map)
        
        # Ensure model is on correct device
        model = model.to(device_map)
        
        logger.info(f"Model loaded successfully on {device}")
        return model
        
    except FileNotFoundError:
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")


def save_model(model: nn.Module, path: str, save_weights_only: bool = False) -> None:
    """
    Save a PyTorch model to file.
    
    Saves either the complete model (architecture + weights) or just the
    state dictionary (weights only). Complete model is easier to load but
    larger in size.
    
    Args:
        model (nn.Module): The model to save
        path (str): Output file path (should end in .pth or .pt)
        save_weights_only (bool): If True, save only state_dict; 
                                 if False, save complete model
        
    Raises:
        IOError: If file cannot be written
        
    Example:
        >>> # Save complete model (recommended for deployment)
        >>> save_model(model, 'models/trained_model.pth')
        >>> 
        >>> # Save only weights (smaller file, requires architecture code)
        >>> save_model(model, 'models/weights.pth', save_weights_only=True)
    """
    try:
        logger.info(f"Saving model to {path}")
        
        if save_weights_only:
            # Save only the state dictionary
            torch.save(model.state_dict(), path)
            logger.info("Model state dictionary saved successfully")
        else:
            # Save complete model
            torch.save(model, path)
            logger.info("Complete model saved successfully")
            
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise IOError(f"Failed to save model: {str(e)}")


def get_model_info(model: nn.Module) -> Dict[str, Union[int, str]]:
    """
    Get information about a model's architecture and parameters.
    
    Args:
        model (nn.Module): The model to analyze
        
    Returns:
        Dict containing model information:
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            - model_size_mb: Approximate model size in MB
            - model_class: Model class name
            
    Example:
        >>> model = create_model('efficientnet', 7)
        >>> info = get_model_info(model)
        >>> print(f"Parameters: {info['total_params']:,}")
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate model size (4 bytes per float32 parameter)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': round(model_size_mb, 2),
        'model_class': model.__class__.__name__
    }