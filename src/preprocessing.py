"""
Image Preprocessing Module for Skin Cancer Detection

This module provides image preprocessing utilities for preparing dermoscopic
images for deep learning model inference and training. All preprocessing
follows medical imaging best practices and ImageNet standards.

Functions:
    get_transform: Returns standard inference preprocessing pipeline
    preprocess_image: Preprocesses single image for model prediction
    get_data_transforms: Returns training and validation preprocessing pipelines

Author: Abdullah Isa Markus
Date: 2025
"""

import torch
from torchvision import transforms
from PIL import Image
import io
import logging
from typing import Tuple, Union, Optional

# Configure logging
logger = logging.getLogger(__name__)

# ImageNet normalization constants (used for transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard input size for models
INPUT_SIZE = (224, 224)


def get_transform() -> transforms.Compose:
    """
    Get standard preprocessing transform for model inference.
    
    Creates a preprocessing pipeline that resizes images to 224x224,
    converts to tensor, and normalizes using ImageNet statistics.
    This transform should be used for single image predictions.
    
    Returns:
        transforms.Compose: Preprocessing pipeline for inference
        
    Note:
        Uses ImageNet normalization for compatibility with pre-trained models.
        All input images are resized to 224x224 regardless of aspect ratio.
        
    Example:
        >>> transform = get_transform()
        >>> image = Image.open('lesion.jpg')
        >>> tensor = transform(image)
        >>> print(tensor.shape)  # torch.Size([3, 224, 224])
    """
    logger.debug("Creating inference preprocessing transform")
    
    return transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def preprocess_image(image: Union[str, Image.Image, bytes]) -> torch.Tensor:
    """
    Preprocess a single image for model prediction.
    
    Handles multiple input formats (file path, PIL Image, or bytes) and
    applies the standard preprocessing pipeline. Automatically adds batch
    dimension for model compatibility.
    
    Args:
        image: Input image in one of these formats:
            - str: File path to image
            - PIL.Image: Loaded PIL image
            - bytes: Raw image bytes
            
    Returns:
        torch.Tensor: Preprocessed image tensor with shape (1, 3, 224, 224)
        
    Raises:
        ValueError: If image format is not supported
        IOError: If image file cannot be loaded
        
    Note:
        Output tensor is ready for model inference without additional processing.
        Automatically converts to RGB if image has different color mode.
        
    Example:
        >>> # From file path
        >>> tensor = preprocess_image('path/to/lesion.jpg')
        >>> 
        >>> # From PIL Image
        >>> pil_image = Image.open('lesion.jpg')
        >>> tensor = preprocess_image(pil_image)
        >>> 
        >>> # From uploaded file bytes
        >>> tensor = preprocess_image(uploaded_file.read())
    """
    try:
        # Handle different input formats
        if isinstance(image, str):
            logger.debug(f"Loading image from path: {image}")
            pil_image = Image.open(image)
        elif isinstance(image, bytes):
            logger.debug("Loading image from bytes")
            pil_image = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            logger.debug("Using provided PIL Image")
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Ensure RGB format (medical images might be grayscale or RGBA)
        if pil_image.mode != 'RGB':
            logger.debug(f"Converting image from {pil_image.mode} to RGB")
            pil_image = pil_image.convert('RGB')
        
        # Apply preprocessing transform
        transform = get_transform()
        image_tensor = transform(pil_image)
        
        # Add batch dimension for model compatibility
        batch_tensor = image_tensor.unsqueeze(0)
        
        logger.debug(f"Image preprocessed successfully. Shape: {batch_tensor.shape}")
        return batch_tensor
        
    except Exception as e:
        logger.error(f"Failed to preprocess image: {str(e)}")
        raise IOError(f"Failed to preprocess image: {str(e)}")


def get_data_transforms(
    augment_training: bool = True,
    input_size: Tuple[int, int] = INPUT_SIZE
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Get training and validation preprocessing transforms.
    
    Creates separate preprocessing pipelines for training and validation.
    Training transform includes data augmentation for improved generalization,
    while validation transform only applies basic preprocessing.
    
    Args:
        augment_training (bool): Whether to apply data augmentation to training data
        input_size (Tuple[int, int]): Target image size (height, width)
        
    Returns:
        Tuple[transforms.Compose, transforms.Compose]: 
            (training_transform, validation_transform)
            
    Note:
        Training augmentations include:
        - Random horizontal flips (50% probability)
        - Random rotations (±10 degrees)
        - Random affine transformations (shear, scale)
        - Color jittering (brightness, contrast)
        
    Example:
        >>> train_transform, val_transform = get_data_transforms()
        >>> 
        >>> # Apply to datasets
        >>> train_dataset = SkinLesionDataset(..., transform=train_transform)
        >>> val_dataset = SkinLesionDataset(..., transform=val_transform)
    """
    logger.info(f"Creating data transforms with input size: {input_size}")
    
    # Base transforms for all data
    base_transforms = [
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    
    if augment_training:
        # Training transforms with augmentation
        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            
            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0, 
                shear=10, 
                scale=(0.8, 1.2),
                fill=0  # Black fill for medical images
            ),
            
            # Color augmentations (mild for medical images)
            transforms.ColorJitter(
                brightness=0.2, 
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            
            # Convert to tensor and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        
        logger.info("Training transform created with augmentation")
    else:
        # Training without augmentation (same as validation)
        train_transform = transforms.Compose(base_transforms)
        logger.info("Training transform created without augmentation")
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose(base_transforms)
    logger.info("Validation transform created")
    
    return train_transform, val_transform


def denormalize_tensor(
    tensor: torch.Tensor, 
    mean: list = IMAGENET_MEAN, 
    std: list = IMAGENET_STD
) -> torch.Tensor:
    """
    Denormalize a tensor for visualization purposes.
    
    Reverses the normalization applied during preprocessing to convert
    tensor back to displayable image format (0-1 range).
    
    Args:
        tensor (torch.Tensor): Normalized tensor with shape (C, H, W) or (N, C, H, W)
        mean (list): Mean values used for normalization
        std (list): Standard deviation values used for normalization
        
    Returns:
        torch.Tensor: Denormalized tensor in [0, 1] range
        
    Example:
        >>> # Denormalize for visualization
        >>> normalized_tensor = preprocess_image('image.jpg')[0]  # Remove batch dim
        >>> display_tensor = denormalize_tensor(normalized_tensor)
        >>> 
        >>> # Convert to PIL for display
        >>> pil_image = transforms.ToPILImage()(display_tensor)
    """
    # Handle both single image and batch
    if tensor.dim() == 4:  # Batch dimension present
        batch_size = tensor.size(0)
        denorm_tensor = tensor.clone()
        for i in range(batch_size):
            for c in range(3):
                denorm_tensor[i, c] = denorm_tensor[i, c] * std[c] + mean[c]
    else:  # Single image
        denorm_tensor = tensor.clone()
        for c in range(3):
            denorm_tensor[c] = denorm_tensor[c] * std[c] + mean[c]
    
    # Clamp to [0, 1] range
    return torch.clamp(denorm_tensor, 0, 1)


def validate_image_format(image_path: str) -> bool:
    """
    Validate if image file is in supported format.
    
    Checks if the image file can be loaded and is in a format
    suitable for medical image analysis.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        bool: True if image is valid, False otherwise
        
    Note:
        Supported formats: JPEG, PNG, TIFF, BMP
        Minimum size: 32x32 pixels
        Maximum size: 4096x4096 pixels
    """
    try:
        with Image.open(image_path) as img:
            # Check format
            if img.format not in ['JPEG', 'PNG', 'TIFF', 'BMP']:
                logger.warning(f"Unsupported format: {img.format}")
                return False
            
            # Check size constraints
            width, height = img.size
            if width < 32 or height < 32:
                logger.warning(f"Image too small: {width}x{height}")
                return False
            if width > 4096 or height > 4096:
                logger.warning(f"Image too large: {width}x{height}")
                return False
            
            return True
            
    except Exception as e:
        logger.error(f"Invalid image file: {str(e)}")
        return False