import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import DataLoader
from src.model import create_model
import os
from tqdm import tqdm
import numpy as np

# Configuration
BATCH_SIZE = 16
EPOCHS = 1
NUM_CLASSES = 7
LEARNING_RATE = 1e-4


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_loader = DataLoader(data_dir='data/images',
                             metadata_path='data/HAM10000_metadata.csv',
                             batch_size=BATCH_SIZE)
    train_loader, val_loader, test_loader = data_loader.create_data_loaders()

    # Create models
    models = {'efficientnet': create_model('efficientnet', NUM_CLASSES)}

    # Training setup
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0
    best_model = None
    best_model_name = None

    # Train each model
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.2,
                                                         patience=3,
                                                         verbose=True)

        # Training loop
        for epoch in range(EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                                optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

            # Print progress
            print(f'Epoch: {epoch+1}/{EPOCHS}')
            print(
                f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            # Learning rate scheduling
            scheduler.step(val_loss)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model = model
                best_model_name = name

    # Save best model
    os.makedirs('models', exist_ok=True)
    torch.save(best_model, 'models/best_model.pth')
    print(
        f"\nBest model ({best_model_name}) saved with validation accuracy: {best_val_acc:.2f}%"
    )


if __name__ == "__main__":
    train_model()
