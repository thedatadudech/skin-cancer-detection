import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from torchvision import transforms
from sklearn.model_selection import train_test_split

class SkinLesionDataset(Dataset):
    """Dataset class for skin lesion images"""
    def __init__(self, image_ids, labels, data_dir, transform=None):
        self.image_ids = image_ids
        self.labels = labels
        self.data_dir = data_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.data_dir, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx])
        return image, label

class DataLoader:
    def __init__(self, data_dir, metadata_path, batch_size=32):
        self.data_dir = data_dir
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def load_metadata(self):
        """Load and process metadata"""
        df = pd.read_csv(self.metadata_path)
        diagnosis_mapping = {name: idx for idx, name in enumerate(df.dx.unique())}
        df['label'] = df.dx.map(diagnosis_mapping)
        return df

    def create_data_loaders(self):
        """Create train, validation, and test dataloaders"""
        df = self.load_metadata()

        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.label, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.1, stratify=train_df.label, random_state=42)

        # Create datasets
        train_dataset = SkinLesionDataset(
            train_df.image_id.values,
            train_df.label.values,
            self.data_dir,
            self.train_transform
        )

        val_dataset = SkinLesionDataset(
            val_df.image_id.values,
            val_df.label.values,
            self.data_dir,
            self.val_transform
        )

        test_dataset = SkinLesionDataset(
            test_df.image_id.values,
            test_df.label.values,
            self.data_dir,
            self.val_transform
        )

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2,
            persistent_workers=True, pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
            persistent_workers=True, pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
            persistent_workers=True, pin_memory=True
        )

        return train_loader, val_loader, test_loader