import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_ROOT = Path("data/processed")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


def get_datasets():
    train_transform, eval_transform = get_transforms()

    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"
    test_dir = DATA_ROOT / "test"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=eval_transform)

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(batch_size: int = BATCH_SIZE) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    train_dataset, val_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders()
    print("Classes:", class_names)
    batch = next(iter(train_loader))
    images, labels = batch
    print("Batch shape:", images.shape, labels.shape)
