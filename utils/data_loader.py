"""
Data loading utilities for the project.
Handles CIFAR-100, Stanford Dogs, and Flowers-102 datasets.

Week 2 implementation - getting the data pipeline working
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image
import yaml
from scipy.io import loadmat

def get_transforms(config, train=True):
    """
    Returns transforms for training or testing.
    Training has augmentation, testing doesn't.
    """

    if train:
        aug = config['augmentation']['train']
        transform = transforms.Compose([
            transforms.RandomResizedCrop(aug['random_resized_crop']),
            transforms.RandomHorizontalFlip(p=aug['random_horizontal_flip']),
            transforms.ColorJitter(
                brightness=aug['color_jitter']['brightness'],
                contrast=aug['color_jitter']['contrast'],
                saturation=aug['color_jitter']['saturation'],
                hue=aug['color_jitter']['hue']
            ),
            transforms.RandomRotation(aug['random_rotation']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug['normalize']['mean'],
                std=aug['normalize']['std']
            )
        ])
    else:
        aug = config['augmentation']['test']
        transform = transforms.Compose([
            transforms.Resize(aug['resize']),
            transforms.CenterCrop(aug['center_crop']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=aug['normalize']['mean'],
                std=aug['normalize']['std']
            )
        ])

    return transform

def get_cifar100_loaders(config, data_dir='./datasets/cifar100'):
    """
    CIFAR-100 data loaders. This one's easy since torchvision handles everything.

    Returns train_loader, val_loader, test_loader
    """

    print("Loading CIFAR-100 dataset...")

    train_transform = get_transforms(config, train=True)
    test_transform = get_transforms(config, train=False)

    train_dataset = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform
    )

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    generator = torch.Generator().manual_seed(config['seed'])
    train_subset, val_subset = random_split(
        train_dataset,
        [train_size, val_size],
        generator=generator
    )

    batch_size = config['datasets']['cifar100']['batch_size']
    num_workers = config['num_workers']

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['pin_memory']
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['pin_memory']
    )

    print(f"CIFAR-100: {len(train_subset)} train, {len(val_subset)} val, {len(test_dataset)} test")

    return train_loader, val_loader, test_loader

def get_stanford_dogs_loaders(config, data_dir='./datasets/stanford_dogs'):
    """
    Stanford Dogs loader. Had to figure out the weird folder structure here.
    The images are in Images/breed_folder/Images/{breed_name}/*.jpg
    """

    print("Loading Stanford Dogs dataset...")

    train_transform = get_transforms(config, train=True)
    test_transform = get_transforms(config, train=False)

    images_path = os.path.join(data_dir, 'Images', 'breed_folder', 'Images')

    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Can't find Stanford Dogs at {images_path}")

    full_dataset = datasets.ImageFolder(
        root=images_path,
        transform=train_transform
    )

    print(f"Found {len(full_dataset)} images in {len(full_dataset.classes)} breeds")

    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    batch_size = config['datasets']['stanford_dogs']['batch_size']
    num_workers = config['num_workers']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['pin_memory']
    )

    print(f"Stanford Dogs: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    return train_loader, val_loader, test_loader

class Flowers102Dataset(Dataset):
    """
    Custom dataset for Flowers-102 since we need to parse the .mat files manually.
    The dataset has imagelabels.mat and setid.mat that define the splits.
    """

    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(data_dir, 'flowers-102', 'jpg')
        labels_file = os.path.join(data_dir, 'flowers-102', 'imagelabels.mat')
        splits_file = os.path.join(data_dir, 'flowers-102', 'setid.mat')

        labels_data = loadmat(labels_file)
        splits_data = loadmat(splits_file)

        self.all_labels = labels_data['labels'][0] - 1

        if split == 'train':
            self.indices = splits_data['trnid'][0] - 1
        elif split == 'val':
            self.indices = splits_data['valid'][0] - 1
        elif split == 'test':
            self.indices = splits_data['tstid'][0] - 1
        else:
            raise ValueError(f"Unknown split: {split}")

        self.image_files = [f"image_{str(i+1).zfill(5)}.jpg" for i in self.indices]
        self.labels = [self.all_labels[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_flowers102_loaders(config, data_dir='./datasets/flowers102'):
    """
    Flowers-102 loader using custom dataset class to parse .mat files.
    """

    print("Loading Flowers-102 dataset...")

    train_transform = get_transforms(config, train=True)
    test_transform = get_transforms(config, train=False)

    train_dataset = Flowers102Dataset(
        data_dir=data_dir,
        split='train',
        transform=train_transform
    )

    val_dataset = Flowers102Dataset(
        data_dir=data_dir,
        split='val',
        transform=test_transform
    )

    test_dataset = Flowers102Dataset(
        data_dir=data_dir,
        split='test',
        transform=test_transform
    )

    batch_size = config['datasets']['flowers102']['batch_size']
    num_workers = config['num_workers']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config['pin_memory']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['pin_memory']
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config['pin_memory']
    )

    print(f"Flowers-102: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")

    return train_loader, val_loader, test_loader

def get_dataloaders(dataset_name, config):
    """
    Main function to get dataloaders for any dataset.
    Just pass the dataset name and it routes to the right loader.

    dataset_name: 'cifar100', 'stanford_dogs', or 'flowers102'
    """

    if dataset_name == 'cifar100':
        return get_cifar100_loaders(config)
    elif dataset_name == 'stanford_dogs':
        return get_stanford_dogs_loaders(config)
    elif dataset_name == 'flowers102':
        return get_flowers102_loaders(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'cifar100', 'stanford_dogs', or 'flowers102'")

def load_config(config_path='configs/config.yaml'):
    """Load config from yaml file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# testing code
if __name__ == '__main__':
    print("Testing data loaders...\n")

    config = load_config()

    print("=" * 50)
    print("Testing CIFAR-100")
    print("=" * 50)
    try:
        train_loader, val_loader, test_loader = get_cifar100_loaders(config)
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print("CIFAR-100 works!\n")
    except Exception as e:
        print(f"CIFAR-100 failed: {e}\n")

    print("=" * 50)
    print("Testing Stanford Dogs")
    print("=" * 50)
    try:
        train_loader, val_loader, test_loader = get_stanford_dogs_loaders(config)
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print("Stanford Dogs works!\n")
    except Exception as e:
        print(f"Stanford Dogs failed: {e}\n")

    print("=" * 50)
    print("Testing Flowers-102")
    print("=" * 50)
    try:
        train_loader, val_loader, test_loader = get_flowers102_loaders(config)
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print("Flowers-102 works!\n")
    except Exception as e:
        print(f"Flowers-102 failed: {e}\n")

    print("All done!")
