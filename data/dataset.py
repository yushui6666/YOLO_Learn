import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Tuple
from utils.coco_utils import COCODataset, collate_fn
from utils.augmentations import Augmentations


def create_dataloader(
    img_dir: str,
    ann_file: str,
    batch_size: int = 16,
    img_size: int = 640,
    num_workers: int = 4,
    is_training: bool = True,
    augmentation_config: dict = None
) -> DataLoader:
    """
    Create a DataLoader for COCO dataset
    Args:
        img_dir: Directory containing images
        ann_file: Path to COCO annotation file
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of worker processes
        is_training: Whether this is for training
        augmentation_config: Dictionary of augmentation parameters
    Returns:
        DataLoader instance
    """
    # Create augmentation transform if training
    transform = None
    if is_training and augmentation_config:
        transform = Augmentations(
            hsv_h=augmentation_config.get('hsv_h', 0.015),
            hsv_s=augmentation_config.get('hsv_s', 0.7),
            hsv_v=augmentation_config.get('hsv_v', 0.4),
            degrees=augmentation_config.get('degrees', 0.0),
            translate=augmentation_config.get('translate', 0.1),
            scale=augmentation_config.get('scale', 0.5),
            shear=augmentation_config.get('shear', 0.0),
            perspective=augmentation_config.get('perspective', 0.0),
            flipud=augmentation_config.get('flipud', 0.0),
            fliplr=augmentation_config.get('fliplr', 0.5),
            mosaic=augmentation_config.get('mosaic', 1.0),
            mixup=augmentation_config.get('mixup', 0.0)
        )
    
    # Create dataset
    dataset = COCODataset(
        img_dir=img_dir,
        ann_file=ann_file,
        img_size=img_size,
        transform=transform,
        is_training=is_training
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=is_training
    )
    
    return dataloader


if __name__ == '__main__':
    # Test dataloader (requires actual COCO data)
    print("DataLoader Example")
    print("="*50)
    print("\nTo use this dataloader, you need COCO format data.")
    print("\nExample usage:")
    print("""
from data.dataset import create_dataloader

# Create training dataloader
train_loader = create_dataloader(
    img_dir='data/coco/train2017',
    ann_file='data/coco/annotations/instances_train2017.json',
    batch_size=16,
    img_size=640,
    num_workers=4,
    is_training=True,
    augmentation_config={
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'fliplr': 0.5,
        'mosaic': 1.0
    }
)

# Iterate
for images, targets, num_objects in train_loader:
    print(f"Images: {images.shape}, dtype: {images.dtype}")
    print(f"Targets: {targets.shape}, dtype: {targets.dtype}")
    print(f"Num objects: {num_objects}")
    break
    """)
