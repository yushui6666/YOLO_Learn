import os
import json
import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional
from pycocotools.coco import COCO


class COCODataset:
    """
    COCO dataset loader for YOLOv8
    """
    def __init__(self, img_dir: str, ann_file: str, img_size: int = 640, 
                 transform=None, is_training: bool = True):
        """
        Args:
            img_dir: Directory containing images
            ann_file: Path to COCO annotation JSON file
            img_size: Target image size
            transform: Optional data augmentation
            is_training: Whether this is for training
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        self.is_training = is_training
        
        # Load COCO annotations
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # Category mapping
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_class_id = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.class_id_to_cat_id = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)
    
    def __len__(self) -> int:
        return len(self.ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Get a single item from the dataset
        Returns:
            (image, targets) where targets is (N, 6) [class, x, y, w, h, conf]
        """
        # Get image info
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Convert to YOLO format
        targets = []
        for ann in anns:
            if ann['iscrowd']:
                continue
            
            bbox = ann['bbox']  # [x, y, w, h] in pixels
            cat_id = ann['category_id']
            
            # Convert to YOLO format [class, x_center, y_center, width, height]
            x_center = (bbox[0] + bbox[2] / 2) / img_info['width']
            y_center = (bbox[1] + bbox[3] / 2) / img_info['height']
            width = bbox[2] / img_info['width']
            height = bbox[3] / img_info['height']
            
            # Clip to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            width = max(0, min(1, width))
            height = max(0, min(1, height))
            
            # Skip invalid boxes
            if width < 0.01 or height < 0.01:
                continue
            
            # Convert category ID to class ID
            class_id = self.cat_id_to_class_id.get(cat_id, -1)
            if class_id == -1:
                continue
            
            targets.append([class_id, x_center, y_center, width, height, 1.0])
        
        # Convert to numpy array
        if len(targets) > 0:
            targets = np.array(targets, dtype=np.float32)
        else:
            targets = np.zeros((0, 6), dtype=np.float32)
        
        # Apply data augmentation if training
        if self.transform and self.is_training:
            img, targets = self.transform(img, targets)
        
        # Resize image to target size
        img = cv2.resize(img, (self.img_size, self.img_size))
        
        # Convert to float and normalize
        img = img.astype(np.float32) / 255.0
        
        # Transpose to CHW format and convert to torch tensor
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        
        return img, targets
    
    def get_class_names(self) -> List[str]:
        """Get list of class names"""
        cats = self.coco.loadCats(self.cat_ids)
        return [cat['name'] for cat in cats]


def collate_fn(batch: List[Tuple[torch.Tensor, np.ndarray]]) -> Tuple[torch.Tensor, List[np.ndarray], List[str]]:
    """
    Custom collate function for DataLoader
    Args:
        batch: List of (image, targets) tuples
    Returns:
        (images, targets_batch, image_ids)
    """
    images = []
    targets_batch = []
    image_ids = []
    
    for idx, (img, targets) in enumerate(batch):
        images.append(img)
        targets_batch.append(targets)
        image_ids.append(str(idx))
    
    # Stack images into a batch tensor
    images = torch.stack(images, dim=0)
    
    return images, targets_batch, image_ids


def create_coco_dataset(img_dir: str, ann_file: str, img_size: int = 640,
                       transform=None, is_training: bool = True) -> COCODataset:
    """
    Factory function to create COCO dataset
    """
    return COCODataset(
        img_dir=img_dir,
        ann_file=ann_file,
        img_size=img_size,
        transform=transform,
        is_training=is_training
    )


if __name__ == '__main__':
    # Test COCO dataset (requires actual COCO data)
    # This is a demonstration of how to use the dataset
    print("COCO Dataset Example")
    print("="*50)
    print("\nTo use this dataset, you need COCO format data:")
    print("- Image directory: 'data/coco/train2017'")
    print("- Annotation file: 'data/coco/annotations/instances_train2017.json'")
    print("\nExample usage:")
    print("""
from utils.coco_utils import COCODataset, collate_fn
import torch.utils.data as data

# Create dataset
dataset = COCODataset(
    img_dir='data/coco/train2017',
    ann_file='data/coco/annotations/instances_train2017.json',
    img_size=640,
    is_training=True
)

# Create dataloader
dataloader = data.DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn
)

# Iterate
for images, targets, num_objects in dataloader:
    print(f"Images shape: {images.shape}")
    print(f"Targets shape: {targets.shape}")
    break
    """)
