"""
COCO Dataset Loader for YOLO Knowledge Distillation
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, List, Tuple, Optional, Callable
import cv2


class COCODataset(Dataset):
    """
    COCO dataset for YOLO training with knowledge distillation.
    
    Supports loading images and annotations in COCO format,
    with data augmentation for training.
    """
    
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        img_size: int = 640,
        augment: bool = False,
        normalize: bool = True
    ):
        """
        Initialize COCO dataset.
        
        Args:
            image_dir: Directory containing image files
            annotation_file: Path to COCO annotation JSON file
            img_size: Target image size
            augment: Whether to apply data augmentation
            normalize: Whether to normalize images to [0, 1]
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.img_size = img_size
        self.augment = augment
        self.normalize = normalize
        
        # Load annotations
        self._load_annotations()
        
        # Define image transformation
        self.transform = self._get_transform()
    
    def _load_annotations(self):
        """Load and parse COCO annotations."""
        with open(self.annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build image to annotations mapping
        self.images = self.coco_data.get('images', [])
        self.annotations = self.coco_data.get('annotations', [])
        self.categories = self.coco_data.get('categories', [])
        
        # Create category id to index mapping
        self.cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(self.categories)}
        self.num_classes = len(self.categories)
        
        # Build image_id to annotations mapping
        self.img_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_anns:
                self.img_id_to_anns[img_id] = []
            self.img_id_to_anns[img_id].append(ann)
        
        # Build image_id to image info mapping
        self.img_id_to_info = {img['id']: img for img in self.images}
        
        print(f"Loaded {len(self.images)} images with {len(self.annotations)} annotations")
        print(f"Number of classes: {self.num_classes}")
    
    def _get_transform(self) -> Callable:
        """Get image transformation function."""
        def transform(img: np.ndarray) -> torch.Tensor:
            # Resize image
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to torch tensor and normalize
            img = torch.from_numpy(img).permute(2, 0, 1).float()
            
            if self.normalize:
                img = img / 255.0
            
            return img
        
        return transform
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image from file."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        return img
    
    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image."""
        if not self.augment:
            return img
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # Random color jittering (brightness, contrast)
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-10, 10)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # Random blur
        if np.random.random() > 0.8:
            ksize = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        
        return img
    
    def _process_annotations(self, img_info: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process annotations for an image.
        
        Returns:
            boxes: Tensor of shape (N, 4) in normalized [x1, y1, x2, y2] format
            labels: Tensor of shape (N,) with class indices
        """
        img_id = img_info['id']
        img_width = img_info['width']
        img_height = img_info['height']
        
        anns = self.img_id_to_anns.get(img_id, [])
        
        if len(anns) == 0:
            # Return empty tensors if no annotations
            return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)
        
        boxes = []
        labels = []
        
        for ann in anns:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Convert to [x1, y1, x2, y2]
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            
            # Normalize to [0, 1]
            x1 /= img_width
            y1 /= img_height
            x2 /= img_width
            y2 /= img_height
            
            # Clip to [0, 1]
            x1 = max(0, min(1, x1))
            y1 = max(0, min(1, y1))
            x2 = max(0, min(1, x2))
            y2 = max(0, min(1, y2))
            
            boxes.append([x1, y1, x2, y2])
            
            # Convert category id to index
            cat_id = ann['category_id']
            if cat_id in self.cat_id_to_idx:
                labels.append(self.cat_id_to_idx[cat_id])
            else:
                labels.append(0)  # Default to background class
        
        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return boxes, labels
    
    def __getitem__(self, index: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Returns:
            dict with keys:
                - image: Tensor of shape (3, H, W)
                - boxes: Tensor of shape (N, 4) normalized boxes
                - labels: Tensor of shape (N,) class indices
                - img_path: Path to the image file
                - img_size: Original image size (width, height)
        """
        img_info = self.images[index]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        # Load image
        img = self._load_image(img_path)
        original_size = (img.shape[1], img.shape[0])  # (width, height)
        
        # Apply augmentation
        img = self._augment_image(img)
        
        # Transform image
        img = self.transform(img)
        
        # Process annotations
        boxes, labels = self._process_annotations(img_info)
        
        # Apply same augmentation to boxes if needed
        if self.augment and np.random.random() > 0.5:
            # Horizontal flip - flip boxes
            boxes[:, [0, 2]] = 1 - boxes[:, [2, 0]]
        
        return {
            'image': img,
            'boxes': boxes,
            'labels': labels,
            'img_path': img_path,
            'img_size': original_size,
            'index': index
        }
    
    def __len__(self) -> int:
        return len(self.images)
    
    def get_class_names(self) -> List[str]:
        """Get list of class names."""
        return [cat['name'] for cat in self.categories]


class CollateFn:
    """
    Custom collate function for batching.
    Pads boxes and labels to handle variable number of objects per image.
    """
    
    def __init__(self, max_objects: int = 100):
        self.max_objects = max_objects
    
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        Collate batch of samples.
        
        Returns:
            dict with batched tensors
        """
        images = torch.stack([item['image'] for item in batch])
        
        # Pad boxes and labels
        batch_size = len(batch)
        boxes_list = []
        labels_list = []
        
        for item in batch:
            boxes = item['boxes'][:self.max_objects]
            labels = item['labels'][:self.max_objects]
            
            # Pad to max_objects
            num_boxes = len(boxes)
            if num_boxes < self.max_objects:
                pad_boxes = torch.zeros((self.max_objects - num_boxes, 4))
                pad_labels = torch.full((self.max_objects - num_boxes,), -1, dtype=torch.long)
                boxes = torch.cat([boxes, pad_boxes], dim=0)
                labels = torch.cat([labels, pad_labels], dim=0)
            
            boxes_list.append(boxes)
            labels_list.append(labels)
        
        return {
            'images': images,
            'boxes': torch.stack(boxes_list),
            'labels': torch.stack(labels_list),
            'img_paths': [item['img_path'] for item in batch],
            'img_sizes': [item['img_size'] for item in batch],
            'indices': [item['index'] for item in batch]
        }


def create_dataloader(
    image_dir: str,
    annotation_file: str,
    batch_size: int = 8,
    img_size: int = 640,
    augment: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create DataLoader for COCO dataset.
    
    Args:
        image_dir: Directory containing image files
        annotation_file: Path to COCO annotation JSON file
        batch_size: Batch size
        img_size: Target image size
        augment: Whether to apply data augmentation
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster loading
    
    Returns:
        DataLoader object
    """
    dataset = COCODataset(
        image_dir=image_dir,
        annotation_file=annotation_file,
        img_size=img_size,
        augment=augment
    )
    
    collate_fn = CollateFn(max_objects=100)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=augment
    )
    
    return dataloader