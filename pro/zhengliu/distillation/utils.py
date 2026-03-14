"""
Utility functions for YOLO Knowledge Distillation
"""
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime


def load_yaml_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file without PyYAML dependency.
    Simple parser for basic YAML structure.
    """
    config = {}
    current_section = None
    
    with open(config_path, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line or line.strip().startswith('#'):
                continue
            
            # Check indentation level
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            
            if ':' in stripped:
                key, _, value = stripped.partition(':')
                key = key.strip()
                value = value.strip()
                
                if indent == 0:
                    # Top-level key
                    if value:
                        config[key] = parse_yaml_value(value)
                    else:
                        config[key] = {}
                        current_section = key
                elif current_section and indent > 0:
                    # Nested key
                    if isinstance(config.get(current_section), dict):
                        config[current_section][key] = parse_yaml_value(value) if value else {}
    
    return config


def parse_yaml_value(value: str):
    """Parse YAML value to appropriate Python type."""
    value = value.strip()
    
    # Remove quotes if present
    if (value.startswith('"') and value.endswith('"')) or \
       (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    
    # Boolean
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    
    # Number
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except ValueError:
        pass
    
    return value


def load_coco_classes(annotations_path: str) -> Tuple[int, List[str]]:
    """
    Load class information from COCO annotation file.
    
    Returns:
        num_classes: Number of classes
        class_names: List of class names
    """
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    categories = data.get('categories', [])
    # COCO categories have 'id' and 'name'
    # Sort by id to ensure consistent ordering
    categories_sorted = sorted(categories, key=lambda x: x['id'])
    
    class_names = [cat['name'] for cat in categories_sorted]
    num_classes = len(class_names)
    
    return num_classes, class_names


def get_device(device_str: str) -> torch.device:
    """
    Get torch device from string specification.
    
    Args:
        device_str: Device string like "0", "cuda", "cpu", or "cuda:0"
    
    Returns:
        torch.device object
    """
    if device_str == "cpu":
        return torch.device("cpu")
    
    if device_str.isdigit():
        device_id = int(device_str)
        if torch.cuda.is_available() and device_id < torch.cuda.device_count():
            return torch.device(f"cuda:{device_id}")
        else:
            if not torch.cuda.is_available():
                print("Warning: CUDA not available, falling back to CPU")
            else:
                print(f"Warning: CUDA device {device_id} not available, falling back to CPU")
            return torch.device("cpu")
    
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_output_directories(output_dir: str) -> Dict[str, str]:
    """
    Create output directory structure.
    
    Returns:
        Dictionary with paths to created directories
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dirs = {
        'checkpoint': os.path.join(output_dir, timestamp, 'checkpoints'),
        'logs': os.path.join(output_dir, timestamp, 'logs'),
        'images': os.path.join(output_dir, timestamp, 'images'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str = ''):
        self.name = name
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.name}: {self.avg:.4f}'


class ProgressMeter:
    """Display training progress."""
    
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.num_batches = num_batches
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        entries = [self.prefix + f'[{batch}/{self.num_batches}]']
        entries += [str(meter) for meter in self.meters]
        print('  '.join(entries))


def save_checkpoint(state: Dict, is_best: bool, save_dir: str, filename: str = 'checkpoint.pth'):
    """Save training checkpoint."""
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(save_dir, 'model_best.pth')
        torch.save(state, best_path)


def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between two sets of boxes.
    
    Args:
        box1: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        box2: Tensor of shape (M, 4) in [x1, y1, x2, y2] format
    
    Returns:
        IoU tensor of shape (N, M)
    """
    # Calculate intersection
    inter_x1 = torch.max(box1[:, None, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[:, 3])
    
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    inter_area = inter_w * inter_h
    
    # Calculate union
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area[:, None] + box2_area - inter_area
    
    # Calculate IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from [x, y, w, h] to [x1, y1, x2, y2] format."""
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1], dim=-1)


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_subheader(text: str):
    """Print formatted subheader."""
    print(f"\n--- {text} ---\n")