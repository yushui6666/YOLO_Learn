import random
import numpy as np
import cv2
import torch
from typing import Tuple, Optional


class Augmentations:
    """
    Data augmentation utilities for YOLOv8 training
    """
    def __init__(self, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, 
                 translate=0.1, scale=0.5, shear=0.0, perspective=0.0,
                 flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0):
        self.hsv_h = hsv_h
        self.hsv_s = hsv_s
        self.hsv_v = hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.flipud = flipud
        self.fliplr = fliplr
        self.mosaic = mosaic
        self.mixup = mixup
    
    def augment_hsv(self, img: np.ndarray) -> np.ndarray:
        """
        Apply HSV color augmentation
        """
        if self.hsv_h == 0 and self.hsv_s == 0 and self.hsv_v == 0:
            return img
        
        # HSV color space augmentation
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Hue
        if self.hsv_h > 0:
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + random.uniform(-self.hsv_h, self.hsv_h) * 180) % 180
        
        # Saturation
        if self.hsv_s > 0:
            img_hsv[:, :, 1] = img_hsv[:, :, 1] * random.uniform(1 - self.hsv_s, 1 + self.hsv_s)
            img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
        
        # Value
        if self.hsv_v > 0:
            img_hsv[:, :, 2] = img_hsv[:, :, 2] * random.uniform(1 - self.hsv_v, 1 + self.hsv_v)
            img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)
        
        return cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    def random_perspective(self, img: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random perspective transform
        """
        if self.degrees == 0 and self.translate == 0 and self.scale == 0 and self.shear == 0 and self.perspective == 0:
            return img, targets
        
        height, width = img.shape[:2]
        
        # Center
        center = np.array([width / 2, height / 2], dtype=np.float32)
        
        # Perspective transform
        if self.perspective > 0:
            # Generate perspective points
            pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            # Add random offset
            pts += (np.random.uniform(-0.5, 0.5, (4, 2)) * np.array([width, height]) * self.perspective)
            
            # Get transform matrix
            M = cv2.getPerspectiveTransform(pts, np.float32([[0, 0], [width, 0], [width, height], [0, height]]))
        else:
            M = None
        
        # Rotation
        if self.degrees > 0:
            angle = random.uniform(-self.degrees, self.degrees)
            a = math.tan(math.radians(angle / 2))
            
            # Rotation matrix
            R = np.array([
                [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
                [math.sin(math.radians(angle)), math.cos(math.radians(angle)), 0],
                [0, 0, 1]
            ])
        else:
            R = np.eye(3)
        
        # Scale
        if self.scale > 0:
            scale = random.uniform(1 - self.scale, 1 + self.scale)
            S = np.array([
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1]
            ])
        else:
            S = np.eye(3)
        
        # Shear
        if self.shear > 0:
            shear_x = random.uniform(-self.shear, self.shear)
            shear_y = random.uniform(-self.shear, self.shear)
            Sh = np.array([
                [1, shear_x, 0],
                [shear_y, 1, 0],
                [0, 0, 1]
            ])
        else:
            Sh = np.eye(3)
        
        # Translation
        if self.translate > 0:
            tx = random.uniform(-0.5, 0.5) * self.translate * width
            ty = random.uniform(-0.5, 0.5) * self.translate * height
            T = np.array([
                [1, 0, tx],
                [0, 1, ty],
                [0, 0, 1]
            ])
        else:
            T = np.eye(3)
        
        # Combine transforms
        M_combined = T @ S @ Sh @ R
        
        # Apply to image
        img = cv2.warpPerspective(img, M_combined, (width, height), borderMode=cv2.BORDER_REPLICATE)
        
        # Apply to targets
        if len(targets) > 0:
            # Convert to homogeneous coordinates
            xy = targets[:, 1:3].copy()
            xy[:, 0] *= width
            xy[:, 1] *= height
            
            xy_homogeneous = np.column_stack([xy, np.ones(len(xy))])
            
            # Apply transform
            xy_transformed = (M_combined @ xy_homogeneous.T).T
            xy_transformed = xy_transformed[:, :2]
            
            # Update targets
            targets[:, 1:3] = xy_transformed[:, :2]
            targets[:, 1] /= width
            targets[:, 2] /= height
        
        return img, targets
    
    def horizontal_flip(self, img: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply horizontal flip
        """
        if self.fliplr > 0 and random.random() < self.fliplr:
            img = cv2.flip(img, 1)
            if len(targets) > 0:
                targets[:, 1] = 1 - targets[:, 1]  # Flip x coordinate
        
        return img, targets
    
    def vertical_flip(self, img: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply vertical flip
        """
        if self.flipud > 0 and random.random() < self.flipud:
            img = cv2.flip(img, 0)
            if len(targets) > 0:
                targets[:, 2] = 1 - targets[:, 2]  # Flip y coordinate
        
        return img, targets
    
    def mosaic_augmentation(self, images: list, targets: list, img_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply mosaic augmentation (4 images)
        """
        if self.mosaic == 0 or random.random() > self.mosaic:
            return images[0], targets[0]
        
        # Get 4 images
        indices = random.choices(range(len(images)), k=4)
        mosaic_images = [images[i] for i in indices]
        mosaic_targets = [targets[i] for i in indices]
        
        # Create mosaic
        yc, xc = img_size, img_size  # Mosaic center
        
        # Calculate positions
        positions = [
            (xc, yc, 0, 0),        # top-left
            (xc, yc, xc, 0),       # top-right
            (xc, yc, 0, yc),       # bottom-left
            (xc, yc, xc, yc)       # bottom-right
        ]
        
        # Initialize mosaic image
        mosaic_img = np.zeros((img_size * 2, img_size * 2, 3), dtype=np.uint8)
        mosaic_target = []
        
        for i, (img, target) in enumerate(zip(mosaic_images, mosaic_targets)):
            h, w = img.shape[:2]
            
            # Scale image to mosaic size
            scale = min((yc * 2) / h, (xc * 2) / w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            h, w = img.shape[:2]
            
            # Calculate position
            xc_pos, yc_pos, x_offset, y_offset = positions[i]
            
            # Adjust offset based on position
            if i == 0:  # top-left
                x_offset = 0
                y_offset = 0
            elif i == 1:  # top-right
                x_offset = xc - w
                y_offset = 0
            elif i == 2:  # bottom-left
                x_offset = 0
                y_offset = yc - h
            else:  # bottom-right
                x_offset = xc - w
                y_offset = yc - h
            
            # Place image in mosaic
            mosaic_img[y_offset:y_offset + h, x_offset:x_offset + w] = img
            
            # Adjust targets
            if len(target) > 0:
                target = target.copy()
                target[:, 1] = (target[:, 1] * w + x_offset) / (img_size * 2)
                target[:, 2] = (target[:, 2] * h + y_offset) / (img_size * 2)
                target[:, 3] = target[:, 3] * w / (img_size * 2)
                target[:, 4] = target[:, 4] * h / (img_size * 2)
                mosaic_target.append(target)
        
        # Crop to img_size
        y1 = random.randint(0, yc)
        y2 = y1 + img_size
        x1 = random.randint(0, xc)
        x2 = x1 + img_size
        
        mosaic_img = mosaic_img[y1:y2, x1:x2]
        
        # Adjust targets after crop
        if len(mosaic_target) > 0:
            mosaic_target = np.vstack(mosaic_target)
            mosaic_target[:, 1] = (mosaic_target[:, 1] * (img_size * 2) - x1) / img_size
            mosaic_target[:, 2] = (mosaic_target[:, 2] * (img_size * 2) - y1) / img_size
            mosaic_target[:, 3] = mosaic_target[:, 3] * (img_size * 2) / img_size
            mosaic_target[:, 4] = mosaic_target[:, 4] * (img_size * 2) / img_size
            
            # Clip to valid range
            mosaic_target[:, 1:5] = np.clip(mosaic_target[:, 1:5], 0, 1)
            
            # Filter out invalid boxes
            valid = (mosaic_target[:, 3] > 0) & (mosaic_target[:, 4] > 0)
            mosaic_target = mosaic_target[valid]
        else:
            mosaic_target = np.zeros((0, 5))
        
        return mosaic_img, mosaic_target
    
    def __call__(self, img: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply all augmentations
        """
        # HSV augmentation
        img = self.augment_hsv(img)
        
        # Perspective transform
        img, targets = self.random_perspective(img, targets)
        
        # Flip
        img, targets = self.horizontal_flip(img, targets)
        img, targets = self.vertical_flip(img, targets)
        
        return img, targets


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), 
              color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Resize image to new shape with padding
    Returns:
        (resized_image, (scale_x, scale_y))
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding
    
    dw /= 2
    dh /= 2
    
    # Resize
    if shape != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Add padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, (r, r)


import math


if __name__ == '__main__':
    # Test augmentations
    aug = Augmentations()
    
    # Create dummy image
    img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    targets = np.array([[0, 0.5, 0.5, 0.2, 0.2]])  # [class, x, y, w, h]
    
    # Apply augmentations
    img_aug, targets_aug = aug(img, targets)
    
    print(f"Original image shape: {img.shape}")
    print(f"Augmented image shape: {img_aug.shape}")
    print(f"Original targets: {targets}")
    print(f"Augmented targets: {targets_aug}")
