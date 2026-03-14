"""
Backbone Utilities for YOLOv8
Provides unified interface and factory functions for different backbone networks
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


class BaseBackbone(nn.Module, ABC):
    """
    Abstract base class for all backbone networks
    All backbones must implement this interface to be compatible with YOLOv8
    """
    
    @property
    @abstractmethod
    def out_channels(self) -> List[int]:
        """
        Return output channels for P3, P4, P5 feature maps
        Returns:
            List[int]: [C3, C4, C5] channel numbers
        """
        pass
    
    @property
    def strides(self) -> List[int]:
        """
        Return strides for P3, P4, P5 feature maps
        Returns:
            List[int]: [8, 16, 32] by default
        """
        return [8, 16, 32]
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return backbone name
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning three scale feature maps
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Tuple of (P3, P4, P5) feature maps
            P3: stride 8, shape (B, C3, H/8, W/8)
            P4: stride 16, shape (B, C4, H/16, W/16)
            P5: stride 32, shape (B, C5, H/32, W/32)
        """
        pass


def build_backbone(
    backbone_name: str,
    in_channels: int = 3,
    width_multiple: float = 1.0,
    depth_multiple: float = 1.0,
    pretrained: bool = True,
    **kwargs
) -> BaseBackbone:
    """
    Factory function to create backbone networks
    
    Args:
        backbone_name: Name of backbone ('CSPDarknet', 'ResNet50', 'ResNet101', 
                                       'MobileNetV3', 'VGG16', 'VGG19')
        in_channels: Number of input channels
        width_multiple: Width scaling factor (for CSPDarknet)
        depth_multiple: Depth scaling factor (for CSPDarknet)
        pretrained: Whether to use ImageNet pretrained weights
        **kwargs: Additional arguments for specific backbones
    
    Returns:
        Backbone network instance
    
    Example:
        >>> backbone = build_backbone('ResNet50', pretrained=True)
        >>> p3, p4, p5 = backbone(x)
    """
    # Import here to avoid circular imports
    try:
        from .backbone import (
            CSPDarknet, ResNetBackbone, MobileNetV3Backbone, VGGBackbone
        )
    except ImportError:
        from backbone import (
            CSPDarknet, ResNetBackbone, MobileNetV3Backbone, VGGBackbone
        )
    
    backbone_name = backbone_name.lower()
    
    if backbone_name in ['cspdarknet', 'yolo', 'yolov8']:
        return CSPDarknet(
            in_channels=in_channels,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple
        )
    
    elif backbone_name in ['resnet50', 'resnet-50', 'resnet_50']:
        return ResNetBackbone(
            model_name='resnet50',
            in_channels=in_channels,
            pretrained=pretrained
        )
    
    elif backbone_name in ['resnet101', 'resnet-101', 'resnet_101']:
        return ResNetBackbone(
            model_name='resnet101',
            in_channels=in_channels,
            pretrained=pretrained
        )
    
    elif backbone_name in ['mobilenetv3', 'mobilenet_v3', 'mobilenet-v3']:
        return MobileNetV3Backbone(
            in_channels=in_channels,
            pretrained=pretrained
        )
    
    elif backbone_name in ['vgg16', 'vgg-16', 'vgg_16']:
        return VGGBackbone(
            model_name='vgg16',
            in_channels=in_channels,
            pretrained=pretrained
        )
    
    elif backbone_name in ['vgg19', 'vgg-19', 'vgg_19']:
        return VGGBackbone(
            model_name='vgg19',
            in_channels=in_channels,
            pretrained=pretrained
        )
    
    else:
        raise ValueError(
            f"Unknown backbone: {backbone_name}. "
            f"Supported: CSPDarknet, ResNet50, ResNet101, MobileNetV3, VGG16, VGG19"
        )


def list_backbones() -> List[str]:
    """
    Return list of supported backbone names
    """
    return [
        'CSPDarknet',
        'ResNet50', 'ResNet101',
        'MobileNetV3',
        'VGG16', 'VGG19'
    ]


def get_backbone_info(backbone_name: str) -> dict:
    """
    Get information about a specific backbone
    
    Args:
        backbone_name: Name of the backbone
    
    Returns:
        Dictionary with backbone information
    """
    info = {
        'cspdarknet': {
            'name': 'CSPDarknet',
            'params': '~11M',
            'out_channels': [256, 512, 1024],
            'description': 'YOLO native backbone, high accuracy'
        },
        'resnet50': {
            'name': 'ResNet50',
            'params': '~25M',
            'out_channels': [512, 1024, 2048],
            'description': 'Classic backbone, rich pretrained weights'
        },
        'resnet101': {
            'name': 'ResNet101',
            'params': '~44M',
            'out_channels': [512, 1024, 2048],
            'description': 'Deeper ResNet, higher accuracy'
        },
        'mobilenetv3': {
            'name': 'MobileNetV3',
            'params': '~5M',
            'out_channels': [40, 112, 960],
            'description': 'Lightweight, best for mobile deployment'
        },
        'vgg16': {
            'name': 'VGG16',
            'params': '~138M',
            'out_channels': [256, 512, 512],
            'description': 'Classic structure, rich features'
        },
        'vgg19': {
            'name': 'VGG19',
            'params': '~144M',
            'out_channels': [256, 512, 512],
            'description': 'Deeper VGG, more features'
        }
    }
    
    return info.get(backbone_name.lower(), {})


if __name__ == '__main__':
    # Test backbone factory
    print("Supported backbones:")
    for name in list_backbones():
        info = get_backbone_info(name)
        print(f"  - {info['name']}: {info['params']} params, "
              f"channels {info['out_channels']}")
    
    # Test creation
    backbone = build_backbone('ResNet50', pretrained=False)
    x = torch.randn(1, 3, 640, 640)
    p3, p4, p5 = backbone(x)
    print(f"\nResNet50 test:")
    print(f"  Input: {x.shape}")
    print(f"  P3: {p3.shape}")
    print(f"  P4: {p4.shape}")
    print(f"  P5: {p5.shape}")
    print(f"  Out channels: {backbone.out_channels}")
