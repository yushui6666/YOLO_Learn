"""
Backbone Networks for YOLOv8
All backbones use standard torchvision implementations with pretrained weights support
"""

import torch
import torch.nn as nn
from typing import List, Tuple

# Import torchvision for pretrained models
try:
    import torchvision.models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ============================================================================
# Standard Conv Module (used by neck and head)
# ============================================================================

class Conv(nn.Module):
    """
    Standard convolution with BN and SiLU activation
    This is a common building block, not a custom architecture
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ============================================================================
# ResNet Backbone
# ============================================================================

class ResNetBackbone(nn.Module):
    """
    ResNet backbone for YOLOv8
    Supports ResNet50 and ResNet101 with pretrained weights
    Uses standard torchvision implementation
    """
    
    def __init__(self, model_name: str = 'resnet50', in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for ResNet backbone.")
        
        self._name = model_name
        
        if model_name.lower() == 'resnet50':
            if pretrained:
                self.backbone = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = tv_models.resnet50(weights=None)
        elif model_name.lower() == 'resnet101':
            if pretrained:
                self.backbone = tv_models.resnet101(weights=tv_models.ResNet101_Weights.IMAGENET1K_V1)
            else:
                self.backbone = tv_models.resnet101(weights=None)
        else:
            raise ValueError(f"Unknown ResNet model: {model_name}")
        
        # Adapt input channels if needed
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, 
                                            kernel_size=old_conv.kernel_size, 
                                            stride=old_conv.stride, 
                                            padding=old_conv.padding, bias=old_conv.bias)
        
        # Output channels for P3, P4, P5 (layer2, layer3, layer4 outputs)
        self._out_channels = [512, 1024, 2048]
        
        # Organize layers for YOLO feature extraction
        self.stem = nn.Sequential(self.backbone.conv1, self.backbone.bn1, 
                                   self.backbone.relu, self.backbone.maxpool)
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4
    
    @property
    def out_channels(self) -> List[int]:
        return self._out_channels
    
    @property
    def name(self) -> str:
        return self._name
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        p3 = self.layer2(x)      # stride 8
        p4 = self.layer3(p3)     # stride 16
        p5 = self.layer4(p4)     # stride 32
        return p3, p4, p5


# ============================================================================
# MobileNetV3 Backbone
# ============================================================================

class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3 backbone for YOLOv8
    Lightweight architecture for mobile/edge deployment
    Uses standard torchvision implementation
    """
    
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for MobileNetV3 backbone.")
        
        self._name = 'MobileNetV3'
        
        if pretrained:
            self.backbone = tv_models.mobilenet_v3_large(weights=tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        else:
            self.backbone = tv_models.mobilenet_v3_large(weights=None)
        
        # Adapt input channels if needed
        if in_channels != 3:
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(in_channels, old_conv.out_channels,
                                                      kernel_size=old_conv.kernel_size,
                                                      stride=old_conv.stride,
                                                      padding=old_conv.padding, bias=False)
        
        # Feature indices for P3, P4, P5
        self.p3_idx = 7   # stride 8
        self.p4_idx = 13  # stride 16
        self.p5_idx = 16  # stride 32
        
        self._out_channels = [40, 112, 960]
    
    @property
    def out_channels(self) -> List[int]:
        return self._out_channels
    
    @property
    def name(self) -> str:
        return self._name
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = None, None, None
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i == self.p3_idx:
                p3 = x
            elif i == self.p4_idx:
                p4 = x
            elif i == self.p5_idx:
                p5 = x
                break
        return p3, p4, p5


# ============================================================================
# VGG Backbone
# ============================================================================

class VGGBackbone(nn.Module):
    """
    VGG backbone for YOLOv8
    Supports VGG16 and VGG19 with pretrained weights
    Uses standard torchvision implementation
    """
    
    def __init__(self, model_name: str = 'vgg16', in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for VGG backbone.")
        
        self._name = model_name
        
        if model_name.lower() == 'vgg16':
            if pretrained:
                vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
            else:
                vgg = tv_models.vgg16(weights=None)
        elif model_name.lower() == 'vgg19':
            if pretrained:
                vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
            else:
                vgg = tv_models.vgg19(weights=None)
        else:
            raise ValueError(f"Unknown VGG model: {model_name}")
        
        self.features = vgg.features
        
        # Adapt input channels if needed
        if in_channels != 3:
            old_conv = self.features[0]
            self.features[0] = nn.Conv2d(in_channels, old_conv.out_channels,
                                         kernel_size=old_conv.kernel_size,
                                         stride=old_conv.stride,
                                         padding=old_conv.padding, bias=old_conv.bias)
        
        # Output channels for P3, P4, P5
        self._out_channels = [256, 512, 512]
        
        # Find pool indices for feature extraction
        self.pool_indices = [i for i, layer in enumerate(self.features) if isinstance(layer, nn.MaxPool2d)]
        self.p3_idx = self.pool_indices[2] if len(self.pool_indices) > 2 else 16
        self.p4_idx = self.pool_indices[3] if len(self.pool_indices) > 3 else 23
        self.p5_idx = self.pool_indices[4] if len(self.pool_indices) > 4 else 30
    
    @property
    def out_channels(self) -> List[int]:
        return self._out_channels
    
    @property
    def name(self) -> str:
        return self._name
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = None, None, None
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.p3_idx:
                p3 = x
            elif i == self.p4_idx:
                p4 = x
            elif i == self.p5_idx:
                p5 = x
        return p3, p4, p5


# ============================================================================
# EfficientNet Backbone (Additional Standard Option)
# ============================================================================

class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone for YOLOv8
    Efficient architecture with compound scaling
    Uses standard torchvision implementation
    """
    
    def __init__(self, model_name: str = 'efficientnet_b0', in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for EfficientNet backbone.")
        
        self._name = model_name
        
        # Map model names to torchvision models and their output channels
        self.model_configs = {
            'efficientnet_b0': (tv_models.efficientnet_b0, [40, 112, 1280]),
            'efficientnet_b1': (tv_models.efficientnet_b1, [40, 112, 1280]),
            'efficientnet_b2': (tv_models.efficientnet_b2, [48, 120, 1408]),
            'efficientnet_b3': (tv_models.efficientnet_b3, [48, 136, 1536]),
            'efficientnet_b4': (tv_models.efficientnet_b4, [56, 160, 1792]),
        }
        
        if model_name.lower() not in self.model_configs:
            raise ValueError(f"Unknown EfficientNet model: {model_name}. "
                           f"Supported: {list(self.model_configs.keys())}")
        
        model_fn, out_channels = self.model_configs[model_name.lower()]
        
        if pretrained:
            self.backbone = model_fn(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 
                                     if model_name.lower() == 'efficientnet_b0' else None)
        else:
            self.backbone = model_fn(weights=None)
        
        # Adapt input channels if needed
        if in_channels != 3:
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(in_channels, old_conv.out_channels,
                                                      kernel_size=old_conv.kernel_size,
                                                      stride=old_conv.stride,
                                                      padding=old_conv.padding, bias=False)
        
        self._out_channels = out_channels
        
        # Feature indices for different EfficientNet variants
        self.p3_idx = 4   # stride 8
        self.p4_idx = 7   # stride 16
        self.p5_idx = -1  # last layer, stride 32
    
    @property
    def out_channels(self) -> List[int]:
        return self._out_channels
    
    @property
    def name(self) -> str:
        return self._name
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p3, p4, p5 = None, None, None
        for i, layer in enumerate(self.backbone.features):
            x = layer(x)
            if i == self.p3_idx:
                p3 = x
            elif i == self.p4_idx:
                p4 = x
        
        # P5 is the last feature map
        p5 = self.backbone.avgpool(x)
        p5 = self.backbone.classifier[:1](p5)  # Get features before final classification
        
        return p3, p4, p5


if __name__ == '__main__':
    # Test backbones
    print("Testing standard backbones...")
    
    x = torch.randn(1, 3, 640, 640)
    
    # Test ResNet50
    print("\n--- ResNet50 ---")
    model = ResNetBackbone('resnet50', pretrained=False)
    p3, p4, p5 = model(x)
    print(f"P3: {p3.shape}, P4: {p4.shape}, P5: {p5.shape}")
    print(f"Out channels: {model.out_channels}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test MobileNetV3
    print("\n--- MobileNetV3 ---")
    model = MobileNetV3Backbone(pretrained=False)
    p3, p4, p5 = model(x)
    print(f"P3: {p3.shape}, P4: {p4.shape}, P5: {p5.shape}")
    print(f"Out channels: {model.out_channels}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test VGG16
    print("\n--- VGG16 ---")
    model = VGGBackbone('vgg16', pretrained=False)
    p3, p4, p5 = model(x)
    print(f"P3: {p3.shape}, P4: {p4.shape}, P5: {p5.shape}")
    print(f"Out channels: {model.out_channels}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")