import torch
import torch.nn as nn
from typing import List, Tuple, Optional

# Import torchvision for pretrained models
try:
    import torchvision.models as tv_models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


# ============================================================================
# Common Building Blocks (used by Neck and Head)
# ============================================================================

def autopad(kernel_size: int, padding: int = None) -> int:
    """
    Auto calculate padding to keep output size same as input
    """
    if padding is None:
        padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
    return padding


class Conv(nn.Module):
    """
    Standard convolution with BN and SiLU activation
    Conv2d -> BatchNorm2d -> SiLU
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """
    Standard bottleneck block
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions
    Used in YOLOv8 Neck for feature fusion
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))
    
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Combines Channel and Spatial attention
    """
    def __init__(self, c1, reduction_ratio=16, kernel_size=7):
        super().__init__()
        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction_ratio, c1, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        # Spatial Attention
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    
    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = self.sigmoid(avg_out + max_out)
        x = x * ca
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        x = x * sa
        
        return x


# ============================================================================
# ResNet Backbone
# ============================================================================

class ResNetBackbone(nn.Module):
    """
    ResNet backbone for YOLOv8
    Supports ResNet18, ResNet34, ResNet50 and ResNet101 with pretrained weights
    """
    
    def __init__(self, model_name: str = 'resnet34', in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision is required for ResNet backbone.")
        
        self._name = model_name
        
        if model_name.lower() == 'resnet18':
            if pretrained:
                self.backbone = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = tv_models.resnet18(weights=None)
        elif model_name.lower() == 'resnet34':
            if pretrained:
                self.backbone = tv_models.resnet34(weights=tv_models.ResNet34_Weights.IMAGENET1K_V1)
            else:
                self.backbone = tv_models.resnet34(weights=None)
        elif model_name.lower() == 'resnet50':
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
        
        if in_channels != 3:
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(in_channels, old_conv.out_channels, 
                                            kernel_size=old_conv.kernel_size, 
                                            stride=old_conv.stride, 
                                            padding=old_conv.padding, bias=old_conv.bias)
        
        # Set output channels based on ResNet variant
        if model_name.lower() in ['resnet18', 'resnet34']:
            self._out_channels = [128, 256, 512]
        else:  # resnet50, resnet101
            self._out_channels = [512, 1024, 2048]
        
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
        p3 = self.layer2(x)
        p4 = self.layer3(p3)
        p5 = self.layer4(p4)
        return p3, p4, p5


# ============================================================================
# MobileNetV3 Backbone
# ============================================================================

class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3 backbone for YOLOv8 - Lightweight for mobile deployment
    
    Note: MobileNetV3 has different channel dimensions than ResNet.
    The neck will automatically adapt these channels to the standard YOLOv8 dimensions.
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
        
        if in_channels != 3:
            old_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(in_channels, old_conv.out_channels,
                                                      kernel_size=old_conv.kernel_size,
                                                      stride=old_conv.stride,
                                                      padding=old_conv.padding, bias=False)
        
        # Feature extraction indices for MobileNetV3-Large
        # MobileNetV3-Large architecture:
        # - Layers 0-2: stride 2 (3 -> 16 channels)
        # - Layers 3-5: stride 2 (16 -> 24 channels)
        # - Layers 6-11: stride 2 (24 -> 40 channels) <- P3 (stride 8)
        # - Layers 12-15: stride 2 (40 -> 112 channels) <- P4 (stride 16)
        # - Layers 16-17: stride 2 (112 -> 960 channels) <- P5 (stride 32)
        self.p3_idx = 6   # After block 6, stride=8, 40 channels
        self.p4_idx = 12  # After block 12, stride=16, 112 channels
        self.p5_idx = 16  # Last block, stride=32, 960 channels
        
        # Output channels for each scale
        # These will be adapted by the neck to standard YOLOv8 channels
        self._out_channels = [40, 112, 960]  # Actual output channels
    
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
    """VGG backbone for YOLOv8 - Supports VGG16 and VGG19"""
    
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
        
        if in_channels != 3:
            old_conv = self.features[0]
            self.features[0] = nn.Conv2d(in_channels, old_conv.out_channels,
                                         kernel_size=old_conv.kernel_size,
                                         stride=old_conv.stride,
                                         padding=old_conv.padding, bias=old_conv.bias)
        
        self._out_channels = [256, 512, 512]
        
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


if __name__ == '__main__':
    # Test ResNet34 backbone (standard replacement for CSPDarknet)
    model = ResNetBackbone(model_name='resnet34', pretrained=False)
    x = torch.randn(1, 3, 640, 640)
    y1, y2, y3 = model(x)
    print(f"ResNet34 Test:")
    print(f"Input shape: {x.shape}")
    print(f"P3 output shape: {y1.shape}")
    print(f"P4 output shape: {y2.shape}")
    print(f"P5 output shape: {y3.shape}")
    print(f"Backbone parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
