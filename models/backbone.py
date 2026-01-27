import torch
import torch.nn as nn
import torch.nn.functional as F
class channel_attrntion(nn.Module):
    def __init__(self,in_channels,reduction_ratio=16):
        super(channel_attrntion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#张量层面池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//reduction_ratio,kernel_size=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction_ratio,in_channels,kernel_size=1,bias=False)
        )#mlp降维并先用一个小网络判断哪些通道重要，再把重要的通道放大，不重要的压小
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)
class space_attention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(space_attention, self).__init__()
        padding = kernel_size // 2 
        self.cov = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg_out = torch.mean(x,dim = 1,keepdim=True)#通道层面进行平均和最大池化
        max_out, _ = torch.max(x,dim=1,keepdim=True)
        x_cat = torch.cat([avg_out,max_out],dim=1)#通道拼接
        out = self.cov(x_cat)
        return x * self.sigmoid(out)
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = channel_attrntion(in_channels, reduction_ratio)
        self.spatial_attention = space_attention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

def autopad(k, p=None):
    """
    Auto padding for convolution
    """
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """
    Standard convolution with BN and SiLU activation
    """
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)#groups是分组卷积，作用是减少参数量和计算量
        self.bn = nn.BatchNorm2d(c2)#批归一化作用是加速收敛，防止过拟合
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2f(nn.Module):
    """
    CSP Bottleneck with 2 convolutions
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c_ = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c_, 1, 1)
        self.cv2 = Conv((2 + n) * self.c_, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c_, self.c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c_, self.c_), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """
    Standard bottleneck
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    """
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class CSPDarknet(nn.Module):
    """
    CSPDarknet backbone for YOLOv8
    """
    def __init__(self, in_channels=3, width_multiple=1.0, depth_multiple=1.0):
        super().__init__()
        
        # Calculate channels based on width_multiple
        def make_divisible(v, divisor=8):
            return int(v + divisor / 2) // divisor * divisor
        
        def make_ch(v):
            return make_divisible(v * width_multiple)
        
        def make_n(n):
            return max(round(n * depth_multiple), 1) if n > 1 else n
        
        # Backbone configuration
        self.stem = Conv(in_channels, make_ch(64), 3, 2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            Conv(make_ch(64), make_ch(128), 3, 2),
            C2f(make_ch(128), make_ch(128), n=make_n(3))
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            Conv(make_ch(128), make_ch(256), 3, 2),
            C2f(make_ch(256), make_ch(256), n=make_n(6))
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            Conv(make_ch(256), make_ch(512), 3, 2),
            C2f(make_ch(512), make_ch(512), n=make_n(6))
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            Conv(make_ch(512), make_ch(1024), 3, 2),
            C2f(make_ch(1024), make_ch(1024), n=make_n(3)),
            SPPF(make_ch(1024), make_ch(1024))
        )
        
        # Output channels
        self.out_channels = [
            make_ch(256),   # P3
            make_ch(512),   # P4
            make_ch(1024)   # P5
        ]

    def forward(self, x):
        x = self.stem(x)       # 3x640x640 -> 32x320x320
        x = self.stage1(x)     # 32x320x320 -> 128x160x160 (stride 4)
        p3 = x                 # P3 feature (stride 4) - NOT used, skip this
        
        x = self.stage2(x)     # 128x160x160 -> 256x80x80 (stride 8)
        p3 = x                 # P3 feature (stride 8) - THIS is the real P3
        
        x = self.stage3(x)     # 256x80x80 -> 512x40x40 (stride 16)
        p4 = x                 # P4 feature (stride 16) - THIS is the real P4
        
        x = self.stage4(x)     # 512x40x40 -> 1024x20x20 (stride 32) with SPPF
        p5 = x                 # P5 feature (stride 32) - THIS is the real P5
        
        # Return features for P3, P4, P5 (stride 8, 16, 32)
        return p3, p4, p5


if __name__ == '__main__':
    # Test backbone
    model = CSPDarknet(in_channels=3, width_multiple=0.5, depth_multiple=0.67)
    x = torch.randn(1, 3, 640, 640)
    y1, y2, y3 = model(x)
    print(f"Input shape: {x.shape}")
    print(f"P3 output shape: {y1.shape}")
    print(f"P4 output shape: {y2.shape}")
    print(f"P5 output shape: {y3.shape}")
    print(f"Backbone parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
