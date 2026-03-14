import torch
import torch.nn as nn

# Handle both relative and absolute imports
try:
    from .backbone import Conv, C2f, CBAM
except ImportError:
    from backbone import Conv, C2f, CBAM


class UpSample(nn.Module):
    """
    Upsample module using nearest neighbor interpolation
    """
    def __init__(self, scale_factor=2):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x):
        return self.upsample(x)


class PANet(nn.Module):
    """
    Path Aggregation Network (PANet) neck for YOLOv8
    Supports automatic channel adaptation for different backbones
    """
    def __init__(self, in_channels, width_multiple=1.0, depth_multiple=1.0, use_cbam=True):
        super().__init__()
        
        # Calculate channels based on width_multiple
        def make_divisible(v, divisor=8):
            return int(v + divisor / 2) // divisor * divisor
        
        def make_ch(v):
            return make_divisible(v * width_multiple)
        
        def make_n(n):
            return max(round(n * depth_multiple), 1) if n > 1 else n
        
        # Target channels (standard YOLOv8 channels)
        target_c3 = make_ch(256)
        target_c4 = make_ch(512)
        target_c5 = make_ch(1024)
        
        # Input channels from backbone: [P3, P4, P5]
        c3, c4, c5 = in_channels
        
        # Channel adaptation layers (1x1 conv to align channels from different backbones)
        self.adapt_c3 = Conv(c3, target_c3, 1, 1) if c3 != target_c3 else nn.Identity()
        self.adapt_c4 = Conv(c4, target_c4, 1, 1) if c4 != target_c4 else nn.Identity()
        self.adapt_c5 = Conv(c5, target_c5, 1, 1) if c5 != target_c5 else nn.Identity()
        
        # Top-down pathway
        # P5 -> P4
        self.up1 = UpSample(scale_factor=2)
        self.h1 = Conv(target_c5 + target_c4, target_c4, 1, 1)
        self.c2f_1 = C2f(target_c4, target_c4, n=make_n(3))
        
        # P4 -> P3
        self.up2 = UpSample(scale_factor=2)
        self.h2 = Conv(target_c4 + target_c3, target_c3, 1, 1)
        self.c2f_2 = C2f(target_c3, target_c3, n=make_n(3))
        
        # Bottom-up pathway
        # P3 -> P4
        self.down1 = nn.Conv2d(target_c3, target_c3, 3, 2, padding=1)
        self.h3 = Conv(target_c3 + target_c4, target_c4, 1, 1)
        self.c2f_3 = C2f(target_c4, target_c4, n=make_n(3))
        
        # P4 -> P5
        self.down2 = nn.Conv2d(target_c4, target_c4, 3, 2, padding=1)
        self.h4 = Conv(target_c4 + target_c5, target_c5, 1, 1)
        self.c2f_4 = C2f(target_c5, target_c5, n=make_n(3))
        
        # CBAM modules for P4 and P5 feature enhancement
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam_p4 = CBAM(target_c4)
            self.cbam_p5 = CBAM(target_c5)
        
        # Output channels
        self.out_channels = [target_c3, target_c4, target_c5]
        self.in_channels = in_channels

    def forward(self, x):
        """
        Args:
            x: List of feature maps from backbone [P3, P4, P5]
        Returns:
            List of output feature maps [P3_out, P4_out, P5_out]
        """
        p3, p4, p5 = x
        
        # Adapt channels from different backbones
        p3 = self.adapt_c3(p3)
        p4 = self.adapt_c4(p4)
        p5 = self.adapt_c5(p5)
        
        # Top-down pathway
        # P5 -> P4
        p5_up = self.up1(p5)
        p5_up_cat = torch.cat([p5_up, p4], dim=1)
        p5_up_cat = self.h1(p5_up_cat)
        p4_td = self.c2f_1(p5_up_cat)
        
        # P4 -> P3
        p4_up = self.up2(p4_td)
        p4_up_cat = torch.cat([p4_up, p3], dim=1)
        p4_up_cat = self.h2(p4_up_cat)
        p3_td = self.c2f_2(p4_up_cat)
        
        # Bottom-up pathway
        # P3 -> P4
        p3_down = self.down1(p3_td)
        p3_down_cat = torch.cat([p3_down, p4_td], dim=1)
        p3_down_cat = self.h3(p3_down_cat)
        p4_bu = self.c2f_3(p3_down_cat)
        
        # P4 -> P5
        p4_down = self.down2(p4_bu)
        p4_down_cat = torch.cat([p4_down, p5], dim=1)
        p4_down_cat = self.h4(p4_down_cat)
        p5_bu = self.c2f_4(p4_down_cat)
        
        # Apply CBAM to enhance P4 and P5 features
        if self.use_cbam:
            p4_bu = self.cbam_p4(p4_bu)
            p5_bu = self.cbam_p5(p5_bu)
        
        return [p3_td, p4_bu, p5_bu]


if __name__ == '__main__':
    # Test neck
    in_channels = [256, 512, 1024]
    model = PANet(in_channels, width_multiple=0.5, depth_multiple=0.67)
    
    x1 = torch.randn(1, 256, 80, 80)    # P3
    x2 = torch.randn(1, 512, 40, 40)    # P4
    x3 = torch.randn(1, 1024, 20, 20)   # P5
    
    y1, y2, y3 = model([x1, x2, x3])
    print(f"P3 input shape: {x1.shape}")
    print(f"P4 input shape: {x2.shape}")
    print(f"P5 input shape: {x3.shape}")
    print(f"P3 output shape: {y1.shape}")
    print(f"P4 output shape: {y2.shape}")
    print(f"P5 output shape: {y3.shape}")
    print(f"Neck parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
