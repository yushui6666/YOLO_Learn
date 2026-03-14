import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle both relative and absolute imports
try:
    from .backbone import Conv, autopad
except ImportError:
    from backbone import Conv, autopad


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) integral part.
    将 4*reg_max 的分布 logits 积分成 4 个距离值 (ltrb)。
    """
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1  # reg_max
        # 固定的 [0, 1, ..., reg_max-1] 投影向量，形状 (1,1,reg_max,1,1)
        proj = torch.arange(c1, dtype=torch.float).view(1, 1, c1, 1, 1)
        self.register_buffer('proj', proj)

    def forward(self, x):
        """
        Args:
            x: (B, 4*reg_max, H, W) 的原始回归输出
        Returns:
            (B, 4, H, W) 的距离值 (ltrb)
        """
        b, _, h, w = x.shape
        # 拆成 (B, 4, reg_max, H, W)
        x = x.view(b, 4, self.c1, h, w)
        # 每个边界上的 reg_max 做 softmax 得到分布
        x = x.softmax(2)
        # 计算期望：sum(p * v)，v 为 [0..reg_max-1]
        proj = self.proj.to(dtype=x.dtype, device=x.device)
        x = (x * proj).sum(2)  # (B, 4, H, W)
        return x



class DetectHead(nn.Module):
    """
    Detection Head for YOLOv8
    Decoupled head with separate classification and regression branches
    """
    def __init__(self, num_classes=None, in_channels=[256, 512, 1024], reg_max=16):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_outputs = num_classes + 4 * reg_max
        
        self.reg_max = reg_max
        self.nc = num_classes
        
        # Stems
        self.stems = nn.ModuleList([
            nn.Sequential(
                Conv(x, x, 3, 1),
                Conv(x, x, 3, 1)
            ) for x in in_channels
        ])
        
        # Classification branch
        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                Conv(x, x, 3, 1),
                Conv(x, x, 3, 1)
            ) for x in in_channels
        ])
        
        self.cls_preds = nn.ModuleList([
            nn.Conv2d(x, num_classes, 1) for x in in_channels
        ])
        
        # Regression branch
        self.reg_convs = nn.ModuleList([
            nn.Sequential(
                Conv(x, x, 3, 1),
                Conv(x, x, 3, 1)
            ) for x in in_channels
        ])
        
        self.reg_preds = nn.ModuleList([
            nn.Conv2d(x, 4 * reg_max, 1) for x in in_channels
        ])
        
        self.dfl = DFL(reg_max)
        
        # Anchor points for different scales
        self.strides = [8, 16, 32]
        self.anchors = self._make_anchors(in_channels)

    def _make_anchors(self, in_channels):
        """
        Generate anchor points for each feature map
        """
        anchors = []
        for i in range(len(in_channels)):
            stride = self.strides[i]
            anchors.append(stride)
        return anchors

    def forward(self, x):
        """
        Args:
            x: List of feature maps [P3, P4, P5]
        Returns:
            Tuple of (cls_outputs, box_outputs) where box_outputs is (B, 4*reg_max, N)
            The box outputs are NOT integrated through DFL yet, allowing DFL loss to be computed during training
        """
        outputs = []
        
        for i in range(len(x)):
            x_i = x[i]
            
            # Stem
            x_i = self.stems[i](x_i)
            
            # Classification branch
            cls_feat = self.cls_convs[i](x_i)
            cls_output = self.cls_preds[i](cls_feat)
            
            # Regression branch
            reg_feat = self.reg_convs[i](x_i)
            reg_output = self.reg_preds[i](reg_feat)
            
            # Reshape outputs (DO NOT apply DFL here - it's done in loss for training and in decode for inference)
            b, _, h, w = reg_output.shape
            cls_output = cls_output.reshape(b, self.num_classes, -1)
            reg_output = reg_output.reshape(b, 4 * self.reg_max, -1)
            
            outputs.append((cls_output, reg_output))
        
        # Concatenate all scales
        cls_outputs = torch.cat([o[0] for o in outputs], dim=-1)
        box_outputs = torch.cat([o[1] for o in outputs], dim=-1)  # (B, 4*reg_max, N)
        
        return cls_outputs, box_outputs
    
    def decode_bboxes(self, box_outputs, anchors):
        """
        Decode box outputs to actual coordinates
        Args:
            box_outputs: (B, 4, N)
            anchors: (B, N, 2) anchor points
        Returns:
            (B, N, 4) boxes in format [x1, y1, x2, y2]
        """
        # box_outputs: [dx, dy, dw, dh]
        B, _, N = box_outputs.shape
        
        # Get distance from anchor point to box boundaries
        dist_ltrb = box_outputs.permute(0, 2, 1)  # (B, N, 4)
        
        # Calculate box coordinates
        lt = anchors - dist_ltrb[..., :2]
        rb = anchors + dist_ltrb[..., 2:]
        
        boxes = torch.cat([lt, rb], dim=-1)
        return boxes


if __name__ == '__main__':
    # Test detection head
    num_classes = 80
    in_channels = [256, 512, 1024]
    model = DetectHead(num_classes=num_classes, in_channels=in_channels, reg_max=16)
    
    # Create dummy inputs
    x1 = torch.randn(2, 256, 80, 80)    # P3
    x2 = torch.randn(2, 512, 40, 40)    # P4
    x3 = torch.randn(2, 1024, 20, 20)   # P5
    
    cls_outputs, box_outputs = model([x1, x2, x3])
    print(f"Classification outputs shape: {cls_outputs.shape}")
    print(f"Box outputs shape: {box_outputs.shape}")
    print(f"Head parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
