import torch
import torch.nn as nn
from torchvision.ops import nms

# Handle both relative and absolute imports
try:
    from .backbone import CSPDarknet
    from .backbone_utils import build_backbone, list_backbones, get_backbone_info
    from .neck import PANet
    from .head import DetectHead
except ImportError:
    from backbone import CSPDarknet
    from backbone_utils import build_backbone, list_backbones, get_backbone_info
    from neck import PANet
    from head import DetectHead


class YOLOv8(nn.Module):
    """
    Complete YOLOv8 model with backbone, neck, and head
    Supports multiple backbone networks with seamless switching
    """
    def __init__(self, num_classes=80, width_multiple=1.0, depth_multiple=1.0, 
                 backbone_name='CSPDarknet', backbone_pretrained=False):
        super().__init__()
        
        self.num_classes = num_classes
        self.width_multiple = width_multiple
        self.depth_multiple = depth_multiple
        self.backbone_name = backbone_name
        
        # Backbone - support multiple backbone types
        if backbone_name.lower() in ['cspdarknet', 'yolo', 'yolov8']:
            self.backbone = CSPDarknet(
                in_channels=3,
                width_multiple=width_multiple,
                depth_multiple=depth_multiple
            )
        else:
            # Use factory function for other backbones
            self.backbone = build_backbone(
                backbone_name=backbone_name,
                in_channels=3,
                width_multiple=width_multiple,
                depth_multiple=depth_multiple,
                pretrained=backbone_pretrained
            )
        
        # Neck - automatically adapts to backbone output channels
        self.neck = PANet(
            in_channels=self.backbone.out_channels,
            width_multiple=width_multiple,
            depth_multiple=depth_multiple
        )
        
        # Head
        self.head = DetectHead(
            num_classes=num_classes,
            in_channels=self.neck.out_channels,
            reg_max=16
        )
        
        # Store strides for anchor point generation
        self.strides = [8, 16, 32]
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        Args:
            x: Input tensor (B, 3, H, W)
            return_features: If True, also return neck features for loss computation
        Returns:
            If return_features=True: Tuple of (cls_outputs, box_outputs, neck_features)
            If return_features=False: Tuple of (cls_outputs, box_outputs)
        """
        # Backbone
        backbone_features = self.backbone(x)
        
        # Neck
        neck_features = self.neck(backbone_features)
        
        # Head
        cls_outputs, box_outputs = self.head(neck_features)
        
        if return_features:
            return cls_outputs, box_outputs, neck_features
        return cls_outputs, box_outputs
    
    def predict(self, x):
        """
        Prediction with raw outputs and anchors
        Args:
            x: Input tensor (B, 3, H, W)
        Returns:
            Tuple of (cls_outputs, box_outputs, anchor_points, anchor_strides)
        """
        # Get model outputs
        cls_outputs, box_outputs = self.forward(x)
        
        # Generate anchor points and strides
        B, _, H, W = x.shape
        anchor_points, anchor_strides = self._make_anchors(B, H, W, device=cls_outputs.device)
        
        return cls_outputs, box_outputs, anchor_points, anchor_strides


    
    def decode_predictions(self, outputs, img_h=640, img_w=640, conf_thres=0.001):
        """
        Decode model outputs to predictions
        Args:
            outputs: Tuple of (cls_outputs, box_outputs) where box_outputs is (B, 4*reg_max, N)
            img_h: Image height (for anchor generation)
            img_w: Image width (for anchor generation)
            conf_thres: Confidence threshold
        Returns:
            List of predictions for each image in batch. Each element is (K, 6):
            [x1, y1, x2, y2, score, class]
        """
        cls_outputs, box_outputs = outputs   # cls: (B, C, N), box: (B, 4*R, N)
        batch_size = cls_outputs.shape[0]

        # 生成 anchor 坐标和对应 stride（与 loss.py 保持一致）
        anchor_points, anchor_strides = self._make_anchors(
            batch_size, img_h, img_w, device=cls_outputs.device
        )  # anchor_points: (B, N, 2), anchor_strides: (B, N)

        predictions = []
        for i in range(batch_size):
            # (C, N) -> (N, C)
            cls_scores = cls_outputs[i].permute(1, 0)  # (N, num_classes)
            cls_probs = torch.sigmoid(cls_scores)

            # 找到每个 anchor 的最大类别及其得分
            max_scores, max_classes = cls_probs.max(dim=1)  # (N,), (N,)

            # 置信度过滤
            mask = max_scores > conf_thres  # (N,)
            if mask.sum() == 0:
                predictions.append(torch.zeros((0, 6), device=cls_outputs.device))
                continue

            scores = max_scores[mask]             # (N_f,)
            classes = max_classes[mask].float()   # (N_f,)

            # DFL 积分：box_outputs[i] 是 (4*R, N)
            box_dist = box_outputs[i].view(4, self.head.reg_max, -1).permute(2, 0, 1)  # (N, 4, R)
            box_dist_filtered = box_dist[mask]  # (N_f, 4, R)

            box_dist_softmax = box_dist_filtered.softmax(-1)  # (N_f, 4, R)
            proj = torch.arange(self.head.reg_max, dtype=torch.float32,
                                device=box_outputs.device).view(1, 1, -1)
            # 得到 ltrb，单位是 stride
            boxes_integrated = (box_dist_softmax * proj).sum(-1)  # (N_f, 4)

            # 取出对应的 anchor 坐标和 stride
            anchor_pts = anchor_points[i][mask]                 # (N_f, 2)
            strides = anchor_strides[i][mask].unsqueeze(1)      # (N_f, 1)

            # stride 单位距离 -> 像素距离
            dist_pixels = boxes_integrated * strides            # (N_f, 4)

            # boxes_integrated: [dl, dt, dr, db]
            lt = anchor_pts - dist_pixels[:, :2]                # (N_f, 2)
            rb = anchor_pts + dist_pixels[:, 2:]                # (N_f, 2)
            decoded_boxes = torch.cat([lt, rb], dim=-1)         # (N_f, 4) [x1,y1,x2,y2]

            # --- 加一个简单的 class-agnostic NMS，减少重复框 ---
            if decoded_boxes.numel() == 0:
                predictions.append(torch.zeros((0, 6), device=cls_outputs.device))
                continue

            # 这里使用类无关 NMS：不区分类别，只按 box + score 去重
            nms_iou_thres = 0.5
            keep = nms(decoded_boxes, scores, nms_iou_thres)

            decoded_boxes = decoded_boxes[keep]
            scores = scores[keep]
            classes = classes[keep]
            # --- NMS 结束 ---

            # 拼成最终输出: [x1, y1, x2, y2, score, class]
            pred = torch.cat(
                [decoded_boxes, scores.unsqueeze(1), classes.unsqueeze(1)],
                dim=1
            )
            predictions.append(pred)
        return predictions

    
    def _make_anchors(self, batch_size, img_h, img_w, device=None):
        if device is None:
            device = next(self.parameters()).device

        anchors_list = []
        strides_list = []

        for stride in self.strides:  # self.strides = [8,16,32]
            h = img_h // stride
            w = img_w // stride

            y = torch.arange(h, dtype=torch.float32, device=device)
            x = torch.arange(w, dtype=torch.float32, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

            # cell centers
            anchor_points = torch.stack(
                [grid_x + 0.5, grid_y + 0.5], dim=-1
            ) * stride
            anchor_points = anchor_points.reshape(-1, 2)

            anchor_stride = torch.full(
                (h * w,), stride, dtype=torch.float32, device=device
            )

            anchors_list.append(anchor_points)
            strides_list.append(anchor_stride)

        anchor_points = torch.cat(anchors_list, dim=0)
        anchor_strides = torch.cat(strides_list, dim=0)

        anchor_points = anchor_points.unsqueeze(0).repeat(batch_size, 1, 1)
        anchor_strides = anchor_strides.unsqueeze(0).repeat(batch_size, 1)

        return anchor_points, anchor_strides

    
    def load_weights(self, weights_path):
        """
        Load pre-trained weights
        Args:
            weights_path: Path to weights file
        """
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        self.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from {weights_path}")
    
    def save_weights(self, save_path, epoch=None, optimizer=None, loss=None):
        """
        Save model weights
        Args:
            save_path: Path to save weights
            epoch: Current epoch (optional)
            optimizer: Optimizer state (optional)
            loss: Current loss (optional)
        """
        checkpoint = {
            'model': self.state_dict(),
            'epoch': epoch,
            'num_classes': self.num_classes,
            'width_multiple': self.width_multiple,
            'depth_multiple': self.depth_multiple
        }
        
        if optimizer is not None:
            checkpoint['optimizer'] = optimizer.state_dict()
        
        if loss is not None:
            checkpoint['loss'] = loss
        
        torch.save(checkpoint, save_path)
        print(f"Saved weights to {save_path}")


def create_model(num_classes=None, width_multiple=None, depth_multiple=None, 
                 backbone_name='CSPDarknet', backbone_pretrained=False):
    """
    Factory function to create YOLOv8 model with configurable backbone
    
    Args:
        num_classes: Number of detection classes
        width_multiple: Width scaling factor
        depth_multiple: Depth scaling factor
        backbone_name: Backbone network name. Options:
            - 'CSPDarknet' (default, YOLO native)
            - 'ResNet50', 'ResNet101'
            - 'MobileNetV3'
            - 'VGG16', 'VGG19'
        backbone_pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        YOLOv8 model
    """
    model = YOLOv8(
        num_classes=num_classes,
        width_multiple=width_multiple,
        depth_multiple=depth_multiple,
        backbone_name=backbone_name,
        backbone_pretrained=backbone_pretrained
    )
    return model


def list_supported_backbones():
    """
    Print all supported backbone networks
    """
    print("Supported backbone networks:")
    for name in list_backbones():
        info = get_backbone_info(name)
        print(f"  - {info.get('name', name)}: {info.get('params', 'N/A')} params, "
              f"channels {info.get('out_channels', 'N/A')}")


if __name__ == '__main__':
    # List supported backbones
    list_supported_backbones()
    
    # Test with default backbone
    print("\n--- Testing CSPDarknet backbone ---")
    model = create_model(num_classes=80, width_multiple=0.5, depth_multiple=0.67, 
                        backbone_name='CSPDarknet')
    x = torch.randn(2, 3, 640, 640)
    cls_outputs, box_outputs, anchor_points, anchor_strides = model.predict(x)
    print(f"Input shape: {x.shape}")
    print(f"Classification outputs shape: {cls_outputs.shape}")
    print(f"Box outputs shape: {box_outputs.shape}")
    print(f"Backbone: {model.backbone_name}")
    print(f"Backbone out channels: {model.backbone.out_channels}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test with ResNet50 backbone (without pretrained for faster testing)
    print("\n--- Testing ResNet50 backbone ---")
    model_resnet = create_model(num_classes=80, width_multiple=0.5, depth_multiple=0.67,
                                backbone_name='ResNet50', backbone_pretrained=False)
    cls_outputs, box_outputs, anchor_points, anchor_strides = model_resnet.predict(x)
    print(f"Backbone: {model_resnet.backbone_name}")
    print(f"Backbone out channels: {model_resnet.backbone.out_channels}")
    print(f"Total parameters: {sum(p.numel() for p in model_resnet.parameters()) / 1e6:.2f}M")
