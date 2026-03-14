import torch
import torch.nn as nn
import torch.nn.functional as F
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU between box1 and box2.
    Args:
        box1: (..., 4)
        box2: (..., 4)
        xywh: if True, boxes are in (x, y, w, h) format, else (x1, y1, x2, y2)
        GIoU/DIoU/CIoU: use corresponding IoU variants
    """
    if xywh:
        # (x, y, w, h) -> (x1, y1, x2, y2)
        box1 = torch.cat(
            [box1[..., :2] - box1[..., 2:] / 2,
             box1[..., :2] + box1[..., 2:] / 2],
            dim=-1,
        )
        box2 = torch.cat(
            [box2[..., :2] - box2[..., 2:] / 2,
             box2[..., :2] + box2[..., 2:] / 2],
            dim=-1,
        )

    x1, y1, x2, y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    x1g, y1g, x2g, y2g = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # areas
    area1 = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    area2 = (x2g - x1g).clamp(0) * (y2g - y1g).clamp(0)

    # intersection
    inter = (torch.min(x2, x2g) - torch.max(x1, x1g)).clamp(0) * \
            (torch.min(y2, y2g) - torch.max(y1, y1g)).clamp(0)

    # union
    union = area1 + area2 - inter + eps
    iou = inter / union

    if not (CIoU or DIoU or GIoU):
        return iou

    # enclosing box
    cw = torch.max(x2, x2g) - torch.min(x1, x1g)
    ch = torch.max(y2, y2g) - torch.min(y1, y1g)

    if CIoU or DIoU:
        # center distance
        c2 = cw ** 2 + ch ** 2 + eps
        rho2 = ((x2 + x1 - x2g - x1g) ** 2 +
                (y2 + y1 - y2g - y1g) ** 2) / 4

        if CIoU:
            # aspect ratio term
            v = (4 / (torch.pi ** 2)) * torch.pow(
                torch.atan((x2 - x1) / (y2 - y1 + eps)) -
                torch.atan((x2g - x1g) / (y2g - y1g + eps)),
                2,
            )
            with torch.no_grad():
                alpha = v / (v - iou + (1 + eps))
            return iou - (rho2 / c2 + alpha * v)
        else:
            return iou - (rho2 / c2)
    else:
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area


class VarifocalLoss(nn.Module):
    """
    Varifocal Loss for classification (单独实现，目前在 YOLOv8Loss 里没用到).

    pred_score:  (B, C, N) logits
    target_score:(B, C, N) IoU-weighted targets in [0,1]
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, target_score, alpha=0.75, gamma=2.0):
        pred_prob = pred_score.sigmoid()
        weight = alpha * pred_prob.pow(gamma) * (1.0 - target_score) + target_score
        loss = F.binary_cross_entropy_with_logits(
            pred_score, target_score, reduction="none"
        ) * weight
        return loss.mean()


class BboxLoss(nn.Module):
    """
    Bounding box loss (CIoU loss).
    Expects inputs of shape (K, 4) in xywh format.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_box, gt_box):
        loss = 1.0 - bbox_iou(pred_box, gt_box, xywh=True, CIoU=True)
        return loss.mean()


class DFLoss(nn.Module):
    """
    Distribution Focal Loss.

    pred_logits: (K, reg_max) logits over discrete bins [0, reg_max-1]
    target:      (K,) continuous distance in [0, reg_max-1]
    """
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_logits, target):
        # clamp target into valid range (avoid right index overflow)
        target = target.clamp(0, self.reg_max - 1 - 1e-3)

        tl = target.floor().long()          # left bin index
        tr = tl + 1                         # right bin index
        wl = tr.float() - target            # left weight
        wr = target - tl.float()            # right weight

        loss_l = F.cross_entropy(pred_logits, tl, reduction="none")
        loss_r = F.cross_entropy(
            pred_logits, tr.clamp(max=self.reg_max - 1), reduction="none"
        )

        loss = wl * loss_l + wr * loss_r
        return loss.mean()


class YOLOv8Loss(nn.Module):
    """
    Complete YOLOv8-like loss function.

    Expects box outputs that are NOT integrated through DFL yet:
    pred_dist shape: (B, 4 * reg_max, N)
    """

    def __init__(self, num_classes=None, box_gain=1.0, cls_gain=1.0,
                 dfl_gain=1.0, obj_gain=1.0, reg_max=16,
                 max_pos_per_gt=10, use_focal_loss=False):
        super().__init__()

        if num_classes is None:
            raise ValueError("num_classes must be specified")

        self.num_classes = num_classes
        self.box_gain = float(box_gain)
        self.cls_gain = float(cls_gain)
        self.dfl_gain = float(dfl_gain)
        self.obj_gain = float(obj_gain)
        self.reg_max = reg_max
        self.max_pos_per_gt = max_pos_per_gt  # Maximum positive samples per GT
        self.use_focal_loss = use_focal_loss

        # loss components
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.bbox_loss = BboxLoss()
        self.dfl_loss = DFLoss(reg_max=self.reg_max)
        self.varifocal_loss = VarifocalLoss()

        # strides for anchor generation
        self.strides = [8, 16, 32]

    def _make_anchors(self, batch_size, img_h, img_w, device):
        """
        Generate anchor points for all scales based on image size.

        Returns:
            anchor_points: (B, N, 2) in absolute pixel coordinates (cell centers)
            anchor_strides: (B, N)
        """
        anchors_list = []
        strides_list = []

        for stride in self.strides:
            h = img_h // stride
            w = img_w // stride

            y = torch.arange(h, dtype=torch.float32, device=device)
            x = torch.arange(w, dtype=torch.float32, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

            # cell centers: (i + 0.5, j + 0.5) * stride
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

    def _make_anchors_from_features(self, batch_size, feature_maps, device):
        """
        Generate anchor points based on actual feature map sizes.

        Returns:
            anchor_points: (B, N, 2) in absolute pixel coordinates (cell centers)
            anchor_strides: (B, N)
        """
        anchors_list = []
        strides_list = []

        for i, fmap in enumerate(feature_maps):
            stride = self.strides[i]
            _, _, h, w = fmap.shape

            y = torch.arange(h, dtype=torch.float32, device=device)
            x = torch.arange(w, dtype=torch.float32, device=device)
            grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

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

    def _assign_targets(self, targets, anchor_points, anchor_strides,
                        batch_size, num_anchors, device,
                        img_h=640, img_w=640, pred_cls=None):
        """
        Assign ground truth boxes to anchor points using top-k strategy.

        Args:
            targets: list of (num_gt, 6) [class, x, y, w, h, anchor_idx], x,y,w,h in [0,1]
            anchor_points: (B, N, 2) in pixel coords (cell centers)
            anchor_strides: (B, N)
            pred_cls: (B, num_classes, N) optional predicted class scores for task-aligned assignment
        Returns:
            cls_targets: (B, num_classes, N)
            box_targets: (B, 4, N) in ltrb format (distances in stride units)
            obj_targets: (B, N)
        """
        assert anchor_points.shape[1] == num_anchors, \
            f"Anchor count mismatch: {anchor_points.shape[1]} vs {num_anchors}"

        cls_targets = torch.zeros(
            batch_size, self.num_classes, num_anchors, device=device
        )
        box_targets = torch.zeros(batch_size, 4, num_anchors, device=device)
        obj_targets = torch.zeros(batch_size, num_anchors, device=device)

        for b in range(batch_size):
            batch_labels = targets[b]
            # remove padding
            if batch_labels.numel() == 0:
                continue
            if batch_labels.ndim == 1:
                batch_labels = batch_labels.view(1, -1)

            if batch_labels.shape[1] >= 1:
                valid_mask = batch_labels[:, 0] >= 0
                batch_labels = batch_labels[valid_mask]

            if batch_labels.numel() == 0:
                continue

            gt_cls = batch_labels[:, 0].long()
            gt_box_xywh = batch_labels[:, 1:5]  # normalized [0,1]

            # to absolute pixels
            gt_box_abs = gt_box_xywh.clone()
            gt_box_abs[:, 0] *= img_w
            gt_box_abs[:, 1] *= img_h
            gt_box_abs[:, 2] *= img_w
            gt_box_abs[:, 3] *= img_h

            batch_anchor_points = anchor_points[b]        # (N, 2)
            batch_anchor_strides = anchor_strides[b]      # (N,)
            ax = batch_anchor_points[:, 0]
            ay = batch_anchor_points[:, 1]
            stride_all = batch_anchor_strides

            num_gt = gt_cls.shape[0]

            for j in range(num_gt):
                cls_id = gt_cls[j].item()
                cx, cy, w, h = gt_box_abs[j]

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                # Step 1: Find candidate anchors inside the GT box
                in_x = (ax > x1) & (ax < x2)
                in_y = (ay > y1) & (ay < y2)
                in_box = in_x & in_y

                candidate_indices = torch.nonzero(in_box, as_tuple=False).squeeze(1)

                if candidate_indices.numel() == 0:
                    # fallback: nearest anchor by center distance
                    dx = ax - cx
                    dy = ay - cy
                    distances = torch.sqrt(dx ** 2 + dy ** 2)
                    closest = int(distances.argmin().item())
                    candidate_indices = torch.tensor(
                        [closest], device=device, dtype=torch.long
                    )

                # Step 2: Calculate matching scores for candidates
                if pred_cls is not None:
                    # Task-aligned assignment: use classification score
                    cls_scores = pred_cls[b, cls_id, candidate_indices].sigmoid()
                else:
                    # Simple assignment: use IoU-based score (approximated by center distance)
                    dx_cand = ax[candidate_indices] - cx
                    dy_cand = ay[candidate_indices] - cy
                    center_dist = torch.sqrt(dx_cand ** 2 + dy_cand ** 2)
                    # Normalize distance by box size
                    max_dist = torch.sqrt(w ** 2 + h ** 2) / 2 + 1e-7
                    cls_scores = 1.0 - center_dist / max_dist

                # Calculate IoU for each candidate
                # Convert anchor points to small boxes for IoU calculation
                anchor_w = stride_all[candidate_indices]
                anchor_h = stride_all[candidate_indices]
                anchor_boxes = torch.stack([
                    ax[candidate_indices] - anchor_w / 2,
                    ay[candidate_indices] - anchor_h / 2,
                    ax[candidate_indices] + anchor_w / 2,
                    ay[candidate_indices] + anchor_h / 2
                ], dim=1)  # (K, 4) in x1y1x2y2 format
                
                gt_box_x1y1x2y2 = torch.tensor([x1, y1, x2, y2], device=device).unsqueeze(0)
                
                ious = bbox_iou(anchor_boxes, gt_box_x1y1x2y2, xywh=False)

                # Combined score: cls_score * IoU (task-aligned)
                match_scores = cls_scores * ious

                # Step 3: Select top-k anchors
                k = min(self.max_pos_per_gt, candidate_indices.numel())
                topk_scores, topk_indices = torch.topk(match_scores, k)

                assigned_indices = candidate_indices[topk_indices]

                # distances to box edges in pixels
                ax_assigned = ax[assigned_indices]
                ay_assigned = ay[assigned_indices]
                s_assigned = stride_all[assigned_indices]

                l = (ax_assigned - x1).clamp(min=0)
                t = (ay_assigned - y1).clamp(min=0)
                r = (x2 - ax_assigned).clamp(min=0)
                b_dist = (y2 - ay_assigned).clamp(min=0)

                # convert to stride units and clamp upper bound
                gt_ltrb = torch.stack([l, t, r, b_dist], dim=1) / s_assigned.view(-1, 1)
                gt_ltrb = gt_ltrb.clamp(max=self.reg_max - 1e-3)  # Prevent overflow

                # update targets
                cls_targets[b, cls_id, assigned_indices] = 1.0
                obj_targets[b, assigned_indices] = 1.0
                box_targets[b, :, assigned_indices] = gt_ltrb.t()

        return cls_targets, box_targets, obj_targets

    @staticmethod
    def _ltrb_to_xywh(ltrb, anchor_points, anchor_strides, img_h, img_w):
        """
        Convert ltrb (in stride units) to normalized xywh.

        Args:
            ltrb: (B, 4, N), distances in stride units
            anchor_points: (B, N, 2) in pixels (cell centers)
            anchor_strides: (B, N)
        Returns:
            xywh: (B, 4, N), normalized by img_w/img_h
        """
        B, four, N = ltrb.shape
        assert four == 4

        ax = anchor_points[..., 0]      # (B, N)
        ay = anchor_points[..., 1]      # (B, N)
        s = anchor_strides              # (B, N)

        l = ltrb[:, 0, :] * s           # (B, N) in pixels
        t = ltrb[:, 1, :] * s
        r = ltrb[:, 2, :] * s
        b = ltrb[:, 3, :] * s

        # reconstruct box center and size
        cx = ax + (r - l) / 2
        cy = ay + (b - t) / 2
        w = l + r
        h = t + b

        xywh = torch.zeros_like(ltrb)
        xywh[:, 0, :] = cx / img_w
        xywh[:, 1, :] = cy / img_h
        xywh[:, 2, :] = w / img_w
        xywh[:, 3, :] = h / img_h

        return xywh

    def forward(self, outputs, targets, img_h=640, img_w=640, features=None):
        """
        Args:
            outputs: (pred_cls, pred_dist)
                pred_cls:  (B, num_classes, N) logits
                pred_dist: (B, 4*reg_max, N) box distribution logits
            targets: list of (num_gt, 6) [class, x, y, w, h, anchor_idx]
            img_h/img_w: training image size
            features: optional feature maps list for anchor generation

        Returns:
            dict with total_loss, box_loss, cls_loss, dfl_loss, obj_loss
        """
        pred_cls, pred_dist = outputs

        device = pred_cls.device
        batch_size = pred_cls.shape[0]
        num_anchors = pred_cls.shape[2]

        # anchor generation (warning if not using features)
        if features is None:
            import warnings
            warnings.warn(
                "Anchor generation using img_h/img_w may cause size mismatches. "
                "Consider passing feature maps to the loss function.",
                UserWarning
            )
            anchor_points, anchor_strides = self._make_anchors(
                batch_size, img_h, img_w, device
            )
        else:
            anchor_points, anchor_strides = self._make_anchors_from_features(
                batch_size, features, device
            )

        # target assignment with task-aligned strategy (using pred_cls for better assignment)
        cls_targets, box_targets, obj_targets = self._assign_targets(
            targets, anchor_points, anchor_strides,
            batch_size, num_anchors, device, img_h, img_w, pred_cls=pred_cls.detach()
        )

        # positive anchors mask（用于 box / DFL / cls）
        pos_mask = obj_targets > 0  # (B, N)

        # Classification loss：只在正样本 anchor 上计算
        if self.use_focal_loss:
            if pos_mask.any():
                # 只取正样本 anchor，对每个 anchor 仍然做 C 维分类
                pos_mask_exp = pos_mask.unsqueeze(1).expand_as(pred_cls)  # (B, C, N)
                pred_pos = pred_cls[pos_mask_exp].view(-1, self.num_classes, 1)   # (K_pos, C, 1)
                tgt_pos = cls_targets[pos_mask_exp].view(-1, self.num_classes, 1) # (K_pos, C, 1)
                # VarifocalLoss 期望 (B, C, N)，这里把每个正样本当作一个 batch
                cls_loss = self.varifocal_loss(pred_pos, tgt_pos)
            else:
                cls_loss = torch.tensor(0.0, device=device)
        else:
            if pos_mask.any():
                # 标准 BCE，但只在正样本 anchor 上做平均
                bce_loss = self.bce(pred_cls, cls_targets)  # (B, C, N)
                pos_mask_exp = pos_mask.unsqueeze(1).expand_as(bce_loss)  # (B, C, N)
                cls_loss = (bce_loss * pos_mask_exp).sum() / pos_mask_exp.sum()
            else:
                cls_loss = torch.tensor(0.0, device=device)

        # Objectness loss: max class score as objectness proxy（所有 anchor 都参与）
        obj_pred = pred_cls.max(dim=1)[0]  # (B, N)
        obj_loss = self.bce(obj_pred, obj_targets).mean()

        # Box and DFL losses (only on positive anchors)
        if pos_mask.any():
            B, C, N = pred_dist.shape
            assert C == 4 * self.reg_max, \
                f"pred_dist channels {C} != 4*reg_max {4*self.reg_max}"

            # DFL integration for CIoU loss
            pred_dist_reshaped = pred_dist.view(B, 4, self.reg_max, N)  # (B, 4, R, N)
            pred_prob = pred_dist_reshaped.softmax(2)                   # along R
            proj = torch.arange(self.reg_max, device=device).view(1, 1, self.reg_max, 1)
            pred_ltrb = (pred_prob * proj).sum(2)                       # (B, 4, N)

            # convert ltrb to xywh (normalized)
            pred_box_xywh = self._ltrb_to_xywh(
                pred_ltrb, anchor_points, anchor_strides, img_h, img_w
            )
            box_targets_xywh = self._ltrb_to_xywh(
                box_targets, anchor_points, anchor_strides, img_h, img_w
            )

            # flatten and index by positive mask
            pred_flat = pred_box_xywh.permute(0, 2, 1).reshape(-1, 4)   # (B*N,4)
            tgt_flat = box_targets_xywh.permute(0, 2, 1).reshape(-1, 4)
            mask_flat = pos_mask.reshape(-1)

            box_loss = self.bbox_loss(pred_flat[mask_flat], tgt_flat[mask_flat])

            # DFL loss - vectorized version (much faster)
            # (B, 4, R, N) -> (B, N, 4, R) -> (B*N, 4, R)
            pred_perm = pred_dist_reshaped.permute(0, 3, 1, 2)          # (B, N, 4, R)
            pred_flat_dfl = pred_perm.reshape(-1, 4, self.reg_max)     # (B*N, 4, R)
            tgt_flat_ltrb = box_targets.permute(0, 2, 1).reshape(-1, 4)  # (B*N, 4)
            
            # 只取正样本
            pos_pred = pred_flat_dfl[mask_flat]   # (K_pos, 4, R)
            pos_tgt = tgt_flat_ltrb[mask_flat]    # (K_pos, 4)
            
            # Clamp targets to valid range
            pos_tgt = pos_tgt.clamp(0, self.reg_max - 1 - 1e-3)
            
            # Compute left/right bin indices for all sides
            tl = pos_tgt.floor().long()      # (K_pos, 4)
            tr = tl + 1                      # (K_pos, 4)
            wl = tr.float() - pos_tgt        # (K_pos, 4)
            wr = pos_tgt - tl.float()        # (K_pos, 4)
            
            # Clamp right bin indices
            tr = tr.clamp(max=self.reg_max - 1)
            
            # Compute losses for all sides
            loss_l = F.cross_entropy(
                pos_pred.reshape(-1, self.reg_max),  # (K_pos*4, R)
                tl.reshape(-1),                      # (K_pos*4,)
                reduction='none'
            )
            loss_r = F.cross_entropy(
                pos_pred.reshape(-1, self.reg_max),
                tr.reshape(-1),
                reduction='none'
            )
            
            # 展平权重并加权求平均
            wl_flat = wl.reshape(-1)  # (K_pos*4,)
            wr_flat = wr.reshape(-1)  # (K_pos*4,)
            
            dfl_loss = (wl_flat * loss_l + wr_flat * loss_r).mean()
        else:
            box_loss = torch.tensor(0.0, device=device)
            dfl_loss = torch.tensor(0.0, device=device)

        # Total loss with obj_loss included
        total_loss = (self.box_gain * box_loss +
                      self.cls_gain * cls_loss +
                      self.dfl_gain * dfl_loss +
                      self.obj_gain * obj_loss)

        return {
            'total_loss': total_loss,
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'dfl_loss': dfl_loss,
            'obj_loss': obj_loss,
        }



if __name__ == '__main__':
    # Simple self-test
    loss_fn = YOLOv8Loss(num_classes=80)

    batch_size = 4
    img_h = img_w = 640

    # 80*80 + 40*40 + 20*20 = 8400 anchors
    num_anchors = 8400
    reg_max = loss_fn.reg_max

    # mock predictions
    pred_cls = torch.randn(batch_size, 80, num_anchors)
    pred_dist = torch.randn(batch_size, 4 * reg_max, num_anchors)

    # mock targets: each image has 1 box at center
    labels = [
        torch.tensor([[0, 0.5, 0.5, 0.3, 0.3, 0]], dtype=torch.float32)
        for _ in range(batch_size)
    ]

    loss_dict = loss_fn((pred_cls, pred_dist), labels, img_h=img_h, img_w=img_w)

    print(f"Total loss: {loss_dict['total_loss'].item():.4f}")
    print(f"Box loss:   {loss_dict['box_loss'].item():.4f}")
    print(f"Class loss: {loss_dict['cls_loss'].item():.4f}")
    print(f"DFL loss:   {loss_dict['dfl_loss'].item():.4f}")
    print(f"Obj  loss:  {loss_dict['obj_loss'].item():.4f}")
