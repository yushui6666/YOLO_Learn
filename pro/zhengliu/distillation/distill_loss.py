"""
Distillation Loss Functions for YOLO Knowledge Distillation

Implements output distillation losses for object detection:
1. Classification distillation loss (KL divergence)
2. Box regression distillation loss (L2/IoU loss)
3. Objectness distillation loss (BCE with temperature)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for YOLO object detection.
    
    Combines:
    1. Classification distillation (KL divergence between soft predictions)
    2. Box regression distillation (L2 loss on box coordinates)
    3. Objectness distillation (BCE loss with temperature scaling)
    4. Ground truth losses (standard detection losses)
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        beta: float = 0.3,
        box_weight: float = 0.05,
        cls_weight: float = 0.5,
        obj_weight: float = 1.0
    ):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature for softening predictions (higher = softer)
            alpha: Weight for distillation loss (teacher supervision)
            beta: Weight for ground truth loss (1 - alpha recommended)
            box_weight: Weight for box regression loss
            cls_weight: Weight for classification loss
            obj_weight: Weight for objectness loss
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        
        # Loss functions
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.bce_with_logits = nn.BCEWithLogitsLoss()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(
        self,
        student_outputs: Dict,
        teacher_outputs: Dict,
        targets: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute distillation loss.
        
        Args:
            student_outputs: Student model predictions
                - cls_pred: Classification logits (B, N, num_classes)
                - box_pred: Box predictions (B, N, 4)
                - obj_pred: Objectness logits (B, N)
            teacher_outputs: Teacher model predictions (same format)
            targets: Ground truth targets (optional, for beta loss)
                - cls_gt: Ground truth classes
                - box_gt: Ground truth boxes
                - obj_gt: Ground truth objectness
        
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        
        # Extract predictions
        student_cls = student_outputs.get('cls_pred')
        student_box = student_outputs.get('box_pred')
        student_obj = student_outputs.get('obj_pred')
        
        teacher_cls = teacher_outputs.get('cls_pred')
        teacher_box = teacher_outputs.get('box_pred')
        teacher_obj = teacher_outputs.get('obj_pred')
        
        # ===== Distillation Loss (Alpha) =====
        # Classification distillation with temperature scaling
        if student_cls is not None and teacher_cls is not None:
            cls_loss_distill = self._kl_div_loss_with_temp(
                student_cls, teacher_cls, self.temperature
            )
        else:
            cls_loss_distill = torch.tensor(0.0, device=student_cls.device if student_cls is not None else 'cpu')
        
        # Box distillation (L2 loss)
        if student_box is not None and teacher_box is not None:
            # Only compute on positive samples (where teacher has objects)
            teacher_obj_sig = torch.sigmoid(teacher_obj) if teacher_obj is not None else None
            if teacher_obj_sig is not None:
                pos_mask = (teacher_obj_sig > 0.5).unsqueeze(-1).expand_as(student_box)
                if pos_mask.sum() > 0:
                    box_loss_distill = self.l1(
                        student_box[pos_mask], 
                        teacher_box[pos_mask].detach()
                    )
                else:
                    box_loss_distill = torch.tensor(0.0, device=student_box.device)
            else:
                box_loss_distill = self.l1(student_box, teacher_box.detach())
        else:
            box_loss_distill = torch.tensor(0.0, device=student_box.device if student_box is not None else 'cpu')
        
        # Objectness distillation
        if student_obj is not None and teacher_obj is not None:
            obj_loss_distill = self._bce_loss_with_temp(
                student_obj, teacher_obj, self.temperature
            )
        else:
            obj_loss_distill = torch.tensor(0.0, device=student_obj.device if student_obj is not None else 'cpu')
        
        # Combine distillation losses
        distill_loss = (
            self.cls_weight * cls_loss_distill +
            self.box_weight * box_loss_distill +
            self.obj_weight * obj_loss_distill
        )
        
        loss_dict['cls_loss_distill'] = cls_loss_distill.item()
        loss_dict['box_loss_distill'] = box_loss_distill.item()
        loss_dict['obj_loss_distill'] = obj_loss_distill.item()
        loss_dict['distill_loss'] = distill_loss.item()
        
        # ===== Ground Truth Loss (Beta) =====
        gt_loss = torch.tensor(0.0, device=distill_loss.device)
        
        if targets is not None:
            gt_cls = targets.get('cls_gt')
            gt_box = targets.get('box_gt')
            gt_obj = targets.get('obj_gt')
            
            # Classification loss with ground truth
            if student_cls is not None and gt_cls is not None:
                cls_loss_gt = self._cross_entropy_loss(student_cls, gt_cls)
            else:
                cls_loss_gt = torch.tensor(0.0, device=student_cls.device if student_cls is not None else 'cpu')
            
            # Box loss with ground truth
            if student_box is not None and gt_box is not None:
                box_loss_gt = self.l1(student_box, gt_box)
            else:
                box_loss_gt = torch.tensor(0.0, device=student_box.device if student_box is not None else 'cpu')
            
            # Objectness loss with ground truth
            if student_obj is not None and gt_obj is not None:
                obj_loss_gt = self.bce_with_logits(student_obj, gt_obj)
            else:
                obj_loss_gt = torch.tensor(0.0, device=student_obj.device if student_obj is not None else 'cpu')
            
            gt_loss = (
                self.cls_weight * cls_loss_gt +
                self.box_weight * box_loss_gt +
                self.obj_weight * obj_loss_gt
            )
            
            loss_dict['cls_loss_gt'] = cls_loss_gt.item()
            loss_dict['box_loss_gt'] = box_loss_gt.item()
            loss_dict['obj_loss_gt'] = obj_loss_gt.item()
            loss_dict['gt_loss'] = gt_loss.item()
        
        # ===== Total Loss =====
        total_loss = self.alpha * distill_loss + self.beta * gt_loss
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def _kl_div_loss_with_temp(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        KL divergence loss with temperature scaling.
        
        Args:
            student_logits: Student classification logits
            teacher_logits: Teacher classification logits
            temperature: Temperature for softening
        
        Returns:
            KL divergence loss
        """
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        
        # KL divergence
        loss = F.kl_div(
            student_soft,
            teacher_soft.detach(),
            reduction='batchmean'
        )
        
        # Scale by temperature^2 as per distillation paper
        return loss * (temperature ** 2)
    
    def _bce_loss_with_temp(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        BCE loss for objectness with temperature scaling.
        
        Args:
            student_logits: Student objectness logits
            teacher_logits: Teacher objectness logits
            temperature: Temperature for softening
        
        Returns:
            BCE loss
        """
        student_soft = torch.sigmoid(student_logits / temperature)
        teacher_soft = torch.sigmoid(teacher_logits / temperature).detach()
        
        loss = F.binary_cross_entropy(student_soft, teacher_soft, reduction='mean')
        return loss * (temperature ** 2)
    
    def _cross_entropy_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross entropy loss for classification.
        
        Args:
            logits: Classification logits (B, N, num_classes)
            targets: Target class indices (B, N) or one-hot (B, N, num_classes)
        
        Returns:
            Cross entropy loss
        """
        if targets.dim() == 2:
            # Targets are class indices
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction='mean'
            )
        else:
            # Targets are one-hot encoded
            targets_flat = targets.view(-1, targets.size(-1))
            logits_flat = logits.view(-1, logits.size(-1))
            return F.cross_entropy(
                logits_flat,
                targets_flat.argmax(dim=-1),
                reduction='mean'
            )


class FeatureDistillationLoss(nn.Module):
    """
    Feature-based distillation loss (optional extension).
    
    Matches intermediate feature maps between teacher and student.
    Uses hint-based distillation where teacher features guide student learning.
    """
    
    def __init__(
        self,
        feature_channels: Dict[str, Tuple[int, int]],
        loss_type: str = 'mse'
    ):
        """
        Initialize feature distillation loss.
        
        Args:
            feature_channels: Dictionary mapping layer names to (teacher_channels, student_channels)
            loss_type: Type of feature loss ('mse', 'l1', 'cosine')
        """
        super().__init__()
        self.feature_channels = feature_channels
        self.loss_type = loss_type
        
        # Projection layers to match channel dimensions
        self.projections = nn.ModuleDict()
        for layer_name, (t_ch, s_ch) in feature_channels.items():
            if t_ch != s_ch:
                self.projections[layer_name] = nn.Conv2d(s_ch, t_ch, kernel_size=1)
        
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'l1':
            self.loss_fn = nn.L1Loss()
        else:
            self.loss_fn = None  # Use cosine similarity
    
    def forward(
        self,
        student_features: Dict[str, torch.Tensor],
        teacher_features: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute feature distillation loss.
        
        Args:
            student_features: Dictionary of student feature maps
            teacher_features: Dictionary of teacher feature maps
        
        Returns:
            total_loss: Combined feature loss
            loss_dict: Dictionary of individual losses per layer
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0)
        
        for layer_name in self.feature_channels:
            if layer_name not in student_features or layer_name not in teacher_features:
                continue
            
            student_feat = student_features[layer_name]
            teacher_feat = teacher_features[layer_name]
            
            # Project student features if needed
            if layer_name in self.projections:
                student_feat = self.projections[layer_name](student_feat)
            
            # Match spatial dimensions
            if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
                student_feat = F.interpolate(
                    student_feat,
                    size=teacher_feat.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Compute loss
            if self.loss_fn is not None:
                layer_loss = self.loss_fn(student_feat, teacher_feat.detach())
            else:
                # Cosine similarity loss
                student_flat = student_feat.flatten(2)
                teacher_flat = teacher_feat.flatten(2)
                cos_sim = F.cosine_similarity(student_flat, teacher_flat.detach(), dim=-1)
                layer_loss = (1 - cos_sim.mean())
            
            loss_dict[f'feat_loss_{layer_name}'] = layer_loss.item()
            total_loss = total_loss + layer_loss
        
        loss_dict['feature_loss_total'] = total_loss.item()
        return total_loss, loss_dict


def create_distillation_criterion(
    temperature: float = 4.0,
    alpha: float = 0.7,
    beta: float = 0.3,
    **kwargs
) -> DistillationLoss:
    """
    Factory function to create distillation loss criterion.
    
    Args:
        temperature: Distillation temperature
        alpha: Distillation loss weight
        beta: Ground truth loss weight
        **kwargs: Additional arguments
    
    Returns:
        DistillationLoss module
    """
    return DistillationLoss(
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        **kwargs
    )