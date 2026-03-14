# models/head.py - 检测头详解

## 概述

`head.py` 实现了 YOLOv8 的解耦检测头 `DetectHead`，包含：
- **DFL (Distribution Focal Loss) 积分模块**：将分布输出转换为边界框坐标
- **解耦检测头**：分类和回归分支独立处理
- **锚点生成**：为每个特征图位置生成锚点
- **边界框解码**：将网络输出转换为最终检测框

---

## 核心概念

### 解耦检测头 (Decoupled Head)

**传统 YOLO (YOLOv3/v4/v5)**：分类和回归共享特征
```
特征 → 单一卷积层 → [分类 + 回归输出]
```

**YOLOv8 解耦头**：分类和回归独立处理
```
特征 ─→ 分类分支 → 分类输出
  └→ 回归分支 → 回归输出
```

**优势：**
- 分类和回归任务互不干扰
- 提高检测精度
- 更灵活的特征学习

---

## 核心模块详解

### 1. DFL (Distribution Focal Loss) 积分 ⭐

```python
class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1  # reg_max = 16
        
        # 固定的投影向量 [0, 1, 2, ..., reg_max-1]
        proj = torch.arange(c1, dtype=torch.float).view(1, 1, c1, 1, 1)
        self.register_buffer('proj', proj)
```

**核心思想：**
- YOLOv8 不直接预测边界框的 4 个值（x, y, w, h）
- 而是预测每个边界的**分布**：[0, 1, ..., 15] 上 16 个离散值的概率
- 通过积分（期望）得到连续的距离值

**前向传播：**
```python
def forward(self, x):
    """
    Args:
        x: (B, 4*reg_max, H, W) 原始回归输出
    Returns:
        (B, 4, H, W) 的距离值 [l, t, r, b]
    """
    b, _, h, w = x.shape
    
    # 重塑为 (B, 4, reg_max, H, W)
    x = x.view(b, 4, self.c1, h, w)
    
    # 在 reg_max 维度上做 softmax 得到分布
    x = x.softmax(2)
    
    # 计算期望：sum(p * v)
    proj = self.proj.to(dtype=x.dtype, device=x.device)
    x = (x * proj).sum(2)  # (B, 4, H, W)
    return x
```

**示例：**
```python
# 假设 reg_max = 16
# 网络输出的分布（经过 softmax）
dist = [0.05, 0.10, 0.15, 0.20, 0.25, 0.10, 0.05, 0.05, 
        0.02, 0.01, 0.01, 0.01, 0.00, 0.00, 0.00, 0.00]

# 投影向量
proj = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# 计算期望（积分）
distance = sum(dist[i] * proj[i] for i in range(16))
# distance ≈ 3.75

# 解释：边界到锚点的距离约为 3.75 个步长单位
```

**为什么使用 DFL？**
1. **更灵活的分布建模**：可以表示任意距离，不受离散值限制
2. **更好的梯度流**：分布学习的梯度更平滑
3. **更高的精度**：通过积分获得亚像素级精度

---

### 2. DetectHead - 解耦检测头

#### 2.1 初始化

```python
class DetectHead(nn.Module):
    def __init__(self, num_classes=None, in_channels=[256, 512, 1024], reg_max=16):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_outputs = num_classes + 4 * reg_max
        self.reg_max = reg_max
        self.nc = num_classes
```

**参数说明：**
- `num_classes`: 检测类别数（COCO 为 80）
- `in_channels`: 3 个尺度的输入通道数 [P3, P4, P5]
- `reg_max`: DFL 的离散值数量（默认 16）

#### 2.2 网络结构

```
对于每个尺度 (P3, P4, P5):
    输入 (C, H, W)
      ↓
  Stems: Conv(3×3) + Conv(3×3)
      ↓
    ┌───┴───┐
    ↓       ↓
分类分支   回归分支
    ↓       ↓
  Convs    Convs
  (2×Conv) (2×Conv)
    ↓       ↓
  Pred_cls Pred_reg
  (1×1)    (1×1, 4*reg_max)
```

**Stem 模块：**
```python
self.stems = nn.ModuleList([
    nn.Sequential(
        Conv(x, x, 3, 1),
        Conv(x, x, 3, 1)
    ) for x in in_channels
])
```
- 两个 3×3 卷积
- 作用：增强特征表示

**分类分支：**
```python
self.cls_convs = nn.ModuleList([
    nn.Sequential(
        Conv(x, x, 3, 1),
        Conv(x, x, 3, 1)
    ) for x in in_channels
])

self.cls_preds = nn.ModuleList([
    nn.Conv2d(x, num_classes, 1) for x in in_channels
])
```
- 两个 3×3 卷积 + 1×1 预测卷积
- 输出：`(B, num_classes, H*W)`

**回归分支：**
```python
self.reg_convs = nn.ModuleList([
    nn.Sequential(
        Conv(x, x, 3, 1),
        Conv(x, x, 3, 1)
    ) for x in in_channels
])

self.reg_preds = nn.ModuleList([
    nn.Conv2d(x, 4 * reg_max, 1) for x in in_channels
])
```
- 两个 3×3 卷积 + 1×1 预测卷积
- 输出：`(B, 4*reg_max, H*W)`

#### 2.3 前向传播

```python
def forward(self, x):
    """
    Args:
        x: List of feature maps [P3, P4, P5]
    Returns:
        Tuple of (cls_outputs, box_outputs)
        cls_outputs: (B, num_classes, N) - 分类 logits
        box_outputs: (B, 4*reg_max, N) - 回归分布（未积分）
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
        
        # Reshape outputs
        b, _, h, w = reg_output.shape
        cls_output = cls_output.reshape(b, self.num_classes, -1)
        reg_output = reg_output.reshape(b, 4 * self.reg_max, -1)
        
        outputs.append((cls_output, reg_output))
    
    # Concatenate all scales
    cls_outputs = torch.cat([o[0] for o in outputs], dim=-1)
    box_outputs = torch.cat([o[1] for o in outputs], dim=-1)
    
    return cls_outputs, box_outputs
```

**输出示例：**
```python
# 输入: 3 个特征图
P3: (B, 256, 80, 80)   → 6400 个位置
P4: (B, 512, 40, 40)   → 1600 个位置
P5: (B, 1024, 20, 20)  → 400 个位置

# 总位置数: 6400 + 1600 + 400 = 8400

# 输出:
cls_outputs:  (B, 80, 8400)     # 80 个类别的分类分数
box_outputs:  (B, 64, 8400)     # 4×16=64 维的回归分布
```

**关键点：**
- `box_outputs` **不经过 DFL 积分**
- 训练时：直接使用分布计算 DFL loss
- 推理时：在 `decode_bboxes` 中进行 DFL 积分

---

### 3. 锚点生成 (Anchor Points)

```python
def _make_anchors(self, in_channels):
    anchors = []
    for i in range(len(in_channels)):
        stride = self.strides[i]
        anchors.append(stride)
    return anchors
```

**注意：** 这只是返回步长列表，实际的锚点坐标在 `yolov8.py` 的 `_make_anchors` 中生成。

**锚点坐标生成（在 YOLOv8 类中）：**
```python
def _make_anchors(self, batch_size, img_h, img_w, device=None):
    anchors_list = []
    strides_list = []
    
    for stride in self.strides:  # [8, 16, 32]
        h = img_h // stride
        w = img_w // stride
        
        y = torch.arange(h, dtype=torch.float32, device=device)
        x = torch.arange(w, dtype=torch.float32, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # 计算单元格中心点
        anchor_points = torch.stack(
            [grid_x + 0.5, grid_y + 0.5], dim=-1
        ) * stride
        anchor_points = anchor_points.reshape(-1, 2)
        
        anchors_list.append(anchor_points)
        strides_list.append(torch.full((h * w,), stride))
    
    # 拼接所有尺度
    anchor_points = torch.cat(anchors_list, dim=0)
    anchor_strides = torch.cat(strides_list, dim=0)
    
    return anchor_points, anchor_strides
```

**示例：**
```python
# 假设输入图像 640×640
# Stride = 8
h = 640 // 8 = 80
w = 640 // 8 = 80

# 生成网格
grid_x = [0.5, 1.5, 2.5, ..., 79.5]  # 80 个点
grid_y = [0.5, 1.5, 2.5, ..., 79.5]  # 80 个点

# 锚点坐标（像素）
anchor_points = grid_x * stride, grid_y * stride
# 第一个锚点: (4, 4)
# 最后一个锚点: (636, 636)

# 步长
anchor_strides = [8, 8, 8, ..., 8]  # 6400 个 8
```

---

### 4. 边界框解码 (decode_bboxes)

```python
def decode_bboxes(self, box_outputs, anchors):
    """
    Decode box outputs to actual coordinates
    Args:
        box_outputs: (B, 4, N) - DFL 积分后的 [l, t, r, b]
        anchors: (B, N, 2) - 锚点坐标
    Returns:
        (B, N, 4) boxes in format [x1, y1, x2, y2]
    """
    B, _, N = box_outputs.shape
    
    # 获取距离 [l, t, r, b]
    dist_ltrb = box_outputs.permute(0, 2, 1)  # (B, N, 4)
    
    # 计算边界框坐标
    lt = anchors - dist_ltrb[..., :2]  # 左上角: 锚点 - [l, t]
    rb = anchors + dist_ltrb[..., 2:]  # 右下角: 锚点 + [r, b]
    
    boxes = torch.cat([lt, rb], dim=-1)
    return boxes
```

**ltrb 格式：**
- `l`: 左边界到锚点的距离（左边距）
- `t`: 上边界到锚点的距离（上边距）
- `r`: 右边界到锚点的距离（右边距）
- `b`: 下边界到锚点的距离（下边距）

**示例：**
```python
# 假设锚点位置 (320, 320)
anchor = (320, 320)

# DFL 积分后的距离（像素）
ltrb = [50, 60, 70, 80]  # [l, t, r, b]

# 计算边界框
x1 = anchor[0] - ltrb[0]  # 320 - 50 = 270
y1 = anchor[1] - ltrb[1]  # 320 - 60 = 260
x2 = anchor[0] + ltrb[2]  # 320 + 70 = 390
y2 = anchor[1] + ltrb[3]  # 320 + 80 = 400

# 最终边界框: [270, 260, 390, 400]
```

---

## 在 YOLOv8 中的完整流程

### 训练时

```python
# 1. 前向传播
cls_outputs, box_outputs = model(images)  # 不进行 DFL 积分

# 2. 传递给损失函数
loss = loss_fn(
    outputs=(cls_outputs, box_outputs),
    targets=targets,
    features=neck_features  # 用于生成锚点
)

# 3. 损失函数内部进行 DFL 积分
# box_outputs: (B, 4*16, N) → DFL → (B, 4, N)
```

### 推理时

```python
# 1. 前向传播
cls_outputs, box_outputs = model(images)

# 2. 解码预测
predictions = model.decode_predictions(
    (cls_outputs, box_outputs),
    img_h=640,
    img_w=640,
    conf_thres=0.25
)

# decode_predictions 内部:
# - 生成锚点坐标
# - 对 box_outputs 进行 DFL 积分
# - 使用 ltrb 和锚点计算边界框
# - 应用 NMS
```

---

## 使用示例

### 示例 1：创建检测头

```python
from models.head import DetectHead
import torch

# YOLOv8 检测头
head = DetectHead(
    num_classes=80,                 # COCO 数据集 80 类
    in_channels=[256, 512, 1024],   # P3, P4, P5 通道数
    reg_max=16                      # DFL 离散值数量
)

# 参数量
params = sum(p.numel() for p in head.parameters()) / 1e6
print(f"Head parameters: {params:.2f}M")
# 输出: Head parameters: 15.45M
```

### 示例 2：前向传播

```python
# 模拟 neck 输出的特征图
x1 = torch.randn(2, 256, 80, 80)    # P3
x2 = torch.randn(2, 512, 40, 40)    # P4
x3 = torch.randn(2, 1024, 20, 20)   # P5

# 前向传播
cls_outputs, box_outputs = head([x1, x2, x3])

print(f"Classification outputs: {cls_outputs.shape}")
print(f"Box outputs: {box_outputs.shape}")

# 输出:
# Classification outputs: torch.Size([2, 80, 8400])
# Box outputs: torch.Size([2, 64, 8400])
```

### 示例 3：单独使用 DFL

```python
from models.head import DFL
import torch

# 创建 DFL 模块
dfl = DFL(c1=16)

# 模拟回归输出
box_dist = torch.randn(1, 64, 80, 80)  # (B, 4*16, H, W)

# DFL 积分
box_integrated = dfl(box_dist)

print(f"Distribution shape: {box_dist.shape}")
print(f"Integrated shape: {box_integrated.shape}")

# 输出:
# Distribution shape: torch.Size([1, 64, 80, 80])
# Integrated shape: torch.Size([1, 4, 80, 80])
```

### 示例 4：解码边界框

```python
import torch

# 假设的锚点和 DFL 输出
anchors = torch.tensor([[
    [320, 320],  # 位置 1
    [100, 100],  # 位置 2
    [500, 500]   # 位置 3
]]).float()

box_outputs = torch.tensor([[
    [50, 60, 70, 80],   # 位置 1 的 [l, t, r, b]
    [30, 40, 50, 60],   # 位置 2 的 [l, t, r, b]
    [40, 50, 60, 70]    # 位置 3 的 [l, t, r, b]
]]).float()

# 解码
boxes = head.decode_bboxes(box_outputs, anchors)

print("Decoded boxes:")
print(boxes[0])

# 输出:
# [[270., 260., 390., 400.],   # [320-50, 320-60, 320+70, 320+80]
#  [ 70.,  60., 150., 160.],   # [100-30, 100-40, 100+50, 100+60]
#  [460., 450., 560., 570.]]   # [500-40, 500-50, 500+60, 500+70]
```

---

## 架构对比

### YOLOv8 Head vs YOLOv5 Head

| 特性 | YOLOv8 | YOLOv5 |
|------|--------|--------|
| 架构 | 解耦头 | 解耦头 |
| 回归方法 | DFL 分布 | 直接预测 |
| 锚点 | 无锚点 | 有锚点 |
| 输出格式 | ltrb + 分布 | xywh + objectness |
| 精度 | 更高 | 较高 |
| 速度 | 更快 | 较快 |

### DFL vs 直接预测

| 方法 | DFL | 直接预测 (xywh) |
|------|-----|----------------|
| 灵活性 | 高（分布建模） | 低（固定格式） |
| 精度 | 亚像素级 | 像素级 |
| 梯度流 | 平滑 | 可能不稳定 |
| 计算量 | 稍高 | 较低 |

---

## 调试技巧

### 1. 检查分类输出分布

```python
import matplotlib.pyplot as plt

# 获取分类输出
cls_outputs, _ = head([x1, x2, x3])
cls_probs = torch.sigmoid(cls_outputs)

# 可视化某个位置的类别概率
pos_idx = 1000  # 任意位置
class_probs = cls_probs[0, :, pos_idx].detach().cpu().numpy()

plt.figure(figsize=(12, 4))
plt.bar(range(80), class_probs)
plt.xlabel('Class ID')
plt.ylabel('Probability')
plt.title('Class Probability Distribution')
plt.savefig('class_probs.png')
```

### 2. 分析 DFL 分布

```python
# 获取回归输出
_, box_outputs = head([x1, x2, x3])

# 重塑为分布格式
box_dist = box_outputs.view(2, 4, 16, -1)
box_dist = box_dist.softmax(2)  # (B, 4, 16, N)

# 可视化某个位置的分布
pos_idx = 500
dist_left = box_dist[0, 0, :, pos_idx].detach().cpu().numpy()  # l
dist_top = box_dist[0, 1, :, pos_idx].detach().cpu().numpy()   # t

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.bar(range(16), dist_left)
ax1.set_xlabel('Distance Bin')
ax1.set_ylabel('Probability')
ax1.set_title('Left Distance Distribution')

ax2.bar(range(16), dist_top)
ax2.set_xlabel('Distance Bin')
ax2.set_ylabel('Probability')
ax2.set_title('Top Distance Distribution')

plt.savefig('dfl_distribution.png')
```

### 3. 可视化解码后的边界框

```python
import cv2
import numpy as np

# 假设的预测
predictions = model.decode_predictions(
    (cls_outputs, box_outputs),
    conf_thres=0.5
)

# 可视化
img = np.zeros((640, 640, 3), dtype=np.uint8)
boxes = predictions[0].cpu().numpy()

for box in boxes:
    x1, y1, x2, y2, score, cls_id = box.astype(int)
    if score > 0:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{cls_id}', (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imwrite('predictions.png', img)
```

---

## 常见问题

### Q1: 为什么 box_outputs 不在前向传播中积分？

**A:**
- **训练时**：使用分布计算 DFL loss，梯度流更好
- **推理时**：在 `decode_predictions` 中积分，避免重复计算
- **灵活性**：可以在不同阶段应用不同的处理

### Q2: ltrb 格式的优势是什么？

**A:**
- **直观**：直接表示边界到锚点的距离
- **稳定**：相对于锚点的偏移，不受图像平移影响
- **高效**：解码简单，只需要加减运算

### Q3: 为什么使用解耦头？

**A:**
- **任务差异**：分类关注"是什么"，回归关注"在哪里"
- **特征需求**：两个任务需要不同的特征表示
- **性能提升**：独立学习各自的特征，精度更高

### Q4: reg_max=16 的选择依据？

**A:**
- **覆盖范围**：16 个离散值覆盖 0-15 个步长单位
- **精度**：每个步长对应 8/16/32 像素，足够精细
- **平衡**：参数量和精度的良好平衡

---

## 性能优化建议

### 1. 调整 reg_max

```python
# 增加精度（适用于大分辨率图像）
head = DetectHead(num_classes=80, reg_max=32)

# 减少计算量（适用于小模型）
head = DetectHead(num_classes=80, reg_max=8)
```

### 2. 优化分类分支

```python
# 减少分类分支的卷积层数
self.cls_convs = nn.ModuleList([
    nn.Sequential(
        Conv(x, x, 3, 1)  # 只使用 1 个卷积
    ) for x in in_channels
])
```

### 3. 使用轻量级激活函数

```python
# 替换 SiLU 为 ReLU（速度更快，精度略降）
from torch.nn import ReLU

self.cls_convs = nn.ModuleList([
    nn.Sequential(
        Conv(x, x, 3, 1, act=ReLU())
    ) for x in in_channels
])
```

---

## 总结

DetectHead 是 YOLOv8 的核心检测模块：

| 组件 | 功能 | 关键特性 |
|------|------|----------|
| DFL | 分布积分 | 将离散分布转换为连续距离 |
| Stems | 特征增强 | 两个 3×3 卷积 |
| 分类分支 | 分类预测 | 2×Conv + 1×1 预测 |
| 回归分支 | 边界框预测 | 2×Conv + 1×1 预测 (4×16 维) |
| 锚点生成 | 位置基准 | 每个特征图位置的中心点 |
| 边界框解码 | 坐标转换 | ltrb + 锚点 → x1y1x2y2 |

**设计理念：**
- **解耦设计**：分类和回归独立处理
- **分布学习**：DFL 提供亚像素级精度
- **无锚点**：基于特征图位置的灵活检测
- **多尺度**：3 个尺度覆盖不同大小的目标

**使用建议：**
- 高精度任务：使用 `reg_max=16` 或更大
- 实时应用：减少分类分支层数
- 小目标检测：关注 P3 的输出
- 大目标检测：关注 P5 的输出
