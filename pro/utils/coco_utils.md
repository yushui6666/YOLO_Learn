# utils/coco_utils.py - COCO 数据集加载器详解

## 📌 功能简介

该文件实现了 COCO（Common Objects in Context）数据集的加载和处理，包括：
- COCO 格式标注文件的解析
- COCO 格式到 YOLO 格式的标注转换
- 数据加载和批次整理
- 类别映射管理

这是训练 YOLOv8 模型的数据输入核心模块。

---

## 🏗️ 核心架构

```
COCODataset (PyTorch Dataset)
├── __init__()              # 初始化，加载 COCO 标注
├── __len__()               # 返回数据集大小
├── __getitem__()           # 获取单个样本
├── get_class_names()       # 获取类别名称列表

辅助函数：
└── collate_fn()            # 批次整理函数
└── create_coco_dataset()   # 工厂函数创建数据集
```

---

## 🔍 核心概念

### 1. COCO 数据集格式

COCO 格式是一种广泛使用的目标检测数据集格式：

**标注文件结构**（JSON）：
```json
{
  "images": [
    {
      "id": 1,
      "file_name": "000001.jpg",
      "width": 640,
      "height": 480
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person"
    }
  ]
}
```

### 2. YOLO 格式标注

YOLO 格式使用归一化的中心坐标和宽高：

```
[class_id, x_center, y_center, width, height]
```

其中所有坐标值都在 [0, 1] 范围内。

### 3. 格式转换

**COCO → YOLO**：
- `bbox = [x, y, w, h]` (像素，左上角坐标)
- `YOLO = [x_center, y_center, w, h]` (归一化，中心坐标)

转换公式：
```python
x_center = (x + w/2) / image_width
y_center = (y + h/2) / image_height
width = w / image_width
height = h / image_height
```

---

## 💻 代码解析

### 1. COCODataset 类初始化

```python
class COCODataset:
    def __init__(self, img_dir: str, ann_file: str, img_size: int = 640, 
                 transform=None, is_training: bool = True):
        """
        Args:
            img_dir: 图像目录
            ann_file: COCO 标注 JSON 文件路径
            img_size: 目标图像尺寸
            transform: 数据增强函数
            is_training: 是否为训练模式
        """
        self.img_dir = img_dir
        self.img_size = img_size
        self.transform = transform
        self.is_training = is_training
        
        # 加载 COCO 标注
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        
        # 类别映射
        self.cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_class_id = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.class_id_to_cat_id = {i: cat_id for i, cat_id in enumerate(self.cat_ids)}
        self.num_classes = len(self.cat_ids)
```

**关键点**：
- 使用 `pycocotools` 库解析 COCO 格式
- 建立 `cat_id_to_class_id` 映射，确保类别 ID 连续
- `self.ids` 保存所有图像 ID 的有序列表

---

### 2. __getitem__ 方法

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, np.ndarray]:
    """
    获取单个样本
    Returns:
        (image, targets) 其中 targets 是 (N, 6) [class, x, y, w, h, conf]
    """
    # 获取图像信息
    img_id = self.ids[idx]
    img_info = self.coco.loadImgs(img_id)[0]
    
    # 加载图像
    img_path = os.path.join(self.img_dir, img_info['file_name'])
    img = cv2.imread(img_path)
    
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 获取标注
    ann_ids = self.coco.getAnnIds(imgIds=img_id)
    anns = self.coco.loadAnns(ann_ids)
```

**关键点**：
- 返回的 targets 格式：`(N, 6)` = `[class, x, y, w, h, conf]`
- `conf` 字段为 1.0，表示这是真实标注
- 图像从 BGR 转换为 RGB 格式

---

### 3. COCO 到 YOLO 格式转换

```python
# 转换到 YOLO 格式 [class, x_center, y_center, width, height]
for ann in anns:
    if ann['iscrowd']:
        continue  # 跳过人群标注
    
    bbox = ann['bbox']  # [x, y, w, h] in pixels
    cat_id = ann['category_id']
    
    # 转换到 YOLO 格式
    x_center = (bbox[0] + bbox[2] / 2) / img_info['width']
    y_center = (bbox[1] + bbox[3] / 2) / img_info['height']
    width = bbox[2] / img_info['width']
    height = bbox[3] / img_info['height']
    
    # 裁剪到 [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    # 跳过无效框
    if width < 0.01 or height < 0.01:
        continue
    
    # 转换类别 ID
    class_id = self.cat_id_to_class_id.get(cat_id, -1)
    if class_id == -1:
        continue
    
    targets.append([class_id, x_center, y_center, width, height, 1.0])
```

**关键点**：
- **坐标转换**：从左上角坐标转换为中心坐标
- **归一化**：所有坐标除以图像宽高
- **裁剪**：使用 `max(0, min(1, x))` 确保坐标在有效范围
- **过滤**：跳过过小的目标框（< 1% 图像尺寸）

---

### 4. 数据增强应用

```python
# 应用数据增强（仅训练时）
if self.transform and self.is_training:
    img, targets = self.transform(img, targets)

# 调整图像到目标尺寸
img = cv2.resize(img, (self.img_size, self.img_size))

# 转换为 float 并归一化
img = img.astype(np.float32) / 255.0

# 转换到 CHW 格式并转为 PyTorch tensor
img = img.transpose(2, 0, 1)
img = torch.from_numpy(img).float()

return img, targets
```

**关键点**：
- 数据增强仅在训练时应用
- 图像归一化到 [0, 1] 范围
- 从 HWC 转换为 CHW 格式（PyTorch 标准）
- targets 保持为 numpy 数组（因为每个图像的目标数量不同）

---

### 5. collate_fn 批次整理函数

```python
def collate_fn(batch: List[Tuple[torch.Tensor, np.ndarray]]) -> Tuple[torch.Tensor, List[np.ndarray], List[str]]:
    """
    自定义批次整理函数
    Args:
        batch: (image, targets) 元组列表
    Returns:
        (images, targets_batch, image_ids)
    """
    images = []
    targets_batch = []
    image_ids = []
    
    for idx, (img, targets) in enumerate(batch):
        images.append(img)
        targets_batch.append(targets)
        image_ids.append(str(idx))
    
    # 堆叠图像为批次 tensor
    images = torch.stack(images, dim=0)
    
    return images, targets_batch, image_ids
```

**关键点**：
- **图像可以堆叠**：因为都是相同尺寸 `(C, H, W)`
- **targets 不能堆叠**：因为每个图像的目标数量不同
- 返回 `image_ids` 用于追踪每个样本的来源

---

### 6. 类别名称获取

```python
def get_class_names(self) -> List[str]:
    """获取类别名称列表"""
    cats = self.coco.loadCats(self.cat_ids)
    return [cat['name'] for cat in cats]
```

**使用场景**：
- 推理时将类别 ID 转换为可读名称
- 评估时生成分类报告
- 可视化时在目标框上标注类别

---

## 🎯 学习要点

### 1. 数据加载流程

```
原始图像 (HWC, BGR)
    ↓
加载图像
    ↓
BGR → RGB 转换
    ↓
解析 COCO 标注
    ↓
COCO 格式 → YOLO 格式转换
    ↓
应用数据增强（训练时）
    ↓
调整到目标尺寸
    ↓
归一化到 [0, 1]
    ↓
HWC → CHW 转换
    ↓
转为 PyTorch tensor
```

---

### 2. 坐标系统

**COCO 格式**：
- 原点：图像左上角
- 坐标：左上角 (x, y)
- 尺寸：像素单位

**YOLO 格式**：
- 原点：图像左上角
- 坐标：中心点 (x_center, y_center)
- 尺寸：归一化 [0, 1]

**转换代码**：
```python
# COCO: [x, y, w, h] (左上角, 像素)
# YOLO: [x_center, y_center, w, h] (中心点, 归一化)

x_center = (x + w/2) / image_width
y_center = (y + h/2) / image_height
width = w / image_width
height = h / image_height
```

---

### 3. 类别 ID 映射

COCO 类别 ID 可能不连续（如 1, 3, 5...），需要映射到连续 ID：

```python
# COCO 类别 ID
cat_ids = [1, 3, 5, 7, 9]  # 可能不连续

# 映射到连续的 class_id (0, 1, 2, 3, 4)
cat_id_to_class_id = {
    1: 0,
    3: 1,
    5: 2,
    7: 3,
    9: 4
}

# 反向映射（推理时使用）
class_id_to_cat_id = {
    0: 1,
    1: 3,
    2: 5,
    3: 7,
    4: 9
}
```

**重要性**：
- 模型输出类别索引从 0 开始
- 连续的类别 ID 提高模型效率
- 反向映射用于输出 COCO 格式结果

---

### 4. 数据增强时机

```python
if self.transform and self.is_training:
    img, targets = self.transform(img, targets)
```

**关键点**：
- **训练时**：应用数据增强提升泛化能力
- **验证/测试时**：不应用增强，保持原始数据
- **推理时**：不应用增强，确保结果一致性

---

## 📊 使用示例

### 1. 创建数据集

```python
from utils.coco_utils import COCODataset, collate_fn
import torch.utils.data as data

# 创建训练集
train_dataset = COCODataset(
    img_dir='data/coco/train2017',
    ann_file='data/coco/annotations/instances_train2017.json',
    img_size=640,
    transform=augmentation_transform,  # 可选的数据增强
    is_training=True
)

# 创建验证集
val_dataset = COCODataset(
    img_dir='data/coco/val2017',
    ann_file='data/coco/annotations/instances_val2017.json',
    img_size=640,
    transform=None,
    is_training=False
)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"类别数量: {train_dataset.num_classes}")
print(f"类别名称: {train_dataset.get_class_names()}")
```

---

### 2. 创建数据加载器

```python
# 训练数据加载器
train_loader = data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,           # 训练时打乱数据
    num_workers=4,         # 多进程加载
    collate_fn=collate_fn, # 自定义批次整理
    pin_memory=True,       # 加速 GPU 传输
    drop_last=True         # 丢弃最后一个不完整的批次
)

# 验证数据加载器
val_loader = data.DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,         # 验证时不打乱
    num_workers=4,
    collate_fn=collate_fn,
    pin_memory=True,
    drop_last=False        # 保留所有数据
)
```

---

### 3. 迭代数据

```python
# 训练循环
for epoch in range(num_epochs):
    for batch_idx, (images, targets, image_ids) in enumerate(train_loader):
        # images: (B, 3, 640, 640)
        # targets: List[(N_i, 6)] 每个图像的目标框
        # image_ids: List[str] 图像 ID 列表
        
        # 将数据移到 GPU
        images = images.to(device)
        targets = [t.to(device) for t in targets]
        
        # 前向传播
        cls_outputs, box_outputs = model(images)
        
        # 计算损失
        loss_dict = loss_fn((cls_outputs, box_outputs), targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, "
                  f"Loss: {loss_dict['total_loss'].item():.4f}")
```

---

### 4. 可视化样本

```python
import matplotlib.pyplot as plt
import cv2

# 获取一个样本
image, targets = val_dataset[0]

# 转换回 numpy 格式
img_np = image.numpy().transpose(1, 2, 0)  # CHW -> HWC
img_np = (img_np * 255).astype(np.uint8)

# 绘制目标框
for box in targets:
    class_id, x, y, w, h, conf = box
    x, y, w, h = x * 640, y * 640, w * 640, h * 640
    
    # 转换到 x1, y1, x2, y2 格式
    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
    
    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_np, f"{class_id}", (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 显示图像
plt.figure(figsize=(10, 10))
plt.imshow(img_np)
plt.axis('off')
plt.show()
```

---

## 🔧 调试技巧

### 1. 检查数据加载

```python
# 检查数据集大小
print(f"数据集大小: {len(dataset)}")

# 获取第一个样本
image, targets = dataset[0]
print(f"图像形状: {image.shape}")  # (3, 640, 640)
print(f"目标框数量: {len(targets)}")
print(f"目标框形状: {targets.shape}")  # (N, 6)
print(f"目标框示例:\n{targets[:3]}")
```

---

### 2. 检查类别分布

```python
from collections import Counter

# 统计每个类别的目标数量
class_counter = Counter()

for idx in range(len(dataset)):
    _, targets = dataset[idx]
    class_ids = targets[:, 0].astype(int)
    class_counter.update(class_ids)

# 打印类别分布
print("类别分布:")
for class_id, count in sorted(class_counter.items()):
    class_name = dataset.get_class_names()[class_id]
    print(f"  {class_name} (ID={class_id}): {count} 个目标")
```

---

### 3. 验证坐标转换

```python
# 手动验证 COCO 到 YOLO 的转换
img_info = dataset.coco.loadImgs(img_id)[0]
ann = dataset.coco.loadAnns(ann_ids)[0]

# COCO 格式
x_coco, y_coco, w_coco, h_coco = ann['bbox']
print(f"COCO 格式: x={x_coco}, y={y_coco}, w={w_coco}, h={h_coco}")

# YOLO 格式
x_center = (x_coco + w_coco/2) / img_info['width']
y_center = (y_coco + h_coco/2) / img_info['height']
width = w_coco / img_info['width']
height = h_coco / img_info['height']
print(f"YOLO 格式: x={x_center:.4f}, y={y_center:.4f}, "
      f"w={width:.4f}, h={height:.4f}")
```

---

## ⚠️ 注意事项

1. **内存管理**：使用 `num_workers` 时注意内存占用
2. **数据格式**：确保 COCO 标注文件格式正确
3. **图像路径**：检查 `img_dir` 和标注文件中的路径是否匹配
4. **类别映射**：验证类别 ID 映射是否正确
5. **批次大小**：根据 GPU 内存调整 batch_size
6. **数据增强**：验证时不要应用数据增强

---

## 📚 参考资料

- COCO 数据集官网：https://cocodataset.org/
- pycocotools 文档：https://github.com/cocodataset/cocoapi
- PyTorch DataLoader：https://pytorch.org/docs/stable/data.html
- COCO 格式说明：https://cocodataset.org/#format-data
