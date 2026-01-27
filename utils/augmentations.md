# utils/augmentations.py - 数据增强工具详解

## 📌 功能简介

该文件实现了 YOLOv8 训练所需的数据增强工具，包括：
- HSV 色彩空间增强
- 几何变换（透视变换、旋转、平移、缩放、剪切）
- 翻转增强
- 马赛克增强
- Letterbox 填充缩放

这些增强方法可以有效提升模型的泛化能力和鲁棒性。

---

## 🏗️ 核心架构

```
Augmentations (数据增强类)
├── augment_hsv()          # HSV 色彩增强
├── random_perspective()   # 随机透视变换
├── horizontal_flip()       # 水平翻转
├── vertical_flip()        # 垂直翻转
├── mosaic_augmentation()  # 马赛克增强
└── __call__()             # 统一接口

辅助函数：
└── letterbox()            # 带填充的图像缩放
```

---

## 🔍 核心概念

### 1. HSV 色彩增强

HSV（Hue, Saturation, Value）色彩空间比 RGB 更适合进行色彩增强：
- **Hue（色相）**：颜色类型，范围 [0, 180]
- **Saturation（饱和度）**：颜色鲜艳程度，范围 [0, 255]
- **Value（明度）**：亮度，范围 [0, 255]

**优势**：在 HSV 空间进行增强不会破坏图像的自然色彩关系。

### 2. 透视变换

通过 3×3 变换矩阵实现图像的几何变换：
- **旋转（Rotation）**：围绕中心点旋转
- **平移（Translation）**：沿 x 和 y 方向移动
- **缩放（Scale）**：放大或缩小
- **剪切（Shear）**：沿坐标轴倾斜

### 3. 马赛克增强

将 4 张训练图像拼接成一张图像，特点：
- 增加背景多样性
- 模拟密集目标场景
- 提升模型对小目标检测能力

### 4. Letterbox 填充

保持宽高比的图像缩放方法：
- 计算缩放比例（取宽高的较小比例）
- 添加灰色填充达到目标尺寸
- 避免图像失真

---

## 💻 代码解析

### 1. Augmentations 类初始化

```python
class Augmentations:
    def __init__(self, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, 
                 degrees=0.0, translate=0.1, scale=0.5, 
                 shear=0.0, perspective=0.0,
                 flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0):
```

**参数说明**：
- `hsv_h/s/v`：HSV 增强的强度
- `degrees`：旋转角度范围（±）
- `translate`：平移比例（相对于图像尺寸）
- `scale`：缩放比例范围（1 ± scale）
- `shear`：剪切角度范围
- `perspective`：透视变换强度
- `flipud/fliplr`：垂直/水平翻转概率
- `mosaic`：马赛克增强概率

---

### 2. HSV 色彩增强

```python
def augment_hsv(self, img: np.ndarray) -> np.ndarray:
    # 转换到 HSV 色彩空间
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # 色相增强（循环偏移）
    if self.hsv_h > 0:
        img_hsv[:, :, 0] = (img_hsv[:, :, 0] + 
                           random.uniform(-self.hsv_h, self.hsv_h) * 180) % 180
    
    # 饱和度增强（乘法调整）
    if self.hsv_s > 0:
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * \
                          random.uniform(1 - self.hsv_s, 1 + self.hsv_s)
        img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1], 0, 255)
    
    # 明度增强（乘法调整）
    if self.hsv_v > 0:
        img_hsv[:, :, 2] = img_hsv[:, :, 2] * \
                          random.uniform(1 - self.hsv_v, 1 + self.hsv_v)
        img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2], 0, 255)
```

**关键点**：
- 色相使用**取模运算**实现循环，保持颜色连续性
- 饱和度和明度使用**乘法调整**，更符合人眼感知
- 使用 `np.clip` 将值限制在有效范围

---

### 3. 透视变换

```python
def random_perspective(self, img: np.ndarray, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # 组合变换矩阵：T @ S @ Sh @ R
    # T: Translation, S: Scale, Sh: Shear, R: Rotation
    M_combined = T @ S @ Sh @ R
    
    # 应用到图像
    img = cv2.warpPerspective(img, M_combined, (width, height), 
                              borderMode=cv2.BORDER_REPLICATE)
    
    # 应用到目标框（齐次坐标变换）
    xy_homogeneous = np.column_stack([xy, np.ones(len(xy))])
    xy_transformed = (M_combined @ xy_homogeneous.T).T
```

**关键点**：
- 变换矩阵**右乘顺序**：先旋转，再剪切，再缩放，最后平移
- 使用**齐次坐标**统一处理平移和线性变换
- `BORDER_REPLICATE` 边界填充避免黑边

---

### 4. 马赛克增强

```python
def mosaic_augmentation(self, images: list, targets: list, img_size: int):
    # 随机选择 4 张图像
    indices = random.choices(range(len(images)), k=4)
    
    # 创建 2×2 马赛克
    yc, xc = img_size, img_size  # 马赛克中心
    
    # 每个子图的位置（左上、右上、左下、右下）
    positions = [(0, 0), (xc, 0), (0, yc), (xc, yc)]
    
    # 随机裁剪区域
    y1 = random.randint(0, yc)
    y2 = y1 + img_size
    x1 = random.randint(0, xc)
    x2 = x1 + img_size
    
    mosaic_img = mosaic_img[y1:y2, x1:x2]
```

**关键点**：
- 4 张图像缩放到相同大小后拼接
- 随机裁剪确保每次马赛克都不同
- 目标框坐标需要**统一转换**到马赛克坐标系

---

### 5. 翻转增强

```python
def horizontal_flip(self, img: np.ndarray, targets: np.ndarray):
    if self.fliplr > 0 and random.random() < self.fliplr:
        img = cv2.flip(img, 1)  # 水平翻转
        if len(targets) > 0:
            targets[:, 1] = 1 - targets[:, 1]  # 翻转 x 坐标（归一化）
    return img, targets

def vertical_flip(self, img: np.ndarray, targets: np.ndarray):
    if self.flipud > 0 and random.random() < self.flipud:
        img = cv2.flip(img, 0)  # 垂直翻转
        if len(targets) > 0:
            targets[:, 2] = 1 - targets[:, 2]  # 翻转 y 坐标（归一化）
    return img, targets
```

**关键点**：
- 翻转后坐标转换：`x' = 1 - x`（归一化坐标）
- 需要同步转换目标框的坐标
- 使用概率控制是否应用翻转

---

### 6. Letterbox 填充

```python
def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)):
    # 计算缩放比例（保持宽高比）
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # 计算填充（上下左右对称）
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    
    # 缩放图像
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # 添加灰色填充
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                            cv2.BORDER_CONSTANT, value=color)
```

**关键点**：
- `r` 取最小比例确保图像完整放入目标尺寸
- 填充使用 `(114, 114, 114)` 灰色（COCO 数据集均值）
- 返回缩放比例用于后续坐标转换

---

## 🎯 学习要点

### 1. 目标框坐标变换

所有几何变换都需要同步调整目标框坐标：

```python
# 示例：透视变换中的坐标转换
xy = targets[:, 1:3].copy()  # 归一化坐标 [x, y]
xy[:, 0] *= width             # 转换到像素坐标
xy[:, 1] *= height

xy_homogeneous = np.column_stack([xy, np.ones(len(xy))])  # 齐次坐标
xy_transformed = (M_combined @ xy_homogeneous.T).T        # 应用变换

# 转换回归一化坐标
targets[:, 1:3] = xy_transformed[:, :2]
targets[:, 1] /= width
targets[:, 2] /= height
```

**关键**：
- 先转换到像素坐标应用变换
- 变换后再转换回归一化坐标
- 使用齐次坐标统一处理平移和旋转

---

### 2. 增强顺序

推荐的增强顺序：
1. **HSV 增强**（不影响几何关系）
2. **透视变换**（几何变换组合）
3. **翻转**（简单的几何变换）
4. **马赛克**（多图像融合，需最后执行）

---

### 3. 随机性控制

```python
# 使用概率控制增强是否执行
if random.random() < self.fliplr:  # 50% 概率水平翻转
    img = cv2.flip(img, 1)

# 使用范围控制增强强度
angle = random.uniform(-self.degrees, self.degrees)  # ±degrees 范围
```

**关键**：
- 不是每次都应用所有增强
- 增强强度在一定范围内随机
- 避免过度增强导致失真

---

### 4. 边界处理

```python
# 使用 BORDER_REPLICATE 避免黑边
img = cv2.warpPerspective(img, M, (width, height), 
                          borderMode=cv2.BORDER_REPLICATE)

# 裁剪到有效范围
mosaic_target[:, 1:5] = np.clip(mosaic_target[:, 1:5], 0, 1)
```

**关键**：
- 几何变换可能产生无效坐标
- 使用 `np.clip` 确保坐标在有效范围内
- 过滤掉无效目标框

---

## 📊 使用示例

```python
from utils.augmentations import Augmentations, letterbox
import cv2
import numpy as np

# 创建增强器
aug = Augmentations(
    hsv_h=0.015,      # 色相增强
    hsv_s=0.7,        # 饱和度增强
    hsv_v=0.4,        # 明度增强
    degrees=10.0,     # 旋转 ±10°
    translate=0.1,    # 平移 10%
    scale=0.5,        # 缩放 ±50%
    fliplr=0.5,       # 50% 概率水平翻转
    mosaic=1.0        # 总是应用马赛克
)

# 读取图像
img = cv2.imread('image.jpg')
targets = np.array([[0, 0.5, 0.5, 0.3, 0.3]])  # [class, x, y, w, h]

# 应用增强
img_aug, targets_aug = aug(img, targets)

# Letterbox 缩放到 640×640
img_resized, (scale_x, scale_y) = letterbox(img_aug, (640, 640))

print(f"原始图像: {img.shape}")
print(f"增强后图像: {img_aug.shape}")
print(f"缩放后图像: {img_resized.shape}")
print(f"缩放比例: ({scale_x:.2f}, {scale_y:.2f})")
```

---

## 🔧 调试技巧

### 1. 可视化增强效果

```python
import matplotlib.pyplot as plt

# 创建增强器
aug = Augmentations()

# 对同一图像应用多次增强
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i in range(8):
    img_aug, _ = aug(img.copy(), targets.copy())
    axes[i//4, i%4].imshow(cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB))
    axes[i//4, i%4].set_title(f'Augmentation {i+1}')
plt.show()
```

### 2. 检查目标框变换

```python
# 原始目标框
print(f"原始目标框: {targets}")

# 应用增强
img_aug, targets_aug = aug(img, targets)
print(f"增强后目标框: {targets_aug}")

# 绘制目标框
for box in targets_aug:
    x, y, w, h = box[1:] * [img_aug.shape[1], img_aug.shape[0], 
                             img_aug.shape[1], img_aug.shape[0]]
    cv2.rectangle(img_aug, 
                 (int(x-w/2), int(y-h/2)), 
                 (int(x+w/2), int(y+h/2)), 
                 (0, 255, 0), 2)
```

---

## ⚠️ 注意事项

1. **马赛克增强**：仅在训练时使用，推理时不应用
2. **坐标格式**：确保目标框使用归一化坐标 [0, 1]
3. **增强强度**：根据数据集调整参数，避免过度增强
4. **性能优化**：马赛克增强计算量较大，可适当降低概率
5. **数据一致性**：确保图像和目标框同步变换

---

## 📚 参考资料

- YOLOv8 官方文档：https://docs.ultralytics.com/
- OpenCV 几何变换：https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
- 数据增强综述：https://arxiv.org/abs/1909.11065
