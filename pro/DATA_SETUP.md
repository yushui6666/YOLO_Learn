# 数据集准备指南

本文档详细说明如何准备和放置数据集。

## 数据集目录结构

数据集应该放在项目根目录下的 `datasets/` 文件夹中（可以自定义位置）。

### 推荐的目录结构

```
项目根目录/
├── datasets/                    # 数据集根目录
│   ├── coco/                   # COCO 格式数据集
│   │   ├── train/
│   │   │   ├── images/         # 训练图像
│   │   │   │   ├── 000000000001.jpg
│   │   │   │   ├── 000000000002.jpg
│   │   │   │   └── ...
│   │   │   └── annotations/   # 训练标注文件
│   │   │       └── instances_train.json
│   │   └── val/
│   │       ├── images/         # 验证图像
│   │       │   ├── 000000000501.jpg
│   │       │   ├── 000000000502.jpg
│   │       │   └── ...
│   │       └── annotations/   # 验证标注文件
│   │           └── instances_val.json
│   │
│   ├── custom/                 # 自定义数据集
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── annotations/
│   │   │       └── instances_train.json
│   │   └── val/
│   │       ├── images/
│   │       └── annotations/
│   │           └── instances_val.json
│   │
│   └── yolo/                   # YOLO 格式数据集
│       ├── train/
│       │   ├── images/
│       │   │   ├── img1.jpg
│       │   │   ├── img2.jpg
│       │   │   └── ...
│       │   └── labels/
│       │       ├── img1.txt
│       │       ├── img2.txt
│       │       └── ...
│       └── val/
│           ├── images/
│           └── labels/
│
├── configs/
├── models/
├── utils/
├── data/
├── train.py
├── evaluate.py
├── infer.py
└── ...
```

## COCO 格式数据集

### 准备步骤

1. **下载 COCO 数据集**

```bash
# 下载 COCO 2017 数据集
cd datasets/coco

# 训练集
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
mv train2017 train/images

# 验证集
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
mv val2017 val/images

# 标注文件
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
mv annotations/instances_train2017.json train/annotations/
mv annotations/instances_val2017.json val/annotations/
```

2. **使用 COCO 数据集训练**

```bash
python train.py \
  --train-data datasets/coco/train \
  --val-data datasets/coco/val \
  --config configs/hyperparameters.yaml
```

### COCO JSON 格式

`instances_train.json` 文件结构：

```json
{
  "images": [
    {
      "id": 1,
      "width": 640,
      "height": 480,
      "file_name": "000000000001.jpg"
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 1,
      "name": "person",
      "supercategory": "person"
    }
  ]
}
```

## 自定义数据集

### 方法一：COCO 格式

将您的数据集组织为 COCO 格式：

```
datasets/custom/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── annotations/
│       └── instances_train.json
└── val/
    ├── images/
    │   ├── image3.jpg
    │   ├── image4.jpg
    │   └── ...
    └── annotations/
        └── instances_val.json
```

创建 COCO 格式的标注文件：

```python
import json
from PIL import Image
import glob

def create_coco_annotation(images_dir, output_json, categories):
    images = []
    annotations = []
    annotation_id = 1
    image_id = 1
    
    # 假设您有对应的标注文件（如 XML、TXT 等）
    # 这里需要根据您的标注格式进行调整
    
    for img_path in glob.glob(f"{images_dir}/*.jpg"):
        img = Image.open(img_path)
        width, height = img.size
        
        # 添加图像信息
        images.append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": img_path.split('/')[-1]
        })
        
        # 添加标注信息（需要根据您的标注格式实现）
        # annotations.append({...})
        
        image_id += 1
    
    # 创建 COCO 格式的 JSON
    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    
    with open(output_json, 'w') as f:
        json.dump(coco_data, f, indent=2)

# 示例使用
categories = [
    {"id": 1, "name": "class1", "supercategory": "object"},
    {"id": 2, "name": "class2", "supercategory": "object"}
]

create_coco_annotation(
    "datasets/custom/train/images",
    "datasets/custom/train/annotations/instances_train.json",
    categories
)
```

### 方法二：YOLO 格式

虽然本实现主要支持 COCO 格式，但您也可以使用 YOLO 格式：

```
datasets/yolo/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── labels/
│       ├── img1.txt
│       ├── img2.txt
│       └── ...
└── val/
    ├── images/
    │   ├── img3.jpg
    │   ├── img4.jpg
    │   └── ...
    └── labels/
        ├── img3.txt
        ├── img4.txt
        └── ...
```

YOLO 标注文件格式（每行一个目标）：
```
class_id center_x center_y width height
```

**注意**：如果使用 YOLO 格式，需要修改 `data/dataset.py` 中的数据加载器。

## 数据集示例

### 示例 1：小规模测试数据集

创建一个小的测试数据集用于快速测试：

```bash
# 创建目录
mkdir -p datasets/test/train/images
mkdir -p datasets/test/train/annotations
mkdir -p datasets/test/val/images
mkdir -p datasets/test/val/annotations
```

然后复制一些图像和对应的标注文件到这些目录。

### 示例 2：VOC 数据集转换

如果您有 VOC 格式的数据集，可以转换为 COCO 格式：

```python
import json
import xml.etree.ElementTree as ET
from pathlib import Path

def voc_to_coco(voc_path, output_path):
    # 实现 VOC 到 COCO 的转换
    pass
```

## 使用数据集

### 训练时指定数据集路径

```bash
# 使用绝对路径
python train.py \
  --train-data /absolute/path/to/datasets/custom/train \
  --val-data /absolute/path/to/datasets/custom/val

# 使用相对路径
python train.py \
  --train-data datasets/coco/train \
  --val-data datasets/coco/val
```

### 评估时指定数据集路径

```bash
python evaluate.py \
  --weights runs/train/best_model.pt \
  --data datasets/coco/val
```

### 推理时指定图像/视频路径

```bash
# 单张图像
python infer.py \
  --weights runs/train/best_model.pt \
  --source datasets/test/image.jpg

# 图像目录
python infer.py \
  --weights runs/train/best_model.pt \
  --source datasets/test/images/

# 视频
python infer.py \
  --weights runs/train/best_model.pt \
  --source datasets/test/video.mp4
```

## 数据集验证

在开始训练前，验证数据集是否正确：

```python
import json
from pathlib import Path

def validate_coco_dataset(data_path):
    annotations_file = Path(data_path) / "annotations" / "instances_train.json"
    images_dir = Path(data_path) / "images"
    
    # 检查文件是否存在
    if not annotations_file.exists():
        print(f"❌ 标注文件不存在: {annotations_file}")
        return False
    
    if not images_dir.exists():
        print(f"❌ 图像目录不存在: {images_dir}")
        return False
    
    # 加载标注文件
    with open(annotations_file, 'r') as f:
        data = json.load(f)
    
    # 检查图像数量
    num_images = len(data.get('images', []))
    num_annotations = len(data.get('annotations', []))
    num_categories = len(data.get('categories', []))
    
    print(f"✅ 数据集验证通过")
    print(f"   图像数量: {num_images}")
    print(f"   标注数量: {num_annotations}")
    print(f"   类别数量: {num_categories}")
    
    # 检查图像文件是否存在
    missing_images = []
    for img_info in data.get('images', []):
        img_path = images_dir / img_info['file_name']
        if not img_path.exists():
            missing_images.append(img_info['file_name'])
    
    if missing_images:
        print(f"❌ 缺失 {len(missing_images)} 张图像")
        for img in missing_images[:10]:
            print(f"   - {img}")
        if len(missing_images) > 10:
            print(f"   ... 还有 {len(missing_images) - 10} 张")
        return False
    
    return True

# 验证训练集
print("验证训练集:")
validate_coco_dataset("datasets/coco/train")

print("\n验证验证集:")
validate_coco_dataset("datasets/coco/val")
```

## 常见问题

### 1. 数据集应该放在哪里？

推荐放在 `datasets/` 文件夹中，但也可以放在任何位置。使用时通过 `--train-data` 和 `--val-data` 参数指定路径。

### 2. 支持哪些数据集格式？

目前主要支持 COCO 格式。如果需要其他格式（如 YOLO、VOC），需要进行格式转换。

### 3. 图像大小有限制吗？

没有严格限制，但训练时会根据配置的 `image_size` 进行缩放。建议使用相近的图像尺寸以获得更好的效果。

### 4. 如何使用自己的数据集？

1. 将数据集组织为 COCO 格式
2. 创建标注文件（JSON 格式）
3. 放置在 `datasets/` 或自定义目录
4. 在 `configs/hyperparameters.yaml` 中设置 `num_classes`
5. 开始训练

### 5. 数据集路径必须是绝对路径吗？

不是，可以使用相对路径。但建议使用绝对路径以避免路径问题。

## 更多资源

- [COCO 数据集官网](https://cocodataset.org/)
- [COCO 数据格式说明](https://cocodataset.org/#format-data)
- [LabelImg - 图像标注工具](https://github.com/heartexlabs/labelImg)
- [Label Studio - 标注平台](https://labelstud.io/)
