import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2


class MetricsCalculator:
    """
    Calculate detection metrics including mAP, precision, recall, F1,
    and AP for small/medium/large objects.
    """

    def __init__(self, num_classes=80,
                 iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.reset()

    def reset(self):
        """Reset all stored predictions and ground truths."""
        self.predictions = defaultdict(list)       # image_id -> list[{'box','label','score'}]
        self.ground_truths = defaultdict(list)     # image_id -> list[{'box','label',...}]
        self.num_gt_per_class = defaultdict(int)   # class_id -> count

    def update(self, pred_boxes, pred_labels, pred_scores,
               gt_boxes, gt_labels, image_ids):
        """
        Update internal buffers with a batch of predictions and ground truths.

        Args:
            pred_boxes: list[np.ndarray], each (Ni, 4)  [x1,y1,x2,y2] in pixels
            pred_labels: list[np.ndarray], each (Ni,)
            pred_scores: list[np.ndarray], each (Ni,)
            gt_boxes: list[np.ndarray], each (Mi, 4)
            gt_labels: list[np.ndarray], each (Mi,)
            image_ids: list-like, len == batch_size, each is unique image id
        """
        for i, img_id in enumerate(image_ids):
            image_id = str(img_id)

            # store predictions
            for box, label, score in zip(pred_boxes[i], pred_labels[i], pred_scores[i]):
                self.predictions[image_id].append({
                    'box': box,
                    'label': int(label),
                    'score': float(score),
                })

            # store ground truths
            for box, label in zip(gt_boxes[i], gt_labels[i]):
                self.ground_truths[image_id].append({
                    'box': box,
                    'label': int(label),
                    'detected': False,   # local matching flag (will be reset in compute_xxx)
                })
                self.num_gt_per_class[int(label)] += 1

    def compute_iou(self, box1, box2):
        """
        Compute IoU between two boxes
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
    def compute_iou_matrix(self, boxes1, boxes2):
        """
        向量化计算两组框的 IoU。
        boxes1: (N, 4)
        boxes2: (M, 4)
        返回: (N, M) 的 IoU 矩阵（numpy 数组）
        """
        boxes1 = np.asarray(boxes1, dtype=np.float32)
        boxes2 = np.asarray(boxes2, dtype=np.float32)

        if boxes1.size == 0 or boxes2.size == 0:
            return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

        # (N, 1, 4) 和 (1, M, 4) 通过广播得到 (N, M, 4)
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        inter_w = np.clip(x2 - x1, 0, None)
        inter_h = np.clip(y2 - y1, 0, None)
        inter_area = inter_w * inter_h  # (N, M)

        area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0, None) * \
                np.clip(boxes1[:, 3] - boxes1[:, 1], 0, None)  # (N,)
        area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0, None) * \
                np.clip(boxes2[:, 3] - boxes2[:, 1], 0, None)  # (M,)

        union = area1[:, None] + area2[None, :] - inter_area
        iou = inter_area / np.clip(union, 1e-10, None)
        return iou

    def _gather_class_detections(self, class_id, conf_threshold):
        """
        收集某一类别的所有预测和 GT，并带上 image_id，
        方便后面在同一张图里做 IoU 匹配。

        返回:
            preds: list[{'image_id', 'box', 'score'}]
            gts:   dict[image_id -> list[{'box', 'detected'}]]
            num_gt: int
        """
        preds = []
        gts = {}
        num_gt = 0

        # 预测
        for image_id, pred_list in self.predictions.items():
            for p in pred_list:
                if p['label'] != class_id:
                    continue
                if p['score'] < conf_threshold:
                    continue
                preds.append({
                    'image_id': image_id,
                    'box': p['box'],
                    'score': p['score'],
                })

        # GT（本地副本）
        for image_id, gt_list in self.ground_truths.items():
            for gt in gt_list:
                if gt['label'] != class_id:
                    continue
                if image_id not in gts:
                    gts[image_id] = []
                gts[image_id].append({
                    'box': gt['box'],
                    'detected': False,
                })
                num_gt += 1

        return preds, gts, num_gt

    def compute_ap(self, predictions, ground_truths, num_gt, iou_threshold=0.5):
        """
        Compute Average Precision for a single class.

        Args:
            predictions: list[{'image_id','box','score'}]
            ground_truths: dict[image_id -> list[{'box','detected'}]]
            num_gt: number of GT boxes for this class
            iou_threshold: IoU threshold for matching
        """
        if num_gt == 0:
            return 0.0

        # sort by score desc
        predictions = sorted(predictions, key=lambda x: -x['score'])

        tp = np.zeros(len(predictions), dtype=np.float32)
        fp = np.zeros(len(predictions), dtype=np.float32)

        # 先按 image 分组，预先为每张图算好 IoU 矩阵
        from collections import defaultdict
        preds_by_image = defaultdict(list)
        for idx, pred in enumerate(predictions):
            preds_by_image[pred['image_id']].append(idx)

        iou_matrices = {}
        for image_id, idxs in preds_by_image.items():
            gt_list = ground_truths.get(image_id, [])
            if len(gt_list) == 0:
                continue
            pred_boxes = np.stack([predictions[i]['box'] for i in idxs], axis=0)
            gt_boxes = np.stack([g['box'] for g in gt_list], axis=0)
            iou_matrices[image_id] = self.compute_iou_matrix(pred_boxes, gt_boxes)

        # 记录每个全局 prediction 在各自图里的行号
        pred_row_index = {}
        for image_id, idxs in preds_by_image.items():
            for row, global_idx in enumerate(idxs):
                pred_row_index[global_idx] = row

        # 逐个 prediction，利用预先算好的 IoU 矩阵做匹配
        for i, pred in enumerate(predictions):
            image_id = pred['image_id']
            gt_list = ground_truths.get(image_id, [])

            if len(gt_list) == 0:
                fp[i] = 1.0
                continue

            iou_mat = iou_matrices.get(image_id)
            if iou_mat is None:
                fp[i] = 1.0
                continue

            row = pred_row_index[i]
            ious = iou_mat[row]  # shape: (num_gt_image,)

            best_idx = int(ious.argmax())
            best_iou = float(ious[best_idx])

            if best_iou >= iou_threshold and not gt_list[best_idx]['detected']:
                tp[i] = 1.0
                gt_list[best_idx]['detected'] = True
            else:
                fp[i] = 1.0

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / (num_gt + 1e-10)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # 11-point interpolation AP
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0.0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0

        return float(ap)


    def compute_box_area(self, box):
        """Compute the area of a bounding box [x1,y1,x2,y2]."""
        return (box[2] - box[0]) * (box[3] - box[1])

    def compute_ap_by_area_range(self, iou_threshold=0.5,
                                 min_area=0.0, max_area=float('inf'),
                                 conf_threshold=0.001):
        """
        在给定面积区间 [min_area, max_area) 内，计算 AP。
        用于 small / medium / large 目标。

        Args:
            iou_threshold: IoU threshold
            min_area: inclusive lower bound of area
            max_area: exclusive upper bound of area
            conf_threshold: confidence threshold for predictions
        """
        ap_per_class = []

        for class_id in range(self.num_classes):
            preds = []
            gts = {}
            num_gt_in_range = 0

            # 预测：按类别 + 置信度 + 面积过滤
            for image_id, pred_list in self.predictions.items():
                for p in pred_list:
                    if p['label'] != class_id:
                        continue
                    if p['score'] < conf_threshold:
                        continue
                    area = self.compute_box_area(p['box'])
                    if min_area <= area < max_area:
                        preds.append({
                            'image_id': image_id,
                            'box': p['box'],
                            'score': p['score'],
                        })

            # GT：按类别 + 面积过滤
            for image_id, gt_list in self.ground_truths.items():
                for gt in gt_list:
                    if gt['label'] != class_id:
                        continue
                    area = self.compute_box_area(gt['box'])
                    if min_area <= area < max_area:
                        if image_id not in gts:
                            gts[image_id] = []
                        gts[image_id].append({
                            'box': gt['box'],
                            'detected': False,
                        })
                        num_gt_in_range += 1

            if num_gt_in_range > 0:
                ap = self.compute_ap(
                    preds,
                    gts,
                    num_gt_in_range,
                    iou_threshold
                )
                ap_per_class.append(ap)

        return float(np.mean(ap_per_class)) if ap_per_class else 0.0

    def compute_per_class_metrics(self, conf_threshold=0.001, iou_threshold=0.5):
        """
        Compute per-class metrics: AP50, precision, recall, F1.
        
        Args:
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for matching
            
        Returns:
            dict with per_class_ap50, per_class_precision, per_class_recall, per_class_f1
        """
        per_class_ap50 = []
        per_class_precision = []
        per_class_recall = []
        per_class_f1 = []
        
        for class_id in range(self.num_classes):
            preds, gts, num_gt = self._gather_class_detections(class_id, conf_threshold)
            
            # 计算 AP@0.5
            gts_copy = {
                img_id: [
                    {'box': gt['box'], 'detected': False}
                    for gt in gt_list
                ]
                for img_id, gt_list in gts.items()
            }
            ap50 = self.compute_ap(preds, gts_copy, num_gt, iou_threshold)
            per_class_ap50.append(float(ap50))
            
            # 计算该类别的 TP, FP, FN
            tp = 0
            fp = 0
            fn = 0
            
            # 重新收集 GT 用于匹配（需要重置 detected 标志）
            gts_for_match = {
                img_id: [
                    {'box': gt['box'], 'detected': False}
                    for gt in gt_list
                ]
                for img_id, gt_list in gts.items()
            }
            
            # 按置信度排序预测
            sorted_preds = sorted(preds, key=lambda x: -x['score'])
            
            for pred in sorted_preds:
                image_id = pred['image_id']
                pred_box = pred['box']
                
                gt_list = gts_for_match.get(image_id, [])
                if len(gt_list) == 0:
                    fp += 1
                    continue
                
                # 找最佳匹配的 GT
                best_iou = 0.0
                best_idx = -1
                for idx, gt in enumerate(gt_list):
                    if gt['detected']:
                        continue
                    iou = self.compute_iou(pred_box, gt['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                
                if best_iou >= iou_threshold and best_idx >= 0:
                    tp += 1
                    gt_list[best_idx]['detected'] = True
                else:
                    fp += 1
            
            # 计算FN（未匹配的GT数量）
            for gt_list in gts_for_match.values():
                for gt in gt_list:
                    if not gt['detected']:
                        fn += 1
            
            # 计算 precision, recall, f1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_precision.append(float(precision))
            per_class_recall.append(float(recall))
            per_class_f1.append(float(f1))
        
        return {
            'per_class_ap50': per_class_ap50,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'per_class_f1': per_class_f1
        }

    def compute_metrics(self, conf_threshold=0.001):
        """
        Compute all metrics:
            - mAP50
            - mAP50-95
            - AP_small, AP_medium, AP_large (by area)
            - precision, recall, F1
        """
        # 先按类别收集好预测和 GT，只做一遍
        per_class_data = []
        for class_id in range(self.num_classes):
            preds, gts, num_gt = self._gather_class_detections(class_id, conf_threshold)
            per_class_data.append((preds, gts, num_gt))

        # mAP over all IoU thresholds
        mAP_values = []
        for iou_thresh in self.iou_thresholds:
            ap_per_class = []

            for class_id, (preds, gts, num_gt) in enumerate(per_class_data):
                # 注意：compute_ap 会修改 gts 里的 'detected' 标志，所以这里要拷贝一份
                gts_copy = {
                    img_id: [
                        {'box': gt['box'], 'detected': False}
                        for gt in gt_list
                    ]
                    for img_id, gt_list in gts.items()
                }
                ap = self.compute_ap(preds, gts_copy, num_gt, iou_thresh)
                ap_per_class.append(ap)

            mAP = np.mean(ap_per_class) if ap_per_class else 0.0
            mAP_values.append(mAP)

        mAP50 = mAP_values[0] if len(mAP_values) > 0 else 0.0
        mAP50_95 = np.mean(mAP_values) if len(mAP_values) > 0 else 0.0

        # 面积分段 AP（如果不太在意，可以直接先注释掉这一块）
        small_max = 32.0 ** 2
        medium_max = 96.0 ** 2

        ap_small = self.compute_ap_by_area_range(
            iou_threshold=0.5,
            min_area=0.0,
            max_area=small_max,
            conf_threshold=conf_threshold,
        )
        ap_medium = self.compute_ap_by_area_range(
            iou_threshold=0.5,
            min_area=small_max,
            max_area=medium_max,
            conf_threshold=conf_threshold,
        )
        ap_large = self.compute_ap_by_area_range(
            iou_threshold=0.5,
            min_area=medium_max,
            max_area=float('inf'),
            conf_threshold=conf_threshold,
        )

        precision, recall, f1 = self.compute_precision_recall_f1(conf_threshold)

        metrics = {
            'mAP50': mAP50,
            'mAP50-95': mAP50_95,
            'AP_small': ap_small,
            'AP_medium': ap_medium,
            'AP_large': ap_large,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
        return metrics


    def compute_precision_recall_f1(self, conf_threshold=0.001):
        """
        Compute precision, recall, and F1 score at IoU=0.5.
        使用向量化 IoU 计算，按图片匹配预测与 GT。
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        iou_thresh = 0.5

        for image_id, gt_list in self.ground_truths.items():
            preds = [
                p for p in self.predictions.get(image_id, [])
                if p['score'] >= conf_threshold
            ]
            gts = list(gt_list)

            num_gt = len(gts)
            num_pred = len(preds)

            if num_pred == 0:
                # 没有预测，这张图所有 GT 都是 FN
                total_fn += num_gt
                continue

            if num_gt == 0:
                # 没有 GT，这张图所有预测都是 FP
                total_fp += num_pred
                continue

            pred_boxes = np.stack([p['box'] for p in preds], axis=0)
            gt_boxes = np.stack([g['box'] for g in gts], axis=0)

            # 所有预测 × 所有 GT 的 IoU
            iou_mat = self.compute_iou_matrix(pred_boxes, gt_boxes)

            # 只允许同类别之间匹配：类别不同的 IoU 置 0
            pred_labels = np.array([p['label'] for p in preds], dtype=np.int32)
            gt_labels = np.array([g['label'] for g in gts], dtype=np.int32)
            label_match = (pred_labels[:, None] == gt_labels[None, :])
            iou_mat = np.where(label_match, iou_mat, 0.0)

            gt_detected = np.zeros(num_gt, dtype=bool)

            # 逐个预测框，选还没匹配过的 GT 中 IoU 最大的一个
            for pi in range(num_pred):
                ious_row = iou_mat[pi].copy()
                # 已被匹配的 GT 不再参与
                ious_row[gt_detected] = 0.0

                best_gt_idx = int(ious_row.argmax())
                best_iou = float(ious_row[best_gt_idx])

                if best_iou >= iou_thresh:
                    total_tp += 1
                    gt_detected[best_gt_idx] = True
                else:
                    total_fp += 1

            total_fn += int((~gt_detected).sum())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        return precision, recall, f1


    def print_metrics(self, metrics):
        """
        Pretty-print metrics dict.
        """
        print("\n" + "=" * 50)
        print("Detection Metrics")
        print("=" * 50)
        print(f"mAP50-95:     {metrics['mAP50-95']:.4f}")
        print(f"AP_small:     {metrics['AP_small']:.4f} (area < 32²)")
        print(f"AP_medium:    {metrics['AP_medium']:.4f} (32² ≤ area < 96²)")
        print(f"AP_large:     {metrics['AP_large']:.4f} (area ≥ 96²)")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1 Score:     {metrics['f1']:.4f}")
        print("=" * 50 + "\n")



def compute_fps(model, input_size=(640, 640), num_iterations=100, device='cuda'):
    """
    Compute inference speed (FPS) of the model
    Args:
        model: YOLOv8 model
        input_size: (height, width) of input
        num_iterations: Number of iterations for timing
        device: Device to run on ('cuda' or 'cpu')
    Returns:
        FPS value
    """
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Timing
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    end_time = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
    
    if device == 'cuda':
        start_time.record()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
    
    if device == 'cuda':
        end_time.record()
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
    else:
        import time
        start = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
        elapsed_time = time.time() - start
    
    fps = num_iterations / elapsed_time
    return fps


if __name__ == '__main__':
    # Test metrics calculator
    metrics_calc = MetricsCalculator(num_classes=5)
    
    # Simulate some predictions and ground truths
    pred_boxes = [
        np.array([[10, 10, 50, 50], [100, 100, 150, 150]]),
        np.array([[20, 20, 60, 60]])
    ]
    pred_labels = [
        np.array([0, 1]),
        np.array([0])
    ]
    pred_scores = [
        np.array([0.9, 0.8]),
        np.array([0.7])
    ]
    
    gt_boxes = [
        np.array([[12, 12, 52, 52], [105, 105, 155, 155]]),
        np.array([[25, 25, 65, 65]])
    ]
    gt_labels = [
        np.array([0, 1]),
        np.array([0])
    ]
    image_ids = [0,1]
    # Update metrics
    metrics_calc.update(pred_boxes, pred_labels, pred_scores,
                        gt_boxes, gt_labels, image_ids)
    
    # Compute and print metrics
    metrics = metrics_calc.compute_metrics(conf_threshold=0.5)
    metrics_calc.print_metrics(metrics)
