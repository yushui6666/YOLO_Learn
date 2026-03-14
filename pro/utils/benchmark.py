"""
YOLOv8 基准测试工具
用于测量模型的推理速度、参数量、GFLOPs、显存占用等资源指标
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    计算模型的参数量
    
    Args:
        model: PyTorch 模型
        
    Returns:
        包含总参数量、可训练参数量的字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': non_trainable_params,
        'total_params_m': total_params / 1e6,  # 百万单位
        'trainable_params_m': trainable_params / 1e6,
    }


def compute_flops(model: nn.Module, input_size: Tuple[int, int, int] = (3, 640, 640)) -> float:
    """
    计算模型的 FLOPs (Floating Point Operations)
    
    Args:
        model: PyTorch 模型
        input_size: 输入尺寸 (C, H, W)
        
    Returns:
        GFLOPs (十亿次浮点运算)
    """
    from thop import profile
    
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size, device=device)
    
    # 使用 thop 库计算 FLOPs
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # flops 是单次推理的运算量，profile 返回的是实际值的 2 倍（因为包含了反向传播）
    gflops = flops / 1e9 / 2  # 转换为 GFLOPs
    
    return gflops


def compute_flops_manual(model: nn.Module, input_size: Tuple[int, int, int] = (3, 640, 640)) -> Dict[str, float]:
    """
    手动估算模型的 FLOPs（不依赖 thop 库的备用方案）
    
    Args:
        model: PyTorch 模型
        input_size: 输入尺寸 (C, H, W)
        
    Returns:
        包含 GFLOPs 估算值的字典
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size, device=device)
    
    total_flops = 0
    
    def count_conv_flops(module: nn.Conv2d, x: torch.Tensor) -> int:
        """计算卷积层的 FLOPs"""
        out_h, out_w = module.output_size if hasattr(module, 'output_size') else (x.shape[2], x.shape[3])
        kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels * module.out_channels
        # 每个输出位置的 FLOPs（乘法和加法）
        return 2 * kernel_flops * out_h * out_w
    
    def count_bn_flops(module: nn.BatchNorm2d, x: torch.Tensor) -> int:
        """计算 BN 层的 FLOPs"""
        return 4 * x.numel()  # 减均值、除方差、乘 gamma、加 beta
    
    def count_linear_flops(module: nn.Linear, x: torch.Tensor) -> int:
        """计算全连接层的 FLOPs"""
        return 2 * module.in_features * module.out_features
    
    def count_activation_flops(module: nn.Module, x: torch.Tensor) -> int:
        """计算激活函数的 FLOPs"""
        if isinstance(module, (nn.ReLU, nn.SiLU, nn.Sigmoid)):
            return x.numel()  # 每个元素一次操作
        elif isinstance(module, nn.Softmax):
            return 3 * x.numel()  # exp + sum + divide
        return 0
    
    # 注册 hook 来统计 FLOPs
    def hook_fn(module, inputs, outputs):
        nonlocal total_flops
        if isinstance(module, nn.Conv2d):
            total_flops += count_conv_flops(module, inputs[0])
        elif isinstance(module, nn.BatchNorm2d):
            total_flops += count_bn_flops(module, inputs[0])
        elif isinstance(module, nn.Linear):
            total_flops += count_linear_flops(module, inputs[0])
        elif isinstance(module, (nn.ReLU, nn.SiLU, nn.Sigmoid, nn.Softmax)):
            total_flops += count_activation_flops(module, inputs[0])
    
    # 注册 hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear, 
                               nn.ReLU, nn.SiLU, nn.Sigmoid, nn.Softmax)):
            hooks.append(module.register_forward_hook(hook_fn))
    
    # 前向传播
    with torch.no_grad():
        model.eval()
        _ = model(dummy_input)
    
    # 移除 hooks
    for hook in hooks:
        hook.remove()
    
    gflops = total_flops / 1e9
    
    return {
        'gflops_manual': gflops,
        'total_flops': total_flops
    }


def measure_inference_time(
    model: nn.Module, 
    input_size: Tuple[int, int] = (640, 640),
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    测量模型的推理时间（拆分预处理、推理、后处理）
    
    Args:
        model: YOLOv8 模型
        input_size: 输入图像尺寸 (H, W)
        num_iterations: 测试迭代次数
        warmup_iterations: 预热迭代次数
        device: 设备 ('cuda' 或 'cpu')
        
    Returns:
        包含各项时间指标的字典
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    # 创建假输入
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(dummy_input)
    
    # 同步 CUDA（如果使用 GPU）
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 时间记录
    total_times = []
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    
    # 开始计时
    for _ in range(num_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 总时间
        start_total = time.perf_counter()
        
        # 预处理（这里主要是 tensor 传输，实际预处理在 dataloader 中）
        start_preprocess = time.perf_counter()
        input_tensor = dummy_input.to(device)
        if device == 'cuda':
            torch.cuda.synchronize()
        end_preprocess = time.perf_counter()
        preprocess_times.append(end_preprocess - start_preprocess)
        
        # 推理
        start_inference = time.perf_counter()
        with torch.no_grad():
            outputs = model(input_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()
        end_inference = time.perf_counter()
        inference_times.append(end_inference - start_inference)
        
        # 后处理（解码预测结果）
        start_postprocess = time.perf_counter()
        predictions = model.decode_predictions(outputs, img_h=input_size[0], img_w=input_size[1])
        if device == 'cuda':
            torch.cuda.synchronize()
        end_postprocess = time.perf_counter()
        postprocess_times.append(end_postprocess - start_postprocess)
        
        end_total = time.perf_counter()
        total_times.append(end_total - start_total)
    
    # 计算统计值
    results = {
        'avg_total_time_ms': np.mean(total_times) * 1000,
        'std_total_time_ms': np.std(total_times) * 1000,
        'avg_preprocess_time_ms': np.mean(preprocess_times) * 1000,
        'std_preprocess_time_ms': np.std(preprocess_times) * 1000,
        'avg_inference_time_ms': np.mean(inference_times) * 1000,
        'std_inference_time_ms': np.std(inference_times) * 1000,
        'avg_postprocess_time_ms': np.mean(postprocess_times) * 1000,
        'std_postprocess_time_ms': np.std(postprocess_times) * 1000,
        'fps': 1.0 / np.mean(total_times),  # 每秒帧数
        'num_iterations': num_iterations,
        'device': device,
        'input_size': input_size,
    }
    
    return results


def measure_memory_usage(
    model: nn.Module,
    input_size: Tuple[int, int] = (640, 640),
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    测量模型的显存/内存占用
    
    Args:
        model: YOLOv8 模型
        input_size: 输入图像尺寸 (H, W)
        device: 设备 ('cuda' 或 'cpu')
        
    Returns:
        包含内存使用指标的字典
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    results = {
        'device': device,
    }
    
    if device == 'cuda':
        # 清空 CUDA 缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 模型权重占用
        model_memory = sum(p.element_size() * p.numel() for p in model.parameters())
        results['model_weights_mb'] = model_memory / (1024 ** 2)
        
        # 前向传播的峰值内存
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        peak_memory = torch.cuda.max_memory_allocated()
        results['peak_memory_mb'] = peak_memory / (1024 ** 2)
        results['peak_memory_gb'] = peak_memory / (1024 ** 3)
        
        # 清空缓存
        torch.cuda.empty_cache()
    else:
        # CPU 内存测量（使用 tracemalloc）
        import tracemalloc
        tracemalloc.start()
        
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1], device=device)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        results['peak_memory_mb'] = peak / (1024 ** 2)
        results['model_weights_mb'] = sum(p.element_size() * p.numel() for p in model.parameters()) / (1024 ** 2)
    
    return results


def get_model_file_size(weights_path: str) -> Dict[str, float]:
    """
    获取模型文件的大小
    
    Args:
        weights_path: 模型权重文件路径
        
    Returns:
        包含文件大小（MB）的字典
    """
    path = Path(weights_path)
    if not path.exists():
        return {'error': f'File not found: {weights_path}'}
    
    size_bytes = path.stat().st_size
    
    return {
        'size_bytes': size_bytes,
        'size_kb': size_bytes / 1024,
        'size_mb': size_bytes / (1024 ** 2),
        'size_gb': size_bytes / (1024 ** 3),
    }


def run_full_benchmark(
    model: nn.Module,
    weights_path: Optional[str] = None,
    input_size: Tuple[int, int] = (640, 640),
    device: Optional[str] = None
) -> Dict:
    """
    运行完整的基准测试
    
    Args:
        model: YOLOv8 模型
        weights_path: 模型权重文件路径（可选）
        input_size: 输入图像尺寸 (H, W)
        device: 设备 ('cuda' 或 'cpu')
        
    Returns:
        包含所有基准测试结果的字典
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Running benchmark on {device}...")
    print(f"Input size: {input_size}")
    
    results = {
        'device': device,
        'input_size': input_size,
    }
    
    # 1. 参数量
    print("\n[1/5] Counting parameters...")
    param_results = count_parameters(model)
    results['parameters'] = param_results
    print(f"  Total params: {param_results['total_params_m']:.2f}M")
    
    # 2. GFLOPs
    print("\n[2/5] Computing GFLOPs...")
    try:
        gflops = compute_flops(model, (3, input_size[0], input_size[1]))
        results['gflops'] = gflops
        print(f"  GFLOPs: {gflops:.2f}")
    except ImportError:
        print("  thop not installed, using manual estimation...")
        manual_flops = compute_flops_manual(model, (3, input_size[0], input_size[1]))
        results['gflops_manual'] = manual_flops
        print(f"  GFLOPs (estimated): {manual_flops['gflops_manual']:.2f}")
    
    # 3. 推理时间
    print("\n[3/5] Measuring inference time...")
    time_results = measure_inference_time(model, input_size, device=device)
    results['inference_time'] = time_results
    print(f"  Avg inference time: {time_results['avg_inference_time_ms']:.2f} ms")
    print(f"  FPS: {time_results['fps']:.2f}")
    
    # 4. 内存占用
    print("\n[4/5] Measuring memory usage...")
    memory_results = measure_memory_usage(model, input_size, device=device)
    results['memory'] = memory_results
    print(f"  Peak memory: {memory_results['peak_memory_mb']:.2f} MB")
    
    # 5. 模型文件大小
    if weights_path:
        print("\n[5/5] Getting model file size...")
        file_results = get_model_file_size(weights_path)
        results['file_size'] = file_results
        if 'size_mb' in file_results:
            print(f"  File size: {file_results['size_mb']:.2f} MB")
    
    return results


def print_benchmark_report(results: Dict) -> str:
    """
    打印基准测试报告
    
    Args:
        results: run_full_benchmark 返回的结果字典
        
    Returns:
        格式化的报告字符串
    """
    report = []
    report.append("=" * 60)
    report.append("YOLOv8 模型基准测试报告")
    report.append("=" * 60)
    report.append("")
    
    # 基本信息
    report.append("【基本信息】")
    report.append(f"  设备：{results['device']}")
    report.append(f"  输入尺寸：{results['input_size'][0]}x{results['input_size'][1]}")
    report.append("")
    
    # 模型复杂度
    report.append("【模型复杂度】")
    if 'parameters' in results:
        p = results['parameters']
        report.append(f"  参数量：{p['total_params_m']:.2f}M ({p['total_params']:,})")
        report.append(f"    - 可训练：{p['trainable_params_m']:.2f}M")
        report.append(f"    - 不可训练：{p['non_trainable_params']:,}")
    
    if 'gflops' in results:
        report.append(f"  GFLOPs: {results['gflops']:.2f}")
    elif 'gflops_manual' in results:
        report.append(f"  GFLOPs (估算): {results['gflops_manual']['gflops_manual']:.2f}")
    report.append("")
    
    # 推理速度
    report.append("【推理速度】")
    if 'inference_time' in results:
        t = results['inference_time']
        report.append(f"  FPS: {t['fps']:.2f}")
        report.append(f"  平均总耗时：{t['avg_total_time_ms']:.2f} ms (±{t['std_total_time_ms']:.2f})")
        report.append(f"  耗时分解:")
        report.append(f"    - 预处理：{t['avg_preprocess_time_ms']:.2f} ms")
        report.append(f"    - 推理：{t['avg_inference_time_ms']:.2f} ms")
        report.append(f"    - 后处理：{t['avg_postprocess_time_ms']:.2f} ms")
    report.append("")
    
    # 资源占用
    report.append("【资源占用】")
    if 'memory' in results:
        m = results['memory']
        report.append(f"  模型权重：{m['model_weights_mb']:.2f} MB")
        report.append(f"  峰值内存：{m['peak_memory_mb']:.2f} MB")
    
    if 'file_size' in results and 'size_mb' in results['file_size']:
        report.append(f"  模型文件：{results['file_size']['size_mb']:.2f} MB")
    report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)


if __name__ == '__main__':
    # 测试代码
    from models.yolov8 import create_model
    
    # 创建测试模型
    model = create_model(
        num_classes=80,
        width_multiple=0.5,
        depth_multiple=0.67,
        backbone_name='CSPDarknet'
    )
    
    # 运行基准测试
    results = run_full_benchmark(model, input_size=(640, 640))
    
    # 打印报告
    report = print_benchmark_report(results)
    print(report)