#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模型工具
=======

提供模型相关的实用功能，包括:
- 模型保存和加载
- 模型转换
- 参数统计
- 模型分析
- 模型优化
"""

import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod

class ModelUtils:
    """模型工具类"""
    
    @staticmethod
    def save_model(model: Union[nn.Module, Any], 
                   filepath: Union[str, Path],
                   metadata: Optional[Dict[str, Any]] = None):
        """
        保存模型
        
        Args:
            model: 要保存的模型
            filepath: 保存路径
            metadata: 额外元数据
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else model,
            'model_class': type(model).__name__,
            'model_module': type(model).__module__,
            'metadata': metadata or {}
        }
        
        # 添加PyTorch特有信息
        if hasattr(model, 'config'):
            save_data['model_config'] = model.config
        
        torch.save(save_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    @staticmethod
    def load_model(filepath: Union[str, Path],
                   model_class: Optional[type] = None,
                   **kwargs) -> Any:
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
            model_class: 模型类（用于重建模型）
            kwargs: 传递给模型构造函数的参数
            
        Returns:
            加载的模型
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        try:
            # 尝试加载PyTorch模型
            if filepath.suffix in ['.pth', '.pt']:
                checkpoint = torch.load(filepath, map_location='cpu')
                
                # 如果有模型类信息，尝试重建模型
                if 'model_class' in checkpoint and model_class is None:
                    module_path = checkpoint.get('model_module', '')
                    class_name = checkpoint['model_class']
                    
                    # 动态导入模块
                    if module_path:
                        try:
                            import importlib
                            module = importlib.import_module(module_path)
                            model_class = getattr(module, class_name)
                        except (ImportError, AttributeError):
                            pass
                
                # 创建模型实例
                if model_class:
                    model = model_class(**kwargs)
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # 直接加载状态
                        model = checkpoint.get('model_state_dict', checkpoint)
                else:
                    # 只返回状态字典
                    model = checkpoint.get('model_state_dict', checkpoint)
                
            else:
                # 通用加载方式
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
            
            print(f"模型加载成功: {filepath}")
            return model
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """统计模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size(model: nn.Module) -> float:
        """获取模型大小（MB）"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_all_mb = (param_size + buffer_size) / 1024 / 1024
        return size_all_mb
    
    @staticmethod
    def get_model_info(model: nn.Module) -> Dict[str, Any]:
        """获取模型详细信息"""
        total_params = ModelUtils.count_parameters(model)
        model_size = ModelUtils.get_model_size(model)
        
        layer_info = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM, nn.Transformer)):
                if hasattr(module, 'weight'):
                    layer_info.append({
                        'name': name,
                        'type': type(module).__name__,
                        'parameters': module.weight.numel(),
                        'shape': list(module.weight.shape)
                    })
        
        return {
            'model_class': type(model).__name__,
            'total_parameters': total_params,
            'model_size_mb': model_size,
            'trainable_parameters': total_params,
            'layers': layer_info,
            'device': next(model.parameters()).device.type if next(model.parameters(), None) else 'cpu'
        }
    
    @staticmethod
    def fuse_conv_bn(conv: nn.Module, bn: nn.Module) -> nn.Module:
        """融合卷积层和批归一化层"""
        if not isinstance(conv, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            raise ValueError("conv必须是卷积层")
        if not isinstance(bn, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            raise ValueError("bn必须是批归一化层")
        
        # 融合权重
        fused_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            bias=True
        )
        
        # 计算融合后的权重和偏置
        fused_conv.weight.data = conv.weight.data * (bn.weight.data / torch.sqrt(bn.running_var + bn.eps)).view(-1, 1, 1, 1)
        if conv.bias is not None:
            fused_conv.bias.data = bn.bias.data + (conv.bias.data - bn.running_mean) * bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
        else:
            fused_conv.bias.data = bn.bias.data - bn.running_mean * bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
        
        return fused_conv
    
    @staticmethod
    def quantize_model(model: nn.Module, quantization_type: str = 'dynamic') -> nn.Module:
        """量化模型"""
        import torch.quantization as quantization
        
        model.eval()
        
        if quantization_type == 'dynamic':
            model = quantization.quantize_dynamic(
                model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
            )
        elif quantization_type == 'static':
            model.qconfig = quantization.get_default_qconfig('fbgemm')
            model = quantization.prepare(model)
            model = quantization.convert(model)
        else:
            raise ValueError(f"不支持的量化类型: {quantization_type}")
        
        return model
    
    @staticmethod
    def prune_model(model: nn.Module, pruning_ratio: float = 0.5) -> nn.Module:
        """模型剪枝"""
        import torch.nn.utils.prune as prune
        
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
                prune.remove(module, 'weight')
        
        return model
    
    @staticmethod
    def convert_to_onnx(model: nn.Module, input_shape: Tuple, 
                       output_path: Union[str, Path], opset_version: int = 11):
        """转换为ONNX格式"""
        model.eval()
        
        # 创建示例输入
        dummy_input = torch.randn(input_shape)
        
        # 导出ONNX
        torch.onnx.export(
            model, dummy_input, str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"模型已导出为ONNX格式: {output_path}")
    
    @staticmethod
    def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
        """比较两个模型的参数"""
        info1 = ModelUtils.get_model_info(model1)
        info2 = ModelUtils.get_model_info(model2)
        
        # 比较参数数量
        param_diff = info1['total_parameters'] - info2['total_parameters']
        size_diff = info1['model_size_mb'] - info2['model_size_mb']
        
        # 比较层结构
        layers1 = {layer['type']: layer['parameters'] for layer in info1['layers']}
        layers2 = {layer['type']: layer['parameters'] for layer in info2['layers']}
        
        return {
            'parameter_difference': param_diff,
            'size_difference_mb': size_diff,
            'model1_info': info1,
            'model2_info': info2,
            'layer_comparison': {
                'model1_layers': layers1,
                'model2_layers': layers2,
                'common_layers': set(layers1.keys()) & set(layers2.keys())
            }
        }

class ModelCheckpoint:
    """模型检查点管理"""
    
    def __init__(self, save_dir: Union[str, Path], 
                 save_best_only: bool = True,
                 save_frequency: int = 1,
                 monitor_metric: str = 'val_loss',
                 mode: str = 'min'):
        """
        初始化模型检查点
        
        Args:
            save_dir: 保存目录
            save_best_only: 是否只保存最佳模型
            save_frequency: 保存频率（epoch）
            monitor_metric: 监控指标
            mode: 监控模式 ('min' 或 'max')
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.current_epoch = 0
        
    def save_checkpoint(self, model: nn.Module, epoch: int, 
                       metrics: Dict[str, float], **kwargs):
        """保存检查点"""
        self.current_epoch = epoch
        
        # 检查是否是最佳模型
        current_metric = metrics.get(self.monitor_metric)
        is_best = False
        
        if current_metric is not None:
            if self.mode == 'min' and current_metric < self.best_metric:
                is_best = True
                self.best_metric = current_metric
            elif self.mode == 'max' and current_metric > self.best_metric:
                is_best = True
                self.best_metric = current_metric
        
        # 保存检查点
        should_save = False
        if is_best and self.save_best_only:
            should_save = True
        elif not self.save_best_only and epoch % self.save_frequency == 0:
            should_save = True
        
        if should_save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'best_metric': self.best_metric,
                'config': kwargs.get('config', {}),
                'optimizer_state_dict': kwargs.get('optimizer_state_dict'),
                'scheduler_state_dict': kwargs.get('scheduler_state_dict')
            }
            
            # 保存当前检查点
            checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            
            # 保存最佳检查点
            if is_best:
                best_path = self.save_dir / "best_model.pth"
                torch.save(checkpoint, best_path)
                print(f"保存最佳模型: {checkpoint_path}")
            
            print(f"检查点已保存: {checkpoint_path}")
    
    def load_best_checkpoint(self, model: nn.Module) -> Dict[str, Any]:
        """加载最佳检查点"""
        best_path = self.save_dir / "best_model.pth"
        if not best_path.exists():
            raise FileNotFoundError("最佳检查点不存在")
        
        checkpoint = torch.load(best_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return checkpoint

class ModelProfiler:
    """模型性能分析器"""
    
    @staticmethod
    def profile_model(model: nn.Module, input_shape: Tuple) -> Dict[str, Any]:
        """分析模型性能"""
        from torch.profiler import profile, record_function, ProfilerActivity
        
        model.eval()
        dummy_input = torch.randn(input_shape)
        
        # 使用PyTorch Profiler进行性能分析
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                model(dummy_input)
        
        # 分析结果
        profile_results = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)
        
        # 计算FLOPs
        flops = ModelProfiler.calculate_flops(model, input_shape)
        
        # 计算推理时间
        import time
        times = []
        for _ in range(100):
            start_time = time.time()
            with torch.no_grad():
                _ = model(dummy_input)
            times.append(time.time() - start_time)
        
        avg_inference_time = np.mean(times) * 1000  # 毫秒
        
        return {
            'flops': flops,
            'avg_inference_time_ms': avg_inference_time,
            'profile_table': profile_results,
            'input_shape': input_shape,
            'model_params': ModelUtils.count_parameters(model)
        }
    
    @staticmethod
    def calculate_flops(model: nn.Module, input_shape: Tuple) -> int:
        """计算FLOPs（简化实现）"""
        flops = 0
        
        def calculate_layer_flops(module, input_shape):
            nonlocal flops
            
            if isinstance(module, nn.Linear):
                flops += input_shape[-1] * module.out_features * input_shape[0]
            elif isinstance(module, nn.Conv2d):
                output_size = (
                    input_shape[2] - module.kernel_size[0] + 2 * module.padding[0]
                ) // module.stride[0] + 1
                flops += (input_shape[1] * module.kernel_size[0] * module.kernel_size[1] *
                         module.out_channels * output_size * output_size * input_shape[0])
            elif isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh)):
                flops += np.prod(input_shape)
            elif isinstance(module, nn.MaxPool2d):
                flops += np.prod(input_shape) // (module.kernel_size ** 2)
        
        for module in model.modules():
            calculate_layer_flops(module, input_shape)
        
        return flops

def save_model_summary(model: nn.Module, input_shape: Tuple, 
                      output_path: Union[str, Path]):
    """保存模型摘要"""
    from torchinfo import summary
    
    summary_str = summary(model, input_shape=input_shape, 
                         col_names=["input_size", "output_size", "num_params"])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(str(summary_str))
    
    print(f"模型摘要已保存到: {output_path}")

class ModelConverter:
    """模型格式转换器"""
    
    @staticmethod
    def pytorch_to_tensorflow(pytorch_model: nn.Module, 
                            input_shape: Tuple,
                            output_path: Union[str, Path]):
        """PyTorch到TensorFlow转换（简化实现）"""
        # 这是一个复杂的过程，通常需要专门的工具
        # 这里只提供基本框架
        
        # 1. 转换为ONNX
        onnx_path = output_path.with_suffix('.onnx')
        ModelUtils.convert_to_onnx(pytorch_model, input_shape, onnx_path)
        
        # 2. ONNX到TensorFlow的转换需要额外工具
        # 可以使用onnx-tensorflow等库
        
        print(f"请使用工具将ONNX模型 {onnx_path} 转换为TensorFlow格式")
    
    @staticmethod
    def tensorflow_to_pytorch(tensorflow_model, 
                            input_shape: Tuple,
                            output_path: Union[str, Path]):
        """TensorFlow到PyTorch转换（简化实现）"""
        # TensorFlow到PyTorch的转换比较复杂
        # 通常需要重新构建模型结构并迁移权重
        
        print("TensorFlow到PyTorch的转换需要手动实现或使用专门工具")