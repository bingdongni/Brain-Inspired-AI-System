"""
层级配置模块
============

定义新皮层各层级的配置参数和枚举类型。
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class LayerType(Enum):
    """皮层层级类型"""
    # 视觉通路
    V1 = "V1"  # 初级视觉皮层 - 边缘检测
    V2 = "V2"  # 次级视觉皮层 - 纹理和简单形状
    V4 = "V4"  # 第四级视觉皮层 - 复杂形状和颜色
    IT = "IT"  # 下颞皮层 - 对象表征
    VOTC = "VOTC"  # 腹侧枕颞皮层 - 概念接口
    
    # 听觉通路
    IC = "IC"  # 下丘 - 听觉皮层下
    MGB = "MGB"  # 内侧膝状体 - 听觉中继
    AC = "AC"  # 听觉皮层
    
    # 前额叶和运动系统
    PFC = "PFC"  # 前额叶皮层
    PMD = "PMd"  # 背侧前运动皮层
    PPC = "PPC"  # 顶叶皮层
    
    # 跨模态接口
    MTL = "MTL"  # 内侧颞叶
    dlATL = "dlATL"  # 背外侧前颞叶


class ProcessingMode(Enum):
    """处理模式"""
    FEEDFORWARD = "feedforward"  # 前馈处理
    FEEDBACK = "feedback"        # 反馈处理
    PREDICTIVE = "predictive"    # 预测编码
    ATTENTION = "attention"      # 注意调制
    ERROR = "error"             # 误差传播
    INTEGRATION = "integration"  # 跨模态整合


@dataclass
class LayerConfig:
    """层级配置参数"""
    
    # 基础网络参数
    layer_type: LayerType
    input_channels: int
    output_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    activation: str = "relu"
    dropout_rate: float = 0.1
    use_batch_norm: bool = True
    
    # V4层间区室化参数
    superficial_ratio: float = 0.6  # 浅层比例（低对比度增强）
    deep_ratio: float = 0.4         # 深层比例（反馈门控）
    
    # 预测编码参数
    prediction_strength: float = 0.8  # 预测强度
    error_amplification: float = 1.2  # 误差放大因子
    
    # 层级特异性参数
    receptive_field: float = 1.0      # 感受野大小
    selectivity_sharpness: float = 2.0  # 选择性 sharpness
    
    # 注意机制参数
    attention_enabled: bool = True
    attention_type: str = "spatial_feature"  # spatial, feature, both
    
    # 跨模态整合参数
    cross_modal_enabled: bool = False
    integration_type: str = "weighted_sum"  # weighted_sum, gated, attention
    
    # 预测误差参数
    error_threshold: float = 0.1  # 误差阈值
    adaptation_rate: float = 0.01  # 适应速率
    
    # 层间连接参数
    feedback_strength: float = 0.5  # 反馈强度
    lateral_inhibition: float = 0.2  # 侧向抑制
    
    def get_spatial_size(self, input_spatial_size: int) -> int:
        """计算输出空间尺寸"""
        return (input_spatial_size + 2 * self.padding - self.kernel_size) // self.stride + 1
    
    def get_receptive_field(self) -> float:
        """计算感受野大小"""
        if self.layer_type == LayerType.V1:
            return 3.0
        elif self.layer_type == LayerType.V2:
            return 7.0
        elif self.layer_type == LayerType.V4:
            return 15.0
        elif self.layer_type == LayerType.IT:
            return 31.0
        else:
            return self.receptive_field
    
    def get_prediction_coverage(self) -> float:
        """获取预测覆盖范围"""
        coverage_map = {
            LayerType.V1: 0.2,
            LayerType.V2: 0.4,
            LayerType.V4: 0.6,
            LayerType.IT: 0.8,
            LayerType.VOTC: 1.0,
            LayerType.AC: 0.6,
            LayerType.MGB: 0.4,
            LayerType.IC: 0.2,
            LayerType.PFC: 0.9,
            LayerType.PMD: 0.7
        }
        return coverage_map.get(self.layer_type, 0.5)


def create_default_layer_config(layer_type: LayerType, input_channels: int, 
                              output_channels: int, kernel_size: int = 3) -> LayerConfig:
    """创建默认层级配置"""
    
    # 不同层级的默认参数
    default_configs = {
        LayerType.V1: {
            'stride': 1, 'padding': 1, 'activation': 'relu',
            'dropout_rate': 0.05, 'prediction_strength': 0.3,
            'error_amplification': 1.0, 'receptive_field': 3.0,
            'selectivity_sharpness': 1.5
        },
        LayerType.V2: {
            'stride': 1, 'padding': 1, 'activation': 'relu',
            'dropout_rate': 0.1, 'prediction_strength': 0.5,
            'error_amplification': 1.1, 'receptive_field': 7.0,
            'selectivity_sharpness': 2.0
        },
        LayerType.V4: {
            'stride': 1, 'padding': 1, 'activation': 'gelu',
            'dropout_rate': 0.15, 'prediction_strength': 0.7,
            'error_amplification': 1.2, 'receptive_field': 15.0,
            'selectivity_sharpness': 2.5,
            'attention_enabled': True
        },
        LayerType.IT: {
            'stride': 1, 'padding': 0, 'activation': 'gelu',
            'dropout_rate': 0.2, 'prediction_strength': 0.8,
            'error_amplification': 1.3, 'receptive_field': 31.0,
            'selectivity_sharpness': 3.0,
            'attention_enabled': True,
            'cross_modal_enabled': True
        },
        LayerType.VOTC: {
            'stride': 1, 'padding': 0, 'activation': 'gelu',
            'dropout_rate': 0.2, 'prediction_strength': 0.9,
            'error_amplification': 1.4, 'receptive_field': 45.0,
            'selectivity_sharpness': 3.5,
            'attention_enabled': True,
            'cross_modal_enabled': True
        },
        LayerType.AC: {
            'stride': 1, 'padding': 1, 'activation': 'relu',
            'dropout_rate': 0.1, 'prediction_strength': 0.6,
            'error_amplification': 1.2, 'receptive_field': 8.0,
            'selectivity_sharpness': 2.0
        },
        LayerType.MGB: {
            'stride': 1, 'padding': 1, 'activation': 'relu',
            'dropout_rate': 0.08, 'prediction_strength': 0.5,
            'error_amplification': 1.1, 'receptive_field': 5.0,
            'selectivity_sharpness': 1.8
        },
        LayerType.IC: {
            'stride': 1, 'padding': 1, 'activation': 'relu',
            'dropout_rate': 0.05, 'prediction_strength': 0.4,
            'error_amplification': 1.0, 'receptive_field': 3.0,
            'selectivity_sharpness': 1.5
        },
        LayerType.PFC: {
            'stride': 1, 'padding': 0, 'activation': 'gelu',
            'dropout_rate': 0.25, 'prediction_strength': 0.9,
            'error_amplification': 1.5, 'receptive_field': 60.0,
            'selectivity_sharpness': 4.0,
            'attention_enabled': True,
            'cross_modal_enabled': True
        },
        LayerType.PMD: {
            'stride': 1, 'padding': 0, 'activation': 'relu',
            'dropout_rate': 0.15, 'prediction_strength': 0.7,
            'error_amplification': 1.3, 'receptive_field': 25.0,
            'selectivity_sharpness': 2.5,
            'attention_enabled': True
        }
    }
    
    # 获取默认配置
    defaults = default_configs.get(layer_type, {})
    
    # 创建配置对象
    config = LayerConfig(
        layer_type=layer_type,
        input_channels=input_channels,
        output_channels=output_channels,
        kernel_size=kernel_size,
        **defaults
    )
    
    return config