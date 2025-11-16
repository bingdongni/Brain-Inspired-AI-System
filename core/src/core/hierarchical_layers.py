"""
分层抽象机制 (Hierarchical Layers)
====================================

实现新皮层的分层信息处理架构，包括：
- V1→V2→V4→IT的视觉层次处理
- IC→MGB→AC的听觉层级预测误差
- 前馈-反馈协同机制
- V4层间区室化注意调制

基于论文：
- "Revealing Detail along the Visual Hierarchy" (Neuron, 2018)
- "Neurons along the auditory pathway exhibit hierarchical organization of prediction error"
- "Laminar compartmentalization of attention modulation in area V4"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math


class LayerType(Enum):
    """皮层层级类型"""
    V1 = "V1"  # 初级视觉皮层 - 边缘检测
    V2 = "V2"  # 次级视觉皮层 - 纹理和简单形状
    V4 = "V4"  # 第四级视觉皮层 - 复杂形状和颜色
    IT = "IT"  # 下颞皮层 - 对象表征
    VOTC = "VOTC"  # 腹侧枕颞皮层 - 概念接口
    IC = "IC"  # 下丘 - 听觉皮层下
    MGB = "MGB"  # 内侧膝状体 - 听觉中继
    AC = "AC"  # 听觉皮层
    PFC = "PFC"  # 前额叶皮层
    PMD = "PMd"  # 背侧前运动皮层


class ProcessingMode(Enum):
    """处理模式"""
    FEEDFORWARD = "feedforward"  # 前馈处理
    FEEDBACK = "feedback"        # 反馈处理
    PREDICTIVE = "predictive"    # 预测编码
    ATTENTION = "attention"      # 注意调制
    ERROR = "error"             # 误差传播


@dataclass
class LayerConfig:
    """层级配置参数"""
    layer_type: LayerType
    input_channels: int
    output_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    activation: str = "relu"
    dropout_rate: float = 0.1
    
    # V4层间区室化参数
    superficial_ratio: float = 0.6  # 浅层比例（低对比度增强）
    deep_ratio: float = 0.4         # 深层比例（反馈门控）
    
    # 预测编码参数
    prediction_strength: float = 0.8  # 预测强度
    error_amplification: float = 1.2  # 误差放大因子
    
    # 层级特异性参数
    receptive_field: float = 1.0      # 感受野大小
    selectivity_sharpness: float = 2.0  # 选择性 sharpness


class HierarchicalLayer(nn.Module):
    """
    分层抽象层级
    
    实现单层皮层的功能，包括：
    - 特征提取和整合
    - 前馈-反馈信号处理
    - 注意调制
    - 预测编码
    """
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        self.layer_type = config.layer_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 主特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(config.input_channels, config.output_channels, 
                     config.kernel_size, config.stride, config.padding),
            self._get_activation(config.activation),
            nn.Dropout2d(config.dropout_rate)
        )
        
        # V4层间区室化 - 浅层（特征增强）
        if config.layer_type == LayerType.V4:
            self.superficial_processor = nn.Sequential(
                nn.Conv2d(config.output_channels, int(config.output_channels * config.superficial_ratio),
                         3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(int(config.output_channels * config.superficial_ratio))
            )
            
            # 深层（反馈门控）
            self.deep_processor = nn.Sequential(
                nn.Conv2d(config.output_channels, int(config.output_channels * config.deep_ratio),
                         3, 1, 1),
                nn.ReLU(),
                nn.BatchNorm2d(int(config.output_channels * config.deep_ratio))
            )
            
            # 注意门控机制
            self.attention_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(config.output_channels, config.output_channels // 4, 1),
                nn.ReLU(),
                nn.Conv2d(config.output_channels // 4, config.output_channels, 1),
                nn.Sigmoid()
            )
        
        # 预测编码组件
        self.predictor = PredictionEncoder(config)
        
        # 反馈抑制层（防止过度反馈）
        self.feedback_regulator = FeedbackRegulator(config)
        
    def _get_activation(self, activation: str):
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.2),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor, 
                feedback: Optional[torch.Tensor] = None,
                attention: Optional[torch.Tensor] = None,
                mode: ProcessingMode = ProcessingMode.FEEDFORWARD) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征图 [batch, channels, height, width]
            feedback: 反馈信号（自上而下）
            attention: 注意权重
            mode: 处理模式
            
        Returns:
            输出特征和辅助信息字典
        """
        batch_size = x.size(0)
        
        # 基础特征提取
        features = self.feature_extractor(x)
        
        # V4层间区室化处理
        if self.layer_type == LayerType.V4:
            # 浅层：低对比度增强
            superficial_out = self.superficial_processor(features)
            
            # 深层：对比度独立的反馈调制
            deep_out = self.deep_processor(features)
            
            # 注意力门控
            if attention is not None:
                attention_weights = self.attention_gate(features)
                features = features * attention_weights
            else:
                attention_weights = torch.ones_like(features)
            
            # 层间整合
            integrated_features = self._integrate_laminar_signals(
                features, superficial_out, deep_out, attention_weights
            )
            
            features = integrated_features
        
        # 预测编码处理
        predictive_output = self.predictor(features, feedback, mode)
        
        # 反馈调节
        if feedback is not None:
            features = self.feedback_regulator(features, feedback)
        
        return {
            'features': features,
            'predictive_output': predictive_output,
            'attention_weights': attention if attention is not None else torch.ones_like(features),
            'prediction_error': predictive_output.get('error', torch.zeros_like(features)),
            'layer_type': self.layer_type.value
        }
    
    def _integrate_laminar_signals(self, base_features: torch.Tensor,
                                  superficial: torch.Tensor, 
                                  deep: torch.Tensor,
                                  attention_weights: torch.Tensor) -> torch.Tensor:
        """整合层间信号（V4特有）"""
        # 浅层特征增强：低对比度响应增强
        enhanced_features = base_features * (1 + 0.3 * superficial)
        
        # 深层反馈门控：对比度独立的调制
        gated_features = deep * attention_weights
        
        # 加权整合
        integrated = 0.5 * enhanced_features + 0.5 * gated_features
        
        return integrated


class PredictionEncoder(nn.Module):
    """预测编码器 - 实现前馈-反馈预测机制"""
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        self.prediction_strength = config.prediction_strength
        
        # 预测模型（自上而下的先验）
        self.predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.output_channels, config.output_channels * 2),
            nn.ReLU(),
            nn.Linear(config.output_channels * 2, config.output_channels),
            nn.Unflatten(1, (config.output_channels, 1, 1))
        )
        
        # 误差检测器
        self.error_detector = nn.Sequential(
            nn.Conv2d(config.output_channels * 2, config.output_channels, 1),
            nn.ReLU(),
            nn.Conv2d(config.output_channels, config.output_channels, 3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, features: torch.Tensor, 
                feedback: Optional[torch.Tensor] = None,
                mode: ProcessingMode = ProcessingMode.FEEDFORWARD) -> Dict[str, torch.Tensor]:
        """预测编码前向传播"""
        
        # 生成预测
        prediction = self.predictor(features)
        
        # 如果有反馈信号，结合反馈
        if feedback is not None:
            prediction = prediction * (1 + 0.5 * F.interpolate(feedback, features.shape[2:]))
        
        if mode == ProcessingMode.PREDICTIVE:
            # 计算预测误差
            error_input = torch.cat([features, prediction], dim=1)
            prediction_error = self.error_detector(error_input) * self.config.error_amplification
            
            return {
                'prediction': prediction,
                'error': prediction_error,
                'confidence': torch.sigmoid(torch.norm(prediction_error, dim=1, keepdim=True))
            }
        else:
            return {
                'features': features * self.prediction_strength + prediction * (1 - self.prediction_strength)
            }


class FeedbackRegulator(nn.Module):
    """反馈调节器 - 防止过度反馈，稳定层级间通信"""
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        self.feedback_strength = 0.5
        
        # 反馈强度调制器
        self.strength_modulator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.output_channels, config.output_channels // 4),
            nn.ReLU(),
            nn.Linear(config.output_channels // 4, 1),
            nn.Sigmoid()
        )
        
        # 反馈过滤层
        self.feedback_filter = nn.Sequential(
            nn.Conv2d(config.output_channels, config.output_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.output_channels, config.output_channels, 1)
        )
        
    def forward(self, features: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        """反馈调节前向传播"""
        # 计算反馈强度
        strength = self.strength_modulator(features)
        
        # 过滤反馈信号
        filtered_feedback = self.feedback_filter(feedback)
        
        # 应用强度调制
        regulated_feedback = filtered_feedback * strength * self.feedback_strength
        
        # 结合到特征中
        regulated_features = features + regulated_feedback
        
        return regulated_features


class ProcessingHierarchy(nn.Module):
    """
    处理层次结构
    
    整个新皮层的分层处理流水线，整合多个皮层区域
    """
    
    def __init__(self, layer_configs: List[LayerConfig]):
        super().__init__()
        self.layer_configs = layer_configs
        self.layers = nn.ModuleList([HierarchicalLayer(config) for config in layer_configs])
        
        # 层级间连接强度矩阵
        self.connection_matrix = self._initialize_connections()
        
        # 注意控制器
        self.attention_controller = AttentionController(len(layer_configs))
        
        # 预测误差累积器
        self.error_accumulator = ErrorAccumulator()
        
    def _initialize_connections(self) -> torch.Tensor:
        """初始化层级间连接强度"""
        n_layers = len(self.layer_configs)
        connections = torch.zeros(n_layers, n_layers)
        
        # 前馈连接（相邻层间）
        for i in range(n_layers - 1):
            connections[i, i + 1] = 1.0
            
        # 反馈连接（长距离）
        for i in range(2, n_layers):
            for j in range(i):
                connections[i, j] = 0.3  # 较弱的反向连接
                
        return connections
    
    def forward(self, inputs: torch.Tensor) -> Dict[str, Union[torch.Tensor, List[Dict]]]:
        """
        层次前向传播
        
        Args:
            inputs: 输入数据 [batch, channels, height, width]
            
        Returns:
            各层输出和层次信息
        """
        layer_outputs = []
        current_input = inputs
        feedback_signals = {}
        
        # 多层前向传播
        for i, (layer, config) in enumerate(zip(self.layers, self.layer_configs)):
            
            # 获取前层反馈信号
            feedback = feedback_signals.get(i, None)
            
            # 层级处理
            layer_output = layer(current_input, feedback=feedback, mode=ProcessingMode.FEEDFORWARD)
            layer_outputs.append(layer_output)
            
            # 更新输入为下一层的输出
            current_input = layer_output['features']
            
            # 存储反馈信号供后续层使用
            if i < len(self.layers) - 1:
                feedback_signals[i + 1] = layer_output.get('prediction', current_input)
        
        # 预测编码模式下的误差检测
        predictive_outputs = []
        for i in range(len(self.layers) - 1, -1, -1):
            config = self.layer_configs[i]
            layer_output = self.layers[i](
                layer_outputs[i]['features'],
                mode=ProcessingMode.PREDICTIVE
            )
            predictive_outputs.append(layer_output)
        
        # 注意力调制
        attention_weights = self.attention_controller(layer_outputs)
        
        return {
            'layer_outputs': layer_outputs,
            'predictive_outputs': predictive_outputs,
            'attention_weights': attention_weights,
            'final_output': current_input,
            'prediction_errors': [out.get('error', torch.zeros_like(out['features'])) 
                                for out in predictive_outputs]
        }
    
    def get_layer_receptive_field(self, layer_idx: int) -> float:
        """获取指定层的感受野大小"""
        if layer_idx >= len(self.layer_configs):
            return 1.0
            
        config = self.layer_configs[layer_idx]
        current_rf = config.receptive_field
        
        # 从输入层到目标层累积感受野
        for i in range(layer_idx):
            prev_config = self.layer_configs[i]
            current_rf += (prev_config.kernel_size - 1) * prev_config.stride
            
        return current_rf


class AttentionController(nn.Module):
    """注意力控制器 - 管理层级间的注意分配"""
    
    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = n_layers
        
        # 层级注意权重生成器
        self.layer_attention = nn.Parameter(torch.ones(n_layers) / n_layers)
        
        # 空间注意生成器
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, layer_outputs: List[Dict]) -> torch.Tensor:
        """生成注意力权重"""
        batch_size = layer_outputs[0]['features'].size(0)
        
        # 层级注意权重
        layer_weights = F.softmax(self.layer_attention, dim=0)
        
        # 生成空间注意图
        features = layer_outputs[-1]['features']  # 使用顶层特征
        spatial_attention = self.spatial_attention(
            F.adaptive_avg_pool2d(features, (1, 1)).expand_as(features)
        )
        
        # 结合层级和空间注意
        attention_weights = layer_weights[-1] * spatial_attention
        
        return attention_weights


class ErrorAccumulator(nn.Module):
    """预测误差累积器 - 收集和聚合层级预测误差"""
    
    def __init__(self):
        super().__init__()
        self.error_history = []
        
    def forward(self, prediction_errors: List[torch.Tensor]) -> torch.Tensor:
        """累积预测误差"""
        if not prediction_errors:
            return torch.tensor(0.0)
            
        # 计算各层误差的加权和
        total_error = torch.zeros_like(prediction_errors[0])
        
        for i, error in enumerate(prediction_errors):
            weight = 0.5 ** (len(prediction_errors) - i - 1)  # 递减权重
            total_error += weight * F.adaptive_avg_pool2d(error, (1, 1))
            
        return total_error


def create_visual_hierarchy(input_channels: int = 3) -> List[LayerConfig]:
    """创建视觉皮层层次配置"""
    return [
        LayerConfig(LayerType.V1, input_channels, 64, 7, stride=2, padding=3),
        LayerConfig(LayerType.V2, 64, 128, 5, stride=2, padding=2),
        LayerConfig(LayerType.V4, 128, 256, 3, stride=1, padding=1),
        LayerConfig(LayerType.IT, 256, 512, 3, stride=1, padding=1),
        LayerConfig(LayerType.VOTC, 512, 1024, 2, stride=1),
    ]


def create_auditory_hierarchy(input_channels: int = 1) -> List[LayerConfig]:
    """创建听觉皮层层次配置"""
    return [
        LayerConfig(LayerType.IC, input_channels, 64, 11, stride=2, padding=5),
        LayerConfig(LayerType.MGB, 64, 128, 7, stride=2, padding=3),
        LayerConfig(LayerType.AC, 128, 256, 5, stride=1, padding=2),
        LayerConfig(LayerType.PFC, 256, 512, 3, stride=1, padding=1),
    ]


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建视觉层次
    visual_configs = create_visual_hierarchy()
    visual_hierarchy = ProcessingHierarchy(visual_configs).to(device)
    
    # 测试输入
    test_input = torch.randn(1, 3, 224, 224).to(device)
    
    # 前向传播
    outputs = visual_hierarchy(test_input)
    
    print(f"视觉层次处理完成")
    print(f"最终输出形状: {outputs['final_output'].shape}")
    print(f"层次数: {len(outputs['layer_outputs'])}")
    print(f"预测误差数量: {len(outputs['prediction_errors'])}")
