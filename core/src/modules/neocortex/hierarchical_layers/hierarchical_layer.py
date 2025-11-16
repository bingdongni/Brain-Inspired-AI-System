"""
分层抽象层级实现
================

实现单层皮层的功能，包括：
- 特征提取和整合
- 前馈-反馈信号处理
- 注意调制
- 预测编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from .layer_config import LayerType, ProcessingMode, LayerConfig, create_default_layer_config


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
        self.feature_extractor = self._build_feature_extractor()
        
        # V4层间区室化处理
        if config.layer_type == LayerType.V4:
            self.superficial_processor = self._build_superficial_processor()
            self.deep_processor = self._build_deep_processor()
            self.attention_gate = self._build_attention_gate()
        
        # 预测编码组件
        self.predictor = PredictiveEncoder(config)
        self.error_amplifier = ErrorAmplifier(config)
        
        # 反馈抑制层（防止过度反馈）
        self.feedback_regulator = FeedbackRegulator(config)
        
        # 层间整合机制
        self.integration_module = self._build_integration_module()
        
    def _build_feature_extractor(self) -> nn.Module:
        """构建特征提取器"""
        if self.config.layer_type in [LayerType.IT, LayerType.VOTC, LayerType.PFC, LayerType.PMD]:
            # 全连接层用于高级皮层
            return nn.Sequential(
                nn.Linear(self.config.input_channels, self.config.output_channels),
                self._get_activation(self.config.activation),
                nn.Dropout(self.config.dropout_rate)
            )
        else:
            # 卷积层用于感知皮层
            return nn.Sequential(
                nn.Conv2d(self.config.input_channels, self.config.output_channels, 
                         self.config.kernel_size, self.config.stride, self.config.padding),
                nn.BatchNorm2d(self.config.output_channels) if self.config.use_batch_norm else nn.Identity(),
                self._get_activation(self.config.activation),
                nn.Dropout2d(self.config.dropout_rate)
            )
    
    def _build_superficial_processor(self) -> nn.Module:
        """构建V4浅层处理器（特征增强）"""
        superficial_channels = int(self.config.output_channels * self.config.superficial_ratio)
        return nn.Sequential(
            nn.Conv2d(self.config.output_channels, superficial_channels, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(superficial_channels),
            nn.Conv2d(superficial_channels, self.config.output_channels, 3, 1, 1),
            nn.Sigmoid()
        )
    
    def _build_deep_processor(self) -> nn.Module:
        """构建V4深层处理器（反馈门控）"""
        deep_channels = int(self.config.output_channels * self.config.deep_ratio)
        return nn.Sequential(
            nn.Conv2d(self.config.output_channels, deep_channels, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(deep_channels),
            nn.Conv2d(deep_channels, self.config.output_channels, 1, 1, 0),
            nn.Sigmoid()
        )
    
    def _build_attention_gate(self) -> nn.Module:
        """构建V4注意门控机制"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.config.output_channels, self.config.output_channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(self.config.output_channels // 4, self.config.output_channels, 1),
            nn.Sigmoid()
        )
    
    def _build_integration_module(self) -> nn.Module:
        """构建层间整合模块"""
        if self.config.cross_modal_enabled:
            if self.config.integration_type == "attention":
                return CrossModalAttention(self.config.input_channels, self.config.output_channels)
            elif self.config.integration_type == "gated":
                return GatedIntegration(self.config.input_channels, self.config.output_channels)
            else:
                return WeightedSumIntegration(self.config.input_channels, self.config.output_channels)
        else:
            return nn.Identity()
    
    def _get_activation(self, activation: str) -> nn.Module:
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
                cross_modal: Optional[torch.Tensor] = None,
                mode: ProcessingMode = ProcessingMode.FEEDFORWARD) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征图
            feedback: 来自上层的反馈信号
            attention: 注意调制信号
            cross_modal: 跨模态输入
            mode: 处理模式
            
        Returns:
            dict: 包含输出特征、预测误差、注意力权重等信息
        """
        batch_size = x.shape[0]
        original_shape = x.shape
        
        # 特征提取
        if self.config.layer_type in [LayerType.IT, LayerType.VOTC, LayerType.PFC, LayerType.PMD]:
            # 全连接层处理
            if len(x.shape) > 2:
                x = x.view(batch_size, -1)
            features = self.feature_extractor(x)
        else:
            # 卷积层处理
            features = self.feature_extractor(x)
        
        # V4层间区室化处理
        if self.config.layer_type == LayerType.V4:
            features = self._process_v4_compartments(features, attention)
        
        # 跨模态整合
        if cross_modal is not None and self.config.cross_modal_enabled:
            features = self.integration_module(features, cross_modal)
        
        # 预测编码处理
        if mode == ProcessingMode.PREDICTIVE:
            predictions = self.predictor(feedback) if feedback is not None else self.predictor(features)
            errors = self.error_amplifier(features, predictions)
            features = features + errors * self.config.error_amplification
        
        # 反馈调节
        if feedback is not None:
            features = self.feedback_regulator(features, feedback)
        
        # 注意调制
        if attention is not None and self.config.attention_enabled:
            features = features * attention
        
        # 生成输出
        output_dict = {
            'features': features,
            'layer_type': self.config.layer_type.value,
            'spatial_size': self._get_spatial_size(original_shape),
            'channels': self.config.output_channels
        }
        
        # 添加模式特定信息
        if mode == ProcessingMode.PREDICTIVE:
            output_dict['predictions'] = predictions if 'predictions' in locals() else None
            output_dict['prediction_errors'] = errors if 'errors' in locals() else None
        
        return output_dict
    
    def _process_v4_compartments(self, features: torch.Tensor, attention: Optional[torch.Tensor] = None) -> torch.Tensor:
        """处理V4层间区室化"""
        # 浅层处理（特征增强）
        superficial_output = self.superficial_processor(features)
        
        # 深层处理（反馈门控）
        deep_output = self.deep_processor(features)
        
        # 注意门控
        if attention is not None:
            attention_weights = self.attention_gate(features)
            superficial_output = superficial_output * attention_weights
        
        # 区室化输出
        combined = features * (1 - self.config.superficial_ratio) + \
                  superficial_output * self.config.superficial_ratio
        
        combined = combined * (1 - self.config.deep_ratio) + \
                  deep_output * self.config.deep_ratio
        
        return combined
    
    def _get_spatial_size(self, shape: torch.Size) -> List[int]:
        """获取空间尺寸"""
        if len(shape) >= 4:
            return list(shape[2:])
        elif len(shape) == 2:
            return [1]
        else:
            return []
    
    def get_receptive_field_info(self) -> Dict[str, float]:
        """获取感受野信息"""
        return {
            'receptive_field_size': self.config.get_receptive_field(),
            'prediction_coverage': self.config.get_prediction_coverage(),
            'layer_depth': self.config.layer_type.value
        }


class PredictiveEncoder(nn.Module):
    """预测编码器"""
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        self.prediction_strength = config.prediction_strength
        
        # 预测网络
        if config.layer_type in [LayerType.IT, LayerType.VOTC, LayerType.PFC, LayerType.PMD]:
            self.predictor = nn.Sequential(
                nn.Linear(config.output_channels, config.output_channels),
                nn.ReLU(),
                nn.Linear(config.output_channels, config.output_channels),
                nn.Tanh()
            )
        else:
            self.predictor = nn.Sequential(
                nn.Conv2d(config.output_channels, config.output_channels, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(config.output_channels, config.output_channels, 3, 1, 1),
                nn.Tanh()
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向预测"""
        predictions = self.predictor(x)
        return predictions * self.prediction_strength


class ErrorAmplifier(nn.Module):
    """误差放大器"""
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        self.error_amplification = config.error_amplification
        self.error_threshold = config.error_threshold
    
    def forward(self, actual: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
        """计算并放大预测误差"""
        errors = actual - predicted
        
        # 只放大超过阈值的误差
        error_magnitude = torch.abs(errors)
        error_mask = error_magnitude > self.error_threshold
        
        # 放大显著误差
        amplified_errors = errors * self.error_amplification
        amplified_errors = amplified_errors * error_mask
        
        return amplified_errors


class FeedbackRegulator(nn.Module):
    """反馈调节器"""
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        self.feedback_strength = config.feedback_strength
        
        # 反馈调节网络
        self.regulator = nn.Sequential(
            nn.Conv2d(config.output_channels * 2, config.output_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        """调节反馈信号"""
        # 确保反馈信号形状匹配
        if features.shape != feedback.shape:
            feedback = F.interpolate(feedback, size=features.shape[2:], mode='bilinear', align_corners=False)
        
        # 特征反馈整合
        combined = torch.cat([features, feedback], dim=1)
        regulation_weights = self.regulator(combined)
        
        # 应用反馈调节
        regulated_features = features + feedback * regulation_weights * self.feedback_strength
        return regulated_features


class CrossModalAttention(nn.Module):
    """跨模态注意力整合"""
    
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.attention_query = nn.Linear(input_channels, output_channels)
        self.attention_key = nn.Linear(input_channels, output_channels)
        self.attention_value = nn.Linear(input_channels, output_channels)
        
    def forward(self, features: torch.Tensor, cross_modal: torch.Tensor) -> torch.Tensor:
        """跨模态注意力整合"""
        # 计算注意力权重
        query = self.attention_query(features)
        key = self.attention_key(cross_modal)
        value = self.attention_value(cross_modal)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = F.softmax(attention_scores / np.sqrt(key.size(-1)), dim=-1)
        
        # 应用注意力
        attended_cross = torch.matmul(attention_weights, value)
        
        # 特征融合
        integrated = features + attended_cross
        return integrated


class GatedIntegration(nn.Module):
    """门控整合模块"""
    
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_channels * 2, output_channels),
            nn.Sigmoid()
        )
        
        self.project = nn.Linear(input_channels, output_channels)
    
    def forward(self, features: torch.Tensor, cross_modal: torch.Tensor) -> torch.Tensor:
        """门控整合"""
        # 计算门控权重
        combined = torch.cat([features, cross_modal], dim=-1)
        gate_weights = self.gate(combined)
        
        # 投影跨模态特征
        projected_cross = self.project(cross_modal)
        
        # 门控融合
        integrated = features + gate_weights * projected_cross
        return integrated


class WeightedSumIntegration(nn.Module):
    """加权求和整合模块"""
    
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        self.weight_generator = nn.Sequential(
            nn.Linear(input_channels * 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.project = nn.Linear(input_channels, output_channels)
    
    def forward(self, features: torch.Tensor, cross_modal: torch.Tensor) -> torch.Tensor:
        """加权求和整合"""
        # 计算融合权重
        combined = torch.cat([features, cross_modal], dim=-1)
        weights = self.weight_generator(combined)
        
        # 投影并加权
        projected_cross = self.project(cross_modal)
        weighted_cross = weights * projected_cross
        weighted_features = (1 - weights) * features
        
        # 融合输出
        integrated = weighted_features + weighted_cross
        return integrated


class FeedbackRegulator(nn.Module):
    """反馈调节器（重新定义）"""
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        self.feedback_strength = config.feedback_strength
        
        # 反馈调节网络
        if config.layer_type in [LayerType.IT, LayerType.VOTC, LayerType.PFC, LayerType.PMD]:
            self.regulator = nn.Sequential(
                nn.Linear(config.output_channels * 2, config.output_channels),
                nn.Sigmoid()
            )
        else:
            self.regulator = nn.Sequential(
                nn.Conv2d(config.output_channels * 2, config.output_channels, 1),
                nn.Sigmoid()
            )
    
    def forward(self, features: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        """调节反馈信号"""
        # 确保反馈信号形状匹配
        if features.shape != feedback.shape:
            if len(features.shape) >= 4:
                feedback = F.interpolate(feedback, size=features.shape[2:], mode='bilinear', align_corners=False)
            else:
                feedback = F.interpolate(feedback, size=features.shape[-1], mode='linear', align_corners=False)
        
        # 特征反馈整合
        combined = torch.cat([features, feedback], dim=1)
        regulation_weights = self.regulator(combined)
        
        # 应用反馈调节
        regulated_features = features + feedback * regulation_weights * self.feedback_strength
        return regulated_features