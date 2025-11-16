"""
视觉层次处理模块
================

实现视觉皮层的专用层次化处理机制，包括：
- V1→V2→V4→IT→VOTC的视觉通路
- 视觉特征层次提取
- 视觉注意力机制
- 视觉预测编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .layer_config import LayerType, ProcessingMode, LayerConfig, create_default_layer_config
from .processing_hierarchy import ProcessingHierarchy
from .hierarchical_layer import HierarchicalLayer


class VisualProcessingStream(nn.Module):
    """
    视觉处理流
    
    实现完整的视觉层次处理通路，从V1到VOTC的层级特征提取和整合。
    """
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建视觉层次配置
        self.layer_configs = create_visual_hierarchy(input_channels, feature_dim)
        
        # 创建视觉处理层次
        self.visual_hierarchy = ProcessingHierarchy(self.layer_configs)
        
        # 视觉注意力系统
        self.spatial_attention = SpatialAttention(feature_dim)
        self.feature_attention = FeatureAttention(feature_dim)
        
        # 视觉预测编码
        self.visual_predictor = VisualPredictor(feature_dim)
        
        # 视觉概念接口
        self.concept_interface = VisualConceptInterface(feature_dim)
    
    def forward(self, x: torch.Tensor,
                attention_region: Optional[torch.Tensor] = None,
                semantic_context: Optional[torch.Tensor] = None,
                mode: ProcessingMode = ProcessingMode.FEEDFORWARD) -> Dict[str, torch.Tensor]:
        """
        视觉处理前向传播
        
        Args:
            x: 输入图像 [batch, channels, height, width]
            attention_region: 注意力区域 [batch, height, width]
            semantic_context: 语义上下文
            mode: 处理模式
            
        Returns:
            dict: 视觉处理结果
        """
        batch_size = x.shape[0]
        
        # 层级处理
        hierarchy_result = self.visual_hierarchy(x, mode=mode)
        
        # 提取层级特征
        v1_output = self._get_layer_output(hierarchy_result, LayerType.V1)
        v2_output = self._get_layer_output(hierarchy_result, LayerType.V2)
        v4_output = self._get_layer_output(hierarchy_result, LayerType.V4)
        it_output = self._get_layer_output(hierarchy_result, LayerType.IT)
        votc_output = self._get_layer_output(hierarchy_result, LayerType.VOTC)
        
        # 注意力处理
        attention_maps = {}
        if mode in [ProcessingMode.ATTENTION, ProcessingMode.FEEDFORWARD]:
            # 空间注意力
            if v4_output is not None:
                spatial_att = self.spatial_attention(v4_output)
                attention_maps['spatial'] = spatial_att
            
            # 特征注意力
            if it_output is not None:
                feature_att = self.feature_attention(it_output)
                attention_maps['feature'] = feature_att
            
            # 区域性注意力
            if attention_region is not None:
                region_att = self._apply_region_attention(v4_output, attention_region)
                attention_maps['region'] = region_att
        
        # 预测编码
        predictions = {}
        if mode == ProcessingMode.PREDICTIVE:
            predictions = self.visual_predictor(hierarchy_result)
        
        # 概念提取
        concept_representation = {}
        if votc_output is not None:
            concept_representation = self.concept_interface(
                votc_output, 
                semantic_context=semantic_context
            )
        
        # 整合输出
        visual_result = {
            'final_output': votc_output['features'] if votc_output else it_output['features'],
            'layer_outputs': hierarchy_result['layer_outputs'],
            'attention_maps': attention_maps,
            'predictions': predictions,
            'concept_representation': concept_representation,
            'visual_summary': {
                'processed_layers': len([out for out in [v1_output, v2_output, v4_output, it_output, votc_output] if out is not None]),
                'attention_types': list(attention_maps.keys()),
                'has_concepts': len(concept_representation) > 0,
                'prediction_accuracy': predictions.get('accuracy', 0.0) if predictions else 0.0
            }
        }
        
        return visual_result
    
    def _get_layer_output(self, hierarchy_result: Dict, layer_type: LayerType) -> Optional[Dict]:
        """从层次结果中提取指定层的输出"""
        for layer_output in hierarchy_result['layer_outputs']:
            if layer_output['layer_type'] == layer_type.value:
                return layer_output
        return None
    
    def _apply_region_attention(self, features: Optional[Dict], region: torch.Tensor) -> torch.Tensor:
        """应用区域性注意力"""
        if features is None:
            return torch.zeros_like(region)
        
        feature_map = features['features']
        
        # 将区域注意力调整到特征图大小
        if feature_map.shape[-2:] != region.shape[-2:]:
            region_resized = F.interpolate(
                region.unsqueeze(1), 
                size=feature_map.shape[-2:], 
                mode='bilinear', 
                align_corners=False
            )
        else:
            region_resized = region.unsqueeze(1)
        
        # 应用区域注意力
        attended_features = feature_map * region_resized
        
        return attended_features
    
    def get_visual_hierarchy_info(self) -> Dict[str, any]:
        """获取视觉层次信息"""
        return {
            'layer_configs': [
                {
                    'type': config.layer_type.value,
                    'input_channels': config.input_channels,
                    'output_channels': config.output_channels,
                    'receptive_field': config.get_receptive_field(),
                    'prediction_coverage': config.get_prediction_coverage()
                }
                for config in self.layer_configs
            ],
            'hierarchy_statistics': self.visual_hierarchy.get_processing_statistics()
        }


class SpatialAttention(nn.Module):
    """空间注意力机制"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 空间注意力生成器
        self.spatial_attention_generator = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 空间特征增强
        self.spatial_enhancer = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """空间注意力前向传播"""
        feature_map = features['features']
        
        # 生成空间注意力权重
        spatial_weights = self.spatial_attention_generator(feature_map)
        
        # 增强空间特征
        enhanced_features = self.spatial_enhancer(feature_map)
        
        # 应用空间注意力
        attended_features = feature_map * spatial_weights + enhanced_features * (1 - spatial_weights)
        
        return {
            'attended_features': attended_features,
            'spatial_weights': spatial_weights,
            'attention_entropy': self._compute_attention_entropy(spatial_weights)
        }
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """计算注意力熵"""
        # 将权重归一化为概率分布
        normalized_weights = attention_weights / (attention_weights.sum() + 1e-8)
        # 计算熵
        entropy = -torch.sum(normalized_weights * torch.log(normalized_weights + 1e-8))
        return entropy


class FeatureAttention(nn.Module):
    """特征注意力机制"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 特征注意力生成器
        self.feature_attention_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim),
            nn.Sigmoid()
        )
        
        # 特征重加权
        self.feature_reweighter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """特征注意力前向传播"""
        feature_map = features['features']
        
        # 生成特征注意力权重
        feature_weights = self.feature_attention_generator(feature_map)
        
        # 特征重加权
        reweighted_features = self.feature_reweighter(feature_weights) * feature_map
        
        return {
            'attended_features': reweighted_features,
            'feature_weights': feature_weights,
            'selectivity': feature_weights.mean()
        }


class VisualPredictor(nn.Module):
    """视觉预测器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 视觉特征预测
        self.feature_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh()
        )
        
        # 预测误差计算器
        self.error_calculator = nn.MSELoss(reduction='none')
        
        # 预测准确性评估器
        self.accuracy_assessor = nn.Sequential(
            nn.Linear(feature_dim * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hierarchy_result: Dict) -> Dict[str, torch.Tensor]:
        """视觉预测前向传播"""
        # 获取层级特征
        layer_outputs = hierarchy_result['layer_outputs']
        
        # 从高层特征预测低层特征
        predictions = {}
        errors = {}
        
        for i, layer_output in enumerate(layer_outputs):
            if i == 0:  # 第一层无法预测
                continue
                
            current_features = layer_output['features']
            layer_type = layer_output['layer_type']
            
            # 从当前层预测前一层的特征
            predicted_features = self.feature_predictor(current_features)
            
            # 计算预测误差
            prev_layer_output = layer_outputs[i-1]
            prev_features = prev_layer_output['features']
            
            # 调整尺寸以匹配
            if predicted_features.shape != prev_features.shape:
                if len(predicted_features.shape) >= 4:
                    predicted_features = F.adaptive_avg_pool2d(predicted_features, prev_features.shape[2:])
                else:
                    predicted_features = F.interpolate(
                        predicted_features, size=prev_features.shape[-1], mode='linear'
                    )
            
            error = self.error_calculator(predicted_features, prev_features)
            
            predictions[layer_type] = predicted_features
            errors[f"{layer_type}_error"] = error.mean()
        
        # 评估整体预测准确性
        final_features = layer_outputs[-1]['features']
        if len(final_features.shape) > 2:
            final_features = F.adaptive_avg_pool2d(final_features, 1).flatten(1)
        
        accuracy_score = self.accuracy_assessor(final_features).mean()
        
        return {
            'predictions': predictions,
            'errors': errors,
            'accuracy': accuracy_score,
            'total_error': torch.stack(list(errors.values())).mean()
        }


class VisualConceptInterface(nn.Module):
    """视觉概念接口"""
    
    def __init__(self, feature_dim: int, num_concepts: int = 100):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_concepts = num_concepts
        
        # 概念检测器
        self.concept_detectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(num_concepts)
        ])
        
        # 概念关系编码器
        self.concept_relations = nn.Sequential(
            nn.Linear(num_concepts, num_concepts // 2),
            nn.ReLU(),
            nn.Linear(num_concepts // 2, num_concepts // 4)
        )
        
        # 语义上下文整合器
        self.context_integrator = nn.Sequential(
            nn.Linear(feature_dim + (semantic_dim if (semantic_dim := getattr(self, 'semantic_dim', 256)) else 256), feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, features: Dict[str, torch.Tensor], 
                semantic_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """视觉概念提取前向传播"""
        feature_map = features['features']
        
        # 全局特征池化
        if len(feature_map.shape) > 2:
            global_features = F.adaptive_avg_pool2d(feature_map, 1).flatten(1)
        else:
            global_features = feature_map
        
        # 检测概念
        concept_activations = []
        for detector in self.concept_detectors:
            activation = detector(global_features)
            concept_activations.append(activation)
        
        concept_activations = torch.cat(concept_activations, dim=1)
        
        # 计算概念关系
        concept_relationships = self.concept_relations(concept_activations)
        
        # 整合语义上下文
        if semantic_context is not None:
            # 确保语义上下文维度匹配
            if semantic_context.shape[-1] != global_features.shape[-1]:
                semantic_context = F.interpolate(
                    semantic_context.unsqueeze(-1), 
                    size=global_features.shape[-1], 
                    mode='linear'
                ).squeeze(-1)
            
            integrated_features = self.context_integrator(
                torch.cat([global_features, semantic_context], dim=-1)
            )
        else:
            integrated_features = global_features
        
        # 概念强度和选择性
        concept_strength = concept_activations.mean(dim=1)
        concept_selectivity = self._compute_selectivity(concept_activations)
        
        return {
            'concept_activations': concept_activations,
            'concept_relationships': concept_relationships,
            'concept_strength': concept_strength,
            'concept_selectivity': concept_selectivity,
            'integrated_representation': integrated_features,
            'active_concepts': (concept_activations > 0.5).sum(dim=1).float(),
            'concept_diversity': self._compute_diversity(concept_activations)
        }
    
    def _compute_selectivity(self, activations: torch.Tensor) -> torch.Tensor:
        """计算概念选择性"""
        # 选择性 = 最强激活 / 平均激活
        max_activation = activations.max(dim=1, keepdim=True)[0]
        mean_activation = activations.mean(dim=1, keepdim=True)
        selectivity = max_activation / (mean_activation + 1e-8)
        return selectivity.squeeze(-1)
    
    def _compute_diversity(self, activations: torch.Tensor) -> torch.Tensor:
        """计算概念多样性"""
        # 多样性 = 激活的概念数量 / 总概念数量
        active_count = (activations > 0.1).sum(dim=1).float()
        diversity = active_count / activations.shape[1]
        return diversity


def create_visual_hierarchy(input_channels: int, feature_dim: int = 512) -> List[LayerConfig]:
    """创建视觉层次配置
    
    Args:
        input_channels: 输入通道数（通常为3表示RGB）
        feature_dim: 特征维度
        
    Returns:
        List[LayerConfig]: 视觉层次配置列表
    """
    configs = []
    
    # V1 - 初级视觉皮层
    v1_config = create_default_layer_config(
        LayerType.V1, input_channels, 64, kernel_size=7
    )
    configs.append(v1_config)
    
    # V2 - 次级视觉皮层
    v2_config = create_default_layer_config(
        LayerType.V2, 64, 128, kernel_size=5
    )
    configs.append(v2_config)
    
    # V4 - 第四级视觉皮层
    v4_config = create_default_layer_config(
        LayerType.V4, 128, 256, kernel_size=3
    )
    v4_config.attention_enabled = True
    configs.append(v4_config)
    
    # IT - 下颞皮层
    it_config = create_default_layer_config(
        LayerType.IT, 256, feature_dim, kernel_size=3
    )
    it_config.attention_enabled = True
    it_config.cross_modal_enabled = True
    configs.append(it_config)
    
    # VOTC - 腹侧枕颞皮层
    votc_config = create_default_layer_config(
        LayerType.VOTC, feature_dim, feature_dim, kernel_size=1
    )
    votc_config.attention_enabled = True
    votc_config.cross_modal_enabled = True
    votc_config.dropout_rate = 0.3
    configs.append(votc_config)
    
    return configs