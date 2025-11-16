"""
跨模态模块实现
==============

实现多感官信息整合和概念形成机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .processing_config import CrossModalConfig, ModuleType


class CrossModalModule(nn.Module):
    """跨模态模块基类"""
    
    def __init__(self, config: CrossModalConfig, input_dims: List[int]):
        super().__init__()
        self.config = config
        self.input_dims = input_dims
        self.feature_dim = config.feature_dim
        
        # 模态投影器
        self.modal_projectors = nn.ModuleList([
            nn.Linear(dim, config.feature_dim) for dim in input_dims
        ])
    
    def forward(self, modal_features: List[torch.Tensor]) -> Dict[str, Any]:
        raise NotImplementedError


class VisualLanguageInterface(CrossModalModule):
    """视觉-语言接口"""
    
    def __init__(self, config: CrossModalConfig, input_dims: List[int]):
        super().__init__(config, input_dims)
        
        # 视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(),
            nn.Linear(config.feature_dim, config.feature_dim // 2)
        )
        
        # 语言编码器
        self.language_encoder = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(),
            nn.Linear(config.feature_dim, config.feature_dim // 2)
        )
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            config.feature_dim // 2, num_heads=8, batch_first=True
        )
        
        # 概念检测器
        self.concept_detector = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(config.feature_dim // 2, 100),  # 100个概念
            nn.Sigmoid()
        )
    
    def forward(self, modal_features: List[torch.Tensor]) -> Dict[str, Any]:
        visual_features = self.visual_encoder(modal_features[0])
        language_features = self.language_encoder(modal_features[1])
        
        # 跨模态注意力
        attended_visual, _ = self.cross_attention(
            language_features.unsqueeze(1), visual_features.unsqueeze(1), visual_features.unsqueeze(1)
        )
        attended_visual = attended_visual.squeeze(1)
        
        # 整合特征
        integrated_features = torch.cat([visual_features, attended_visual], dim=1)
        
        # 概念检测
        concept_activations = self.concept_detector(integrated_features)
        
        return {
            'integrated_features': integrated_features,
            'concept_activations': concept_activations,
            'visual_features': visual_features,
            'language_features': language_features,
            'interface_type': 'visual_language'
        }


class AuditoryVisualInterface(CrossModalModule):
    """听觉-视觉接口"""
    
    def __init__(self, config: CrossModalConfig, input_dims: List[int]):
        super().__init__(config, input_dims)
        
        # 听觉特征提取
        self.audio_extractor = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(),
            nn.Linear(config.feature_dim, config.feature_dim // 2)
        )
        
        # 视觉特征提取
        self.visual_extractor = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(),
            nn.Linear(config.feature_dim, config.feature_dim // 2)
        )
        
        # 时间同步器
        self.temporal_synchronizer = nn.LSTM(
            config.feature_dim, config.feature_dim // 2, batch_first=True
        )
        
        # 跨模态整合器
        self.integration_network = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(),
            nn.Linear(config.feature_dim, config.feature_dim)
        )
    
    def forward(self, modal_features: List[torch.Tensor]) -> Dict[str, Any]:
        audio_features = self.audio_extractor(modal_features[0])
        visual_features = self.visual_extractor(modal_features[1])
        
        # 时间同步
        sync_output, _ = self.temporal_synchronizer(
            torch.cat([audio_features.unsqueeze(1), visual_features.unsqueeze(1)], dim=1)
        )
        
        # 跨模态整合
        integrated = self.integration_network(sync_output.mean(dim=1))
        
        return {
            'integrated_features': integrated,
            'audio_features': audio_features,
            'visual_features': visual_features,
            'synchronized_features': sync_output,
            'interface_type': 'auditory_visual'
        }


class MultimodalIntegrator(CrossModalModule):
    """多模态整合器"""
    
    def __init__(self, config: CrossModalConfig, input_dims: List[int]):
        super().__init__(config, input_dims)
        
        # 模态特定编码器
        self.modality_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, config.feature_dim),
                nn.ReLU(),
                nn.Linear(config.feature_dim, config.feature_dim // 2)
            ) for dim in input_dims
        ])
        
        # 注意力融合器
        if config.fusion_method == "attention":
            self.fusion_attention = nn.MultiheadAttention(
                config.feature_dim // 2, num_heads=8, batch_first=True
            )
        
        # 概念形成器
        self.concept_former = nn.Sequential(
            nn.Linear(config.feature_dim // 2 * len(input_dims), config.feature_dim),
            nn.ReLU(),
            nn.Linear(config.feature_dim, 50),
            nn.Sigmoid()
        )
        
        # 对齐质量评估器
        self.alignment_evaluator = nn.Sequential(
            nn.Linear(config.feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, modal_features: List[torch.Tensor]) -> Dict[str, Any]:
        # 编码各模态
        encoded_modalities = []
        for i, (encoder, features) in enumerate(zip(self.modality_encoders, modal_features)):
            encoded = encoder(features)
            encoded_modalities.append(encoded)
        
        # 模态融合
        if self.config.fusion_method == "attention":
            # 注意力融合
            query = encoded_modalities[0].unsqueeze(1)
            keys = [mod.unsqueeze(1) for mod in encoded_modalities[1:]]
            values = keys.copy()
            
            fused_features, attention_weights = self.fusion_attention(query, keys[0], values[0])
            fused_features = fused_features.squeeze(1)
            
        elif self.config.fusion_method == "concatenation":
            # 拼接融合
            fused_features = torch.cat(encoded_modalities, dim=1)
            
        else:  # weighted_sum
            # 加权求和
            weights = torch.ones(len(encoded_modalities)) / len(encoded_modalities)
            fused_features = sum(w * features for w, features in zip(weights, encoded_modalities))
        
        # 概念形成
        concepts = self.concept_former(fused_features)
        
        # 对齐质量评估
        alignment_quality = self.alignment_evaluator(fused_features)
        
        return {
            'integrated_features': fused_features,
            'concepts': concepts,
            'alignment_quality': alignment_quality,
            'modal_encodings': encoded_modalities,
            'fusion_method': self.config.fusion_method
        }


# 工厂函数
def create_crossmodal_module(input_dims: List[int],
                           target_dim: int,
                           fusion_method: str = "attention",
                           **kwargs):
    """创建跨模态模块"""
    config = CrossModalConfig(
        module_type=ModuleType.CROSSMODAL,
        feature_dim=target_dim,
        fusion_method=fusion_method
    )
    
    if fusion_method == "attention" and len(input_dims) == 2:
        # 假设是视觉-语言接口
        return VisualLanguageInterface(config, input_dims)
    elif fusion_method == "temporal" and len(input_dims) == 2:
        # 假设是听觉-视觉接口
        return AuditoryVisualInterface(config, input_dims)
    else:
        return MultimodalIntegrator(config, input_dims)