"""
注意调制机制
==============

实现V4层间区室化注意调制，包括：
- 空间注意力机制
- 特征注意力机制
- 注意门控机制
- 注意权重计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


class AttentionModulator(nn.Module):
    """注意调制器
    
    统一的注意调制接口，整合空间注意力和特征注意力。
    """
    
    def __init__(self, feature_dim: int, attention_type: str = "both"):
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_type = attention_type
        
        # 空间注意力
        if attention_type in ["spatial", "both"]:
            self.spatial_attention = SpatialAttention(feature_dim)
        
        # 特征注意力
        if attention_type in ["feature", "both"]:
            self.feature_attention = FeatureAttention(feature_dim)
        
        # 注意融合机制
        if attention_type == "both":
            self.attention_fusion = AttentionFusion(feature_dim)
        
        # 注意强度控制器
        self.attention_controller = AttentionController(feature_dim)
    
    def forward(self, features: torch.Tensor,
                spatial_hints: Optional[torch.Tensor] = None,
                feature_hints: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        注意调制前向传播
        
        Args:
            features: 输入特征 [batch, channels, height, width] 或 [batch, channels]
            spatial_hints: 空间注意力提示
            feature_hints: 特征注意力提示
            context: 上下文信息
            
        Returns:
            dict: 注意调制结果
        """
        batch_size = features.shape[0]
        
        # 输入特征标准化
        if len(features.shape) > 2:
            input_shape = features.shape
            features_flat = features.view(batch_size, features.shape[1], -1)
        else:
            features_flat = features.unsqueeze(-1)
            input_shape = features.shape
        
        # 计算注意权重
        attention_weights = {}
        attended_features = {}
        
        # 空间注意力
        if hasattr(self, 'spatial_attention'):
            spatial_result = self.spatial_attention(features)
            attention_weights['spatial'] = spatial_result['spatial_weights']
            if spatial_hints is not None:
                # 应用空间提示
                attended_features['spatial'] = self._apply_spatial_hints(
                    spatial_result['attended_features'], spatial_hints
                )
            else:
                attended_features['spatial'] = spatial_result['attended_features']
        
        # 特征注意力
        if hasattr(self, 'feature_attention'):
            feature_result = self.feature_attention(features)
            attention_weights['feature'] = feature_result['feature_weights']
            if feature_hints is not None:
                # 应用特征提示
                attended_features['feature'] = self._apply_feature_hints(
                    feature_result['attended_features'], feature_hints
                )
            else:
                attended_features['feature'] = feature_result['attended_features']
        
        # 注意融合
        if hasattr(self, 'attention_fusion'):
            # 准备融合输入
            if 'spatial' in attended_features and 'feature' in attended_features:
                fusion_input = {
                    'spatial': attended_features['spatial'],
                    'feature': attended_features['feature']
                }
            elif 'spatial' in attended_features:
                fusion_input = {'spatial': attended_features['spatial']}
            elif 'feature' in attended_features:
                fusion_input = {'feature': attended_features['feature']}
            else:
                fusion_input = {'features': features}
            
            fusion_result = self.attention_fusion(fusion_input, context)
            final_attended = fusion_result['fused_features']
            fused_weights = fusion_result['fusion_weights']
        else:
            # 如果没有融合机制，选择主要注意力结果
            if attended_features:
                final_attended = list(attended_features.values())[0]
                fused_weights = attention_weights.get(list(attention_weights.keys())[0])
            else:
                final_attended = features
                fused_weights = torch.ones_like(features[:, :1, ...])
        
        # 注意强度控制
        controlled_result = self.attention_controller(final_attended, context)
        
        return {
            'attended_features': controlled_result['controlled_features'],
            'attention_weights': attention_weights,
            'fusion_weights': fused_weights if hasattr(self, 'attention_fusion') else None,
            'attention_intensity': controlled_result['intensity'],
            'attention_quality': controlled_result['quality'],
            'attention_summary': {
                'types_used': list(attention_weights.keys()),
                'total_attention_strength': sum(w.mean() for w in attention_weights.values()),
                'attention_coherence': self._compute_attention_coherence(attention_weights)
            }
        }
    
    def _apply_spatial_hints(self, features: torch.Tensor, hints: torch.Tensor) -> torch.Tensor:
        """应用空间注意力提示"""
        # 确保提示形状匹配
        if features.shape[-2:] != hints.shape[-2:]:
            hints = F.interpolate(hints.unsqueeze(1), size=features.shape[-2:], 
                                mode='bilinear', align_corners=False).squeeze(1)
        
        # 应用提示作为额外的空间注意力
        hinted_features = features * hints
        return hinted_features
    
    def _apply_feature_hints(self, features: torch.Tensor, hints: torch.Tensor) -> torch.Tensor:
        """应用特征注意力提示"""
        # 特征提示通常是一个权重向量
        if len(hints.shape) == 1:
            hints = hints.unsqueeze(0).expand(features.shape[0], -1)
        
        # 应用到特征维度
        if len(features.shape) > 1:
            if hints.shape[-1] == features.shape[-3]:  # 特征维度匹配
                # 扩展提示到空间维度
                spatial_size = features.shape[-2] * features.shape[-1]
                hints_expanded = hints.unsqueeze(-1).expand(-1, -1, spatial_size)
                hints_reshaped = hints_expanded.view(features.shape)
                
                hinted_features = features * hints_reshaped
            else:
                hinted_features = features
        else:
            hinted_features = features
        
        return hinted_features
    
    def _compute_attention_coherence(self, attention_weights: Dict[str, torch.Tensor]) -> float:
        """计算注意一致性"""
        if len(attention_weights) < 2:
            return 1.0
        
        weights_list = list(attention_weights.values())
        
        # 计算注意力分布的相关性
        coherence_scores = []
        for i in range(len(weights_list) - 1):
            for j in range(i + 1, len(weights_list)):
                # 展平权重以便计算相关性
                w1 = weights_list[i].flatten()
                w2 = weights_list[j].flatten()
                
                # 计算余弦相似性
                similarity = F.cosine_similarity(w1, w2, dim=0)
                coherence_scores.append(similarity.item())
        
        return np.mean(coherence_scores) if coherence_scores else 0.0


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
        
        # 空间特征增强器
        self.spatial_enhancer = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, 3, 1, 1)
        )
        
        # 多尺度空间注意力
        self.multi_scale_attention = nn.ModuleList([
            nn.Conv2d(feature_dim, 1, kernel_size, padding=kernel_size//2)
            for kernel_size in [1, 3, 7, 15]
        ])
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """空间注意力前向传播"""
        batch_size, channels, height, width = features.shape
        
        # 基础空间注意力
        base_attention = self.spatial_attention_generator(features)
        
        # 多尺度空间注意力
        multi_scale_attentions = []
        for attention_conv in self.multi_scale_attention:
            attention_map = attention_conv(features)
            attention_map = torch.sigmoid(attention_map)
            multi_scale_attentions.append(attention_map)
        
        # 组合多尺度注意力
        combined_attention = torch.stack(multi_scale_attentions, dim=0).mean(dim=0)
        combined_attention = torch.sigmoid(combined_attention)
        
        # 融合基础注意力和多尺度注意力
        final_attention = 0.6 * base_attention + 0.4 * combined_attention
        
        # 增强空间特征
        enhanced_features = self.spatial_enhancer(features)
        
        # 应用空间注意力
        attended_features = features * final_attention + enhanced_features * (1 - final_attention)
        
        # 计算注意统计
        attention_entropy = self._compute_attention_entropy(final_attention)
        attention_sparsity = self._compute_attention_sparsity(final_attention)
        
        return {
            'attended_features': attended_features,
            'spatial_weights': final_attention,
            'attention_entropy': attention_entropy,
            'attention_sparsity': attention_sparsity,
            'multi_scale_attentions': multi_scale_attentions
        }
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """计算注意力熵"""
        # 将注意力权重归一化为概率分布
        eps = 1e-8
        normalized_weights = attention_weights / (attention_weights.sum() + eps)
        entropy = -torch.sum(normalized_weights * torch.log(normalized_weights + eps))
        return entropy / (attention_weights.numel() * np.log(attention_weights.numel() + eps))  # 归一化熵
    
    def _compute_attention_sparsity(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """计算注意力稀疏性"""
        # 使用Gini系数作为稀疏性度量
        sorted_weights = torch.sort(attention_weights.flatten())[0]
        n = sorted_weights.numel()
        index = torch.arange(1, n + 1).float()
        gini = (torch.sum((2 * index - n - 1) * sorted_weights)) / (n * torch.sum(sorted_weights))
        return 1 - gini  # 转换为稀疏性度量（值越大越稀疏）


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
        
        # 特征重加权器
        self.feature_reweighter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # 特征选择器
        self.feature_selector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """特征注意力前向传播"""
        batch_size, channels, height, width = features.shape
        
        # 全局特征池化
        global_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # 生成特征注意力权重
        feature_weights = self.feature_attention_generator(features)
        
        # 特征重加权
        reweighted_features = self.feature_reweighter(feature_weights).unsqueeze(-1).unsqueeze(-1)
        attended_features = features * reweighted_features
        
        # 特征选择
        selection_weights = self.feature_selector(global_features).unsqueeze(-1).unsqueeze(-1)
        selected_features = features * selection_weights
        
        # 融合重加权和选择性特征
        fused_features = 0.5 * attended_features + 0.5 * selected_features
        
        # 计算特征统计
        feature_selectivity = self._compute_selectivity(feature_weights)
        feature_diversity = self._compute_diversity(attended_features)
        
        return {
            'attended_features': fused_features,
            'feature_weights': feature_weights,
            'selection_weights': selection_weights,
            'selectivity': feature_selectivity,
            'diversity': feature_diversity,
            'reweighted_features': attended_features,
            'selected_features': selected_features
        }
    
    def _compute_selectivity(self, weights: torch.Tensor) -> torch.Tensor:
        """计算特征选择性"""
        # 选择性 = 最强特征权重 / 平均特征权重
        max_weight = weights.max(dim=1, keepdim=True)[0]
        mean_weight = weights.mean(dim=1, keepdim=True)
        selectivity = max_weight / (mean_weight + 1e-8)
        return selectivity.mean()
    
    def _compute_diversity(self, features: torch.Tensor) -> torch.Tensor:
        """计算特征多样性"""
        # 计算特征间的相似性，多样性 = 1 - 平均相似性
        batch_size = features.shape[0]
        
        # 全局池化特征
        pooled_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # 计算特征间相关性
        correlation_matrix = torch.corrcoef(pooled_features)
        
        # 提取非对角线元素（排除自相关）
        off_diagonal = correlation_matrix[~torch.eye(correlation_matrix.shape[0], dtype=torch.bool)]
        avg_correlation = off_diagonal.abs().mean()
        
        diversity = 1 - avg_correlation
        return diversity


class AttentionFusion(nn.Module):
    """注意融合机制"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 融合权重计算器
        self.fusion_weights = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 2),
            nn.Softmax(dim=1)
        )
        
        # 融合特征生成器
        self.fusion_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 融合质量评估器
        self.quality_assessor = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, attention_inputs: Dict[str, torch.Tensor], 
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """注意融合前向传播"""
        if len(attention_inputs) == 1:
            # 只有一个注意力输入，直接返回
            key = list(attention_inputs.keys())[0]
            return {
                'fused_features': attention_inputs[key],
                'fusion_weights': torch.ones(1, 1),
                'fusion_quality': torch.tensor(1.0)
            }
        
        # 准备融合输入
        if 'spatial' in attention_inputs and 'feature' in attention_inputs:
            spatial_features = attention_inputs['spatial']
            feature_features = attention_inputs['feature']
            
            # 池化到相同大小
            if len(spatial_features.shape) > 2:
                spatial_pooled = F.adaptive_avg_pool2d(spatial_features, 1).flatten(1)
            else:
                spatial_pooled = spatial_features
            
            if len(feature_features.shape) > 2:
                feature_pooled = F.adaptive_avg_pool2d(feature_features, 1).flatten(1)
            else:
                feature_pooled = feature_features
            
            # 计算融合权重
            combined_input = torch.cat([spatial_pooled, feature_pooled], dim=1)
            fusion_weights = self.fusion_weights(combined_input)
            
            # 生成融合特征
            fused_features_input = torch.cat([spatial_features, feature_features], dim=1)
            fused_features = self.fusion_generator(fused_features_input)
            
            # 评估融合质量
            quality_score = self.quality_assessor(fused_features).mean()
        else:
            # 其他融合情况
            features_list = list(attention_inputs.values())
            fused_features = torch.stack(features_list, dim=1).mean(dim=1)
            fusion_weights = torch.ones(1, len(attention_inputs)) / len(attention_inputs)
            quality_score = torch.tensor(0.5)
        
        return {
            'fused_features': fused_features,
            'fusion_weights': fusion_weights,
            'fusion_quality': quality_score
        }


class AttentionController(nn.Module):
    """注意控制器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 注意力强度控制器
        self.intensity_controller = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 注意力质量评估器
        self.quality_evaluator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 注意力调节器
        self.attention_modulator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Tanh()
        )
    
    def forward(self, features: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """注意控制前向传播"""
        batch_size = features.shape[0]
        
        # 计算特征统计
        feature_mean = features.mean()
        feature_std = features.std()
        
        # 基础强度控制
        base_intensity = torch.sigmoid((feature_mean - 0.5) / (feature_std + 1e-8))
        
        # 上下文调节
        if context is not None:
            context_intensity = self.intensity_controller(context)
            final_intensity = 0.7 * base_intensity + 0.3 * context_intensity
        else:
            final_intensity = base_intensity
        
        # 应用注意力调节
        modulation_weights = self.attention_modulator(features.mean(dim=[2, 3]) if len(features.shape) > 2 else features.mean(dim=1))
        controlled_features = features * modulation_weights.unsqueeze(-1).unsqueeze(-1)
        
        # 计算注意质量
        quality_score = self.quality_evaluator(controlled_features.mean())
        
        return {
            'controlled_features': controlled_features,
            'intensity': final_intensity.mean(),
            'quality': quality_score.mean(),
            'modulation_weights': modulation_weights
        }