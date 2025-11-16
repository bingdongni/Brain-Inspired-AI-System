"""
注意模块实现
=============

实现新皮层的注意机制，包括：
- 空间注意力：选择视觉空间位置
- 特征注意力：选择特征维度
- 时间注意力：选择时间序列片段
- 任务注意力：基于任务目标的注意分配
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .processing_config import (
    AttentionConfig, AttentionType, ProcessingMode, ModuleType
)


class AttentionModule(nn.Module):
    """
    注意模块
    
    实现基于新皮层注意控制机制的智能注意力分配。
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        
        # 创建注意引擎
        if config.attention_type == AttentionType.SPATIAL:
            self.attention_engine = SpatialAttentionEngine(config)
        elif config.attention_type == AttentionType.FEATURE:
            self.attention_engine = FeatureAttentionEngine(config)
        elif config.attention_type == AttentionType.TEMPORAL:
            self.attention_engine = TemporalAttentionEngine(config)
        elif config.attention_type == AttentionType.TASK:
            self.attention_engine = TaskAttentionEngine(config)
        elif config.attention_type == AttentionType.COMBINED:
            self.attention_engine = CombinedAttentionEngine(config)
        else:
            raise ValueError(f"不支持的注意力类型: {config.attention_type}")
        
        # 注意质量控制器
        self.attention_controller = AttentionController(config)
        
        # 注意记忆系统
        self.attention_memory = AttentionMemorySystem(config)
        
        # 统计信息
        self.stats = {
            'total_attention_events': 0,
            'successful_attentions': 0,
            'attention_patterns': [],
            'avg_attention_intensity': 0.0
        }
    
    def forward(self, features: torch.Tensor,
                task_context: Optional[Dict[str, torch.Tensor]] = None,
                spatial_hints: Optional[torch.Tensor] = None,
                feature_hints: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        注意模块前向传播
        
        Args:
            features: 输入特征 [batch, feature_dim] 或 [batch, channels, height, width]
            task_context: 任务上下文信息
            spatial_hints: 空间注意力提示
            feature_hints: 特征注意力提示
            
        Returns:
            dict: 注意力处理结果
        """
        batch_size = features.shape[0]
        self.stats['total_attention_events'] += batch_size
        
        # 预处理特征
        processed_features = self._preprocess_features(features)
        
        # 应用注意引擎
        attention_result = self.attention_engine(
            processed_features,
            task_context=task_context,
            spatial_hints=spatial_hints,
            feature_hints=feature_hints
        )
        
        # 注意力控制
        controlled_attention = self.attention_controller(
            attention_result, processed_features
        )
        
        # 更新注意力记忆
        memory_update = self.attention_memory.update(
            processed_features, controlled_attention
        )
        
        # 应用注意力到原始特征
        final_features = self._apply_attention(
            features, controlled_attention['attention_maps']
        )
        
        # 更新统计
        attention_intensity = controlled_attention['attention_intensity'].mean().item()
        self.stats['avg_attention_intensity'] = (
            (self.stats['avg_attention_intensity'] * (self.stats['total_attention_events'] - batch_size) +
             attention_intensity * batch_size) / self.stats['total_attention_events']
        )
        
        if attention_intensity > self.config.selectivity_threshold:
            self.stats['successful_attentions'] += 1
        
        return {
            'attended_features': final_features,
            'attention_maps': controlled_attention['attention_maps'],
            'attention_weights': controlled_attention['attention_weights'],
            'attention_intensity': controlled_attention['attention_intensity'],
            'attention_quality': controlled_attention['attention_quality'],
            'attention_metadata': {
                'type': self.config.attention_type.value,
                'selectivity_threshold': self.config.selectivity_threshold,
                'top_k_active': controlled_attention.get('top_k_active', 0),
                'attention_sparsity': controlled_attention.get('attention_sparsity', 0.0)
            },
            'memory_state': memory_update,
            'statistics': {
                'total_events': self.stats['total_attention_events'],
                'success_rate': self.stats['successful_attentions'] / max(self.stats['total_attention_events'], 1),
                'avg_intensity': self.stats['avg_attention_intensity'],
                'recent_patterns': self.stats['attention_patterns'][-5:] if self.stats['attention_patterns'] else []
            }
        }
    
    def _preprocess_features(self, features: torch.Tensor) -> torch.Tensor:
        """预处理特征"""
        # 特征维度标准化
        if len(features.shape) == 2:
            return features
        
        if len(features.shape) == 4:
            # 2D特征图：展平空间维度
            batch_size, channels, height, width = features.shape
            return features.view(batch_size, channels, height * width)
        
        return features
    
    def _apply_attention(self, original_features: torch.Tensor, 
                        attention_maps: Dict[str, torch.Tensor]) -> torch.Tensor:
        """将注意力应用到原始特征"""
        if not attention_maps:
            return original_features
        
        # 合并所有注意力地图
        combined_attention = None
        for attention_type, attention_map in attention_maps.items():
            if combined_attention is None:
                combined_attention = attention_map
            else:
                combined_attention = combined_attention + attention_map
        
        # 应用注意力
        if len(original_features.shape) == 4:  # [batch, channels, height, width]
            if len(combined_attention.shape) == 3:  # [batch, height*width, 1]
                attention_weighted = combined_attention.transpose(1, 2).unsqueeze(-1)
            else:
                attention_weighted = combined_attention
            
            attended_features = original_features * attention_weighted
        else:
            attended_features = original_features * combined_attention
        
        return attended_features


class CombinedAttentionEngine(nn.Module):
    """组合注意力引擎
    
    整合多种注意机制的综合注意力系统。
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        
        # 子注意力引擎
        self.spatial_engine = SpatialAttentionEngine(config) if config.spatial_attention else None
        self.feature_engine = FeatureAttentionEngine(config) if config.feature_attention else None
        self.temporal_engine = TemporalAttentionEngine(config) if config.temporal_attention else None
        
        # 注意力融合器
        self.attention_fusion = AttentionFusion(config)
        
        # 竞争调节器
        self.competition_regulator = CompetitionRegulator(config)
        
        # 注意力协调器
        self.attention_coordinator = AttentionCoordinator(config)
    
    def forward(self, features: torch.Tensor,
                task_context: Optional[Dict[str, torch.Tensor]] = None,
                spatial_hints: Optional[torch.Tensor] = None,
                feature_hints: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """组合注意引擎前向传播"""
        batch_size = features.shape[0]
        attention_results = {}
        
        # 并行运行各种注意机制
        if self.spatial_engine is not None:
            spatial_result = self.spatial_engine(
                features, task_context=task_context, spatial_hints=spatial_hints
            )
            attention_results['spatial'] = spatial_result
        
        if self.feature_engine is not None:
            feature_result = self.feature_engine(
                features, task_context=task_context, feature_hints=feature_hints
            )
            attention_results['feature'] = feature_result
        
        if self.temporal_engine is not None:
            temporal_result = self.temporal_engine(
                features, task_context=task_context
            )
            attention_results['temporal'] = temporal_result
        
        # 竞争调节
        if len(attention_results) > 1:
            regulated_results = self.competition_regulator(attention_results)
        else:
            regulated_results = attention_results
        
        # 注意力融合
        fused_attention = self.attention_fusion(regulated_results)
        
        # 注意力协调
        coordinated_attention = self.attention_coordinator(
            fused_attention, task_context
        )
        
        return coordinated_attention


class SpatialAttentionEngine(nn.Module):
    """空间注意力引擎"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        
        # 空间注意力生成器
        self.spatial_attention_generator = nn.Sequential(
            nn.Conv2d(config.feature_dim, config.feature_dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(config.feature_dim // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 多尺度空间注意力
        self.multi_scale_generator = nn.ModuleList([
            nn.Conv2d(config.feature_dim, 1, kernel_size, padding=kernel_size//2)
            for kernel_size in [1, 3, 7, 15]
        ])
        
        # 空间注意力增强器
        self.spatial_enhancer = nn.Sequential(
            nn.Conv2d(config.feature_dim, config.feature_dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(config.feature_dim, config.feature_dim, 3, 1, 1)
        )
    
    def forward(self, features: torch.Tensor,
                task_context: Optional[Dict[str, torch.Tensor]] = None,
                spatial_hints: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """空间注意力前向传播"""
        batch_size, channels, height, width = features.shape
        
        # 基础空间注意力
        base_attention = self.spatial_attention_generator(features)
        
        # 多尺度空间注意力
        multi_scale_attentions = []
        for scale_generator in self.multi_scale_generator:
            attention_map = scale_generator(features)
            attention_map = torch.sigmoid(attention_map)
            multi_scale_attentions.append(attention_map)
        
        # 组合多尺度注意力
        combined_attention = torch.stack(multi_scale_attentions, dim=0).mean(dim=0)
        combined_attention = torch.sigmoid(combined_attention)
        
        # 融合基础注意力和多尺度注意力
        final_attention = 0.6 * base_attention + 0.4 * combined_attention
        
        # 应用空间提示
        if spatial_hints is not None:
            # 确保提示形状匹配
            if final_attention.shape != spatial_hints.shape:
                spatial_hints = F.interpolate(
                    spatial_hints.unsqueeze(1), 
                    size=final_attention.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(1)
            
            # 增强基于提示的注意力
            final_attention = final_attention * 0.7 + spatial_hints * 0.3
        
        # 应用任务上下文
        if task_context is not None:
            task_weight = task_context.get('spatial_relevance', 0.5)
            final_attention = final_attention * task_weight + final_attention * (1 - task_weight)
        
        # 增强空间特征
        enhanced_features = self.spatial_enhancer(features)
        
        # 注意力统计
        attention_sparsity = self._compute_attention_sparsity(final_attention)
        attention_entropy = self._compute_attention_entropy(final_attention)
        
        return {
            'attention_map': final_attention,
            'enhanced_features': enhanced_features,
            'spatial_attention': final_attention,
            'multi_scale_attentions': multi_scale_attentions,
            'attention_statistics': {
                'sparsity': attention_sparsity,
                'entropy': attention_entropy,
                'max_activation': final_attention.max(),
                'mean_activation': final_attention.mean()
            }
        }
    
    def _compute_attention_sparsity(self, attention_map: torch.Tensor) -> torch.Tensor:
        """计算注意力稀疏性"""
        # 使用Gini系数
        sorted_weights = torch.sort(attention_map.flatten())[0]
        n = sorted_weights.numel()
        index = torch.arange(1, n + 1).float()
        gini = (torch.sum((2 * index - n - 1) * sorted_weights)) / (n * torch.sum(sorted_weights))
        return 1 - gini
    
    def _compute_attention_entropy(self, attention_map: torch.Tensor) -> torch.Tensor:
        """计算注意力熵"""
        eps = 1e-8
        normalized = attention_map / (attention_map.sum() + eps)
        entropy = -torch.sum(normalized * torch.log(normalized + eps))
        return entropy / (attention_map.numel() * np.log(attention_map.numel() + eps))


class FeatureAttentionEngine(nn.Module):
    """特征注意力引擎"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        
        # 特征注意力生成器
        self.feature_attention_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.feature_dim, config.feature_dim // 4),
            nn.ReLU(),
            nn.Linear(config.feature_dim // 4, config.feature_dim),
            nn.Sigmoid()
        )
        
        # 特征选择器
        self.feature_selector = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.Tanh(),
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.Sigmoid()
        )
        
        # 特征重加权器
        self.feature_reweighter = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.ReLU(),
            nn.Linear(config.feature_dim, config.feature_dim)
        )
    
    def forward(self, features: torch.Tensor,
                task_context: Optional[Dict[str, torch.Tensor]] = None,
                feature_hints: Optional[torch.Tensor] = None) -> Dict[str, Any]:
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
        
        # 应用特征提示
        if feature_hints is not None:
            # 特征提示通常是一个权重向量
            if len(feature_hints.shape) == 1:
                hints = feature_hints.unsqueeze(0).expand(batch_size, -1)
            else:
                hints = feature_hints
            
            # 应用提示
            hint_weights = hints.unsqueeze(-1).unsqueeze(-1)
            fused_features = fused_features * hint_weights + fused_features * (1 - hint_weights)
        
        # 应用任务上下文
        if task_context is not None:
            task_weight = task_context.get('feature_relevance', 0.5)
            context_features = task_context.get('context_features', global_features)
            
            if context_features.shape[-1] == global_features.shape[-1]:
                # 基于上下文调整特征权重
                context_weight = self.feature_attention_generator(context_features.unsqueeze(-1).unsqueeze(-1))
                fused_features = fused_features * context_weight * task_weight + fused_features * (1 - task_weight)
        
        # 计算特征统计
        feature_selectivity = self._compute_selectivity(feature_weights)
        feature_diversity = self._compute_diversity(fused_features)
        
        return {
            'feature_attention': feature_weights,
            'attended_features': fused_features,
            'selection_weights': selection_weights,
            'feature_statistics': {
                'selectivity': feature_selectivity,
                'diversity': feature_diversity,
                'attention_coherence': self._compute_coherence(feature_weights)
            }
        }
    
    def _compute_selectivity(self, weights: torch.Tensor) -> torch.Tensor:
        """计算特征选择性"""
        max_weight = weights.max(dim=1, keepdim=True)[0]
        mean_weight = weights.mean(dim=1, keepdim=True)
        selectivity = max_weight / (mean_weight + 1e-8)
        return selectivity.mean()
    
    def _compute_diversity(self, features: torch.Tensor) -> torch.Tensor:
        """计算特征多样性"""
        batch_size = features.shape[0]
        pooled_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # 计算特征间相关性
        correlation_matrix = torch.corrcoef(pooled_features)
        off_diagonal = correlation_matrix[~torch.eye(correlation_matrix.shape[0], dtype=torch.bool)]
        avg_correlation = off_diagonal.abs().mean()
        
        diversity = 1 - avg_correlation
        return diversity
    
    def _compute_coherence(self, weights: torch.Tensor) -> torch.Tensor:
        """计算注意力连贯性"""
        # 基于权重的方差评估连贯性
        weight_variance = weights.var(dim=1).mean()
        coherence = 1 / (1 + weight_variance)
        return coherence


class TaskAttentionEngine(nn.Module):
    """任务注意力引擎"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        
        # 任务相关性评估器
        self.task_relevance_evaluator = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(config.feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 任务特定注意力生成器
        self.task_specific_generator = nn.Sequential(
            nn.Linear(config.feature_dim + 1, config.feature_dim),  # +1 for task embedding
            nn.ReLU(),
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.Sigmoid()
        )
        
        # 任务记忆系统
        self.task_memory = nn.GRU(
            config.feature_dim, config.feature_dim // 2, batch_first=True
        )
    
    def forward(self, features: torch.Tensor,
                task_context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """任务注意力前向传播"""
        batch_size = features.shape[0]
        
        # 评估任务相关性
        relevance_scores = self.task_relevance_evaluator(features)
        
        # 生成任务特定注意力
        if task_context is not None:
            task_embedding = task_context.get('task_embedding', torch.zeros(batch_size, 1))
            task_features = torch.cat([features.mean(dim=[2, 3]) if len(features.shape) > 2 else features, task_embedding], dim=1)
        else:
            task_features = features.mean(dim=[2, 3]) if len(features.shape) > 2 else features
        
        task_specific_attention = self.task_specific_generator(task_features)
        
        # 应用任务记忆
        task_memory_input = features.mean(dim=[2, 3]) if len(features.shape) > 2 else features
        memory_output, _ = self.task_memory(task_memory_input.unsqueeze(1))
        
        # 基于记忆调整注意力
        memory_adjusted_attention = task_specific_attention * torch.sigmoid(memory_output.squeeze(1))
        
        # 计算任务完成度
        task_completion = self._compute_task_completion(
            memory_adjusted_attention, relevance_scores
        )
        
        # 应用Top-K选择
        top_k_attention = self._apply_top_k_selection(
            memory_adjusted_attention, self.config.top_k_attention
        )
        
        return {
            'task_attention': memory_adjusted_attention,
            'relevance_scores': relevance_scores,
            'task_memory': memory_output,
            'task_completion': task_completion,
            'top_k_attention': top_k_attention,
            'task_statistics': {
                'completion_rate': task_completion.mean(),
                'attention_intensity': memory_adjusted_attention.mean(),
                'task_focus': relevance_scores.mean()
            }
        }
    
    def _compute_task_completion(self, attention: torch.Tensor, 
                               relevance: torch.Tensor) -> torch.Tensor:
        """计算任务完成度"""
        # 基于注意力和相关性计算完成度
        completion = attention * relevance
        return completion
    
    def _apply_top_k_selection(self, attention: torch.Tensor, k: int) -> torch.Tensor:
        """应用Top-K注意力选择"""
        batch_size = attention.shape[0]
        
        # 选择Top-K特征
        top_k_values, top_k_indices = torch.topk(attention, k, dim=1)
        
        # 创建稀疏注意力向量
        sparse_attention = torch.zeros_like(attention)
        sparse_attention.scatter_(1, top_k_indices, top_k_values)
        
        return sparse_attention


# 辅助模块

class AttentionFusion(nn.Module):
    """注意力融合器"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        self.fusion_network = nn.Sequential(
            nn.Linear(config.feature_dim * 3, config.feature_dim),  # spatial + feature + task
            nn.ReLU(),
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, attention_results: Dict[str, Dict]) -> Dict[str, Any]:
        """融合多种注意力结果"""
        if len(attention_results) == 1:
            key = list(attention_results.keys())[0]
            return attention_results[key]
        
        # 提取各注意力结果
        attentions = []
        for attention_type, result in attention_results.items():
            if 'attention_map' in result:
                attention = result['attention_map']
            elif 'feature_attention' in result:
                attention = result['feature_attention']
            elif 'task_attention' in result:
                attention = result['task_attention']
            else:
                continue
            attentions.append(attention.flatten(1))
        
        # 融合注意力
        if attentions:
            combined_attention = torch.cat(attentions, dim=1)
            fused_attention = self.fusion_network(combined_attention)
            
            return {
                'fused_attention': fused_attention,
                'fusion_weights': torch.ones(len(attentions)) / len(attentions),
                'fusion_method': 'weighted_average'
            }
        else:
            return {}


class CompetitionRegulator(nn.Module):
    """竞争调节器"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
    
    def forward(self, attention_results: Dict[str, Dict]) -> Dict[str, Any]:
        """调节注意力之间的竞争"""
        # 简单的竞争调节：基于重要性加权
        regulated_results = {}
        
        for attention_type, result in attention_results.items():
            # 提取注意力强度
            if 'attention_map' in result:
                intensity = result['attention_map'].mean()
            elif 'feature_attention' in result:
                intensity = result['feature_attention'].mean()
            elif 'task_attention' in result:
                intensity = result['task_attention'].mean()
            else:
                intensity = 0.5
            
            # 应用竞争因子
            competition_factor = 1 - intensity * (1 - self.config.competition_factor)
            regulated_results[attention_type] = result
            
            # 调整结果
            for key, value in result.items():
                if isinstance(value, torch.Tensor) and 'attention' in key.lower():
                    regulated_results[attention_type][key] = value * competition_factor
        
        return regulated_results


class AttentionCoordinator(nn.Module):
    """注意力协调器"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        self.coordinator = nn.Sequential(
            nn.Linear(config.feature_dim, config.feature_dim // 2),
            nn.ReLU(),
            nn.Linear(config.feature_dim // 2, config.feature_dim)
        )
    
    def forward(self, attention_result: Dict[str, Any],
                task_context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """协调注意力结果"""
        # 从注意力结果中提取特征
        if 'fused_attention' in attention_result:
            attention_feature = attention_result['fused_attention']
        elif 'attention_map' in attention_result:
            attention_feature = attention_result['attention_map'].flatten(1)
        elif 'feature_attention' in attention_result:
            attention_feature = attention_result['feature_attention']
        else:
            attention_feature = torch.zeros(1, self.config.feature_dim)
        
        # 协调处理
        coordinated_feature = self.coordinator(attention_feature)
        
        # 应用任务上下文
        if task_context is not None:
            task_weight = task_context.get('coordination_weight', 0.5)
            coordinated_feature = coordinated_feature * task_weight + attention_feature * (1 - task_weight)
        
        # 构建最终的注意力地图
        final_attention_maps = {}
        if 'fused_attention' in attention_result:
            final_attention_maps['combined'] = attention_result['fused_attention']
        for key, value in attention_result.items():
            if 'attention' in key and isinstance(value, torch.Tensor):
                final_attention_maps[key] = value
        
        return {
            'attention_maps': final_attention_maps,
            'attention_weights': coordinated_feature,
            'attention_intensity': coordinated_feature.mean(),
            'attention_quality': self._compute_quality(attention_result),
            'coordinated': True
        }
    
    def _compute_quality(self, attention_result: Dict[str, Any]) -> torch.Tensor:
        """计算注意力质量"""
        # 基于注意力的一致性和强度计算质量
        attention_values = []
        for key, value in attention_result.items():
            if isinstance(value, torch.Tensor) and 'attention' in key.lower():
                attention_values.append(value.mean().item())
        
        if attention_values:
            return torch.tensor(np.mean(attention_values))
        else:
            return torch.tensor(0.5)


class AttentionController(nn.Module):
    """注意力控制器"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
    
    def forward(self, attention_result: Dict[str, Any], 
                features: torch.Tensor) -> Dict[str, Any]:
        """控制注意力结果"""
        # 提取注意力地图
        attention_maps = {}
        attention_weights = []
        
        for key, value in attention_result.items():
            if 'attention' in key and isinstance(value, torch.Tensor):
                attention_maps[key] = value
                attention_weights.append(value.mean().item())
        
        if not attention_maps:
            # 创建默认注意力
            batch_size = features.shape[0]
            default_attention = torch.ones(batch_size, 1) * 0.5
            attention_maps['default'] = default_attention
            attention_weights = [0.5]
        
        # 计算注意力强度
        attention_intensity = torch.tensor(np.mean(attention_weights))
        
        # 计算注意力质量
        attention_quality = torch.tensor(self._compute_attention_quality(attention_maps))
        
        # 应用稀疏性约束
        top_k_active = self._count_top_k_active(attention_maps, self.config.top_k_attention)
        attention_sparsity = self._compute_global_sparsity(attention_maps)
        
        return {
            'attention_maps': attention_maps,
            'attention_weights': attention_weights,
            'attention_intensity': attention_intensity,
            'attention_quality': attention_quality,
            'top_k_active': top_k_active,
            'attention_sparsity': attention_sparsity
        }
    
    def _compute_attention_quality(self, attention_maps: Dict[str, torch.Tensor]) -> float:
        """计算注意力质量"""
        if not attention_maps:
            return 0.0
        
        quality_scores = []
        for attention_map in attention_maps.values():
            # 基于分布均匀性评估质量
            mean_attention = attention_map.mean()
            std_attention = attention_map.std()
            
            # 质量 = 一致性 - 过度集中
            consistency = 1 / (1 + std_attention)
            focus_penalty = max(0, mean_attention - 0.5) * 2
            
            quality = consistency - focus_penalty * 0.1
            quality_scores.append(max(0, min(1, quality)))
        
        return np.mean(quality_scores)
    
    def _count_top_k_active(self, attention_maps: Dict[str, torch.Tensor], k: int) -> int:
        """计算Top-K活跃注意力数量"""
        total_active = 0
        for attention_map in attention_maps.values():
            top_k_values = torch.topk(attention_map.flatten(), k)[0]
            active_count = (top_k_values > self.config.selectivity_threshold).sum()
            total_active += active_count.item()
        
        return total_active
    
    def _compute_global_sparsity(self, attention_maps: Dict[str, torch.Tensor]) -> float:
        """计算全局注意力稀疏性"""
        all_attentions = []
        for attention_map in attention_maps.values():
            all_attentions.extend(attention_map.flatten().tolist())
        
        if not all_attentions:
            return 0.0
        
        # 计算非零注意力比例作为稀疏性度量
        non_zero_ratio = sum(1 for x in all_attentions if x > 0.1) / len(all_attentions)
        return 1 - non_zero_ratio


class AttentionMemorySystem(nn.Module):
    """注意力记忆系统"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.cache_size
        
        self.register_buffer('memory_attentions', torch.zeros(self.memory_size, config.feature_dim))
        self.register_buffer('memory_contexts', torch.zeros(self.memory_size, 10))  # 简化的上下文向量
        self.register_buffer('memory_utilities', torch.ones(self.memory_size))
        self.register_buffer('memory_index', torch.tensor(0))
        self.register_buffer('memory_full', torch.tensor(False))
    
    def update(self, features: torch.Tensor, 
               attention_result: Dict[str, Any]) -> Dict[str, Any]:
        """更新注意力记忆"""
        batch_size = features.shape[0]
        
        # 提取注意力特征
        if 'attention_weights' in attention_result:
            attention_features = attention_result['attention_weights']
        else:
            attention_features = features.mean(dim=[2, 3]) if len(features.shape) > 2 else features
        
        # 更新记忆
        for i in range(batch_size):
            idx = int(self.memory_index % self.memory_size)
            
            if len(attention_features.shape) == 2:
                self.memory_attentions[idx] = attention_features[i]
            else:
                self.memory_attentions[idx] = attention_features.flatten()[i * self.config.feature_dim:(i + 1) * self.config.feature_dim]
            
            # 更新效用
            utility = attention_result.get('attention_quality', torch.tensor(0.5))
            self.memory_utilities[idx] = utility.item()
            
            self.memory_index = (self.memory_index + 1) % self.memory_size
            
            if self.memory_index == 0:
                self.memory_full = True
        
        # 计算记忆利用率
        utilization = (self.memory_index / self.memory_size) if not self.memory_full else 1.0
        
        return {
            'memory_utilization': torch.tensor(utilization),
            'memory_quality': self.memory_utilities[:int(self.memory_index)].mean()
        }


# 工厂函数将在后续创建