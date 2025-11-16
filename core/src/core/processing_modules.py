"""
专业处理模块 (Processing Modules)
===============================

实现新皮层的专业功能模块，包括：
- 预测编码模块 (Prediction Coding)
- 注意控制模块 (Attention Control) 
- 决策形成模块 (Decision Making)
- 跨模态接口模块 (Cross-Modal Interface)

基于论文：
- "Distributed representations of prediction error signals across the cortical hierarchy are synergistic"
- "Laminar compartmentalization of attention modulation in area V4"
- "The dynamics and geometry of choice in the premotor cortex"
- "Object knowledge representation in the visual cortex requires communication with the language system"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math


class PredictionType(Enum):
    """预测类型"""
    TEMPORAL = "temporal"      # 时间预测
    SPATIAL = "spatial"        # 空间预测  
    SEMANTIC = "semantic"      # 语义预测
    MULTI_SCALE = "multi_scale" # 多尺度预测


class AttentionType(Enum):
    """注意类型"""
    SPATIAL = "spatial"        # 空间注意
    FEATURE = "feature"        # 特征注意
    TEMPORAL = "temporal"      # 时间注意
    TOP_DOWN = "top_down"      # 自上而下注意
    BOTTOM_UP = "bottom_up"    # 自下而上注意


class DecisionMode(Enum):
    """决策模式"""
    EVIDENCE_ACCUMULATION = "evidence_accumulation"  # 证据累积
    ATTRACTOR_DYNAMICS = "attractor_dynamics"       # 吸引子动力学
    VOTING = "voting"                               # 投票机制
    COMPETITIVE = "competitive"                     # 竞争机制


@dataclass 
class ProcessingConfig:
    """处理模块配置"""
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int = 2
    dropout_rate: float = 0.1
    
    # 预测编码参数
    prediction_horizon: int = 5  # 预测时间步数
    prediction_weight: float = 0.8  # 预测权重
    
    # 注意参数  
    attention_heads: int = 8  # 注意力头数
    attention_dropout: float = 0.1
    
    # 决策参数
    decision_threshold: float = 0.7  # 决策阈值
    confidence_threshold: float = 0.5  # 置信度阈值
    

class PredictionModule(nn.Module):
    """
    预测编码模块
    
    实现基于预测编码的信息处理机制：
    - 生成预测信号
    - 计算预测误差
    - 更新内部模型
    - 协同误差传播
    """
    
    def __init__(self, config: ProcessingConfig):
        super().__init__()
        self.config = config
        self.prediction_horizon = config.prediction_horizon
        self.prediction_weight = config.prediction_weight
        
        # 预测生成器
        self.predictor = nn.Sequential(
            *[nn.Sequential(
                nn.Linear(config.input_dim if i == 0 else config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate)
            ) for i in range(config.num_layers)],
            nn.Linear(config.hidden_dim, config.input_dim * self.prediction_horizon)
        )
        
        # 时间序列编码器（用于时间预测）
        self.temporal_encoder = nn.LSTM(
            config.input_dim, config.hidden_dim // 2, 
            num_layers=1, batch_first=True, bidirectional=True
        )
        
        # 误差检测器
        self.error_detector = nn.Sequential(
            nn.Linear(config.input_dim * 2, config.hidden_dim),  # 拼接输入和预测
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
            nn.Sigmoid()  # 误差强度
        )
        
        # 协同矩阵（跨层级误差协同）
        self.synergy_matrix = nn.Parameter(torch.randn(config.input_dim, config.input_dim))
        
        # 模型更新器
        self.model_updater = nn.Sequential(
            nn.Linear(config.input_dim * 2, config.hidden_dim),
            nn.ReLU(), 
            nn.Linear(config.hidden_dim, config.input_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        预测编码前向传播
        
        Args:
            x: 输入序列 [batch, seq_len, input_dim]
            context: 上下文信息
            
        Returns:
            预测结果和误差信息
        """
        batch_size, seq_len, input_dim = x.shape
        
        # 生成多步预测
        predictions = self.predictor(x)  # [batch, input_dim * horizon]
        predictions = predictions.view(batch_size, self.prediction_horizon, input_dim)
        
        # 时间序列处理（如果输入是序列）
        if seq_len > 1:
            temporal_features, _ = self.temporal_encoder(x)
            context_features = temporal_features[:, -1, :]  # 使用最后时间步
        else:
            context_features = x.squeeze(1)
            
        # 结合上下文
        if context is not None:
            context_features = torch.cat([context_features, context], dim=-1)
            
        # 计算误差（仅对有效预测步）
        errors = []
        confidences = []
        
        for t in range(min(self.prediction_horizon, seq_len)):
            # 当前时间步的真实值
            current_input = x[:, t, :]
            
            # 对应的预测
            if t < predictions.shape[1]:
                predicted = predictions[:, t, :]
            else:
                # 使用最后一个预测作为多步预测
                predicted = predictions[:, -1, :]
            
            # 计算预测误差
            error_input = torch.cat([current_input, predicted], dim=-1)
            error_strength = self.error_detector(error_input)  # [batch, input_dim]
            
            # 协同误差计算
            synergistic_error = torch.matmul(error_strength, self.synergy_matrix)
            final_error = error_strength + 0.3 * synergistic_error
            
            # 置信度计算
            confidence = 1.0 - torch.mean(torch.abs(final_error), dim=-1, keepdim=True)
            confidence = torch.clamp(confidence, 0, 1)
            
            errors.append(final_error)
            confidences.append(confidence)
            
        # 模型更新
        if len(errors) > 0:
            avg_error = torch.stack(errors).mean(dim=0)
            model_update = self.model_updater(
                torch.cat([context_features, avg_error], dim=-1)
            )
        else:
            model_update = torch.zeros_like(x)
            
        return {
            'predictions': predictions,
            'prediction_errors': torch.stack(errors) if errors else torch.zeros_like(x),
            'confidence': torch.stack(confidences) if confidences else torch.ones_like(x),
            'model_update': model_update,
            'temporal_features': temporal_features if seq_len > 1 else context_features
        }


class AttentionModule(nn.Module):
    """
    注意控制模块
    
    实现多层次的注意机制：
    - 空间注意（V4浅层：低对比度增强）
    - 特征注意（V4深层：反馈门控）
    - 自上而下注意调制
    - 自下而上显著性检测
    """
    
    def __init__(self, config: ProcessingConfig):
        super().__init__()
        self.config = config
        self.attention_heads = config.attention_heads
        
        # V4浅层注意处理器（低对比度增强）
        self.superficial_attention = nn.MultiheadAttention(
            embed_dim=config.input_dim,
            num_heads=self.attention_heads // 2,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # V4深层注意处理器（反馈门控）
        self.deep_attention = nn.MultiheadAttention(
            embed_dim=config.input_dim,
            num_heads=self.attention_heads // 2,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # 空间注意生成器
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(config.input_dim * 49, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 49),  # 7x7 空间网格
            nn.Softmax(dim=-1)
        )
        
        # 特征注意生成器
        self.feature_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
            nn.Sigmoid()
        )
        
        # 自上而下控制器
        self.top_down_controller = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.input_dim),
            nn.Sigmoid()
        )
        
        # 显著性检测器
        self.saliency_detector = nn.Sequential(
            nn.Conv2d(config.input_dim, config.input_dim // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(config.input_dim // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # 层间注意门控
        self.inter_layer_gating = nn.Parameter(torch.ones(config.input_dim))
        
    def forward(self, features: torch.Tensor, 
                query: Optional[torch.Tensor] = None,
                attention_type: AttentionType = AttentionType.SPATIAL) -> Dict[str, torch.Tensor]:
        """
        注意控制前向传播
        
        Args:
            features: 输入特征 [batch, channels, height, width] 或 [batch, seq_len, dim]
            query: 查询向量（用于注意力计算）
            attention_type: 注意类型
            
        Returns:
            注意权重和调制后的特征
        """
        batch_size = features.shape[0]
        
        # 重塑特征（如果是空间格式）
        if len(features.shape) == 4:
            # [batch, channels, height, width] -> [batch, seq_len, dim]
            h, w = features.shape[2:]
            features_2d = features.view(batch_size, -1, features.shape[1]).transpose(1, 2)
            spatial_dims = (h, w)
        else:
            features_2d = features
            spatial_dims = None
            
        # 生成查询向量
        if query is None:
            query = features_2d.mean(dim=1, keepdim=True)
            
        # V4层间注意处理
        # 浅层：低对比度增强
        superficial_output, superficial_weights = self.superficial_attention(
            query, features_2d, features_2d
        )
        
        # 深层：对比度独立的反馈门控  
        deep_output, deep_weights = self.deep_attention(
            query, features_2d, features_2d
        )
        
        # 空间注意（浅层功能）
        if spatial_dims is not None:
            spatial_attention_weights = self.spatial_attention(features)
            spatial_attention_weights = spatial_attention_weights.view(
                batch_size, 1, spatial_dims[0], spatial_dims[1]
            )
        else:
            spatial_attention_weights = torch.ones_like(features)
            
        # 特征注意（深层功能）
        feature_attention_weights = self.feature_attention(features)
        
        # 自上而下调制
        top_down_modulation = self.top_down_controller(query.squeeze(1))
        
        # 显著性检测
        if spatial_dims is not None:
            saliency_map = self.saliency_detector(features)
        else:
            saliency_map = torch.ones_like(features[:, :, :1])
            
        # 层间整合
        if len(features.shape) == 4:
            # 空间特征处理
            enhanced_features = features * spatial_attention_weights
            gated_features = features * feature_attention_weights.unsqueeze(-1).unsqueeze(-1)
            top_down_features = features * top_down_modulation.view(-1, 1, 1, 1)
            salient_features = features * saliency_map
        else:
            # 序列特征处理
            enhanced_features = features_2d * spatial_attention_weights.view(batch_size, -1, 1)
            gated_features = features_2d * feature_attention_weights.unsqueeze(1)
            top_down_features = features_2d * top_down_modulation.unsqueeze(1)
            salient_features = features_2d * saliency_map.squeeze(-1)
            
        # 注意门控调节
        modulated_features = (
            0.3 * enhanced_features + 
            0.3 * gated_features + 
            0.2 * top_down_features + 
            0.2 * salient_features
        )
        
        # 应用层间门控
        if len(features.shape) == 4:
            modulated_features = modulated_features * self.inter_layer_gating.view(1, -1, 1, 1)
        else:
            modulated_features = modulated_features * self.inter_layer_gating.unsqueeze(1)
            
        return {
            'attended_features': modulated_features,
            'spatial_attention': spatial_attention_weights,
            'feature_attention': feature_attention_weights,
            'top_down_modulation': top_down_modulation,
            'saliency_map': saliency_map,
            'superficial_weights': superficial_weights,
            'deep_weights': deep_weights
        }


class DecisionModule(nn.Module):
    """
    决策形成模块
    
    实现基于群体动力学的决策机制：
    - 证据累积
    - 吸引子动力学
    - 几何决策空间
    - 选择变量统一
    """
    
    def __init__(self, config: ProcessingConfig):
        super().__init__()
        self.config = config
        self.decision_threshold = config.decision_threshold
        self.confidence_threshold = config.confidence_threshold
        
        # 证据累积器
        self.evidence_accumulator = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.Tanh()  # 累积证据（-1到1之间）
        )
        
        # 吸引子动力学模型
        self.attractor_dynamics = AttractorNetwork(config.output_dim, config.hidden_dim)
        
        # 选择变量投影器（将异质性神经元映射到统一选择空间）
        self.choice_variable_projector = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 2),  # 二维选择空间
            nn.Tanh()
        )
        
        # 决策几何分析器
        self.geometric_analyzer = DecisionGeometry(config.output_dim)
        
        # 置信度估计器
        self.confidence_estimator = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, neural_responses: torch.Tensor, 
                decision_mode: DecisionMode = DecisionMode.ATTRACTOR_DYNAMICS) -> Dict[str, torch.Tensor]:
        """
        决策形成前向传播
        
        Args:
            neural_responses: 神经元群体响应 [batch, num_neurons]
            decision_mode: 决策模式
            
        Returns:
            决策结果和相关信息
        """
        batch_size = neural_responses.shape[0]
        
        # 选择变量投影（统一异质性神经元响应）
        choice_variables = self.choice_variable_projector(neural_responses)
        
        # 证据累积
        accumulated_evidence = self.evidence_accumulator(neural_responses)
        
        if decision_mode == DecisionMode.EVIDENCE_ACCUMULATION:
            # 纯证据累积模式
            decision_output = accumulated_evidence
            
        elif decision_mode == DecisionMode.ATTRACTOR_DYNAMICS:
            # 吸引子动力学模式
            dynamics_output = self.attractor_dynamics(choice_variables, accumulated_evidence)
            decision_output = dynamics_output
            
        elif decision_mode == DecisionMode.VOTING:
            # 投票机制
            decision_output = torch.sign(accumulated_evidence)
            
        else:  # COMPETITIVE
            # 竞争机制
            decision_output = F.softmax(accumulated_evidence, dim=-1)
            
        # 几何分析
        geometric_info = self.geometric_analyzer(choice_variables)
        
        # 置信度估计
        confidence = self.confidence_estimator(decision_output)
        
        # 最终决策
        final_decision = torch.where(
            confidence > self.confidence_threshold,
            torch.sign(decision_output),
            torch.zeros_like(decision_output)
        )
        
        # 决策稳定性检查
        decision_stability = torch.abs(final_decision) > self.decision_threshold
        
        return {
            'decision': final_decision,
            'confidence': confidence,
            'evidence': accumulated_evidence,
            'choice_variables': choice_variables,
            'choice_geometry': geometric_info,
            'decision_stability': decision_stability,
            'attractor_state': dynamics_output if decision_mode == DecisionMode.ATTRACTOR_DYNAMICS else None
        }


class CrossModalModule(nn.Module):
    """
    跨模态接口模块
    
    实现多模态信息整合和概念形成：
    - 视觉-语言接口（VOTC-语言系统）
    - 跨模态特征融合
    - 概念抽象形成
    - 双编码系统支持
    """
    
    def __init__(self, config: ProcessingConfig):
        super().__init__()
        self.config = config
        
        # 视觉编码器
        self.visual_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # 语言编码器
        self.language_encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim)
        )
        
        # 跨模态注意力融合
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # 双编码权重（基于VOTC-语言系统白质连接）
        self.dual_coding_weights = nn.Parameter(torch.tensor([0.6, 0.4]))  # 视觉主导，语言辅助
        
        # 概念形成器
        self.concept_former = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.Tanh()
        )
        
        # 抽象层次编码器
        self.abstract_encoder = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.Tanh()
        )
        
    def forward(self, visual_input: torch.Tensor, 
                language_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        跨模态融合前向传播
        
        Args:
            visual_input: 视觉输入 [batch, visual_dim]
            language_input: 语言输入 [batch, language_dim]
            
        Returns:
            跨模态融合结果和概念信息
        """
        # 分别编码视觉和语言信息
        visual_features = self.visual_encoder(visual_input)
        language_features = self.language_encoder(language_input)
        
        # 跨模态注意力融合
        # 视觉作为查询，语言作为键值
        fused_visual, visual_attention = self.cross_modal_attention(
            visual_features.unsqueeze(1),  # query
            language_features.unsqueeze(1),  # key, value
            language_features.unsqueeze(1)
        )
        
        # 语言作为查询，视觉作为键值  
        fused_language, language_attention = self.cross_modal_attention(
            language_features.unsqueeze(1),  # query
            visual_features.unsqueeze(1),   # key, value
            visual_features.unsqueeze(1)
        )
        
        fused_visual = fused_visual.squeeze(1)
        fused_language = fused_language.squeeze(1)
        
        # 双编码整合（基于白质连接强度）
        dual_coded_features = (
            self.dual_coding_weights[0] * fused_visual + 
            self.dual_coding_weights[1] * fused_language
        )
        
        # 概念形成
        concept_features = torch.cat([fused_visual, fused_language], dim=-1)
        concept_representation = self.concept_former(concept_features)
        
        # 抽象编码
        abstract_representation = self.abstract_encoder(concept_representation)
        
        # 跨模态一致性检查
        cross_modal_consistency = F.cosine_similarity(
            F.normalize(fused_visual, dim=-1), 
            F.normalize(fused_language, dim=-1), 
            dim=-1
        )
        
        return {
            'visual_features': fused_visual,
            'language_features': fused_language,
            'dual_coded_features': dual_coded_features,
            'concept_representation': concept_representation,
            'abstract_representation': abstract_representation,
            'visual_attention': visual_attention,
            'language_attention': language_attention,
            'cross_modal_consistency': cross_modal_consistency,
            'concept_strength': torch.sigmoid(
                torch.norm(concept_representation, dim=-1, keepdim=True)
            )
        }


# 辅助网络类

class AttractorNetwork(nn.Module):
    """吸引子网络 - 实现决策的吸引子动力学"""
    
    def __init__(self, choice_dim: int, hidden_dim: int):
        super().__init__()
        self.choice_dim = choice_dim
        
        # 吸引子势能景观
        self.potential_field = nn.Sequential(
            nn.Linear(choice_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, choice_dim),
            nn.Tanh()
        )
        
        # 动力学更新规则
        self.dynamics_update = nn.Sequential(
            nn.Linear(choice_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, choice_dim)
        )
        
    def forward(self, choice_variables: torch.Tensor, 
                evidence: torch.Tensor) -> torch.Tensor:
        """吸引子动力学更新"""
        # 计算势能梯度
        potential = self.potential_field(choice_variables)
        
        # 动力学更新
        dynamics_input = torch.cat([choice_variables, evidence], dim=-1)
        update = self.dynamics_update(dynamics_input)
        
        # 吸引子更新（模拟梯度下降）
        updated_state = choice_variables + 0.1 * (update - 0.5 * potential)
        
        return updated_state


class DecisionGeometry(nn.Module):
    """决策几何分析器 - 分析选择变量的几何结构"""
    
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        
    def forward(self, choice_variables: torch.Tensor) -> Dict[str, torch.Tensor]:
        """几何分析"""
        batch_size = choice_variables.shape[0]
        
        # 计算选择变量的统计特征
        choice_mean = choice_variables.mean(dim=0)
        choice_std = choice_variables.std(dim=0)
        choice_magnitude = torch.norm(choice_variables, dim=-1)
        
        # 决策边界分析（二维情况下）
        if choice_variables.shape[-1] == 2:
            # 计算到决策边界的距离
            boundary_distance = torch.abs(choice_variables.sum(dim=-1)) / np.sqrt(2)
        else:
            boundary_distance = torch.zeros(batch_size, 1)
            
        # 聚类密度（用于评估决策的稳定性）
        pairwise_distances = torch.cdist(choice_variables, choice_variables)
        cluster_density = torch.exp(-pairwise_distances.mean(dim=-1, keepdim=True))
        
        return {
            'choice_mean': choice_mean,
            'choice_std': choice_std,
            'choice_magnitude': choice_magnitude,
            'boundary_distance': boundary_distance,
            'cluster_density': cluster_density
        }


# 工厂函数

def create_prediction_module(input_dim: int, hidden_dim: int = 256) -> PredictionModule:
    """创建预测模块"""
    config = ProcessingConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim
    )
    return PredictionModule(config)


def create_attention_module(input_dim: int, hidden_dim: int = 256) -> AttentionModule:
    """创建注意模块"""
    config = ProcessingConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=input_dim
    )
    return AttentionModule(config)


def create_decision_module(input_dim: int, output_dim: int = 2) -> DecisionModule:
    """创建决策模块"""
    config = ProcessingConfig(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim
    )
    return DecisionModule(config)


def create_crossmodal_module(input_dim: int, output_dim: int = 128) -> CrossModalModule:
    """创建跨模态模块"""
    config = ProcessingConfig(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim
    )
    return CrossModalModule(config)


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 测试预测模块
    prediction_module = create_prediction_module(128).to(device)
    test_input = torch.randn(4, 10, 128).to(device)  # batch, seq_len, dim
    pred_output = prediction_module(test_input)
    
    print(f"预测模块测试:")
    print(f"预测形状: {pred_output['predictions'].shape}")
    print(f"误差形状: {pred_output['prediction_errors'].shape}")
    print(f"置信度形状: {pred_output['confidence'].shape}")
    
    # 测试注意模块
    attention_module = create_attention_module(256).to(device)
    test_features = torch.randn(4, 256, 14, 14).to(device)
    attention_output = attention_module(test_features)
    
    print(f"\n注意模块测试:")
    print(f"注意特征形状: {attention_output['attended_features'].shape}")
    print(f"空间注意形状: {attention_output['spatial_attention'].shape}")
    
    # 测试决策模块
    decision_module = create_decision_module(512, 2).to(device)
    test_responses = torch.randn(4, 512).to(device)
    decision_output = decision_module(test_responses)
    
    print(f"\n决策模块测试:")
    print(f"决策形状: {decision_output['decision'].shape}")
    print(f"置信度形状: {decision_output['confidence'].shape}")
    print(f"选择变量形状: {decision_output['choice_variables'].shape}")
    
    # 测试跨模态模块
    crossmodal_module = create_crossmodal_module(256, 128).to(device)
    visual_input = torch.randn(4, 256).to(device)
    language_input = torch.randn(4, 256).to(device)
    crossmodal_output = crossmodal_module(visual_input, language_input)
    
    print(f"\n跨模态模块测试:")
    print(f"视觉特征形状: {crossmodal_output['visual_features'].shape}")
    print(f"语言特征形状: {crossmodal_output['language_features'].shape}")
    print(f"概念表示形状: {crossmodal_output['concept_representation'].shape}")
    print(f"抽象表示形状: {crossmodal_output['abstract_representation'].shape}")
