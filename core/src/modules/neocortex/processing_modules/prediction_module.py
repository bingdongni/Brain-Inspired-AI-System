"""
预测模块实现
=============

实现新皮层的预测功能，包括：
- 基于层级结构的多时间尺度预测
- 时间序列预测和模式识别
- 语义预测和概念预测
- 自适应预测学习和记忆
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .processing_config import (
    PredictionConfig, PredictionType, ProcessingMode, ModuleType
)


class PredictionModule(nn.Module):
    """
    预测模块
    
    实现基于新皮层预测编码机制的层级预测功能。
    """
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        self.feature_dim = config.feature_dim
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")
        
        # 创建预测引擎
        if config.prediction_type == PredictionType.HIERARCHICAL:
            self.prediction_engine = HierarchicalPredictor(
                config.feature_dim, 
                config.hierarchical_levels,
                config.prediction_horizon
            )
        elif config.prediction_type == PredictionType.TEMPORAL:
            self.prediction_engine = TemporalPredictor(
                config.feature_dim,
                config.temporal_window,
                config.prediction_horizon
            )
        elif config.prediction_type == PredictionType.SEMANTIC:
            self.prediction_engine = SemanticPredictor(
                config.feature_dim,
                config.prediction_horizon
            )
        else:
            raise ValueError(f"不支持的预测类型: {config.prediction_type}")
        
        # 预测质量评估器
        self.quality_assessor = PredictionQualityAssessor(config.feature_dim)
        
        # 预测记忆系统
        self.prediction_memory = PredictionMemorySystem(config)
        
        # 自适应学习器
        self.adaptive_learner = AdaptivePredictionLearner(config)
        
        # 统计信息
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'avg_confidence': 0.0,
            'prediction_errors': []
        }
    
    def forward(self, features: torch.Tensor, 
                context: Optional[Dict[str, torch.Tensor]] = None,
                target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        预测模块前向传播
        
        Args:
            features: 当前特征 [batch, feature_dim] 或 [batch, channels, height, width]
            context: 上下文信息（时间、历史等）
            target: 目标特征（用于训练和评估）
            
        Returns:
            dict: 预测结果
        """
        batch_size = features.shape[0]
        self.stats['total_predictions'] += batch_size
        
        # 准备输入
        processed_features = self._preprocess_features(features)
        
        # 生成预测
        prediction_result = self.prediction_engine(
            processed_features, context
        )
        
        # 评估预测质量
        quality_assessment = self.quality_assessor(
            processed_features, prediction_result['predictions']
        )
        
        # 记忆和适应
        memory_update = self.prediction_memory.update(
            processed_features, prediction_result, quality_assessment
        )
        
        adaptive_update = self.adaptive_learner.update(
            processed_features, target, prediction_result, quality_assessment
        )
        
        # 计算最终置信度
        final_confidence = self._compute_final_confidence(
            quality_assessment, memory_update, adaptive_update
        )
        
        # 更新统计
        if target is not None:
            error = F.mse_loss(prediction_result['predictions'], target)
            self.stats['prediction_errors'].append(error.item())
            
            if error < 0.1:  # 阈值可调
                self.stats['successful_predictions'] += 1
        
        self.stats['avg_confidence'] = (
            (self.stats['avg_confidence'] * (self.stats['total_predictions'] - batch_size) +
             final_confidence * batch_size) / self.stats['total_predictions']
        )
        
        return {
            'predictions': prediction_result['predictions'],
            'prediction_horizons': prediction_result.get('horizons', []),
            'confidence': final_confidence,
            'quality_assessment': quality_assessment,
            'memory_state': memory_update,
            'adaptive_state': adaptive_update,
            'uncertainty': prediction_result.get('uncertainty', torch.zeros_like(final_confidence)),
            'prediction_metadata': {
                'type': self.config.prediction_type.value,
                'horizon': self.config.prediction_horizon,
                'hierarchical_levels': self.config.hierarchical_levels,
                'model_uncertainty': prediction_result.get('model_uncertainty', 0.0),
                'data_uncertainty': prediction_result.get('data_uncertainty', 0.0)
            },
            'statistics': {
                'total_predictions': self.stats['total_predictions'],
                'success_rate': self.stats['successful_predictions'] / max(self.stats['total_predictions'], 1),
                'avg_confidence': self.stats['avg_confidence'],
                'recent_errors': self.stats['prediction_errors'][-10:] if self.stats['prediction_errors'] else []
            }
        }
    
    def _preprocess_features(self, features: torch.Tensor) -> torch.Tensor:
        """预处理特征"""
        if len(features.shape) > 2:
            # 2D特征图处理
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        
        # 确保特征维度匹配
        if features.shape[-1] != self.feature_dim:
            features = F.interpolate(
                features.unsqueeze(-1), 
                size=self.feature_dim, 
                mode='linear'
            ).squeeze(-1)
        
        return features
    
    def _compute_final_confidence(self, quality_assessment: Dict, 
                                 memory_update: Dict, adaptive_update: Dict) -> torch.Tensor:
        """计算最终置信度"""
        # 综合多个置信度来源
        quality_confidence = quality_assessment['confidence']
        memory_confidence = memory_update.get('confidence', quality_confidence)
        adaptive_confidence = adaptive_update.get('confidence', quality_confidence)
        
        # 加权组合
        final_confidence = (
            0.5 * quality_confidence + 
            0.3 * memory_confidence + 
            0.2 * adaptive_confidence
        )
        
        return final_confidence


class HierarchicalPredictor(nn.Module):
    """层级预测器
    
    实现基于新皮层层级结构的预测，从低层到高层逐级预测。
    """
    
    def __init__(self, feature_dim: int, num_levels: int, horizon: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_levels = num_levels
        self.horizon = horizon
        
        # 层级预测网络
        self.level_predictors = nn.ModuleList([
            LevelPredictor(feature_dim, level) 
            for level in range(num_levels)
        ])
        
        # 层级整合器
        self.hierarchy_integrator = HierarchyIntegrator(feature_dim, num_levels)
        
        # 预测控制器
        self.prediction_controller = PredictionController(feature_dim)
    
    def forward(self, features: torch.Tensor, 
                context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """层级预测前向传播"""
        batch_size = features.shape[0]
        
        # 各层级预测
        level_predictions = []
        level_confidences = []
        
        for i, predictor in enumerate(self.level_predictors):
            # 当前层级预测
            level_pred = predictor(features)
            level_predictions.append(level_pred)
            
            # 预测置信度
            confidence = self._compute_level_confidence(level_pred, i)
            level_confidences.append(confidence)
        
        # 层级整合
        integrated_prediction = self.hierarchy_integrator(
            level_predictions, level_confidences
        )
        
        # 预测控制
        controlled_prediction = self.prediction_controller(
            integrated_prediction, context
        )
        
        # 计算预测不确定性
        uncertainty = self._compute_uncertainty(level_predictions, level_confidences)
        
        return {
            'predictions': controlled_prediction,
            'level_predictions': level_predictions,
            'level_confidences': level_confidences,
            'uncertainty': uncertainty,
            'horizons': list(range(1, self.horizon + 1))
        }
    
    def _compute_level_confidence(self, prediction: torch.Tensor, level: int) -> torch.Tensor:
        """计算层级预测置信度"""
        # 基于预测的稳定性计算置信度
        prediction_variance = prediction.var(dim=1)
        stability = 1 / (1 + prediction_variance)
        
        # 层级权重（高层预测通常更稳定但可能不够精确）
        level_weights = [0.8, 0.6, 0.4][:level + 1]
        level_weight = level_weights[min(level, len(level_weights) - 1)]
        
        confidence = stability * level_weight
        return confidence.mean()
    
    def _compute_uncertainty(self, predictions: List[torch.Tensor], 
                           confidences: List[torch.Tensor]) -> torch.Tensor:
        """计算预测不确定性"""
        # 预测间方差作为不确定性度量
        if len(predictions) > 1:
            pred_stack = torch.stack(predictions)
            prediction_variance = pred_stack.var(dim=0).mean()
            
            # 置信度方差
            conf_variance = torch.stack(confidences).var()
            
            uncertainty = 0.7 * prediction_variance + 0.3 * conf_variance
        else:
            uncertainty = torch.tensor(0.5)
        
        return uncertainty


class TemporalPredictor(nn.Module):
    """时间序列预测器
    
    基于LSTM/GRU实现时间序列的短期和长期预测。
    """
    
    def __init__(self, feature_dim: int, window_size: int, horizon: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        self.horizon = horizon
        
        # 多层LSTM用于时间序列建模
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(feature_dim, feature_dim // (i + 1), batch_first=True)
            for i in range(min(3, horizon // 3 + 1))
        ])
        
        # 预测投影层
        self.prediction_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        # 时间注意力机制
        self.temporal_attention = TemporalAttention(feature_dim)
        
        # 多尺度预测器
        self.multiscale_predictors = nn.ModuleDict({
            f'horizon_{h}': TemporalPredictorHead(feature_dim, h)
            for h in range(1, horizon + 1)
        })
    
    def forward(self, features: torch.Tensor, 
                context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """时间序列预测前向传播"""
        batch_size = features.shape[0]
        
        # 确保输入格式为时间序列
        if len(features.shape) == 2:
            # 单时间步，扩展为序列
            features = features.unsqueeze(1)
        elif len(features.shape) == 3:
            # 已经是 [batch, seq_len, features]
            pass
        else:
            # 其他形状，尝试调整
            features = features.view(batch_size, -1, self.feature_dim)
        
        # 多层LSTM处理
        lstm_outputs = []
        current_input = features
        
        for lstm_layer in self.lstm_layers:
            lstm_out, _ = lstm_layer(current_input)
            lstm_outputs.append(lstm_out)
            
            # 残差连接
            if current_input.shape == lstm_out.shape:
                current_input = current_input + lstm_out
            else:
                current_input = lstm_out
        
        # 时间注意力
        attended_output = self.temporal_attention(lstm_outputs[-1])
        
        # 多尺度预测
        horizon_predictions = {}
        confidences = {}
        
        for h in range(1, self.horizon + 1):
            if str(h) in self.multiscale_predictors:
                pred_head = self.multiscale_predictors[str(h)]
                pred = pred_head(attended_output)
                horizon_predictions[f'h_{h}'] = pred
                
                # 计算置信度
                conf = self._compute_temporal_confidence(pred, h)
                confidences[f'h_{h}'] = conf
        
        # 选择最佳预测
        if horizon_predictions:
            # 选择与目标时间步最接近的预测
            best_horizon = min(horizon_predictions.keys(), 
                             key=lambda x: abs(int(x.split('_')[1]) - self.horizon // 2))
            best_prediction = horizon_predictions[best_horizon]
            best_confidence = confidences[best_horizon]
        else:
            best_prediction = attended_output[:, -1, :]
            best_confidence = torch.tensor(0.5)
        
        # 时间不确定性
        temporal_uncertainty = self._compute_temporal_uncertainty(horizon_predictions)
        
        return {
            'predictions': best_prediction,
            'horizon_predictions': horizon_predictions,
            'horizon_confidences': confidences,
            'best_horizon': best_horizon if horizon_predictions else 'h_1',
            'uncertainty': temporal_uncertainty
        }
    
    def _compute_temporal_confidence(self, prediction: torch.Tensor, horizon: int) -> torch.Tensor:
        """计算时间预测置信度"""
        # 基于预测稳定性和时间距离
        pred_stability = 1 / (1 + prediction.var(dim=1))
        
        # 时间衰减因子
        time_decay = np.exp(-0.1 * horizon)
        
        confidence = pred_stability * time_decay
        return confidence.mean()
    
    def _compute_temporal_uncertainty(self, predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算时间不确定性"""
        if len(predictions) < 2:
            return torch.tensor(0.5)
        
        pred_list = list(predictions.values())
        pred_stack = torch.stack(pred_list)
        
        # 预测间方差
        prediction_variance = pred_stack.var(dim=0).mean()
        
        return prediction_variance


class SemanticPredictor(nn.Module):
    """语义预测器
    
    基于概念和语义关系的预测，实现语义层面的理解和预测。
    """
    
    def __init__(self, feature_dim: int, horizon: int, num_concepts: int = 100):
        super().__init__()
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.num_concepts = num_concepts
        
        # 概念检测器
        self.concept_detector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, num_concepts),
            nn.Sigmoid()
        )
        
        # 语义关系建模器
        self.semantic_modeler = SemanticRelationModeler(feature_dim, num_concepts)
        
        # 概念预测器
        self.concept_predictor = ConceptPredictor(num_concepts, horizon)
        
        # 语义到特征的映射
        self.semantic_to_feature = nn.Sequential(
            nn.Linear(num_concepts, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 预测质量评估
        self.quality_assessor = nn.Sequential(
            nn.Linear(feature_dim * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, 
                context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """语义预测前向传播"""
        batch_size = features.shape[0]
        
        # 检测当前概念
        concept_activations = self.concept_detector(features)
        
        # 建模语义关系
        semantic_state = self.semantic_modeler(concept_activations)
        
        # 预测未来概念
        future_concepts = self.concept_predictor(semantic_state)
        
        # 概念到特征的映射
        semantic_predictions = {}
        for horizon, future_concept in future_concepts.items():
            feature_pred = self.semantic_to_feature(future_concept)
            semantic_predictions[horizon] = feature_pred
        
        # 评估预测质量
        quality_scores = {}
        for horizon, pred in semantic_predictions.items():
            quality_input = torch.cat([features, pred], dim=1)
            quality = self.quality_assessor(quality_input)
            quality_scores[horizon] = quality
        
        # 选择最佳预测
        if semantic_predictions:
            best_horizon = max(quality_scores.keys(), key=lambda h: quality_scores[h].mean())
            best_prediction = semantic_predictions[best_horizon]
            best_quality = quality_scores[best_horizon]
        else:
            best_prediction = features
            best_quality = torch.tensor(0.5)
        
        return {
            'predictions': best_prediction,
            'concept_predictions': future_concepts,
            'semantic_state': semantic_state,
            'concept_activations': concept_activations,
            'quality_scores': quality_scores,
            'best_horizon': best_horizon if semantic_predictions else 'semantic_1',
            'semantic_uncertainty': self._compute_semantic_uncertainty(future_concepts)
        }
    
    def _compute_semantic_uncertainty(self, concept_predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算语义不确定性"""
        if len(concept_predictions) < 2:
            return torch.tensor(0.5)
        
        pred_list = list(concept_predictions.values())
        pred_stack = torch.stack(pred_list)
        
        # 概念预测方差
        concept_variance = pred_stack.var(dim=0).mean()
        
        return concept_variance


# 辅助类定义

class LevelPredictor(nn.Module):
    """单层级预测器"""
    
    def __init__(self, feature_dim: int, level: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.level = level
        
        # 层级特异性预测网络
        if level == 0:
            # 低层：详细特征预测
            self.network = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim),
                nn.Tanh()
            )
        elif level == 1:
            # 中层：模式预测
            self.network = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Tanh()
            )
        else:
            # 高层：抽象预测
            self.network = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim),
                nn.Tanh()
            )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class HierarchyIntegrator(nn.Module):
    """层级整合器"""
    
    def __init__(self, feature_dim: int, num_levels: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_levels = num_levels
        
        # 层级权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim * num_levels, num_levels),
            nn.Softmax(dim=1)
        )
        
        # 特征融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(feature_dim * num_levels, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, predictions: List[torch.Tensor], 
                confidences: List[torch.Tensor]) -> torch.Tensor:
        # 拼接所有预测
        combined_predictions = torch.cat(predictions, dim=1)
        
        # 生成整合权重
        weights = self.weight_generator(combined_predictions)
        
        # 加权融合
        weighted_predictions = []
        for i, (pred, weight) in enumerate(zip(predictions, weights.split(1, dim=1))):
            weighted_predictions.append(pred * weight[:, i:i+1])
        
        # 融合特征
        fused_features = torch.cat(weighted_predictions, dim=1)
        integrated = self.fusion_network(fused_features)
        
        return integrated


class PredictionController(nn.Module):
    """预测控制器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.controller = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, feature_dim)
        )
    
    def forward(self, prediction: torch.Tensor, 
                context: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        controlled_pred = self.controller(prediction)
        
        if context is not None:
            # 可以根据上下文调整预测
            pass
        
        return controlled_pred


class PredictionQualityAssessor(nn.Module):
    """预测质量评估器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.assessor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 3),  # [置信度, 稳定性, 一致性]
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        combined = torch.cat([features, predictions], dim=1)
        scores = self.assessor(combined)
        
        return {
            'confidence': scores[:, 0:1],
            'stability': scores[:, 1:2],
            'consistency': scores[:, 2:3]
        }


class TemporalAttention(nn.Module):
    """时间注意力机制"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.attention = nn.MultiheadAttention(
            feature_dim, num_heads=8, batch_first=True
        )
    
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(sequence, sequence, sequence)
        return attended


class TemporalPredictorHead(nn.Module):
    """时间预测头"""
    
    def __init__(self, feature_dim: int, horizon: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.horizon = horizon
        
        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features[:, -self.horizon:, :])


class SemanticRelationModeler(nn.Module):
    """语义关系建模器"""
    
    def __init__(self, feature_dim: int, num_concepts: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_concepts = num_concepts
        
        self.relation_encoder = nn.GRU(
            num_concepts, feature_dim, batch_first=True
        )
    
    def forward(self, concept_activations: torch.Tensor) -> torch.Tensor:
        # 使用GRU编码概念序列关系
        output, hidden = self.relation_encoder(concept_activations)
        return hidden[-1]


class ConceptPredictor(nn.Module):
    """概念预测器"""
    
    def __init__(self, num_concepts: int, horizon: int):
        super().__init__()
        self.num_concepts = num_concepts
        self.horizon = horizon
        
        self.predictor = nn.GRU(
            num_concepts, num_concepts, batch_first=True
        )
    
    def forward(self, semantic_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 扩展为序列
        sequence = semantic_state.unsqueeze(1).expand(-1, self.horizon, -1)
        
        # 预测
        output, _ = self.predictor(sequence)
        
        predictions = {}
        for i in range(self.horizon):
            predictions[f'semantic_{i+1}'] = output[:, i, :]
        
        return predictions


class PredictionMemorySystem(nn.Module):
    """预测记忆系统"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        self.memory_size = config.prediction_memory_size
        
        self.register_buffer('memory_features', torch.zeros(self.memory_size, config.feature_dim))
        self.register_buffer('memory_predictions', torch.zeros(self.memory_size, config.feature_dim))
        self.register_buffer('memory_confidences', torch.ones(self.memory_size))
        self.register_buffer('memory_index', torch.tensor(0))
        self.register_buffer('memory_full', torch.tensor(False))
    
    def update(self, features: torch.Tensor, prediction_result: Dict, 
               quality_assessment: Dict) -> Dict[str, Any]:
        batch_size = features.shape[0]
        
        for i in range(batch_size):
            idx = int(self.memory_index % self.memory_size)
            
            self.memory_features[idx] = features[i]
            self.memory_predictions[idx] = prediction_result['predictions'][i]
            self.memory_confidences[idx] = quality_assessment['confidence'][i].item()
            
            self.memory_index = (self.memory_index + 1) % self.memory_size
            
            if self.memory_index == 0:
                self.memory_full = True
        
        return {
            'confidence': quality_assessment['confidence'].mean(),
            'memory_utilization': (self.memory_index / self.memory_size) if not self.memory_full else 1.0
        }


class AdaptivePredictionLearner(nn.Module):
    """自适应预测学习器"""
    
    def __init__(self, config: PredictionConfig):
        super().__init__()
        self.config = config
        
        # 简单自适应机制
        self.adaptation_rate = config.adaptation_rate
        
        # 预测误差跟踪
        self.register_buffer('prediction_errors', torch.zeros(100))
        self.register_buffer('error_index', torch.tensor(0))
    
    def update(self, features: torch.Tensor, target: Optional[torch.Tensor],
               prediction_result: Dict, quality_assessment: Dict) -> Dict[str, Any]:
        # 更新误差统计
        if target is not None:
            error = F.mse_loss(prediction_result['predictions'], target).item()
            
            idx = int(self.error_index % self.prediction_errors.numel())
            self.prediction_errors[idx] = error
            self.error_index = (self.error_index + 1) % self.prediction_errors.numel()
        
        # 计算适应置信度
        recent_errors = self.prediction_errors[self.prediction_errors > 0]
        if len(recent_errors) > 0:
            avg_error = recent_errors.mean()
            adaptation_confidence = torch.sigmoid(-avg_error * 10)
        else:
            adaptation_confidence = torch.tensor(0.5)
        
        return {
            'confidence': adaptation_confidence,
            'adaptation_rate': self.adaptation_rate,
            'recent_error_avg': avg_error if 'avg_error' in locals() else torch.tensor(0.0)
        }