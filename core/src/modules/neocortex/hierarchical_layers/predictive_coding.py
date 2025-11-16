"""
预测编码机制
=============

实现新皮层的预测编码机制，包括：
- 预测编码器
- 误差放大器
- 预测层级组织
- 预测误差传播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


class PredictiveEncoder(nn.Module):
    """预测编码器
    
    实现层次化的预测编码机制，每个层级预测下一层级的输入。
    """
    
    def __init__(self, feature_dim: int, prediction_levels: int = 3, 
                 temporal_window: int = 5):
        super().__init__()
        self.feature_dim = feature_dim
        self.prediction_levels = prediction_levels
        self.temporal_window = temporal_window
        
        # 多层次预测网络
        self.prediction_networks = nn.ModuleList([
            PredictionNetwork(feature_dim, level + 1) 
            for level in range(prediction_levels)
        ])
        
        # 时间序列预测器
        self.temporal_predictor = TemporalPredictor(
            feature_dim, temporal_window
        )
        
        # 预测记忆模块
        self.prediction_memory = PredictionMemory(feature_dim)
        
        # 自适应预测控制器
        self.adaptive_controller = AdaptivePredictionController(feature_dim)
    
    def forward(self, features: torch.Tensor, 
                hierarchy_level: int = 0,
                temporal_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        预测编码前向传播
        
        Args:
            features: 当前层级特征
            hierarchy_level: 层级编号（用于选择预测网络）
            temporal_context: 时间上下文
            
        Returns:
            dict: 预测编码结果
        """
        batch_size = features.shape[0]
        
        # 选择适当的预测网络
        if hierarchy_level < len(self.prediction_networks):
            prediction_network = self.prediction_networks[hierarchy_level]
        else:
            prediction_network = self.prediction_networks[-1]
        
        # 生成分层预测
        hierarchical_predictions = []
        for level in range(min(hierarchy_level + 1, len(self.prediction_networks))):
            if level == 0:
                pred = prediction_network(features)
            else:
                # 基于上一层预测生成更高级预测
                if hierarchical_predictions:
                    pred = self._generate_higher_level_prediction(
                        hierarchical_predictions[-1], level
                    )
                else:
                    pred = prediction_network(features)
            hierarchical_predictions.append(pred)
        
        # 时间序列预测
        if temporal_context is not None:
            temporal_prediction = self.temporal_predictor(
                temporal_context, features
            )
        else:
            temporal_prediction = None
        
        # 自适应预测控制
        adaptive_result = self.adaptive_controller(
            features, hierarchical_predictions
        )
        
        # 主要预测结果
        if hierarchical_predictions:
            main_prediction = hierarchical_predictions[-1]
        else:
            main_prediction = features
        
        # 计算预测置信度
        prediction_confidence = self._compute_prediction_confidence(
            features, main_prediction
        )
        
        # 更新预测记忆
        memory_update = self.prediction_memory.update(
            features, main_prediction
        )
        
        return {
            'prediction': main_prediction,
            'hierarchical_predictions': hierarchical_predictions,
            'temporal_prediction': temporal_prediction,
            'adaptive_weights': adaptive_result['weights'],
            'prediction_confidence': prediction_confidence,
            'memory_state': memory_update,
            'prediction_quality': adaptive_result['quality']
        }
    
    def _generate_higher_level_prediction(self, lower_prediction: torch.Tensor, 
                                        level: int) -> torch.Tensor:
        """生成更高层级的预测"""
        # 渐进式抽象
        abstraction_factor = 0.5 ** level
        
        # 全局池化（增加抽象程度）
        if len(lower_prediction.shape) > 2:
            pooled_prediction = F.adaptive_avg_pool2d(
                lower_prediction, 
                max(1, lower_prediction.shape[2] // (2 ** level))
            )
        else:
            pooled_prediction = lower_prediction
        
        # 抽象特征提取
        abstracted = pooled_prediction * abstraction_factor
        
        return abstracted
    
    def _compute_prediction_confidence(self, actual: torch.Tensor, 
                                     predicted: torch.Tensor) -> torch.Tensor:
        """计算预测置信度"""
        # 计算预测误差
        if predicted.shape != actual.shape:
            # 调整尺寸
            if len(predicted.shape) >= 4:
                predicted = F.adaptive_avg_pool2d(predicted, actual.shape[2:])
            elif len(predicted.shape) == 3:
                predicted = F.interpolate(
                    predicted, size=actual.shape[-1], mode='linear'
                )
        
        # 使用余弦相似性作为置信度度量
        actual_flat = actual.flatten(1)
        predicted_flat = predicted.flatten(1)
        
        similarity = F.cosine_similarity(actual_flat, predicted_flat, dim=1)
        confidence = (similarity + 1) / 2  # 归一化到[0,1]
        
        return confidence.mean()


class PredictionNetwork(nn.Module):
    """预测网络
    
    单层预测网络，将当前层级特征预测为下一层输入。
    """
    
    def __init__(self, feature_dim: int, level: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.level = level
        
        # 预测网络结构随层级变化
        if level == 1:
            # 底层预测 - 详细的局部特征
            self.network = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(feature_dim, feature_dim, 3, 1, 1),
                nn.Tanh()
            )
        elif level == 2:
            # 中层预测 - 特征整合
            self.network = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim * 2, 1),
                nn.ReLU(),
                nn.Conv2d(feature_dim * 2, feature_dim, 1),
                nn.Tanh()
            )
        else:
            # 高层预测 - 抽象特征
            self.network = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim),
                nn.Tanh()
            )
        
        # 预测权重生成器
        self.weight_generator = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """预测前向传播"""
        # 生成预测
        prediction = self.network(features)
        
        # 计算预测权重
        weights = self.weight_generator(features)
        
        # 应用权重
        weighted_prediction = prediction * weights
        
        return weighted_prediction


class TemporalPredictor(nn.Module):
    """时间序列预测器"""
    
    def __init__(self, feature_dim: int, window_size: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.window_size = window_size
        
        # LSTM用于时间序列预测
        self.lstm = nn.LSTM(
            feature_dim, feature_dim // 2, 
            batch_first=True, num_layers=2, dropout=0.1
        )
        
        # 预测投影层
        self.prediction_projection = nn.Sequential(
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Tanh()
        )
        
        # 时间注意力
        self.temporal_attention = TemporalAttention(feature_dim // 2)
    
    def forward(self, temporal_context: torch.Tensor, 
                current_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """时间序列预测前向传播"""
        # 确保输入格式正确
        if len(temporal_context.shape) == 2:
            temporal_context = temporal_context.unsqueeze(1)
        
        # 时间注意力处理
        attended_context, attention_weights = self.temporal_attention(temporal_context)
        
        # LSTM预测
        lstm_output, (hidden, cell) = self.lstm(attended_context)
        
        # 预测下一个时间步
        if len(lstm_output.shape) > 2:
            next_hidden = lstm_output[:, -1, :]
        else:
            next_hidden = lstm_output
        
        # 投影到特征空间
        next_prediction = self.prediction_projection(next_hidden)
        
        # 调整形状匹配当前特征
        if len(current_features.shape) > 2:
            batch_size = current_features.shape[0]
            spatial_size = current_features.shape[2:]
            prediction_expanded = next_prediction.unsqueeze(-1).unsqueeze(-1)
            prediction_tiled = prediction_expanded.expand(-1, -1, *spatial_size)
        else:
            prediction_tiled = next_prediction
        
        return {
            'temporal_prediction': prediction_tiled,
            'attention_weights': attention_weights,
            'prediction_horizon': torch.tensor(self.window_size),
            'temporal_confidence': self._compute_temporal_confidence(next_prediction)
        }
    
    def _compute_temporal_confidence(self, prediction: torch.Tensor) -> torch.Tensor:
        """计算时间预测置信度"""
        # 基于预测值的变化程度评估置信度
        prediction_std = prediction.std(dim=1)
        confidence = 1 / (1 + prediction_std)  # 标准差越小，置信度越高
        return confidence.mean()


class TemporalAttention(nn.Module):
    """时间注意力机制"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 时间注意力查询
        self.query = nn.Linear(hidden_dim, hidden_dim)
        
        # 时间注意力键
        self.key = nn.Linear(hidden_dim, hidden_dim)
        
        # 时间注意力值
        self.value = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, temporal_context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """时间注意力前向传播"""
        # 计算注意力分数
        query = self.query(temporal_context)
        key = self.key(temporal_context)
        value = self.value(temporal_context)
        
        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = F.softmax(
            attention_scores / np.sqrt(self.hidden_dim), dim=-1
        )
        
        # 应用注意力
        attended_context = torch.matmul(attention_weights, value)
        
        return attended_context, attention_weights


class AdaptivePredictionController(nn.Module):
    """自适应预测控制器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 预测权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),  # 原始特征 + 预测特征
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # 预测质量评估器
        self.quality_evaluator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 自适应门控
        self.adaptive_gate = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, 
                predictions: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """自适应预测控制前向传播"""
        batch_size = features.shape[0]
        
        # 池化特征用于权重生成
        if len(features.shape) > 2:
            features_pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            features_pooled = features
        
        # 池化预测特征
        predictions_pooled = []
        for pred in predictions:
            if len(pred.shape) > 2:
                pred_pooled = F.adaptive_avg_pool2d(pred, 1).flatten(1)
            else:
                pred_pooled = pred
            predictions_pooled.append(pred_pooled)
        
        # 计算每个预测的权重
        weights = []
        for i, pred in enumerate(predictions):
            # 组合特征和预测
            if i < len(predictions_pooled):
                combined = torch.cat([features_pooled, predictions_pooled[i]], dim=1)
            else:
                combined = features_pooled
            
            # 生成权重
            weight = self.weight_generator(combined)
            weights.append(weight)
        
        # 评估预测质量
        quality_scores = []
        for pred in predictions:
            if len(pred.shape) > 2:
                pred_pooled = F.adaptive_avg_pool2d(pred, 1).flatten(1)
            else:
                pred_pooled = pred
            
            quality = self.quality_evaluator(pred_pooled)
            quality_scores.append(quality)
        
        # 自适应门控
        gate_input = features_pooled.mean(dim=1, keepdim=True)
        adaptive_gate = self.adaptive_gate(gate_input)
        
        # 计算最终质量分数
        if quality_scores:
            final_quality = torch.stack(quality_scores).mean()
        else:
            final_quality = torch.tensor(0.5)
        
        return {
            'weights': weights,
            'quality': final_quality,
            'adaptive_gate': adaptive_gate,
            'num_predictions': len(predictions)
        }


class PredictionMemory(nn.Module):
    """预测记忆模块"""
    
    def __init__(self, feature_dim: int, memory_size: int = 100):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        
        # 记忆存储
        self.register_buffer('memory_features', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_predictions', torch.zeros(memory_size, feature_dim))
        self.register_buffer('memory_weights', torch.ones(memory_size))
        self.register_buffer('memory_index', torch.tensor(0))
        self.register_buffer('memory_full', torch.tensor(False))
        
        # 记忆控制器
        self.memory_controller = nn.Sequential(
            nn.Linear(feature_dim * 2, 1),
            nn.Sigmoid()
        )
    
    def update(self, features: torch.Tensor, prediction: torch.Tensor) -> Dict[str, torch.Tensor]:
        """更新预测记忆"""
        batch_size = features.shape[0]
        
        # 池化特征
        if len(features.shape) > 2:
            features_pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            features_pooled = features.mean(dim=0)
        
        if len(prediction.shape) > 2:
            prediction_pooled = F.adaptive_avg_pool2d(prediction, 1).flatten(1)
        else:
            prediction_pooled = prediction.mean(dim=0)
        
        # 计算记忆价值
        memory_value = self.memory_controller(
            torch.cat([features_pooled, prediction_pooled], dim=-1)
        )
        
        # 更新记忆
        current_index = int(self.memory_index % self.memory_size)
        
        # 指数衰减更新权重
        decay_factor = 0.95
        self.memory_weights[current_index] = self.memory_weights[current_index] * decay_factor + memory_value * (1 - decay_factor)
        
        # 更新记忆内容
        self.memory_features[current_index] = features_pooled
        self.memory_predictions[current_index] = prediction_pooled
        
        # 更新索引
        self.memory_index = (self.memory_index + 1) % self.memory_size
        self.memory_full = self.memory_index == 0
        
        return {
            'memory_updated': True,
            'current_index': current_index,
            'memory_value': memory_value.mean(),
            'memory_utilization': (self.memory_weights > 0.1).float().mean()
        }
    
    def retrieve(self, query_features: torch.Tensor, top_k: int = 5) -> Dict[str, torch.Tensor]:
        """检索相关记忆"""
        if len(query_features.shape) > 2:
            query_pooled = F.adaptive_avg_pool2d(query_features, 1).flatten(1)
        else:
            query_pooled = query_features
        
        # 计算相似性
        similarities = F.cosine_similarity(
            query_pooled.unsqueeze(0), 
            self.memory_features[:int(self.memory_index)], 
            dim=1
        )
        
        # 获取top-k相似记忆
        top_indices = torch.topk(similarities, min(top_k, len(similarities)))[1]
        
        retrieved_features = self.memory_features[top_indices]
        retrieved_predictions = self.memory_predictions[top_indices]
        retrieved_weights = self.memory_weights[top_indices]
        
        return {
            'retrieved_features': retrieved_features,
            'retrieved_predictions': retrieved_predictions,
            'retrieved_weights': retrieved_weights,
            'similarities': similarities[top_indices]
        }


class ErrorAmplifier(nn.Module):
    """误差放大器
    
    基于预测误差的重要性放大显著的预测偏差。
    """
    
    def __init__(self, feature_dim: int, amplification_factor: float = 1.5):
        super().__init__()
        self.feature_dim = feature_dim
        self.amplification_factor = amplification_factor
        
        # 误差重要性评估器
        self.importance_evaluator = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 误差门控器
        self.error_gate = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, actual: torch.Tensor, predicted: torch.Tensor) -> Dict[str, torch.Tensor]:
        """误差放大前向传播"""
        # 计算原始误差
        raw_error = actual - predicted
        
        # 计算误差重要性
        if len(actual.shape) > 2:
            actual_pooled = F.adaptive_avg_pool2d(actual, 1).flatten(1)
        else:
            actual_pooled = actual
        
        importance_scores = self.importance_evaluator(actual_pooled)
        
        # 计算门控权重
        error_magnitude = raw_error.abs().mean(dim=1, keepdim=True)
        gate_weights = self.error_gate(error_magnitude)
        
        # 放大重要误差
        amplified_error = raw_error * importance_scores * gate_weights * self.amplification_factor
        
        # 计算误差统计
        error_stats = {
            'raw_error_magnitude': raw_error.abs().mean(),
            'amplified_error_magnitude': amplified_error.abs().mean(),
            'amplification_ratio': amplified_error.abs().mean() / (raw_error.abs().mean() + 1e-8),
            'importance_score': importance_scores.mean(),
            'gate_weight': gate_weights.mean()
        }
        
        return {
            'amplified_error': amplified_error,
            'raw_error': raw_error,
            'importance_scores': importance_scores,
            'gate_weights': gate_weights,
            'error_stats': error_stats
        }