"""
听觉层次处理模块
================

实现听觉皮层的专用层次化处理机制，包括：
- IC→MGB→AC的听觉通路
- 听觉预测误差处理
- 听觉层级组织
- 声音特征提取
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


class AuditoryProcessingStream(nn.Module):
    """
    听觉处理流
    
    实现完整的听觉层次处理通路，从IC到AC的层级音频特征提取和预测误差处理。
    """
    
    def __init__(self, input_channels: int = 1, feature_dim: int = 256):
        super().__init__()
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建听觉层次配置
        self.layer_configs = create_auditory_hierarchy(input_channels, feature_dim)
        
        # 创建听觉处理层次
        self.auditory_hierarchy = ProcessingHierarchy(self.layer_configs)
        
        # 听觉预测编码系统
        self.predictive_coding = AuditoryPredictiveCoding(feature_dim)
        
        # 听觉预测误差计算器
        self.error_calculator = AuditoryErrorCalculator(feature_dim)
        
        # 音频特征提取器
        self.audio_features = AudioFeatureExtractor(feature_dim)
        
        # 声音对象识别器
        self.sound_object_recognizer = SoundObjectRecognizer(feature_dim)
    
    def forward(self, x: torch.Tensor,
                temporal_context: Optional[torch.Tensor] = None,
                mode: ProcessingMode = ProcessingMode.FEEDFORWARD) -> Dict[str, torch.Tensor]:
        """
        听觉处理前向传播
        
        Args:
            x: 输入音频特征 [batch, channels, time, frequency]
            temporal_context: 时间上下文
            mode: 处理模式
            
        Returns:
            dict: 听觉处理结果
        """
        batch_size = x.shape[0]
        
        # 首先提取音频特征
        audio_features = self.audio_features(x)
        
        # 层级处理
        hierarchy_result = self.auditory_hierarchy(audio_features, mode=mode)
        
        # 提取层级输出
        ic_output = self._get_layer_output(hierarchy_result, LayerType.IC)
        mgb_output = self._get_layer_output(hierarchy_result, LayerType.MGB)
        ac_output = self._get_layer_output(hierarchy_result, LayerType.AC)
        
        # 预测编码处理
        prediction_results = {}
        error_results = {}
        
        if mode == ProcessingMode.PREDICTIVE:
            # 听觉预测
            prediction_results = self.predictive_coding(hierarchy_result)
            
            # 计算预测误差
            error_results = self.error_calculator(hierarchy_result, prediction_results)
        
        # 声音对象识别
        sound_objects = {}
        if ac_output is not None:
            sound_objects = self.sound_object_recognizer(ac_output, temporal_context)
        
        # 计算听觉层级统计
        hierarchy_stats = self._compute_auditory_statistics(hierarchy_result)
        
        # 整合输出
        auditory_result = {
            'final_output': ac_output['features'] if ac_output else mgb_output['features'] if mgb_output else ic_output['features'],
            'layer_outputs': hierarchy_result['layer_outputs'],
            'audio_features': audio_features,
            'prediction_results': prediction_results,
            'error_results': error_results,
            'sound_objects': sound_objects,
            'auditory_summary': {
                'processed_layers': len([out for out in [ic_output, mgb_output, ac_output] if out is not None]),
                'hierarchy_depth': len(self.layer_configs),
                'prediction_accuracy': prediction_results.get('accuracy', 0.0) if prediction_results else 0.0,
                'error_magnitude': error_results.get('average_error', 0.0) if error_results else 0.0,
                'detected_objects': sound_objects.get('num_objects', 0) if sound_objects else 0
            }
        }
        
        return auditory_result
    
    def _get_layer_output(self, hierarchy_result: Dict, layer_type: LayerType) -> Optional[Dict]:
        """从层次结果中提取指定层的输出"""
        for layer_output in hierarchy_result['layer_outputs']:
            if layer_output['layer_type'] == layer_type.value:
                return layer_output
        return None
    
    def _compute_auditory_statistics(self, hierarchy_result: Dict) -> Dict[str, float]:
        """计算听觉层级统计"""
        layer_outputs = hierarchy_result['layer_outputs']
        
        stats = {
            'hierarchy_depth': len(layer_outputs),
            'average_activation': 0.0,
            'prediction_error_propagation': 0.0,
            'temporal_consistency': 0.0
        }
        
        if layer_outputs:
            # 计算平均激活水平
            activations = []
            for output in layer_outputs:
                features = output['features']
                if len(features.shape) > 1:
                    activations.append(features.mean().item())
            
            if activations:
                stats['average_activation'] = np.mean(activations)
            
            # 计算预测误差传播（如果有预测误差）
            if 'prediction_errors' in hierarchy_result:
                errors = hierarchy_result['prediction_errors']
                if errors:
                    stats['prediction_error_propagation'] = torch.stack(errors).mean().item()
        
        return stats
    
    def get_auditory_hierarchy_info(self) -> Dict[str, any]:
        """获取听觉层次信息"""
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
            'hierarchy_statistics': self.auditory_hierarchy.get_processing_statistics()
        }


class AuditoryPredictiveCoding(nn.Module):
    """听觉预测编码"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 听觉特征预测网络
        self.temporal_predictor = nn.LSTM(
            feature_dim, feature_dim, batch_first=True, 
            num_layers=2, dropout=0.1
        )
        
        # 声音模式预测器
        self.pattern_predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh()
        )
        
        # 预测准确性评估
        self.accuracy_evaluator = nn.Sequential(
            nn.Linear(feature_dim * 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hierarchy_result: Dict) -> Dict[str, torch.Tensor]:
        """听觉预测编码前向传播"""
        layer_outputs = hierarchy_result['layer_outputs']
        
        predictions = {}
        
        for i, layer_output in enumerate(layer_outputs):
            if i == 0:
                continue  # 第一层没有前驱
            
            layer_type = layer_output['layer_type']
            current_features = layer_output['features']
            
            # 从当前层预测前一层的特征
            if hasattr(self, 'temporal_predictor') and len(current_features.shape) >= 3:
                # 时间序列预测
                predicted_features = self._temporal_predict(current_features, layer_type)
            else:
                # 模式预测
                predicted_features = self.pattern_predictor(current_features)
            
            predictions[layer_type] = predicted_features
        
        # 计算整体预测准确性
        if layer_outputs:
            final_features = layer_outputs[-1]['features']
            if len(final_features.shape) > 2:
                final_features = F.adaptive_avg_pool2d(final_features, 1).flatten(1)
            
            accuracy = self.accuracy_evaluator(final_features).mean()
        else:
            accuracy = torch.tensor(0.0)
        
        return {
            'predictions': predictions,
            'accuracy': accuracy,
            'temporal_coherence': self._compute_temporal_coherence(predictions)
        }
    
    def _temporal_predict(self, features: torch.Tensor, layer_type: str) -> torch.Tensor:
        """时间序列预测"""
        batch_size = features.shape[0]
        
        # 调整形状以适应LSTM
        if len(features.shape) == 4:  # [batch, channels, time, freq]
            features = features.transpose(1, 2).contiguous()  # [batch, time, channels*freq]
            features = features.view(batch_size, features.shape[1], -1)
        elif len(features.shape) == 3:  # [batch, channels, time]
            features = features.transpose(1, 2)  # [batch, time, channels]
        
        # LSTM预测
        predicted_sequence, _ = self.temporal_predictor(features)
        
        # 返回预测的下一个时间步
        return predicted_sequence
    
    def _compute_temporal_coherence(self, predictions: Dict) -> torch.Tensor:
        """计算时间一致性"""
        if not predictions:
            return torch.tensor(0.0)
        
        coherence_values = []
        for pred in predictions.values():
            if len(pred.shape) >= 3:
                # 计算相邻时间步的相似性
                similarity = F.cosine_similarity(pred[:, :-1], pred[:, 1:], dim=-1)
                coherence_values.append(similarity.mean())
        
        return torch.stack(coherence_values).mean() if coherence_values else torch.tensor(0.0)


class AuditoryErrorCalculator(nn.Module):
    """听觉预测误差计算器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 误差放大器
        self.error_amplifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 误差权重计算
        self.error_weights = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )
        
        # 层级误差聚合器
        self.error_aggregator = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),  # IC, MGB, AC
            nn.ReLU(),
            nn.Linear(feature_dim, 1)
        )
    
    def forward(self, hierarchy_result: Dict, predictions: Dict) -> Dict[str, torch.Tensor]:
        """听觉预测误差计算前向传播"""
        layer_outputs = hierarchy_result['layer_outputs']
        
        # 提取各层实际输出
        actual_outputs = {}
        for layer_output in layer_outputs:
            layer_type = layer_output['layer_type']
            actual_outputs[layer_type] = layer_output['features']
        
        # 计算各层预测误差
        layer_errors = {}
        for layer_type, predicted in predictions.items():
            if layer_type in actual_outputs:
                actual = actual_outputs[layer_type]
                
                # 调整尺寸以匹配
                predicted_resized = self._resize_prediction(predicted, actual.shape)
                
                # 计算误差
                error = actual - predicted_resized
                
                # 放大重要误差
                error_weights = self.error_weights(actual)
                amplified_error = error * error_weights
                
                layer_errors[layer_type] = amplified_error
        
        # 层级误差聚合
        if len(layer_errors) >= 2:
            # 按层级顺序聚合误差
            ic_error = layer_errors.get('IC', torch.zeros_like(list(layer_errors.values())[0]))
            mgb_error = layer_errors.get('MGB', torch.zeros_like(ic_error))
            ac_error = layer_errors.get('AC', torch.zeros_like(ic_error))
            
            # 聚合误差特征
            if len(ic_error.shape) > 2:
                ic_error_pooled = F.adaptive_avg_pool2d(ic_error, 1).flatten(1)
                mgb_error_pooled = F.adaptive_avg_pool2d(mgb_error, 1).flatten(1)
                ac_error_pooled = F.adaptive_avg_pool2d(ac_error, 1).flatten(1)
            else:
                ic_error_pooled = ic_error
                mgb_error_pooled = mgb_error
                ac_error_pooled = ac_error
            
            # 拼接误差
            combined_errors = torch.cat([ic_error_pooled, mgb_error_pooled, ac_error_pooled], dim=-1)
            aggregated_error = self.error_aggregator(combined_errors)
            
            average_error = torch.stack([err.mean() for err in layer_errors.values()]).mean()
        else:
            aggregated_error = torch.zeros(1)
            average_error = torch.tensor(0.0)
        
        return {
            'layer_errors': layer_errors,
            'aggregated_error': aggregated_error,
            'average_error': average_error,
            'error_hierarchy_progression': self._compute_error_progression(layer_errors)
        }
    
    def _resize_prediction(self, predicted: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """调整预测结果的尺寸以匹配实际输出"""
        if predicted.shape == target_shape:
            return predicted
        
        # 根据维度数量选择插值方法
        if len(predicted.shape) >= 4 and len(target_shape) >= 4:
            return F.interpolate(
                predicted, size=target_shape[2:], mode='bilinear', align_corners=False
            )
        elif len(predicted.shape) == 3 and len(target_shape) == 3:
            return F.interpolate(
                predicted, size=target_shape[2], mode='linear', align_corners=False
            )
        else:
            # 如果形状不匹配，使用简单的插值或填充
            min_size = min(predicted.shape[-1], target_shape[-1])
            if len(predicted.shape) >= 3:
                predicted_cropped = predicted[..., :min_size]
            else:
                predicted_cropped = predicted[:, :min_size]
            return predicted_cropped
    
    def _compute_error_progression(self, layer_errors: Dict) -> torch.Tensor:
        """计算层级误差进展"""
        if not layer_errors:
            return torch.tensor(0.0)
        
        # 层级顺序
        layer_order = ['IC', 'MGB', 'AC']
        error_magnitudes = []
        
        for layer in layer_order:
            if layer in layer_errors:
                error_magnitude = torch.abs(layer_errors[layer]).mean()
                error_magnitudes.append(error_magnitude)
        
        if len(error_magnitudes) >= 2:
            # 计算误差增长趋势
            error_progression = torch.stack(error_magnitudes[1:]) - torch.stack(error_magnitudes[:-1])
            return error_progression.mean()
        else:
            return torch.tensor(0.0)


class AudioFeatureExtractor(nn.Module):
    """音频特征提取器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 频谱特征提取
        self.spectral_extractor = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=256, stride=128),  # 简化的频谱分析
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=128, stride=64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(512)  # 标准化到固定长度
        )
        
        # 时域特征提取
        self.temporal_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, stride=8),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256)
        )
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(128 + 64, feature_dim),  # 128(频谱) + 64(时域)
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """音频特征提取前向传播"""
        # x 形状: [batch, channels, time] 或 [batch, channels, time, frequency]
        
        if len(x.shape) == 4:  # 有频率维度
            # 合并频率维度
            x = x.mean(dim=-1)  # [batch, channels, time]
        
        if x.shape[1] == 1:  # 单声道
            x = x.squeeze(1)  # [batch, time]
        
        # 频谱特征提取
        spectral_features = self.spectral_extractor(x.unsqueeze(1))  # [batch, 128, 512]
        spectral_features = spectral_features.flatten(1)  # [batch, 128*512]
        
        # 时域特征提取
        temporal_features = self.temporal_extractor(x.unsqueeze(1))  # [batch, 64, 256]
        temporal_features = temporal_features.flatten(1)  # [batch, 64*256]
        
        # 特征融合
        combined_features = torch.cat([spectral_features, temporal_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # 重塑为标准格式
        batch_size = x.shape[0]
        output_features = fused_features.view(batch_size, self.feature_dim, -1)
        
        return output_features


class SoundObjectRecognizer(nn.Module):
    """声音对象识别器"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 声音对象检测器
        self.object_detector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
        # 声音类别分类器
        self.category_classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 10),  # 10个声音类别
            nn.Softmax(dim=1)
        )
        
        # 时间一致性检查器
        self.temporal_consistency = nn.LSTM(
            feature_dim, feature_dim // 2, batch_first=True, num_layers=1
        )
    
    def forward(self, ac_output: Dict[str, torch.Tensor], 
                temporal_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """声音对象识别前向传播"""
        features = ac_output['features']
        
        # 特征池化
        if len(features.shape) > 2:
            pooled_features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            pooled_features = features
        
        # 检测声音对象
        object_scores = self.object_detector(pooled_features)
        
        # 分类声音类别
        category_probabilities = self.category_classifier(pooled_features)
        
        # 时间一致性分析
        temporal_coherence = 0.0
        if temporal_context is not None:
            if len(temporal_context.shape) == 2:
                temporal_context = temporal_context.unsqueeze(1)
            
            temporal_output, _ = self.temporal_consistency(temporal_context)
            temporal_coherence = temporal_output.mean()
        
        # 计算对象数量和置信度
        object_threshold = 0.5
        num_objects = (object_scores > object_threshold).sum(dim=1).float()
        object_confidence = object_scores.mean()
        
        # 找到最可能的类别
        dominant_category = torch.argmax(category_probabilities, dim=1)
        category_confidence = category_probabilities.max(dim=1)[0].mean()
        
        return {
            'object_scores': object_scores,
            'category_probabilities': category_probabilities,
            'dominant_category': dominant_category,
            'num_objects': num_objects,
            'object_confidence': object_confidence,
            'category_confidence': category_confidence,
            'temporal_coherence': temporal_coherence,
            'recognition_quality': (object_confidence + category_confidence) / 2
        }


def create_auditory_hierarchy(input_channels: int, feature_dim: int = 256) -> List[LayerConfig]:
    """创建听觉层次配置
    
    Args:
        input_channels: 输入通道数（通常为1表示单声道）
        feature_dim: 特征维度
        
    Returns:
        List[LayerConfig]: 听觉层次配置列表
    """
    configs = []
    
    # IC - 下丘（听觉皮层下）
    ic_config = create_default_layer_config(
        LayerType.IC, input_channels, 64, kernel_size=5
    )
    ic_config.prediction_strength = 0.4  # IC层预测能力较弱
    ic_config.error_amplification = 1.0
    configs.append(ic_config)
    
    # MGB - 内侧膝状体（听觉中继）
    mgb_config = create_default_layer_config(
        LayerType.MGB, 64, 128, kernel_size=3
    )
    mgb_config.prediction_strength = 0.5
    mgb_config.error_amplification = 1.1
    configs.append(mgb_config)
    
    # AC - 听觉皮层
    ac_config = create_default_layer_config(
        LayerType.AC, 128, feature_dim, kernel_size=3
    )
    ac_config.prediction_strength = 0.6
    ac_config.error_amplification = 1.2
    ac_config.attention_enabled = True
    configs.append(ac_config)
    
    return configs