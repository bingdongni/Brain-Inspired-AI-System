"""
处理层次管理模块
================

管理多个分层抽象层级的序列处理和协调。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import copy

from .hierarchical_layer import HierarchicalLayer
from .layer_config import LayerConfig, LayerType, ProcessingMode, create_default_layer_config


class ProcessingHierarchy(nn.Module):
    """
    处理层次管理
    
    管理多个分层抽象层级的序列处理，包括：
    - 层间连接管理
    - 前馈-反馈协调
    - 层级间信息传递
    - 动态路由
    """
    
    def __init__(self, layer_configs: List[LayerConfig]):
        super().__init__()
        self.layer_configs = layer_configs
        self.num_layers = len(layer_configs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建层级
        self.layers = nn.ModuleList([
            HierarchicalLayer(config) for config in layer_configs
        ])
        
        # 层间连接矩阵
        self.connection_matrix = self._build_connection_matrix()
        
        # 动态路由组件
        self.dynamic_router = DynamicRouter(self.num_layers)
        
        # 层级监控
        self.layer_monitors = nn.ModuleList([
            LayerMonitor(config) for config in layer_configs
        ])
    
    def _build_connection_matrix(self) -> torch.Tensor:
        """构建层间连接矩阵"""
        # 根据层级类型定义默认连接强度
        connection_strength = {
            (LayerType.V1, LayerType.V2): 1.0,
            (LayerType.V2, LayerType.V4): 0.9,
            (LayerType.V4, LayerType.IT): 0.8,
            (LayerType.IT, LayerType.VOTC): 0.7,
            (LayerType.IC, LayerType.MGB): 0.9,
            (LayerType.MGB, LayerType.AC): 0.8,
            (LayerType.PFC, LayerType.IT): 0.6,
            (LayerType.PFC, LayerType.V4): 0.4,
            (LayerType.PMD, LayerType.PFC): 0.5
        }
        
        # 创建连接矩阵
        matrix = torch.zeros(self.num_layers, self.num_layers)
        
        for i, config_i in enumerate(self.layer_configs):
            for j, config_j in enumerate(self.layer_configs):
                if i < j:  # 前馈连接
                    strength = connection_strength.get(
                        (config_i.layer_type, config_j.layer_type), 
                        0.3
                    )
                    matrix[i, j] = strength
                elif i > j:  # 反馈连接
                    strength = connection_strength.get(
                        (config_j.layer_type, config_i.layer_type), 
                        0.2
                    ) * 0.5  # 反馈连接通常较弱
                    matrix[i, j] = strength
        
        return matrix
    
    def forward(self, x: torch.Tensor, 
                hierarchy_inputs: Optional[Dict[str, torch.Tensor]] = None,
                mode: ProcessingMode = ProcessingMode.FEEDFORWARD,
                attention_signals: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        前向传播
        
        Args:
            x: 输入特征
            hierarchy_inputs: 分层输入（用于不同层级的额外输入）
            mode: 处理模式
            attention_signals: 注意力信号
            
        Returns:
            dict: 包含各层输出、层级表示等信息
        """
        layer_outputs = []
        feedback_signals = []
        prediction_errors = []
        
        current_input = x
        batch_size = x.shape[0]
        
        # 收集层级监控信息
        layer_info = []
        
        # 前向传播
        for i, layer in enumerate(self.layers):
            config = self.layer_configs[i]
            layer_type = config.layer_type.value
            
            # 层级特定输入
            hierarchical_input = None
            if hierarchy_inputs and layer_type in hierarchy_inputs:
                hierarchical_input = hierarchy_inputs[layer_type]
            
            # 层级注意力信号
            attention = None
            if attention_signals and layer_type in attention_signals:
                attention = attention_signals[layer_type]
            
            # 层间连接强度
            connection_strengths = self.connection_matrix[i]
            
            # 处理当前层级
            layer_output = layer(
                current_input,
                feedback=feedback_signals[-1] if feedback_signals else None,
                attention=attention,
                cross_modal=hierarchical_input,
                mode=mode
            )
            
            layer_outputs.append(layer_output)
            
            # 监控层级信息
            monitor_info = self.layer_monitors[i](layer_output)
            layer_info.append(monitor_info)
            
            # 准备下一层输入
            if mode == ProcessingMode.FEEDFORWARD:
                # 应用层间连接强度
                next_input = layer_output['features']
                if i < self.num_layers - 1:
                    # 应用连接权重
                    connection_weight = connection_strengths[i + 1] if i + 1 < self.num_layers else 1.0
                    current_input = next_input * connection_weight
                else:
                    current_input = next_input
        
        # 反馈传播（如果需要）
        if mode == ProcessingMode.FEEDBACK:
            feedback_signals = self._propagate_feedback(layer_outputs, mode)
        
        # 预测编码处理
        if mode == ProcessingMode.PREDICTIVE:
            prediction_errors = self._compute_prediction_errors(layer_outputs)
        
        # 动态路由调整
        if mode == ProcessingMode.INTEGRATION:
            routing_info = self.dynamic_router(layer_outputs)
            layer_outputs = self._apply_routing(layer_outputs, routing_info)
        
        # 准备输出
        output_dict = {
            'layer_outputs': layer_outputs,
            'layer_info': layer_info,
            'connection_matrix': self.connection_matrix,
            'final_output': layer_outputs[-1]['features'] if layer_outputs else x,
            'hierarchy_summary': {
                'num_layers': self.num_layers,
                'layer_types': [config.layer_type.value for config in self.layer_configs],
                'total_params': sum(p.numel() for p in self.parameters()),
                'active_layers': len([info for info in layer_info if info['is_active']]),
                'average_activation': torch.stack([info['activation_level'] for info in layer_info]).mean().item()
            }
        }
        
        # 添加模式特定信息
        if prediction_errors:
            output_dict['prediction_errors'] = prediction_errors
            
        if feedback_signals:
            output_dict['feedback_signals'] = feedback_signals
            
        if mode == ProcessingMode.INTEGRATION:
            output_dict['routing_info'] = routing_info
        
        return output_dict
    
    def _propagate_feedback(self, layer_outputs: List[Dict], mode: ProcessingMode) -> List[torch.Tensor]:
        """传播反馈信号"""
        feedback_signals = []
        
        # 从高层向低层传播反馈
        for i in reversed(range(len(layer_outputs))):
            if i == len(layer_outputs) - 1:
                # 顶层生成反馈
                feedback = layer_outputs[i]['features']
            else:
                # 根据层级特征生成反馈
                current_features = layer_outputs[i]['features']
                next_features = layer_outputs[i + 1]['features']
                
                # 简单的反馈生成（实际中可能更复杂）
                feedback = current_features + 0.1 * (current_features - F.adaptive_avg_pool2d(next_features, current_features.shape[2:]))
            
            feedback_signals.append(feedback)
        
        return feedback_signals
    
    def _compute_prediction_errors(self, layer_outputs: List[Dict]) -> List[torch.Tensor]:
        """计算预测误差"""
        prediction_errors = []
        
        for layer_output in layer_outputs:
            if 'prediction_errors' in layer_output:
                prediction_errors.append(layer_output['prediction_errors'])
        
        return prediction_errors
    
    def _apply_routing(self, layer_outputs: List[Dict], routing_info: Dict) -> List[Dict]:
        """应用动态路由"""
        routed_outputs = []
        
        for i, (layer_output, routing_weights) in enumerate(zip(layer_outputs, routing_info['routing_weights'])):
            # 应用路由权重
            modified_features = layer_output['features'] * routing_weights
            routed_output = layer_output.copy()
            routed_output['features'] = modified_features
            routed_output['routing_weights'] = routing_weights
            routed_outputs.append(routed_output)
        
        return routed_outputs
    
    def get_layer_by_type(self, layer_type: LayerType) -> Optional[HierarchicalLayer]:
        """根据类型获取层级"""
        for i, config in enumerate(self.layer_configs):
            if config.layer_type == layer_type:
                return self.layers[i]
        return None
    
    def get_receptive_fields(self) -> List[Dict[str, float]]:
        """获取所有层级的感受野信息"""
        receptive_fields = []
        for layer in self.layers:
            receptive_fields.append(layer.get_receptive_field_info())
        return receptive_fields
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        stats = {
            'total_layers': self.num_layers,
            'layer_types': [config.layer_type.value for config in self.layer_configs],
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'connection_matrix_stats': {
                'mean_strength': self.connection_matrix.mean().item(),
                'max_strength': self.connection_matrix.max().item(),
                'feedforward_connections': (self.connection_matrix > 0).sum().item() - self.num_layers,  # 减去自连接
                'feedback_connections': (self.connection_matrix < 0).sum().item()
            }
        }
        
        # 添加层级特定统计
        for i, config in enumerate(self.layer_configs):
            stats[f'layer_{i}_{config.layer_type.value}'] = {
                'input_channels': config.input_channels,
                'output_channels': config.output_channels,
                'receptive_field': config.get_receptive_field(),
                'prediction_coverage': config.get_prediction_coverage()
            }
        
        return stats


class DynamicRouter(nn.Module):
    """动态路由模块"""
    
    def __init__(self, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.router = nn.Sequential(
            nn.Linear(num_layers * 128, num_layers),
            nn.Softmax(dim=1)
        )
    
    def forward(self, layer_outputs: List[Dict]) -> Dict[str, torch.Tensor]:
        """动态路由计算"""
        # 收集层级特征
        layer_features = []
        for output in layer_outputs:
            features = output['features']
            if len(features.shape) > 2:
                # 全局池化
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            layer_features.append(features)
        
        # 路由权重计算
        combined_features = torch.cat(layer_features, dim=1)
        routing_weights = self.router(combined_features)
        
        return {
            'routing_weights': routing_weights,
            'routing_entropy': self._compute_entropy(routing_weights)
        }
    
    def _compute_entropy(self, weights: torch.Tensor) -> torch.Tensor:
        """计算路由熵"""
        return -torch.sum(weights * torch.log(weights + 1e-8), dim=1)


class LayerMonitor(nn.Module):
    """层级监控模块"""
    
    def __init__(self, config: LayerConfig):
        super().__init__()
        self.config = config
        
        # 活动监控
        self.activity_monitor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.output_channels, 1),
            nn.Sigmoid()
        )
        
        # 选择性监控
        self.selectivity_monitor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(config.output_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, layer_output: Dict) -> Dict[str, torch.Tensor]:
        """监控层级输出"""
        features = layer_output['features']
        
        # 监控活动水平
        activity_level = self.activity_monitor(features)
        
        # 监控选择性
        selectivity = self.selectivity_monitor(features)
        
        # 判断是否活跃
        is_active = activity_level > 0.1
        
        return {
            'activation_level': activity_level.mean(),
            'selectivity': selectivity.mean(),
            'is_active': is_active.mean() > 0.5,
            'spatial_variance': self._compute_spatial_variance(features),
            'temporal_stability': self._compute_temporal_stability(features)
        }
    
    def _compute_spatial_variance(self, features: torch.Tensor) -> torch.Tensor:
        """计算空间方差"""
        if len(features.shape) < 4:
            return torch.tensor(0.0)
        
        # 计算通道间的空间方差
        mean_features = features.mean(dim=1, keepdim=True)
        variance = ((features - mean_features) ** 2).mean()
        return variance
    
    def _compute_temporal_stability(self, features: torch.Tensor) -> torch.Tensor:
        """计算时间稳定性（简化版）"""
        # 这里简化处理，实际可能需要序列数据
        return torch.tensor(1.0)  # 假设稳定