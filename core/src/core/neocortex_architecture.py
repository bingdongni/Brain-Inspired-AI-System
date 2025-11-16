"""
新皮层模拟器架构 (Neocortex Simulator Architecture)
==================================================

整合所有新皮层模拟器核心组件的主架构：
- 任务优化神经网络 (TONN: Task-Optimized Neural Network)
- 模块化新皮层架构
- 端到端学习与推理
- 多层次协同处理

基于论文：
- "Task-Optimized Neural Networks" 概念设计
- 新皮层层次化信息处理机制综述
- 预测编码、注意控制、决策形成的协同机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import math
import copy

# 导入所有核心模块
from .hierarchical_layers import (
    ProcessingHierarchy, LayerConfig, LayerType, ProcessingMode,
    create_visual_hierarchy, create_auditory_hierarchy
)
from .processing_modules import (
    PredictionModule, AttentionModule, DecisionModule, CrossModalModule,
    PredictionType, AttentionType, DecisionMode, ProcessingConfig,
    create_prediction_module, create_attention_module, 
    create_decision_module, create_crossmodal_module
)
from .abstraction import (
    AbstractionEngine, ConceptUnit, SemanticAbstraction,
    ConceptConfig, ConceptType, AbstractionLevel,
    create_abstraction_engine, create_concept_units
)
from .sparse_activation import (
    ConsolidationEngine, SparseActivation, WeightConsolidation, EngramCell,
    ConsolidationConfig, EngramConfig, CellType, MemoryState,
    create_consolidation_engine
)


class ArchitectureType(Enum):
    """架构类型"""
    TONN = "tonn"                      # 任务优化神经网络
    HIERARCHICAL = "hierarchical"      # 层次化架构
    MODULAR = "modular"               # 模块化架构
    HYBRID = "hybrid"                 # 混合架构


class ProcessingStage(Enum):
    """处理阶段"""
    INPUT = "input"
    HIERARCHICAL_PROCESSING = "hierarchical_processing"
    PREDICTION = "prediction"
    ATTENTION = "attention"
    ABSTRACTION = "abstraction"
    CONSOLIDATION = "consolidation"
    DECISION = "decision"
    OUTPUT = "output"


@dataclass
class NeocortexConfig:
    """新皮层配置"""
    # 基础架构参数
    architecture_type: ArchitectureType = ArchitectureType.TONN
    input_dim: int = 1024
    hidden_dim: int = 512
    output_dim: int = 128
    
    # 层次结构参数
    num_hierarchical_layers: int = 5
    layer_types: List[LayerType] = field(default_factory=lambda: [
        LayerType.V1, LayerType.V2, LayerType.V4, LayerType.IT, LayerType.VOTC
    ])
    
    # 处理模块参数
    prediction_enabled: bool = True
    attention_enabled: bool = True
    abstraction_enabled: bool = True
    consolidation_enabled: bool = True
    decision_enabled: bool = True
    crossmodal_enabled: bool = False
    
    # 学习参数
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    dropout_rate: float = 0.1
    
    # 协同参数
    hierarchical_weight: float = 0.25
    prediction_weight: float = 0.15
    attention_weight: float = 0.15
    abstraction_weight: float = 0.20
    consolidation_weight: float = 0.15
    decision_weight: float = 0.10
    
    # 特殊功能参数
    num_concepts: int = 50
    num_engram_cells: int = 100
    target_sparsity: float = 0.1


class NeocortexSimulator(nn.Module):
    """
    新皮层模拟器主架构
    
    整合所有核心组件，实现完整的端到端新皮层模拟：
    - 分层抽象处理
    - 预测编码
    - 注意控制
    - 知识抽象
    - 稀疏激活与巩固
    - 决策形成
    - 跨模态整合
    """
    
    def __init__(self, config: NeocortexConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化所有处理模块
        self._initialize_modules()
        
        # 架构协调器
        self.architecture_coordinator = ArchitectureCoordinator(config)
        
        # 端到端优化器
        self.end_to_end_optimizer = EndToEndOptimizer(config)
        
        # 性能监控器
        self.performance_monitor = PerformanceMonitor()
        
    def _initialize_modules(self):
        """初始化所有处理模块"""
        
        # 1. 分层处理模块
        if self.config.architecture_type in [ArchitectureType.TONN, ArchitectureType.HIERARCHICAL]:
            self.hierarchical_processor = self._create_hierarchical_processor()
        
        # 2. 预测编码模块
        if self.config.prediction_enabled:
            self.prediction_module = create_prediction_module(self.config.input_dim)
        
        # 3. 注意控制模块
        if self.config.attention_enabled:
            self.attention_module = create_attention_module(self.config.hidden_dim)
        
        # 4. 知识抽象模块
        if self.config.abstraction_enabled:
            self.abstraction_engine = create_abstraction_engine(
                self.config.hidden_dim, self.config.num_concepts
            )
        
        # 5. 稀疏激活与巩固模块
        if self.config.consolidation_enabled:
            self.consolidation_engine = create_consolidation_engine(
                self.config.hidden_dim, self.config.num_engram_cells
            )
        
        # 6. 决策形成模块
        if self.config.decision_enabled:
            self.decision_module = create_decision_module(self.config.hidden_dim)
        
        # 7. 跨模态接口模块
        if self.config.crossmodal_enabled:
            self.crossmodal_module = create_crossmodal_module(self.config.hidden_dim)
        
        # 8. 特征变换器（模块间协调）
        self.feature_transformer = FeatureTransformer(self.config)
        
        # 9. 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.config.output_dim, self.config.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.output_dim // 2, 64),
            nn.Tanh()
        )
        
    def _create_hierarchical_processor(self) -> ProcessingHierarchy:
        """创建分层处理器"""
        layer_configs = []
        
        for i, layer_type in enumerate(self.config.layer_types):
            # 根据层级确定参数
            input_channels = self.config.input_dim if i == 0 else self.config.hidden_dim
            output_channels = self.config.hidden_dim
            
            config = LayerConfig(
                layer_type=layer_type,
                input_channels=input_channels,
                output_channels=output_channels,
                kernel_size=3 + (i % 2) * 2,  # 交替使用3x3和5x5核
                stride=1 if i < 2 else 2,     # 前两层 stride=1，后面的 stride=2
                padding=1,
                dropout_rate=self.config.dropout_rate
            )
            layer_configs.append(config)
        
        return ProcessingHierarchy(layer_configs)
    
    def forward(self, inputs: Dict[str, torch.Tensor],
                context: Optional[Dict] = None) -> Dict[str, Union[torch.Tensor, List, Dict]]:
        """
        新皮层模拟器前向传播
        
        Args:
            inputs: 输入字典，包含不同模态的数据
                    {
                        'visual': torch.Tensor,    # 视觉输入
                        'audio': torch.Tensor,     # 听觉输入
                        'text': torch.Tensor,      # 文本输入
                        'multimodal': torch.Tensor # 多模态输入
                    }
            context: 上下文信息
            
        Returns:
            完整的处理结果
        """
        stage_outputs = {}
        stage_info = {}
        
        # 输入阶段
        if 'multimodal' in inputs:
            current_features = inputs['multimodal']
        elif 'visual' in inputs:
            current_features = inputs['visual']
        else:
            # 默认使用第一个可用的输入
            current_features = list(inputs.values())[0]
            
        stage_outputs[ProcessingStage.INPUT] = current_features
        stage_info[ProcessingStage.INPUT] = {'input_shape': current_features.shape}
        
        # 层次化处理阶段
        if hasattr(self, 'hierarchical_processor'):
            hierarchical_results = self.hierarchical_processor(current_features)
            current_features = hierarchical_results['final_output']
            stage_outputs[ProcessingStage.HIERARCHICAL_PROCESSING] = hierarchical_results
            stage_info[ProcessingStage.HIERARCHICAL_PROCESSING] = {
                'num_layers': len(self.config.layer_types),
                'layer_types': [lt.value for lt in self.config.layer_types]
            }
        
        # 预测编码阶段
        if self.config.prediction_enabled:
            # 准备序列输入（如果有时间维度）
            if len(current_features.shape) == 2:
                # 展平为序列
                seq_input = current_features.unsqueeze(1)  # [batch, 1, dim]
            else:
                seq_input = current_features.view(current_features.size(0), -1).unsqueeze(1)
                
            prediction_results = self.prediction_module(seq_input)
            current_features = prediction_results.get('features', current_features)
            stage_outputs[ProcessingStage.PREDICTION] = prediction_results
            stage_info[ProcessingStage.PREDICTION] = {
                'prediction_horizon': prediction_results['predictions'].shape[1],
                'avg_confidence': prediction_results['confidence'].mean().item()
            }
        
        # 注意控制阶段
        if self.config.attention_enabled:
            # 重塑为空间格式（如果需要）
            if len(current_features.shape) == 2:
                # 假设是特征向量，重塑为空间格式
                spatial_features = current_features.view(
                    current_features.size(0), -1, int(math.sqrt(current_features.size(1))), 
                    int(math.sqrt(current_features.size(1)))
                )
            else:
                spatial_features = current_features
                
            attention_results = self.attention_module(spatial_features)
            current_features = attention_results['attended_features']
            
            # 重塑回原始维度
            current_features = current_features.view(current_features.size(0), -1)
            stage_outputs[ProcessingStage.ATTENTION] = attention_results
            stage_info[ProcessingStage.ATTENTION] = {
                'spatial_attention_shape': attention_results['spatial_attention'].shape,
                'feature_attention_shape': attention_results['feature_attention'].shape
            }
        
        # 特征变换（确保维度匹配）
        current_features = self.feature_transformer(current_features)
        
        # 知识抽象阶段
        if self.config.abstraction_enabled:
            abstraction_results = self.abstraction_engine(current_features, context)
            current_features = abstraction_results['final_abstraction']['integrated_representation']
            stage_outputs[ProcessingStage.ABSTRACTION] = abstraction_results
            stage_info[ProcessingStage.ABSTRACTION] = {
                'active_concepts': abstraction_results['abstraction_summary']['num_active_concepts'],
                'abstraction_level': abstraction_results['abstraction_summary']['abstraction_level']
            }
        
        # 稀疏激活与巩固阶段
        if self.config.consolidation_enabled:
            consolidation_results = self.consolidation_engine(current_features, context)
            current_features = consolidation_results['final_output']['integrated_output']
            stage_outputs[ProcessingStage.CONSOLIDATION] = consolidation_results
            stage_info[ProcessingStage.CONSOLIDATION] = {
                'active_engram_cells': consolidation_results['consolidation_summary']['active_engram_cells'],
                'formed_engrams': consolidation_results['consolidation_summary']['formed_engrams'],
                'memory_strength': consolidation_results['consolidation_summary']['memory_strength']
            }
        
        # 决策形成阶段
        if self.config.decision_enabled:
            # 准备决策输入
            decision_input = current_features
            if len(decision_input.shape) > 2:
                decision_input = decision_input.view(decision_input.size(0), -1)
                
            decision_results = self.decision_module(decision_input)
            current_features = decision_results['decision']
            stage_outputs[ProcessingStage.DECISION] = decision_results
            stage_info[ProcessingStage.DECISION] = {
                'decision_shape': decision_results['decision'].shape,
                'avg_confidence': decision_results['confidence'].mean().item()
            }
        
        # 输出阶段
        output = self.output_layer(current_features)
        stage_outputs[ProcessingStage.OUTPUT] = output
        stage_info[ProcessingStage.OUTPUT] = {'output_shape': output.shape}
        
        # 架构协调
        coordination_results = self.architecture_coordinator(stage_outputs)
        
        # 性能监控
        performance_metrics = self.performance_monitor(stage_outputs)
        
        return {
            'stage_outputs': stage_outputs,
            'stage_info': stage_info,
            'coordination': coordination_results,
            'performance_metrics': performance_metrics,
            'final_output': output,
            'summary': self._generate_processing_summary(stage_info, performance_metrics)
        }
    
    def _generate_processing_summary(self, stage_info: Dict, 
                                   performance_metrics: Dict) -> Dict[str, Any]:
        """生成处理摘要"""
        
        summary = {
            'total_stages': len(stage_info),
            'active_stages': list(stage_info.keys()),
            'architecture_type': self.config.architecture_type.value,
            'processing_efficiency': performance_metrics.get('overall_efficiency', 0.0)
        }
        
        # 收集各阶段的关键指标
        key_metrics = {}
        for stage, info in stage_info.items():
            if stage == ProcessingStage.PREDICTION:
                key_metrics['prediction_confidence'] = info.get('avg_confidence', 0)
            elif stage == ProcessingStage.ABSTRACTION:
                key_metrics['active_concepts'] = info.get('active_concepts', 0)
                key_metrics['abstraction_level'] = info.get('abstraction_level', 0)
            elif stage == ProcessingStage.CONSOLIDATION:
                key_metrics['memory_strength'] = info.get('memory_strength', 0)
            elif stage == ProcessingStage.DECISION:
                key_metrics['decision_confidence'] = info.get('avg_confidence', 0)
        
        summary['key_metrics'] = key_metrics
        
        return summary


class FeatureTransformer(nn.Module):
    """特征变换器 - 确保模块间的维度匹配"""
    
    def __init__(self, config: NeocortexConfig):
        super().__init__()
        self.config = config
        
        # 输入变换
        self.input_transformer = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # 中间层变换
        self.middle_transformer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # 输出变换
        self.output_transformer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.Tanh()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """特征变换前向传播"""
        batch_size = features.shape[0]
        
        # 确保特征维度正确
        if features.shape[-1] != self.config.hidden_dim:
            if features.shape[-1] == self.config.input_dim:
                features = self.input_transformer(features)
            else:
                # 通用变换到隐藏维度
                features = self.middle_transformer(features)
        
        return features


class ArchitectureCoordinator(nn.Module):
    """架构协调器 - 协调各模块的工作"""
    
    def __init__(self, config: NeocortexConfig):
        super().__init__()
        self.config = config
        
        # 模块权重学习器
        self.module_weights = nn.Parameter(torch.tensor([
            config.hierarchical_weight,
            config.prediction_weight,
            config.attention_weight,
            config.abstraction_weight,
            config.consolidation_weight,
            config.decision_weight
        ]))
        
        # 协调策略学习器
        self.coordination_strategy = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, stage_outputs: Dict[ProcessingStage, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """架构协调"""
        
        # 归一化模块权重
        weights = F.softmax(self.module_weights, dim=0)
        
        # 提取各模块输出
        module_outputs = []
        active_modules = []
        
        module_mapping = {
            ProcessingStage.HIERARCHICAL_PROCESSING: 0,
            ProcessingStage.PREDICTION: 1,
            ProcessingStage.ATTENTION: 2,
            ProcessingStage.ABSTRACTION: 3,
            ProcessingStage.CONSOLIDATION: 4,
            ProcessingStage.DECISION: 5
        }
        
        for stage, output in stage_outputs.items():
            if stage in module_mapping:
                module_idx = module_mapping[stage]
                if isinstance(output, dict):
                    # 处理复杂输出类型
                    if 'final_output' in output:
                        module_output = output['final_output']
                    elif 'features' in output:
                        module_output = output['features']
                    else:
                        module_output = torch.zeros_like(output.get('integrated_output', 
                                                                  torch.tensor(0.0)))
                else:
                    module_output = output
                    
                module_outputs.append(module_output)
                active_modules.append(stage.value)
        
        # 协调策略
        strategy_weights = self.coordination_strategy(weights.unsqueeze(0))
        
        # 计算协调输出
        if module_outputs:
            # 加权融合
            coordinated_output = torch.zeros_like(module_outputs[0])
            for i, (output, weight) in enumerate(zip(module_outputs, weights)):
                coordinated_output += weight[i] * output
            
            coordination_quality = torch.tensor(len(module_outputs) / 6.0)  # 模块利用率
        else:
            coordinated_output = torch.zeros(1, 10)  # 默认输出
            coordination_quality = torch.tensor(0.0)
        
        return {
            'module_weights': weights,
            'coordinated_output': coordinated_output,
            'active_modules': active_modules,
            'coordination_strategy': strategy_weights.squeeze(0),
            'coordination_quality': coordination_quality,
            'module_utilization': len(module_outputs) / 6.0
        }


class EndToEndOptimizer(nn.Module):
    """端到端优化器 - 优化整个架构"""
    
    def __init__(self, config: NeocortexConfig):
        super().__init__()
        self.config = config
        
        # 损失函数组合器
        self.loss_combiner = nn.Sequential(
            nn.Linear(6, 3),  # 6个模块的损失
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        
        # 梯度平衡器
        self.gradient_balancer = nn.Sequential(
            nn.Linear(6, 6),
            nn.Softmax(dim=-1)
        )
        
    def compute_losses(self, outputs: Dict[str, Any], 
                      targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """计算端到端损失"""
        
        losses = {}
        
        # 层次化处理损失
        if hasattr(self, 'hierarchical_processor'):
            hier_output = outputs['stage_outputs'].get(ProcessingStage.HIERARCHICAL_PROCESSING)
            if hier_output and 'prediction_errors' in hier_output:
                losses['hierarchical'] = torch.mean(torch.stack(
                    torch.norm(error, dim=-1) for error in hier_output['prediction_errors']
                ))
        
        # 预测编码损失
        if self.config.prediction_enabled:
            pred_output = outputs['stage_outputs'].get(ProcessingStage.PREDICTION)
            if pred_output and 'prediction_errors' in pred_output:
                losses['prediction'] = torch.mean(torch.norm(
                    pred_output['prediction_errors'], dim=-1
                ))
        
        # 注意控制损失（最小化不必要的激活）
        if self.config.attention_enabled:
            attn_output = outputs['stage_outputs'].get(ProcessingStage.ATTENTION)
            if attn_output:
                attention_sparsity = torch.mean(attn_output['spatial_attention'])
                losses['attention'] = -torch.log(attention_sparsity + 1e-8)  # 鼓励稀疏
        
        # 抽象损失（最大化概念形成）
        if self.config.abstraction_enabled:
            abs_output = outputs['stage_outputs'].get(ProcessingStage.ABSTRACTION)
            if abs_output:
                abstraction_quality = abs_output['final_abstraction']['abstraction_quality']
                losses['abstraction'] = -torch.log(abstraction_quality + 1e-8)  # 最大化质量
        
        # 巩固损失（平衡巩固与遗忘）
        if self.config.consolidation_enabled:
            cons_output = outputs['stage_outputs'].get(ProcessingStage.CONSOLIDATION)
            if cons_output:
                memory_strength = cons_output['consolidation_summary']['memory_strength']
                losses['consolidation'] = -torch.log(memory_strength + 1e-8)  # 最大化记忆强度
        
        # 决策损失
        if self.config.decision_enabled and targets is not None:
            decision_output = outputs['stage_outputs'].get(ProcessingStage.DECISION)
            if decision_output:
                prediction = decision_output['decision']
                losses['decision'] = F.mse_loss(prediction, targets)
        
        # 总体损失
        if losses:
            # 加权组合损失
            loss_weights = torch.tensor([1.0] * len(losses))
            total_loss = sum(weight * loss for weight, loss in zip(loss_weights, losses.values()))
            losses['total'] = total_loss
        
        return losses


class PerformanceMonitor(nn.Module):
    """性能监控器"""
    
    def __init__(self):
        super().__init__()
        
        # 效率计算器
        self.efficiency_calculator = nn.Sequential(
            nn.Linear(4, 8),  # 处理时间、准确率、内存使用、计算复杂度
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 质量评估器
        self.quality_assessor = nn.Sequential(
            nn.Linear(3, 6),  # 抽象质量、巩固质量、决策质量
            nn.ReLU(),
            nn.Linear(6, 1),
            nn.Sigmoid()
        )
        
    def forward(self, stage_outputs: Dict[ProcessingStage, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """性能监控前向传播"""
        
        # 计算处理效率
        processing_efficiency = self._calculate_processing_efficiency(stage_outputs)
        
        # 评估输出质量
        output_quality = self._assess_output_quality(stage_outputs)
        
        # 计算资源使用
        resource_usage = self._calculate_resource_usage(stage_outputs)
        
        # 综合性能指标
        performance_input = torch.tensor([
            processing_efficiency.item(),
            output_quality.item(),
            resource_usage['computational_complexity'],
            resource_usage['memory_efficiency']
        ])
        
        overall_performance = self.efficiency_calculator(performance_input.unsqueeze(0))
        
        return {
            'processing_efficiency': processing_efficiency,
            'output_quality': output_quality,
            'resource_usage': resource_usage,
            'overall_efficiency': overall_performance.squeeze(-1)
        }
    
    def _calculate_processing_efficiency(self, stage_outputs: Dict[ProcessingStage, torch.Tensor]) -> torch.Tensor:
        """计算处理效率"""
        # 简化的效率计算：基于活跃模块数量
        active_stages = len(stage_outputs)
        max_stages = len(ProcessingStage)
        
        efficiency = torch.tensor(active_stages / max_stages)
        return efficiency
    
    def _assess_output_quality(self, stage_outputs: Dict[ProcessingStage, torch.Tensor]) -> torch.Tensor:
        """评估输出质量"""
        # 评估各阶段输出的统计特征
        quality_scores = []
        
        for stage, output in stage_outputs.items():
            if isinstance(output, dict):
                # 复杂输出类型
                if 'features' in output:
                    features = output['features']
                elif 'final_output' in output:
                    features = output['final_output']
                else:
                    features = torch.zeros_like(list(output.values())[0])
            else:
                features = output
                
            # 计算特征的质量指标
            if len(features.shape) > 1:
                # 多维特征：计算方差和幅度
                variance_score = torch.var(features)
                magnitude_score = torch.mean(torch.abs(features))
                quality_score = torch.sigmoid(variance_score + magnitude_score)
            else:
                # 一维特征：直接使用值
                quality_score = torch.sigmoid(torch.mean(features))
                
            quality_scores.append(quality_score)
        
        if quality_scores:
            avg_quality = torch.stack(quality_scores).mean()
        else:
            avg_quality = torch.tensor(0.5)
            
        return avg_quality
    
    def _calculate_resource_usage(self, stage_outputs: Dict[ProcessingStage, torch.Tensor]) -> Dict[str, float]:
        """计算资源使用"""
        
        # 计算总参数数量（简化）
        total_params = sum(
            sum(p.numel() for p in stage.output.parameters()) 
            if hasattr(stage, 'parameters') else 0
            for stage in stage_outputs.values()
        )
        
        # 计算计算复杂度（基于输出张量大小）
        total_compute = sum(
            torch.prod(torch.tensor(output.shape)).item()
            for output in stage_outputs.values()
            if isinstance(output, torch.Tensor)
        )
        
        # 归一化指标
        computational_complexity = min(total_compute / 1e6, 1.0)  # 归一化到[0,1]
        memory_efficiency = 1.0 / (1.0 + total_params / 1e6)  # 内存效率
        
        return {
            'total_parameters': total_params,
            'computational_complexity': computational_complexity,
            'memory_efficiency': memory_efficiency
        }


# 高级架构类

class TONN(nn.Module):
    """
    任务优化神经网络 (Task-Optimized Neural Network)
    
    专为特定任务优化的新皮层架构变体
    """
    
    def __init__(self, config: NeocortexConfig):
        super().__init__()
        self.config = config
        
        # 基础新皮层模拟器
        self.neocortex_simulator = NeocortexSimulator(config)
        
        # 任务特定适配器
        self.task_adapter = TaskSpecificAdapter(config)
        
        # 任务优化器
        self.task_optimizer = TaskOptimizer(config)
        
    def forward(self, inputs: Dict[str, torch.Tensor], 
                task_context: Optional[Dict] = None) -> Dict[str, Any]:
        """TONN前向传播"""
        
        # 基础新皮层处理
        base_results = self.neocortex_simulator(inputs)
        
        # 任务特定适配
        adapted_results = self.task_adapter(base_results, task_context)
        
        # 任务优化
        optimized_results = self.task_optimizer(adapted_results)
        
        return {
            'base_processing': base_results,
            'task_adaptation': adapted_results,
            'task_optimization': optimized_results,
            'final_output': optimized_results['optimized_output']
        }


class ModularNeocortex(nn.Module):
    """
    模块化新皮层架构
    
    支持动态模块加载和配置的灵活架构
    """
    
    def __init__(self, config: NeocortexConfig):
        super().__init__()
        self.config = config
        
        # 模块注册表
        self.module_registry = self._build_module_registry()
        
        # 动态模块加载器
        self.dynamic_loader = DynamicModuleLoader()
        
        # 模块编排器
        self.module_orchestrator = ModuleOrchestrator(config)
        
    def _build_module_registry(self) -> Dict[str, nn.Module]:
        """构建模块注册表"""
        registry = {}
        
        # 注册可用的模块
        if self.config.architecture_type == ArchitectureType.TONN:
            registry['hierarchical'] = ProcessingHierarchy(create_visual_hierarchy())
            registry['prediction'] = create_prediction_module(self.config.input_dim)
            registry['attention'] = create_attention_module(self.config.hidden_dim)
            registry['abstraction'] = create_abstraction_engine(self.config.hidden_dim)
            registry['consolidation'] = create_consolidation_engine(self.config.hidden_dim)
            registry['decision'] = create_decision_module(self.config.hidden_dim)
        
        return registry
    
    def configure_modules(self, active_modules: List[str]):
        """配置活跃模块"""
        self.active_modules = active_modules
        self.loaded_modules = self.dynamic_loader.load_modules(
            self.module_registry, active_modules
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """模块化新皮层前向传播"""
        
        if not hasattr(self, 'loaded_modules'):
            self.configure_modules(['hierarchical', 'attention', 'decision'])
        
        # 模块编排执行
        results = self.module_orchestrator(self.loaded_modules, inputs)
        
        return results


# 辅助类

class TaskSpecificAdapter(nn.Module):
    """任务特定适配器"""
    
    def __init__(self, config: NeocortexConfig):
        super().__init__()
        self.config = config
        
        # 任务特征提取器
        self.task_feature_extractor = nn.Sequential(
            nn.Linear(config.output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
    def forward(self, base_results: Dict[str, Any], 
                task_context: Optional[Dict] = None) -> Dict[str, Any]:
        """任务适配"""
        
        base_output = base_results['final_output']
        
        # 提取任务相关特征
        task_features = self.task_feature_extractor(base_output)
        
        # 结合任务上下文
        if task_context is not None:
            task_context_tensor = torch.tensor([
                task_context.get('difficulty', 0.5),
                task_context.get('urgency', 0.5),
                task_context.get('complexity', 0.5)
            ]).unsqueeze(0).expand(base_output.size(0), -1)
            
            task_features = torch.cat([task_features, task_context_tensor], dim=-1)
        
        return {
            'adapted_features': task_features,
            'task_context': task_context
        }


class TaskOptimizer(nn.Module):
    """任务优化器"""
    
    def __init__(self, config: NeocortexConfig):
        super().__init__()
        self.config = config
        
        # 优化策略学习器
        self.optimization_strategy = nn.Sequential(
            nn.Linear(config.output_dim + 3, config.hidden_dim),  # +3 for context
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.output_dim),
            nn.Tanh()
        )
        
    def forward(self, adapted_results: Dict[str, Any]) -> Dict[str, Any]:
        """任务优化"""
        
        adapted_features = adapted_results['adapted_features']
        
        # 学习最优特征变换
        optimized_features = self.optimization_strategy(adapted_features)
        
        return {
            'optimized_output': optimized_features,
            'optimization_applied': True
        }


class DynamicModuleLoader:
    """动态模块加载器"""
    
    def load_modules(self, registry: Dict[str, nn.Module], 
                    active_modules: List[str]) -> Dict[str, nn.Module]:
        """动态加载模块"""
        loaded = {}
        for module_name in active_modules:
            if module_name in registry:
                loaded[module_name] = registry[module_name]
        return loaded


class ModuleOrchestrator:
    """模块编排器"""
    
    def __init__(self, config: NeocortexConfig):
        self.config = config
        
    def __call__(self, modules: Dict[str, nn.Module], 
                inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """编排模块执行"""
        
        # 简化的编排逻辑
        current_input = list(inputs.values())[0]
        
        results = {}
        for name, module in modules.items():
            if name == 'hierarchical' and hasattr(module, 'forward'):
                output = module(current_input)
                current_input = output['final_output']
            elif hasattr(module, 'forward'):
                output = module(current_input)
                current_input = output.get('features', current_input)
            
            results[name] = output
        
        return {
            'module_results': results,
            'final_output': current_input
        }


# 工厂函数

def create_neocortex_simulator(config: Optional[NeocortexConfig] = None) -> NeocortexSimulator:
    """创建新皮层模拟器"""
    if config is None:
        config = NeocortexConfig()
    return NeocortexSimulator(config)


def create_tonn(config: Optional[NeocortexConfig] = None) -> TONN:
    """创建TONN"""
    if config is None:
        config = NeocortexConfig(architecture_type=ArchitectureType.TONN)
    return TONN(config)


def create_modular_neocortex(config: Optional[NeocortexConfig] = None) -> ModularNeocortex:
    """创建模块化新皮层"""
    if config is None:
        config = NeocortexConfig(architecture_type=ArchitectureType.MODULAR)
    return ModularNeocortex(config)


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建配置
    config = NeocortexConfig(
        architecture_type=ArchitectureType.TONN,
        input_dim=512,
        hidden_dim=256,
        output_dim=128,
        prediction_enabled=True,
        attention_enabled=True,
        abstraction_enabled=True,
        consolidation_enabled=True,
        decision_enabled=True
    )
    
    # 创建新皮层模拟器
    neocortex = create_neocortex_simulator(config).to(device)
    
    # 创建测试输入
    test_inputs = {
        'visual': torch.randn(2, 512).to(device),
        'multimodal': torch.randn(2, 512).to(device)
    }
    
    # 前向传播
    results = neocortex(test_inputs)
    
    print(f"新皮层模拟器测试:")
    print(f"输入形状: {test_inputs['visual'].shape}")
    print(f"总处理阶段: {len(results['stage_outputs'])}")
    print(f"最终输出形状: {results['final_output'].shape}")
    print(f"架构类型: {results['summary']['architecture_type']}")
    print(f"处理效率: {results['summary']['processing_efficiency']:.3f}")
    
    # 显示各阶段信息
    for stage, info in results['stage_info'].items():
        print(f"{stage.value}: {info}")
    
    # 性能指标
    perf_metrics = results['performance_metrics']
    print(f"\n性能指标:")
    print(f"处理效率: {perf_metrics['processing_efficiency']:.3f}")
    print(f"输出质量: {perf_metrics['output_quality']:.3f}")
    print(f"整体效率: {perf_metrics['overall_efficiency']:.3f}")
