"""
新皮层模拟器架构
================

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
from ..hierarchical_layers import (
    ProcessingHierarchy, LayerConfig, LayerType, ProcessingMode,
    create_visual_hierarchy, create_auditory_hierarchy,
    VisualProcessingStream, AuditoryProcessingStream
)

from ..processing_modules import (
    PredictionModule, AttentionModule, DecisionModule, CrossModalModule,
    PredictionConfig, AttentionConfig, DecisionConfig, CrossModalConfig,
    create_prediction_module, create_attention_module, 
    create_decision_module, create_crossmodal_module
)

from ..abstraction import (
    AbstractionEngine, ConceptUnit, SemanticAbstraction,
    AbstractionConfig, ConceptType, AbstractionLevel,
    create_abstraction_engine
)

from ..sparse_activation import (
    ConsolidationEngine, SparseActivation, WeightConsolidation, EngramCell,
    ConsolidationConfig, EngramConfig, CellType, MemoryState,
    create_consolidation_engine
)


class ArchitectureType(Enum):
    """架构类型"""
    TONN = "tonn"                    # 任务优化神经网络
    MODULAR = "modular"              # 模块化架构
    HIERARCHICAL = "hierarchical"    # 层次架构
    HYBRID = "hybrid"               # 混合架构


@dataclass
class NeocortexConfig:
    """新皮层配置"""
    
    # 基础参数
    architecture_type: ArchitectureType = ArchitectureType.TONN
    input_dim: int = 512
    hidden_dim: int = 256
    output_dim: int = 128
    
    # 模块启用标志
    prediction_enabled: bool = True
    attention_enabled: bool = True
    abstraction_enabled: bool = True
    consolidation_enabled: bool = True
    decision_enabled: bool = True
    crossmodal_enabled: bool = True
    
    # 专用参数
    num_concepts: int = 100
    num_engram_cells: int = 200
    hierarchical_levels: int = 4
    
    # 视觉和听觉处理
    visual_processing: bool = True
    auditory_processing: bool = True
    
    # 学习参数
    learning_rate: float = 1e-4
    consolidation_rate: float = 0.01
    memory_capacity: int = 1000
    
    # 性能参数
    batch_size: int = 32
    max_sequence_length: int = 512
    enable_gpu: bool = True
    
    # 调试和监控
    enable_logging: bool = True
    enable_profiling: bool = False
    performance_monitoring: bool = True
    
    def get_device(self) -> torch.device:
        """获取计算设备"""
        if self.enable_gpu and torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")


class NeocortexSimulator(nn.Module):
    """
    新皮层模拟器
    
    整合所有新皮层核心组件的综合模拟器。
    """
    
    def __init__(self, config: NeocortexConfig):
        super().__init__()
        self.config = config
        self.device = config.get_device()
        
        # 初始化各模块
        self._initialize_modules()
        
        # 全局记忆系统
        self.global_memory = GlobalMemorySystem(config)
        
        # 性能监控器
        if config.performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
        
        # 统计信息
        self.processing_stats = {
            'total_batches_processed': 0,
            'avg_processing_time': 0.0,
            'module_utilization': {},
            'memory_usage': {},
            'accuracy_metrics': {}
        }
    
    def _initialize_modules(self):
        """初始化所有模块"""
        
        # 层次化处理
        if self.config.visual_processing:
            self.visual_processor = VisualProcessingStream(
                input_channels=3, feature_dim=self.config.hidden_dim
            )
        
        if self.config.auditory_processing:
            self.auditory_processor = AuditoryProcessingStream(
                input_channels=1, feature_dim=self.config.hidden_dim
            )
        
        # 专业处理模块
        if self.config.prediction_enabled:
            self.prediction_module = create_prediction_module(
                feature_dim=self.config.hidden_dim,
                prediction_type="hierarchical",
                prediction_horizon=10
            )
        
        if self.config.attention_enabled:
            self.attention_module = create_attention_module(
                feature_dim=self.config.hidden_dim,
                attention_type="combined",
                selectivity_threshold=0.5
            )
        
        if self.config.decision_enabled:
            self.decision_module = create_decision_module(
                input_dim=self.config.hidden_dim,
                output_dim=self.config.output_dim,
                decision_mode="probabilistic"
            )
        
        if self.config.abstraction_enabled:
            self.abstraction_engine = create_abstraction_engine(
                feature_dim=self.config.hidden_dim,
                abstraction_type="hierarchical",
                max_concepts=self.config.num_concepts
            )
        
        if self.config.consolidation_enabled:
            self.consolidation_engine = create_consolidation_engine(
                feature_dim=self.config.hidden_dim,
                num_engram_cells=self.config.num_engram_cells,
                sparsity_level=0.95
            )
        
        if self.config.crossmodal_enabled:
            self.crossmodal_module = create_crossmodal_module(
                input_dims=[self.config.hidden_dim, self.config.hidden_dim],
                target_dim=self.config.hidden_dim,
                fusion_method="attention"
            )
        
        # 层级处理层次
        self.processing_hierarchy = self._build_processing_hierarchy()
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
            nn.Tanh()
        )
    
    def _build_processing_hierarchy(self) -> Optional[ProcessingHierarchy]:
        """构建处理层次"""
        if not any([self.config.visual_processing, self.config.auditory_processing]):
            return None
        
        # 创建层级配置
        layer_configs = []
        
        # 添加视觉层级（如果启用）
        if self.config.visual_processing:
            visual_configs = create_visual_hierarchy(
                input_channels=3, feature_dim=self.config.hidden_dim
            )
            layer_configs.extend(visual_configs)
        
        # 添加听觉层级（如果启用）
        if self.config.auditory_processing:
            auditory_configs = create_auditory_hierarchy(
                input_channels=1, feature_dim=self.config.hidden_dim
            )
            layer_configs.extend(auditory_configs)
        
        if layer_configs:
            return ProcessingHierarchy(layer_configs)
        return None
    
    def forward(self, inputs: Dict[str, torch.Tensor],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        新皮层模拟器前向传播
        
        Args:
            inputs: 输入字典，包含视觉、听觉、多模态等输入
            context: 上下文信息
            
        Returns:
            dict: 处理结果
        """
        start_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        end_time = torch.cuda.Event(enable_timing=True) if self.device.type == 'cuda' else None
        
        if start_time:
            start_time.record()
        
        batch_size = inputs[list(inputs.keys())[0]].shape[0] if inputs else 1
        
        # 存储各阶段输出
        stage_outputs = {}
        stage_info = {}
        
        # 1. 感知层处理
        perceptual_output = self._process_perceptual_layer(inputs)
        stage_outputs['perceptual'] = perceptual_output
        
        # 2. 层次化处理
        if self.processing_hierarchy:
            hierarchy_result = self._process_hierarchy(perceptual_output)
            stage_outputs['hierarchy'] = hierarchy_result
            perceptual_features = hierarchy_result['final_output']
        else:
            perceptual_features = perceptual_output['integrated_features']
        
        # 3. 抽象层处理
        if self.config.abstraction_enabled and hasattr(self, 'abstraction_engine'):
            abstraction_result = self._process_abstraction(perceptual_features)
            stage_outputs['abstraction'] = abstraction_result
            abstract_features = abstraction_result['concepts']
        else:
            abstract_features = perceptual_features
            abstraction_result = {}
        
        # 4. 预测处理
        if self.config.prediction_enabled and hasattr(self, 'prediction_module'):
            prediction_result = self._process_prediction(perceptual_features, context)
            stage_outputs['prediction'] = prediction_result
            predicted_features = prediction_result.get('predictions', perceptual_features)
        else:
            predicted_features = perceptual_features
            prediction_result = {}
        
        # 5. 注意处理
        if self.config.attention_enabled and hasattr(self, 'attention_module'):
            attention_result = self._process_attention(perceptual_features, context)
            stage_outputs['attention'] = attention_result
            attended_features = attention_result.get('attended_features', predicted_features)
        else:
            attended_features = predicted_features
            attention_result = {}
        
        # 6. 巩固处理
        if self.config.consolidation_enabled and hasattr(self, 'consolidation_engine'):
            consolidation_result = self._process_consolidation(attended_features)
            stage_outputs['consolidation'] = consolidation_result
            consolidated_features = consolidation_result.get('consolidated_features', attended_features)
        else:
            consolidated_features = attended_features
            consolidation_result = {}
        
        # 7. 决策处理
        if self.config.decision_enabled and hasattr(self, 'decision_module'):
            decision_result = self._process_decision(consolidated_features)
            stage_outputs['decision'] = decision_result
            decision_output = decision_result.get('decisions', consolidated_features)
        else:
            decision_output = consolidated_features
            decision_result = {}
        
        # 8. 生成最终输出
        final_output = self.output_layer(consolidated_features)
        
        if end_time:
            end_time.record()
        
        if start_time and end_time:
            torch.cuda.synchronize()
            processing_time = start_time.elapsed_time(end_time)
            self._update_processing_time(processing_time)
        
        # 构建结果摘要
        summary = self._build_summary(stage_outputs, context)
        
        # 更新统计信息
        self._update_statistics(stage_outputs, batch_size)
        
        # 性能监控
        performance_metrics = {}
        if self.config.performance_monitoring and hasattr(self, 'performance_monitor'):
            performance_metrics = self.performance_monitor.record_batch(
                stage_outputs, processing_time if 'processing_time' in locals() else 0.0
            )
        
        return {
            'final_output': final_output,
            'stage_outputs': stage_outputs,
            'stage_info': stage_info,
            'summary': summary,
            'performance_metrics': performance_metrics,
            'processing_stats': self.processing_stats
        }
    
    def _process_perceptual_layer(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """处理感知层"""
        outputs = {}
        
        # 视觉处理
        if 'visual' in inputs and self.config.visual_processing:
            visual_input = inputs['visual']
            visual_output = self.visual_processor(visual_input)
            outputs['visual'] = visual_output
        
        # 听觉处理
        if 'auditory' in inputs and self.config.auditory_processing:
            auditory_input = inputs['auditory']
            auditory_output = self.auditory_processor(auditory_input)
            outputs['auditory'] = auditory_output
        
        # 多模态整合
        if len(outputs) > 1 and self.config.crossmodal_enabled:
            multimodal_input = [output['final_output'] for output in outputs.values()]
            crossmodal_output = self.crossmodal_module(multimodal_input)
            outputs['crossmodal'] = crossmodal_output
            integrated_features = crossmodal_output['integrated_features']
        elif outputs:
            integrated_features = list(outputs.values())[0]['final_output']
        else:
            integrated_features = inputs.get('multimodal', torch.randn(1, self.config.hidden_dim))
        
        return {
            'modal_outputs': outputs,
            'integrated_features': integrated_features
        }
    
    def _process_hierarchy(self, perceptual_output: Dict[str, Any]) -> Dict[str, Any]:
        """处理层次化层"""
        if not self.processing_hierarchy:
            return {'final_output': perceptual_output['integrated_features']}
        
        # 准备层次输入
        hierarchy_inputs = {}
        if 'visual' in perceptual_output['modal_outputs']:
            hierarchy_inputs['V1'] = perceptual_output['modal_outputs']['visual']['final_output']
        
        if 'auditory' in perceptual_output['modal_outputs']:
            hierarchy_inputs['IC'] = perceptual_output['modal_outputs']['auditory']['final_output']
        
        # 层次处理
        hierarchy_result = self.processing_hierarchy(
            perceptual_output['integrated_features'],
            hierarchy_inputs=hierarchy_inputs
        )
        
        return hierarchy_result
    
    def _process_abstraction(self, features: torch.Tensor) -> Dict[str, Any]:
        """处理抽象层"""
        abstraction_result = self.abstraction_engine(features)
        return abstraction_result
    
    def _process_prediction(self, features: torch.Tensor, 
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理预测层"""
        prediction_result = self.prediction_module(features, context=context)
        return prediction_result
    
    def _process_attention(self, features: torch.Tensor,
                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理注意层"""
        attention_result = self.attention_module(features, task_context=context)
        return attention_result
    
       def _process_consolidation(self, features: torch.Tensor) -> Dict[str, Any]:
        """处理巩固层"""
        consolidation_result = self.consolidation_engine(features)
        return consolidation_result
    
    def _process_decision(self, features: torch.Tensor) -> Dict[str, Any]:
        """处理决策层"""
        decision_result = self.decision_module(features)
        return decision_result
    
    def _build_summary(self, stage_outputs: Dict[str, Any], 
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """构建结果摘要"""
        summary = {
            'architecture_type': self.config.architecture_type.value,
            'num_stages': len(stage_outputs),
            'enabled_modules': [],
            'processing_efficiency': 0.0,
            'output_quality': 0.0
        }
        
        # 检查启用的模块
        for stage_name, stage_output in stage_outputs.items():
            if stage_name in ['prediction', 'attention', 'abstraction', 'consolidation', 'decision']:
                summary['enabled_modules'].append(stage_name)
        
        # 计算处理效率
        if stage_outputs:
            efficiency_scores = []
            for stage_output in stage_outputs.values():
                if isinstance(stage_output, dict):
                    # 提取效率相关指标
                    if 'prediction_confidence' in stage_output:
                        efficiency_scores.append(stage_output['prediction_confidence'].mean().item())
                    elif 'attention_quality' in stage_output:
                        efficiency_scores.append(stage_output['attention_quality'].mean().item())
                    elif 'abstraction_quality' in stage_output:
                        efficiency_scores.append(stage_output['abstraction_quality'])
                    elif 'sparse_efficiency' in stage_output:
                        efficiency_scores.append(stage_output['sparse_efficiency'].item())
            
            if efficiency_scores:
                summary['processing_efficiency'] = sum(efficiency_scores) / len(efficiency_scores)
        
        # 计算输出质量
        if 'abstraction' in stage_outputs:
            summary['output_quality'] = stage_outputs['abstraction'].get('abstraction_quality', 0.5)
        elif 'attention' in stage_outputs:
            summary['output_quality'] = stage_outputs['attention'].get('attention_quality', 0.5).mean().item()
        else:
            summary['output_quality'] = 0.6  # 默认质量
        
        return summary
    
    def _update_processing_time(self, processing_time: float):
        """更新处理时间统计"""
        self.processing_stats['total_batches_processed'] += 1
        
        # 计算平均处理时间
        current_avg = self.processing_stats['avg_processing_time']
        total_batches = self.processing_stats['total_batches_processed']
        
        self.processing_stats['avg_processing_time'] = (
            (current_avg * (total_batches - 1) + processing_time) / total_batches
        )
    
    def _update_statistics(self, stage_outputs: Dict[str, Any], batch_size: int):
        """更新统计信息"""
        # 更新模块利用率
        for stage_name in stage_outputs.keys():
            if stage_name not in self.processing_stats['module_utilization']:
                self.processing_stats['module_utilization'][stage_name] = 0
            
            self.processing_stats['module_utilization'][stage_name] += batch_size
        
        # 更新内存使用统计
        if self.config.consolidation_enabled and 'consolidation' in stage_outputs:
            consolidation_output = stage_outputs['consolidation']
            if 'memory_state' in consolidation_output:
                memory_state = consolidation_output['memory_state']
                self.processing_stats['memory_usage'] = {
                    'active_cells': memory_state.get('active_cells', 0),
                    'consolidated_cells': memory_state.get('consolidated_cells', 0),
                    'total_cells': memory_state.get('total_cells', 0)
                }
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'NeocortexSimulator',
            'architecture_type': self.config.architecture_type.value,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'enabled_modules': [attr for attr in dir(self) 
                              if not attr.startswith('_') and hasattr(getattr(self, attr), '__call__')],
            'device': str(self.device),
            'config': {
                'input_dim': self.config.input_dim,
                'hidden_dim': self.config.hidden_dim,
                'output_dim': self.config.output_dim,
                'num_concepts': self.config.num_concepts,
                'num_engram_cells': self.config.num_engram_cells
            }
        }
    
    def enter_sleep_mode(self):
        """进入睡眠巩固模式"""
        if self.config.consolidation_enabled:
            self.consolidation_engine.enter_sleep_mode()
            print("新皮层进入睡眠巩固模式")
    
    def exit_sleep_mode(self):
        """退出睡眠巩固模式"""
        if self.config.consolidation_enabled:
            self.consolidation_engine.exit_sleep_mode()
            print("新皮层退出睡眠巩固模式")


class GlobalMemorySystem(nn.Module):
    """全局记忆系统"""
    
    def __init__(self, config: NeocortexConfig):
        super().__init__()
        self.config = config
        
        # 短期记忆
        self.short_term_memory = nn.GRU(
            config.hidden_dim, config.hidden_dim, batch_first=True
        )
        
        # 长期记忆编码器
        self.long_term_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 记忆整合器
        self.memory_integrator = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
    
    def forward(self, current_features: torch.Tensor, 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """记忆系统前向传播"""
        batch_size = current_features.shape[0]
        
        # 短期记忆处理
        short_term_output, _ = self.short_term_memory(current_features.unsqueeze(1))
        short_term_features = short_term_output.squeeze(1)
        
        # 长期记忆编码
        long_term_encoded = self.long_term_encoder(current_features)
        
        # 记忆整合
        combined_memory = torch.cat([short_term_features, long_term_encoded], dim=1)
        integrated_memory = self.memory_integrator(combined_memory)
        
        return {
            'integrated_memory': integrated_memory,
            'short_term_memory': short_term_features,
            'long_term_encoded': long_term_encoded
        }


class PerformanceMonitor(nn.Module):
    """性能监控器"""
    
    def __init__(self):
        super().__init__()
        self.performance_history = []
        self.metrics_window = 100
    
    def record_batch(self, stage_outputs: Dict[str, Any], 
                    processing_time: float) -> Dict[str, float]:
        """记录批次性能"""
        metrics = {
            'processing_time': processing_time,
            'stages_active': len(stage_outputs),
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
        
        # 添加各阶段指标
        for stage_name, stage_output in stage_outputs.items():
            if isinstance(stage_output, dict):
                if stage_name == 'prediction' and 'confidence' in stage_output:
                    metrics[f'{stage_name}_confidence'] = stage_output['confidence'].mean().item()
                elif stage_name == 'attention' and 'attention_intensity' in stage_output:
                    metrics[f'{stage_name}_intensity'] = stage_output['attention_intensity'].mean().item()
                elif stage_name == 'abstraction' and 'abstraction_quality' in stage_output:
                    metrics[f'{stage_name}_quality'] = stage_output['abstraction_quality']
                elif stage_name == 'consolidation' and 'sparse_efficiency' in stage_output:
                    metrics[f'{stage_name}_efficiency'] = stage_output['sparse_efficiency'].item()
        
        # 维护历史记录
        self.performance_history.append(metrics)
        if len(self.performance_history) > self.metrics_window:
            self.performance_history.pop(0)
        
        return metrics


class TONN(NeocortexSimulator):
    """
    任务优化神经网络 (TONN)
    
    基于特定任务优化的新皮层架构变体。
    """
    
    def __init__(self, config: NeocortexConfig):
        config.architecture_type = ArchitectureType.TONN
        super().__init__(config)
        
        # 任务特定优化层
        self.task_optimization_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.output_dim),
            nn.Tanh()
        )
        
        # 任务特定记忆
        self.task_memory = nn.GRU(
            config.output_dim, config.output_dim, batch_first=True
        )
    
    def forward(self, inputs: Dict[str, torch.Tensor],
                task_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """TONN前向传播"""
        # 基础处理
        base_result = super().forward(inputs, task_context)
        
        # 任务特定优化
        optimized_output = self.task_optimization_layer(base_result['final_output'])
        
        # 任务记忆处理
        if task_context and 'task_sequence' in task_context:
            task_seq = task_context['task_sequence']
            task_memory_output, _ = self.task_memory(optimized_output.unsqueeze(1))
            final_output = task_memory_output.squeeze(1)
        else:
            final_output = optimized_output
        
        # 更新结果
        base_result['final_output'] = final_output
        base_result['tonn_optimized'] = True
        base_result['task_optimization_applied'] = True
        
        return base_result


class ModularNeocortex(NeocortexSimulator):
    """
    模块化新皮层
    
    可配置模块的新皮层架构。
    """
    
    def __init__(self, config: NeocortexConfig):
        config.architecture_type = ArchitectureType.MODULAR
        super().__init__(config)
        
        self.active_modules = set()
        self.module_configs = {}
    
    def configure_modules(self, module_names: List[str]):
        """配置活动模块"""
        self.active_modules = set(module_names)
        
        # 根据配置启用/禁用模块
        self._update_module_configuration()
    
    def _update_module_configuration(self):
        """更新模块配置"""
        # 根据active_modules更新各模块的启用状态
        for module_name in ['prediction', 'attention', 'abstraction', 'consolidation', 'decision']:
            if module_name in self.active_modules:
                setattr(self.config, f'{module_name}_enabled', True)
            else:
                setattr(self.config, f'{module_name}_enabled', False)
        
        # 重新初始化模块
        self._initialize_modules()
    
    def forward(self, inputs: Dict[str, torch.Tensor],
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """模块化新皮层前向传播"""
        # 根据活动模块执行相应的处理
        base_result = super().forward(inputs, context)
        
        # 添加模块活动信息
        base_result['active_modules'] = list(self.active_modules)
        base_result['modular_configuration'] = True
        
        return base_result


def create_neocortex_simulator(config: NeocortexConfig) -> NeocortexSimulator:
    """创建新皮层模拟器"""
    if config.architecture_type == ArchitectureType.TONN:
        return TONN(config)
    elif config.architecture_type == ArchitectureType.MODULAR:
        return ModularNeocortex(config)
    else:
        return NeocortexSimulator(config)