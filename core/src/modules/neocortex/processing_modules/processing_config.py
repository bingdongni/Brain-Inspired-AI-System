"""
处理模块配置
=============

定义新皮层专业处理模块的配置参数和类型定义。
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List


class ModuleType(Enum):
    """模块类型"""
    PREDICTION = "prediction"      # 预测模块
    ATTENTION = "attention"        # 注意模块
    DECISION = "decision"          # 决策模块
    CROSSMODAL = "crossmodal"      # 跨模态模块


class ProcessingMode(Enum):
    """处理模式"""
    OFFLINE = "offline"           # 离线处理
    ONLINE = "online"             # 在线处理
    ADAPTIVE = "adaptive"         # 自适应处理
    REINFORCEMENT = "reinforcement" # 强化学习模式


class PredictionType(Enum):
    """预测类型"""
    HIERARCHICAL = "hierarchical"     # 层级预测
    TEMPORAL = "temporal"            # 时间序列预测
    SEMANTIC = "semantic"            # 语义预测
    CAUSAL = "causal"                # 因果预测


class AttentionType(Enum):
    """注意力类型"""
    SPATIAL = "spatial"              # 空间注意力
    FEATURE = "feature"              # 特征注意力
    TEMPORAL = "temporal"            # 时间注意力
    TASK = "task"                   # 任务注意力
    COMBINED = "combined"            # 组合注意力


class DecisionMode(Enum):
    """决策模式"""
    PROBABILISTIC = "probabilistic"  # 概率性决策
    NEURAL = "neural"               # 神经网络决策
    COGNITIVE = "cognitive"         # 认知决策
    REINFORCEMENT = "reinforcement"  # 强化学习决策


@dataclass
class ProcessingConfig:
    """通用处理配置"""
    module_type: ModuleType
    feature_dim: int
    processing_mode: ProcessingMode = ProcessingMode.OFFLINE
    
    # 性能配置
    use_gpu: bool = True
    batch_size: int = 32
    max_sequence_length: int = 512
    
    # 学习配置
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1
    
    # 监控配置
    enable_logging: bool = True
    enable_profiling: bool = False
    
    # 缓存配置
    use_cache: bool = True
    cache_size: int = 1000


@dataclass
class PredictionConfig(ProcessingConfig):
    """预测模块配置"""
    prediction_type: PredictionType = PredictionType.HIERARCHICAL
    
    # 预测参数
    prediction_horizon: int = 10       # 预测步长
    hierarchical_levels: int = 3       # 层级数量
    temporal_window: int = 5          # 时间窗口
    
    # 预测质量
    confidence_threshold: float = 0.7  # 置信度阈值
    uncertainty_estimation: bool = True
    
    # 自适应参数
    adaptation_rate: float = 0.1       # 适应速率
    forgetting_factor: float = 0.95    # 遗忘因子
    
    # 记忆配置
    prediction_memory_size: int = 500   # 预测记忆大小
    template_matching: bool = True


@dataclass
class AttentionConfig(ProcessingConfig):
    """注意模块配置"""
    attention_type: AttentionType = AttentionType.COMBINED
    
    # 注意力参数
    attention_span: int = 7            # 注意力跨度
    selectivity_threshold: float = 0.5  # 选择性阈值
    competition_factor: float = 0.8    # 竞争因子
    
    # 多层次注意
    spatial_attention: bool = True     # 空间注意力
    feature_attention: bool = True     # 特征注意力
    temporal_attention: bool = True    # 时间注意力
    
    # 注意力控制
    top_k_attention: int = 3           # Top-K注意力
    attention_sparsity: float = 0.1    # 注意力稀疏性
    attention_decay: float = 0.9       # 注意力衰减
    
    # 任务相关
    task_relevance_weight: float = 0.7  # 任务相关性权重
    context_integration: bool = True


@dataclass
class DecisionConfig(ProcessingConfig):
    """决策模块配置"""
    decision_mode: DecisionMode = DecisionMode.PROBABILISTIC
    
    # 决策参数
    decision_threshold: float = 0.6     # 决策阈值
    confidence_threshold: float = 0.7   # 置信度阈值
    
    # 群体动力学参数
    neural_noise_level: float = 0.1     # 神经噪音水平
    decision_dynamics: str = "attractor"  # 决策动力学类型
    
    # 学习参数
    exploration_rate: float = 0.1       # 探索率
    exploitation_weight: float = 0.7    # 利用权重
    
    # 记忆和经验
    experience_replay: bool = True      # 经验回放
    memory_capacity: int = 1000         # 记忆容量
    learning_from_feedback: bool = True


@dataclass
class CrossModalConfig(ProcessingConfig):
    """跨模态模块配置"""
    
    # 模态配置
    input_modalities: List[str] = None   # 输入模态
    target_modalities: List[str] = None  # 目标模态
    
    # 整合参数
    fusion_method: str = "attention"     # 融合方法
    alignment_method: str = "attention"   # 对齐方法
    
    # 跨模态映射
    mapping_layers: int = 2             # 映射层数
    shared_representation: bool = True  # 共享表示
    
    # 概念形成
    concept_detection: bool = True      # 概念检测
    concept_formation: bool = True      # 概念形成
    concept_abstraction: bool = True    # 概念抽象
    
    # 同步和时序
    temporal_synchronization: bool = True
    phase_alignment: bool = True
    
    # 对齐质量
    alignment_threshold: float = 0.7     # 对齐阈值
    consistency_weight: float = 0.8      # 一致性权重


class ModuleMetadata:
    """模块元数据"""
    
    def __init__(self, name: str, version: str, description: str):
        self.name = name
        self.version = version
        self.description = description
        self.capabilities = []
        self.requirements = []
        self.dependencies = []
    
    def add_capability(self, capability: str):
        """添加能力"""
        self.capabilities.append(capability)
    
    def add_requirement(self, requirement: str):
        """添加需求"""
        self.requirements.append(requirement)
    
    def add_dependency(self, dependency: str):
        """添加依赖"""
        self.dependencies.append(dependency)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'capabilities': self.capabilities,
            'requirements': self.requirements,
            'dependencies': self.dependencies
        }


class ProcessingStats:
    """处理统计"""
    
    def __init__(self):
        self.total_processed = 0
        self.successful_predictions = 0
        self.attention_activations = 0
        self.decisions_made = 0
        self.crossmodal_integrations = 0
        
        self.processing_times = []
        self.accuracy_scores = []
        self.confidence_scores = []
        self.energy_consumption = 0.0
    
    def update_prediction_stats(self, success: bool, processing_time: float, 
                               accuracy: float, confidence: float):
        """更新预测统计"""
        self.total_processed += 1
        if success:
            self.successful_predictions += 1
        self.processing_times.append(processing_time)
        self.accuracy_scores.append(accuracy)
        self.confidence_scores.append(confidence)
    
    def update_attention_stats(self, activations: int, processing_time: float):
        """更新注意统计"""
        self.attention_activations += activations
        self.processing_times.append(processing_time)
    
    def update_decision_stats(self, decisions: int, processing_time: float):
        """更新决策统计"""
        self.decisions_made += decisions
        self.processing_times.append(processing_time)
    
    def update_crossmodal_stats(self, integrations: int, processing_time: float):
        """更新跨模态统计"""
        self.crossmodal_integrations += integrations
        self.processing_times.append(processing_time)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取统计摘要"""
        return {
            'total_processed': self.total_processed,
            'success_rate': self.successful_predictions / max(self.total_processed, 1),
            'avg_processing_time': sum(self.processing_times) / max(len(self.processing_times), 1),
            'avg_accuracy': sum(self.accuracy_scores) / max(len(self.accuracy_scores), 1),
            'avg_confidence': sum(self.confidence_scores) / max(len(self.confidence_scores), 1),
            'attention_activations': self.attention_activations,
            'decisions_made': self.decisions_made,
            'crossmodal_integrations': self.crossmodal_integrations,
            'energy_consumption': self.energy_consumption
        }