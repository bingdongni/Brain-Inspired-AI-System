"""
核心模块包
===========

这个包包含了大脑启发AI系统的核心组件，包括基础模块类、神经网络库、
训练框架以及系统架构设计。

主要模块:
- base_module: 基础模块抽象类
- brain_system: 大脑系统核心类  
- neural_network: 神经网络库
- training_framework: 训练框架
- architecture: 模块化架构设计
- interfaces: 系统接口定义

新皮层模拟器核心模块 (Neocortex Simulator Core):
- hierarchical_layers: 分层抽象机制 (V1→V2→V4→IT视觉通路, IC→MGB→AC听觉通路)
- processing_modules: 专业处理模块 (预测、注意、决策、跨模态处理)
- abstraction: 知识抽象算法 (概念形成、语义抽象、规则提取)
- sparse_activation: 稀疏激活和慢速权重巩固机制
- neocortex_architecture: 模块化神经网络架构 (TONN、ModularNeocortex)

作者: Brain-Inspired AI Team
版本: 1.0.0
日期: 2025-11-16
"""

from .base_module import BaseModule, ModuleState, ModuleType
from .brain_system import BrainSystem, BrainRegion
from .neural_network import NeuralLayer, NetworkArchitecture, ActivationFunction
from .training_framework import TrainingFramework, TrainingConfig, LossFunction
from .architecture import ModularArchitecture, ComponentRegistry, DependencyGraph
from .interfaces import IModule, INeuralComponent, ITrainingComponent

# 新皮层模拟器核心模块 - 从 modules/neocortex/ 目录导入
try:
    # 分层抽象机制
    from ..modules.neocortex.hierarchical_layers import (
        HierarchicalLayer, LayerConfig, LayerType, ProcessingMode,
        ProcessingHierarchy, VisualHierarchy, AuditoryHierarchy,
        AttentionModulation, PredictiveCoding
    )
    from ..modules.neocortex.hierarchical_layers.layer_config import (
        LayerType, ProcessingMode
    )
except ImportError:
    HierarchicalLayer = None
    LayerConfig = None
    LayerType = None
    ProcessingMode = None
    ProcessingHierarchy = None
    VisualHierarchy = None
    AuditoryHierarchy = None
    AttentionModulation = None
    PredictiveCoding = None

# 专业处理模块 - 使用可选导入处理torch依赖
try:
    from ..modules.neocortex.processing_modules import (
        PredictionModule, AttentionModule, DecisionModule, CrossModalModule,
        ProcessingConfig, module_factory
    )
    from ..modules.neocortex.processing_modules.processing_config import (
        PredictionType, AttentionType, DecisionMode, ProcessingConfig
    )
except ImportError:
    # 处理torch不可用的情况
    PredictionModule = None
    AttentionModule = None
    DecisionModule = None
    CrossModalModule = None
    ProcessingConfig = None
    module_factory = None
    PredictionType = None
    AttentionType = None
    DecisionMode = None

# 知识抽象算法
try:
    from ..modules.neocortex.abstraction import (
        AbstractionEngine, ConceptUnit, SemanticAbstraction
    )
    from ..modules.neocortex.abstraction.abstraction_config import (
        ConceptType, AbstractionLevel, ConceptConfig, AbstractionConfig
    )
except ImportError:
    AbstractionEngine = None
    ConceptUnit = None
    SemanticAbstraction = None
    ConceptType = None
    AbstractionLevel = None
    ConceptConfig = None
    AbstractionConfig = None

# 稀疏激活和权重巩固机制
try:
    from ..modules.neocortex.sparse_activation import (
        SparseActivation, WeightConsolidation, ConsolidationEngine, EngramCell,
        ConsolidationConfig, EngramConfig, CellType, MemoryState
    )
except ImportError:
    SparseActivation = None
    WeightConsolidation = None
    ConsolidationEngine = None
    EngramCell = None
    ConsolidationConfig = None
    EngramConfig = None
    CellType = None
    MemoryState = None

# 模块化神经网络架构
try:
    from ..modules.neocortex.neocortex_architecture import (
        NeocortexSimulator, TONN, ModularNeocortex, NeocortexConfig, ArchitectureType
    )
except ImportError:
    NeocortexSimulator = None
    TONN = None
    ModularNeocortex = None
    NeocortexConfig = None
    ArchitectureType = None

__version__ = "1.0.0"
__author__ = "Brain-Inspired AI Team"

# 包级别的公共接口
__all__ = [
    # 基础模块
    'BaseModule',
    'ModuleState', 
    'ModuleType',
    
    # 大脑系统
    'BrainSystem',
    'BrainRegion',
    'SynapticConnection',
    
    # 神经网络
    'NeuralLayer',
    'NetworkArchitecture', 
    'ActivationFunction',
    
    # 训练框架
    'TrainingFramework',
    'TrainingConfig',
    'LossFunction',
    
    # 架构设计
    'ModularArchitecture',
    'ComponentRegistry',
    'DependencyGraph',
    
    # 接口定义
    'IModule',
    'INeuralComponent', 
    'ITrainingComponent',
    
    # 新皮层模拟器核心模块
    # 分层抽象机制
    'HierarchicalLayer',
    'LayerConfig', 
    'LayerType',
    'ProcessingMode',
    'ProcessingHierarchy',
    'VisualHierarchy',
    'AuditoryHierarchy', 
    'AttentionModulation',
    'PredictiveCoding',
    
    # 专业处理模块
    'PredictionModule',
    'AttentionModule',
    'DecisionModule',
    'CrossModalModule',
    'ProcessingConfig',
    'PredictionType',
    'AttentionType', 
    'DecisionMode',
    
    # 知识抽象算法
    'AbstractionEngine',
    'ConceptUnit',
    'SemanticAbstraction',
    'ConceptType',
    'AbstractionLevel',
    'ConceptConfig',
    'AbstractionConfig',
    
    # 稀疏激活和权重巩固
    'SparseActivation',
    'WeightConsolidation',
    'ConsolidationEngine',
    'EngramCell',
    'ConsolidationConfig',
    'EngramConfig',
    'CellType',
    'MemoryState',
    
    # 模块化神经网络架构
    'NeocortexSimulator',
    'TONN',
    'ModularNeocortex',
    'NeocortexConfig',
    'ArchitectureType',
]

def get_version():
    """获取当前版本号"""
    return __version__

def get_system_info():
    """获取系统信息"""
    return {
        'version': __version__,
        'author': __author__,
        'modules': len(__all__),
        'components': [
            'BaseModule',
            'BrainSystem', 
            'NeuralNetwork',
            'TrainingFramework',
            'ModularArchitecture',
            'NeocortexSimulator',  # 新皮层模拟器
            'HierarchicalLayer',   # 分层抽象机制
            'AbstractionEngine',   # 知识抽象
            'SparseActivation'     # 稀疏激活
        ]
    }