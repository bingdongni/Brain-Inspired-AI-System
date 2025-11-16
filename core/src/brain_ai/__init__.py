#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brain-Inspired AI Framework
===========================

基于生物大脑启发的深度学习框架，包含海马体和新皮层的模拟实现。

主要特性:
- 海马体记忆系统模拟
- 新皮层层次化处理
- 动态路由机制
- 持续学习能力
- 模块化架构设计

安装和使用:
    pip install brain-inspired-ai
    
    # 快速开始
    import brain_ai
    from brain_ai import Hippocampus, Neocortex
    
    # 创建海马体实例
    hippocampus = Hippocampus()
    memory = hippocampus.store("key", "value")
    
    # 创建新皮层实例
    neocortex = Neocortex()
    result = neocortex.process(input_data)

主要模块:
    - brain_ai.hippocampus: 海马体记忆系统
    - brain_ai.neocortex: 新皮层处理架构
    - brain_ai.core: 核心抽象和接口
    - brain_ai.modules: 各种功能模块
    - brain_ai.cli: 命令行接口
    - brain_ai.utils: 工具函数

API文档: https://brain-ai.readthedocs.io/
GitHub: https://github.com/brain-ai/brain-inspired-ai
"""

__version__ = "1.0.0"
__author__ = "Brain-Inspired AI Team"
__email__ = "team@brain-ai.org"
__license__ = "MIT"
__url__ = "https://github.com/brain-ai/brain-inspired-ai"

# 核心导入
try:
    from .core import (
        BrainSystem,
        BaseModule,
        Architecture,
        HierarchicalLayer,
        ProcessingModule
    )
except ImportError as e:
    print(f"Warning: Could not import core modules: {e}")

# 海马体模块
try:
    from .hippocampus import (
        HippocampusSimulator,
        EpisodicMemory,
        FastLearning,
        PatternSeparation
    )
except ImportError as e:
    print(f"Warning: Could not import hippocampus modules: {e}")

# 新皮层模块
try:
    from .neocortex import (
        NeocortexArchitecture,
        HierarchicalProcessor,
        AttentionModule,
        DecisionModule
    )
except ImportError as e:
    print(f"Warning: Could not import neocortex modules: {e}")

# 持续学习模块（从workspace目录导入）
try:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from elastic_weight_consolidation import (
        ElasticWeightConsolidation,
        FisherMatrix,
        EWCTrainer,
        IntelligentProtection
    )
    from generative_replay import (
        GenerativeReplay,
        GenerativeReplayTrainer,
        ExperienceReplay,
        BilateralConsolidation
    )
    from dynamic_expansion import (
        DynamicExpansion,
        ProgressiveNeuralNetwork,
        DynamicCapacityGrowth,
        NewTaskLearning
    )
    from knowledge_transfer import (
        KnowledgeTransfer,
        KnowledgeDistillation,
        LearningWithoutForgetting,
        MetaLearning
    )
except ImportError as e:
    print(f"Warning: Could not import lifelong learning modules: {e}")

# 动态路由模块
try:
    from modules.dynamic_routing import (
        DynamicRoutingSystem,
        RealTimeRoutingController,
        ActorCriticRouter,
        DynamicWeightRouter,
        NeuralInspiredRouter
    )
except ImportError as e:
    print(f"Warning: Could not import dynamic routing modules: {e}")

# 海马体模块
try:
    from modules.hippocampus import (
        HippocampalSimulator,
        TransformerMemoryEncoder,
        DifferentiableMemoryDictionary,
        PatternSeparationNetwork,
        EpisodicLearningSystem,
        EpisodicMemorySystem
    )
except ImportError as e:
    print(f"Warning: Could not import hippocampus modules: {e}")

# 记忆接口模块
try:
    from memory_interface.memory_interface_core import (
        MemoryInterface,
        MemoryInterfaceCore,
        MemoryInterfaceExample
    )
    from memory_interface.attention_mechanism.attention_controller import (
        AttentionController,
        AttentionMemory,
        AttentionReader,
        AttentionWriter
    )
    from memory_interface.communication.communication_controller import (
        CommunicationController,
        HippocampusInterface,
        NeocortexInterface,
        MessageTypes,
        ProtocolHandler
    )
    from memory_interface.consolidation.consolidation_engine import (
        ConsolidationEngine,
        ConsolidationController
    )
    from memory_interface.control_flow.flow_controller import (
        FlowController,
        LoadBalancer,
        PriorityManager
    )
    from memory_interface.integration.integrated_memory_system import (
        IntegratedMemorySystem,
        MemoryIntegrator
    )
except ImportError as e:
    print(f"Warning: Could not import memory interface modules: {e}")

# 新皮层模块
try:
    from modules.neocortex.neocortex_architecture import (
        NeocortexArchitecture
    )
    from modules.neocortex.sparse_activation import (
        SparseActivation
    )
    from modules.neocortex.abstraction.abstraction_core import (
        AbstractionCore
    )
    from modules.neocortex.hierarchical_layers.hierarchical_layer import (
        HierarchicalLayer,
        ProcessingHierarchy,
        AttentionModulation,
        PredictiveCoding
    )
    from modules.neocortex.processing_modules.decision_module import (
        DecisionModule,
        AttentionModule,
        CrossModalModule,
        PredictionModule
    )
except ImportError as e:
    print(f"Warning: Could not import neocortex modules: {e}")

# 工具模块
try:
    from .utils import (
        ConfigManager,
        Logger,
        MetricsCollector,
        DataProcessor
    )
except ImportError as e:
    print(f"Warning: Could not import utils modules: {e}")

__all__ = [
    # 核心模块
    "BrainSystem",
    "BaseModule", 
    "Architecture",
    "HierarchicalLayer",
    "ProcessingModule",
    
    # 海马体模块
    "HippocampusSimulator",
    "EpisodicMemory", 
    "FastLearning",
    "PatternSeparation",
    
    # 新皮层模块
    "NeocortexArchitecture",
    "HierarchicalProcessor",
    "AttentionModule",
    "DecisionModule",
    
    # 持续学习模块
    "ElasticWeightConsolidation",
    "GenerativeReplay", 
    "DynamicExpansion",
    "KnowledgeTransfer",
    
    # 动态路由模块
    "DynamicRoutingController",
    "AdaptiveAllocation",
    "EfficiencyOptimization", 
    "ReinforcementRouting",
    
    # 记忆接口模块
    "MemoryInterface",
    "AttentionMechanism",
    "CommunicationController",
    "ConsolidationEngine",
    
    # 工具模块
    "ConfigManager",
    "Logger",
    "MetricsCollector",
    "DataProcessor"
]