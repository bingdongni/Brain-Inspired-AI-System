#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
新皮层模拟架构
=============

基于大脑新皮层结构设计的层次化处理架构，实现高级认知功能模拟。

主要组件:
- 层次化处理层
- 稀疏激活机制
- 注意力调制
- 预测编码
- 决策模块
- 交叉模态整合

理论基础:
- 大脑新皮层的6层结构
- 层次化抽象机制
- 稀疏表示学习
- 预测性编码理论
- 注意力选择机制
"""

try:
    from .neocortex_architecture import (
        NeocortexArchitecture,
        create_neocortex_model,
        get_neocortex_config
    )
except ImportError as e:
    print(f"Warning: Could not import neocortex_architecture: {e}")

try:
    from .sparse_activation import (
        SparseActivation,
        SparsityController,
        AdaptiveSparsity,
        NeuralCoding
    )
except ImportError as e:
    print(f"Warning: Could not import sparse_activation: {e}")

# 抽象层次
try:
    from .abstraction import (
        AbstractionCore,
        AbstractionConfig,
        HierarchicalAbstraction,
        ConceptualRepresentation
    )
except ImportError as e:
    print(f"Warning: Could not import abstraction: {e}")

# 层次化层
try:
    from .hierarchical_layers import (
        HierarchicalLayer,
        ProcessingHierarchy,
        AttentionModulation,
        PredictiveCoding,
        LayerConfig,
        VisualHierarchy,
        AuditoryHierarchy
    )
except ImportError as e:
    print(f"Warning: Could not import hierarchical_layers: {e}")

# 处理模块
try:
    from .processing_modules import (
        DecisionModule,
        AttentionModule,
        CrossModalModule,
        PredictionModule,
        ProcessingConfig,
        ModuleFactory
    )
except ImportError as e:
    print(f"Warning: Could not import processing_modules: {e}")

__version__ = "1.0.0"
__author__ = "Brain-Inspired AI Research Team"

__all__ = [
    # 核心架构
    "NeocortexArchitecture",
    "create_neocortex_model",
    "get_neocortex_config",
    
    # 稀疏激活
    "SparseActivation",
    "SparsityController", 
    "AdaptiveSparsity",
    "NeuralCoding",
    
    # 抽象层次
    "AbstractionCore",
    "AbstractionConfig",
    "HierarchicalAbstraction",
    "ConceptualRepresentation",
    
    # 层次化层
    "HierarchicalLayer",
    "ProcessingHierarchy",
    "AttentionModulation",
    "PredictiveCoding",
    "LayerConfig",
    "VisualHierarchy",
    "AuditoryHierarchy",
    
    # 处理模块
    "DecisionModule",
    "AttentionModule",
    "CrossModalModule", 
    "PredictionModule",
    "ProcessingConfig",
    "ModuleFactory"
]

# 默认配置
DEFAULT_CONFIGS = {
    "small": {
        "hidden_dim": 256,
        "num_layers": 6,
        "num_attention_heads": 4,
        "sparsity_ratio": 0.1,
        "abstraction_levels": 3
    },
    "base": {
        "hidden_dim": 512,
        "num_layers": 12,
        "num_attention_heads": 8,
        "sparsity_ratio": 0.05,
        "abstraction_levels": 4
    },
    "large": {
        "hidden_dim": 1024,
        "num_layers": 24,
        "num_attention_heads": 16,
        "sparsity_ratio": 0.02,
        "abstraction_levels": 5
    }
}

def create_neocortex_with_config(config_name: str = "base", custom_params: dict = None):
    """根据配置名称创建新皮层模型"""
    
    if config_name not in DEFAULT_CONFIGS:
        raise ValueError(f"不支持的配置: {config_name}. 支持的配置: {list(DEFAULT_CONFIGS.keys())}")
    
    config = DEFAULT_CONFIGS[config_name].copy()
    if custom_params:
        config.update(custom_params)
    
    return NeocortexArchitecture(config=config)

def get_neocortex_module_info():
    """获取新皮层模块信息"""
    return {
        "name": "新皮层模拟架构",
        "version": __version__,
        "description": "基于大脑新皮层结构的层次化处理架构",
        "components": {
            "architecture": "新皮层总体架构",
            "sparse_activation": "稀疏激活机制",
            "abstraction": "层次化抽象机制",
            "hierarchical_layers": "层次化处理层",
            "processing_modules": "功能处理模块"
        },
        "features": [
            "6层新皮层结构模拟",
            "层次化信息抽象",
            "稀疏表示学习",
            "注意力调制机制",
            "预测编码",
            "多模态整合",
            "决策制定"
        ]
    }

def example_usage():
    """使用示例"""
    print("=== 新皮层架构使用示例 ===")
    
    # 创建新皮层模型
    model = create_neocortex_with_config("base")
    
    # 模拟输入数据
    input_data = {
        "visual": "图像特征",
        "audio": "音频特征", 
        "text": "文本特征"
    }
    
    # 处理输入
    result = model.process(input_data)
    
    print(f"处理结果: {result}")
    print("示例运行完成!")

if __name__ == "__main__":
    example_usage()