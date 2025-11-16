"""
海马体模拟器核心模块
基于Science期刊最新研究的海马体记忆机制理论实现

主要组件:
- Transformer-based记忆编码器
- 可微分神经字典
- 模式分离机制
- 快速一次性学习功能
- 情景记忆存储和检索系统

基于理论研究报告:
- 多突触末梢(MSBs)的特异性增加机制
- 非同步激活的记忆编码
- 输入特异性增强和空间受限机制
- CA3-CA1通路的重构和连接重塑
"""

from .core.simulator import (
    HippocampalSimulator,
    create_hippocampus_simulator,
    get_hippocampus_config,
    quick_hippocampus_demo
)

from .encoders.transformer_encoder import (
    TransformerMemoryEncoder,
    EpisodicMemoryEncoder,
    MultiSynapticEngram,
    PositionalEncoding,
    AttentionMechanism,
    create_memory_encoder
)

from .memory_cell.differentiable_dict import (
    DifferentiableMemoryDictionary,
    SynapticStorage,
    DifferentiableMemoryKey,
    SynapticConsolidation,
    MemoryConsolidationScheduler
)

from .pattern_separation.mechanism import (
    PatternSeparationNetwork,
    CA3PatternSeparator,
    InputSpecificityEnhancer,
    SynapticRemodeling,
    HierarchicalPatternSeparation,
    SparseCodingLayer
)

from .learning.rapid_learning import (
    EpisodicLearningSystem,
    RapidEncodingUnit,
    SingleTrialLearner,
    FastAssociativeMemory
)

from .memory_system.episodic_storage import (
    EpisodicMemorySystem,
    TemporalContextEncoder,
    EpisodicMemoryCell,
    EpisodicMemory,
    HippocampalIndexing
)

__version__ = "1.0.0"
__author__ = "Brain-Inspired AI Research Team"

__all__ = [
    # 核心模拟器
    "HippocampalSimulator",
    "create_hippocampus_simulator",
    "get_hippocampus_config",
    "quick_hippocampus_demo",
    
    # 记忆编码器
    "TransformerMemoryEncoder",
    "EpisodicMemoryEncoder",
    "MultiSynapticEngram",
    "PositionalEncoding",
    "AttentionMechanism",
    "create_memory_encoder",
    
    # 神经字典
    "DifferentiableMemoryDictionary",
    "SynapticStorage",
    "DifferentiableMemoryKey",
    "SynapticConsolidation",
    "MemoryConsolidationScheduler",
    
    # 模式分离
    "PatternSeparationNetwork",
    "CA3PatternSeparator",
    "InputSpecificityEnhancer",
    "SynapticRemodeling",
    "HierarchicalPatternSeparation",
    "SparseCodingLayer",
    
    # 快速学习
    "EpisodicLearningSystem",
    "RapidEncodingUnit",
    "SingleTrialLearner",
    "FastAssociativeMemory",
    
    # 情景记忆系统
    "EpisodicMemorySystem",
    "TemporalContextEncoder",
    "EpisodicMemoryCell",
    "EpisodicMemory",
    "HippocampalIndexing"
]

# 配置常量
DEFAULT_CONFIGS = {
    "small": {
        "hidden_dim": 256,
        "memory_dim": 256,
        "num_transformer_layers": 3,
        "num_attention_heads": 4,
        "num_ca3_modules": 4,
        "storage_capacity": 5000
    },
    "base": {
        "hidden_dim": 512,
        "memory_dim": 512,
        "num_transformer_layers": 6,
        "num_attention_heads": 8,
        "num_ca3_modules": 8,
        "storage_capacity": 10000
    },
    "large": {
        "hidden_dim": 1024,
        "memory_dim": 1024,
        "num_transformer_layers": 12,
        "num_attention_heads": 16,
        "num_ca3_modules": 16,
        "storage_capacity": 20000
    }
}

# 模块信息
MODULE_INFO = {
    "name": "海马体模拟器",
    "version": __version__,
    "description": "基于最新神经科学研究的海马体记忆机制模拟器",
    "components": {
        "memory_encoder": "基于Transformer的记忆编码器，实现多突触末梢机制",
        "memory_dictionary": "可微分神经字典，支持情景记忆存储和检索",
        "pattern_separator": "模式分离网络，模拟CA3-CA1通路重构",
        "learning_system": "快速一次性学习系统，支持单次试验学习",
        "episodic_system": "情景记忆存储和检索系统"
    },
    "scientific_basis": {
        "source": "Science期刊 - 小鼠海马体记忆印迹的突触架构",
        "doi": "10.1126/science.ado8316",
        "key_mechanisms": [
            "多突触末梢(MSBs)的特异性增加",
            "非同步激活的记忆编码",
            "输入特异性增强",
            "空间受限的突触放大",
            "CA3-CA1通路重构",
            "连接重塑机制"
        ]
    }
}


def get_module_info():
    """获取模块信息"""
    return MODULE_INFO


def get_supported_configs():
    """获取支持的配置"""
    return DEFAULT_CONFIGS.copy()


def create_simulator_with_config(config_name: str = "base", custom_params: dict = None):
    """根据配置名称创建模拟器"""
    
    if config_name not in DEFAULT_CONFIGS:
        raise ValueError(f"不支持的配置: {config_name}. 支持的配置: {list(DEFAULT_CONFIGS.keys())}")
    
    config = DEFAULT_CONFIGS[config_name].copy()
    if custom_params:
        config.update(custom_params)
    
    return HippocampalSimulator(input_dim=config["hidden_dim"], config=config)


# 使用示例函数
def example_usage():
    """使用示例"""
    
    print("=== 海马体模拟器使用示例 ===")
    
    # 1. 创建模拟器
    simulator = create_hippocampus_simulator(
        input_dim=256,
        config=get_hippocampus_config("base")
    )
    
    # 2. 准备输入数据
    input_data = torch.randn(1, 256)
    
    # 3. 编码记忆
    encoding_result = simulator.encode_memory(input_data, metadata={"type": "demo"})
    
    # 4. 存储记忆
    memory_id = simulator.store_memory(
        encoding_result['final_encoding'],
        metadata={"timestamp": "2025-11-16", "importance": 0.8}
    )
    
    # 5. 检索记忆
    retrieval_result = simulator.retrieve_memory(encoding_result['final_encoding'])
    
    # 6. 巩固记忆
    consolidation_result = simulator.consolidate_memories()
    
    # 7. 查看系统状态
    status = simulator.get_system_status()
    
    print("示例运行完成!")
    print(f"编码形状: {encoding_result['final_encoding'].shape}")
    print(f"记忆ID: {memory_id}")
    print(f"检索置信度: {retrieval_result['retrieval_confidence']:.3f}")
    
    return simulator


if __name__ == "__main__":
    import torch
    
    # 运行使用示例
    example_usage()
    
    # 运行快速演示
    quick_hippocampus_demo()