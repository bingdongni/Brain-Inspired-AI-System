"""
模块工厂函数
============

提供专业处理模块的创建工厂函数，方便用户快速构建所需模块。
"""

from typing import Dict, List, Optional, Any, Type
import torch
import torch.nn as nn

from .processing_config import (
    ProcessingConfig, PredictionConfig, AttentionConfig, 
    DecisionConfig, CrossModalConfig, ModuleType
)

from .prediction_module import PredictionModule
from .attention_module import AttentionModule
from .processing_config import DecisionModule, CrossModalModule


def create_prediction_module(feature_dim: int, 
                           prediction_type: str = "hierarchical",
                           prediction_horizon: int = 10,
                           **kwargs) -> PredictionModule:
    """
    创建预测模块
    
    Args:
        feature_dim: 特征维度
        prediction_type: 预测类型 ("hierarchical", "temporal", "semantic")
        prediction_horizon: 预测时间步长
        **kwargs: 其他配置参数
        
    Returns:
        PredictionModule: 预测模块实例
    """
    from .processing_config import PredictionType
    
    # 解析预测类型
    type_map = {
        "hierarchical": PredictionType.HIERARCHICAL,
        "temporal": PredictionType.TEMPORAL,
        "semantic": PredictionType.SEMANTIC,
        "causal": PredictionType.CAUSAL
    }
    
    prediction_type_enum = type_map.get(prediction_type, PredictionType.HIERARCHICAL)
    
    # 创建配置
    config = PredictionConfig(
        module_type=ModuleType.PREDICTION,
        feature_dim=feature_dim,
        prediction_type=prediction_type_enum,
        prediction_horizon=prediction_horizon,
        **kwargs
    )
    
    return PredictionModule(config)


def create_attention_module(feature_dim: int,
                          attention_type: str = "combined",
                          selectivity_threshold: float = 0.5,
                          **kwargs) -> AttentionModule:
    """
    创建注意模块
    
    Args:
        feature_dim: 特征维度
        attention_type: 注意力类型 ("spatial", "feature", "temporal", "task", "combined")
        selectivity_threshold: 选择性阈值
        **kwargs: 其他配置参数
        
    Returns:
        AttentionModule: 注意力模块实例
    """
    from .processing_config import AttentionType
    
    # 解析注意力类型
    type_map = {
        "spatial": AttentionType.SPATIAL,
        "feature": AttentionType.FEATURE,
        "temporal": AttentionType.TEMPORAL,
        "task": AttentionType.TASK,
        "combined": AttentionType.COMBINED
    }
    
    attention_type_enum = type_map.get(attention_type, AttentionType.COMBINED)
    
    # 创建配置
    config = AttentionConfig(
        module_type=ModuleType.ATTENTION,
        feature_dim=feature_dim,
        attention_type=attention_type_enum,
        selectivity_threshold=selectivity_threshold,
        **kwargs
    )
    
    return AttentionModule(config)


def create_decision_module(input_dim: int,
                         output_dim: int,
                         decision_mode: str = "probabilistic",
                         decision_threshold: float = 0.6,
                         **kwargs):
    """
    创建决策模块
    
    Args:
        input_dim: 输入维度
        output_dim: 输出维度
        decision_mode: 决策模式 ("probabilistic", "neural", "cognitive", "reinforcement")
        decision_threshold: 决策阈值
        **kwargs: 其他配置参数
        
    Returns:
        DecisionModule: 决策模块实例
    """
    # 创建配置
    config = DecisionConfig(
        module_type=ModuleType.DECISION,
        feature_dim=input_dim,
        decision_mode=DecisionMode(decision_mode),
        decision_threshold=decision_threshold,
        **kwargs
    )
    
    # 设置输出维度
    config.output_dim = output_dim
    
    return DecisionModule(config)


def create_crossmodal_module(input_dims: List[int],
                           target_dim: int,
                           fusion_method: str = "attention",
                           alignment_method: str = "attention",
                           **kwargs):
    """
    创建跨模态模块
    
    Args:
        input_dims: 各输入模态的维度列表
        target_dim: 目标整合维度
        fusion_method: 融合方法 ("attention", "concatenation", "weighted_sum")
        alignment_method: 对齐方法 ("attention", "projection", "correlation")
        **kwargs: 其他配置参数
        
    Returns:
        CrossModalModule: 跨模态模块实例
    """
    # 创建配置
    config = CrossModalConfig(
        module_type=ModuleType.CROSSMODAL,
        feature_dim=target_dim,
        input_modalities=[f"modality_{i}" for i in range(len(input_dims))],
        fusion_method=fusion_method,
        alignment_method=alignment_method,
        **kwargs
    )
    
    return CrossModalModule(config, input_dims)


def create_processing_pipeline(modules_config: List[Dict[str, Any]]) -> nn.ModuleDict:
    """
    创建处理管道
    
    Args:
        modules_config: 模块配置列表，每个元素包含：
            - type: 模块类型 ("prediction", "attention", "decision", "crossmodal")
            - name: 模块名称
            - params: 模块参数
            
    Returns:
        nn.ModuleDict: 包含所有模块的模块字典
    """
    pipeline = nn.ModuleDict()
    
    for config in modules_config:
        module_type = config['type']
        module_name = config['name']
        module_params = config.get('params', {})
        
        if module_type == 'prediction':
            module = create_prediction_module(**module_params)
        elif module_type == 'attention':
            module = create_attention_module(**module_params)
        elif module_type == 'decision':
            module = create_decision_module(**module_params)
        elif module_type == 'crossmodal':
            module = create_crossmodal_module(**module_params)
        else:
            raise ValueError(f"不支持的模块类型: {module_type}")
        
        pipeline[module_name] = module
    
    return pipeline


def get_default_processing_config(module_type: str, 
                                feature_dim: int,
                                mode: str = "offline") -> ProcessingConfig:
    """
    获取默认处理配置
    
    Args:
        module_type: 模块类型
        feature_dim: 特征维度
        mode: 处理模式
        
    Returns:
        ProcessingConfig: 配置对象
    """
    from .processing_config import ProcessingMode, ModuleType
    
    mode_enum = ProcessingMode(mode)
    type_enum = ModuleType(module_type)
    
    return ProcessingConfig(
        module_type=type_enum,
        feature_dim=feature_dim,
        processing_mode=mode_enum
    )


def create_pretrained_module(module_type: str, 
                           model_path: Optional[str] = None,
                           **kwargs):
    """
    创建预训练模块
    
    Args:
        module_type: 模块类型
        model_path: 模型路径
        **kwargs: 其他参数
        
    Returns:
        预训练模块实例
    """
    # 首先创建基础模块
    if module_type == 'prediction':
        module = create_prediction_module(**kwargs)
    elif module_type == 'attention':
        module = create_attention_module(**kwargs)
    else:
        raise ValueError(f"不支持的预训练模块类型: {module_type}")
    
    # 如果提供了模型路径，加载预训练权重
    if model_path is not None:
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            module.load_state_dict(state_dict)
        except Exception as e:
            print(f"警告：无法加载预训练权重: {e}")
    
    return module


def get_module_info(module_type: str) -> Dict[str, Any]:
    """
    获取模块信息
    
    Args:
        module_type: 模块类型
        
    Returns:
        Dict[str, Any]: 模块信息
    """
    info_map = {
        'prediction': {
            'name': '预测模块',
            'description': '实现基于新皮层预测编码机制的层级预测功能',
            'inputs': ['features', 'context', 'target'],
            'outputs': ['predictions', 'confidence', 'uncertainty'],
            'parameters': ['feature_dim', 'prediction_horizon', 'hierarchical_levels'],
            'supported_types': ['hierarchical', 'temporal', 'semantic', 'causal']
        },
        'attention': {
            'name': '注意模块',
            'description': '实现基于新皮层注意控制机制的智能注意力分配',
            'inputs': ['features', 'task_context', 'spatial_hints', 'feature_hints'],
            'outputs': ['attended_features', 'attention_maps', 'attention_intensity'],
            'parameters': ['feature_dim', 'selectivity_threshold', 'top_k_attention'],
            'supported_types': ['spatial', 'feature', 'temporal', 'task', 'combined']
        },
        'decision': {
            'name': '决策模块',
            'description': '实现基于群体动力学的决策形成机制',
            'inputs': ['input_features', 'options', 'context'],
            'outputs': ['decisions', 'confidences', 'decision_process'],
            'parameters': ['input_dim', 'output_dim', 'decision_threshold'],
            'supported_modes': ['probabilistic', 'neural', 'cognitive', 'reinforcement']
        },
        'crossmodal': {
            'name': '跨模态模块',
            'description': '实现多感官信息整合和概念形成',
            'inputs': ['modal_features_1', 'modal_features_2', ...],
            'outputs': ['integrated_features', 'crossmodal_alignment', 'concepts'],
            'parameters': ['input_dims', 'target_dim', 'fusion_method'],
            'supported_methods': ['attention', 'concatenation', 'weighted_sum']
        }
    }
    
    return info_map.get(module_type, {})


def benchmark_module(module: nn.Module, 
                    input_shape: torch.Size,
                    num_runs: int = 100) -> Dict[str, float]:
    """
    基准测试模块性能
    
    Args:
        module: 要测试的模块
        input_shape: 输入形状
        num_runs: 运行次数
        
    Returns:
        Dict[str, float]: 性能统计
    """
    import time
    
    module.eval()
    device = next(module.parameters()).device
    
    # 生成测试输入
    test_input = torch.randn(input_shape).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = module(test_input)
    
    # 基准测试
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = module(test_input)
            end_time = time.time()
            times.append(end_time - start_time)
    
    # 计算统计
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    # 内存使用（如果是GPU）
    if device.type == 'cuda':
        torch.cuda.synchronize()
        memory_used = torch.cuda.memory_allocated(device) / 1024 / 1024  # MB
    else:
        memory_used = 0.0
    
    return {
        'average_time_ms': avg_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'std_time_ms': std_time * 1000,
        'fps': 1.0 / avg_time if avg_time > 0 else 0,
        'memory_mb': memory_used
    }