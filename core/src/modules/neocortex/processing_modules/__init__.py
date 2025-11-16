"""
新皮层专业处理模块
==================

实现新皮层的专业认知功能处理，包括：
- 预测模块：基于层级预测编码的未来状态预测
- 注意模块：多层次注意力控制和调制
- 决策模块：基于群体动力学的决策形成
- 跨模态模块：多感官信息整合和概念形成

基于论文：
- "Predictive coding in the visual cortex" (Friston, 2010)
- "Attention and memory" (Desimone & Duncan, 1995)
- "Decision making as a window on cognition" (Shadlen & Newsome, 2001)
- "Cross-modal plasticity: where and how" (Bavelier & Neville, 2002)
"""

from .prediction_module import (
    PredictionModule, PredictionType, PredictionEngine,
    HierarchicalPredictor, TemporalPredictor, SemanticPredictor
)

from .attention_module import (
    AttentionModule, AttentionType, AttentionEngine,
    SpatialAttentionEngine, FeatureAttentionEngine, TaskAttentionEngine
)

from .decision_module import (
    DecisionModule, DecisionMode, DecisionEngine,
    ProbabilisticDecisionEngine, NeuralDecisionEngine, CognitiveDecisionEngine
)

from .crossmodal_module import (
    CrossModalModule, CrossModalEngine, 
    VisualLanguageInterface, AuditoryVisualInterface, MultimodalIntegrator
)

from .processing_config import (
    ProcessingConfig, ModuleType, ProcessingMode, 
    PredictionConfig, AttentionConfig, DecisionConfig, CrossModalConfig
)

from .module_factory import (
    create_prediction_module, create_attention_module,
    create_decision_module, create_crossmodal_module,
    create_processing_pipeline
)

__all__ = [
    # 预测模块
    'PredictionModule',
    'PredictionType',
    'PredictionEngine', 
    'HierarchicalPredictor',
    'TemporalPredictor',
    'SemanticPredictor',
    
    # 注意模块
    'AttentionModule',
    'AttentionType',
    'AttentionEngine',
    'SpatialAttentionEngine',
    'FeatureAttentionEngine', 
    'TaskAttentionEngine',
    
    # 决策模块
    'DecisionModule',
    'DecisionMode',
    'DecisionEngine',
    'ProbabilisticDecisionEngine',
    'NeuralDecisionEngine',
    'CognitiveDecisionEngine',
    
    # 跨模态模块
    'CrossModalModule',
    'CrossModalEngine',
    'VisualLanguageInterface',
    'AuditoryVisualInterface',
    'MultimodalIntegrator',
    
    # 配置和工厂
    'ProcessingConfig',
    'ModuleType',
    'ProcessingMode',
    'PredictionConfig',
    'AttentionConfig', 
    'DecisionConfig',
    'CrossModalConfig',
    'create_prediction_module',
    'create_attention_module',
    'create_decision_module',
    'create_crossmodal_module',
    'create_processing_pipeline'
]