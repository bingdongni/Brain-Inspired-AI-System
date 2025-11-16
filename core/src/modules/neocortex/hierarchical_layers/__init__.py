"""
新皮层分层抽象机制
=================

实现新皮层的层次化信息处理架构，包括：
- V1→V2→V4→IT的视觉层次处理
- IC→MGB→AC的听觉层级预测误差
- 前馈-反馈协同机制
- V4层间区室化注意调制

基于论文：
- "Revealing Detail along the Visual Hierarchy" (Neuron, 2018)
- "Neurons along the auditory pathway exhibit hierarchical organization of prediction error"
- "Laminar compartmentalization of attention modulation in area V4"
"""

from .layer_config import LayerType, ProcessingMode, LayerConfig
from .hierarchical_layer import HierarchicalLayer
from .processing_hierarchy import ProcessingHierarchy
from .visual_hierarchy import create_visual_hierarchy, VisualProcessingStream
from .auditory_hierarchy import create_auditory_hierarchy, AuditoryProcessingStream
from .attention_modulation import AttentionModulator, SpatialAttention, FeatureAttention
from .predictive_coding import PredictiveEncoder, ErrorAmplifier

__all__ = [
    'LayerType',
    'ProcessingMode', 
    'LayerConfig',
    'HierarchicalLayer',
    'ProcessingHierarchy',
    'create_visual_hierarchy',
    'create_auditory_hierarchy',
    'VisualProcessingStream',
    'AuditoryProcessingStream',
    'AttentionModulator',
    'SpatialAttention',
    'FeatureAttention',
    'PredictiveEncoder',
    'ErrorAmplifier'
]