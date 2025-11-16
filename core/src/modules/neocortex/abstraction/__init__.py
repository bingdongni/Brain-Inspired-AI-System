"""
新皮层知识抽象算法
==================

实现新皮层的知识抽象和概念形成机制，包括：
- 概念单元：检测和形成抽象概念
- 语义抽象：构建语义关系网络
- 知识图谱：建立知识关联
- 抽象推理：基于抽象概念的推理

基于论文：
- "Neuronal correlate of abstract concepts" (Quiroga et al., 2005)
- "A theory of how the brain might work" (Hawkins et al., 2017)
- "The hippocampal system as a navigational map" (O'Keefe & Nadel, 1978)
"""

from .abstraction_core import (
    AbstractionEngine, ConceptUnit, SemanticAbstraction, KnowledgeGraph,
    ConceptConfig, ConceptType, AbstractionLevel, create_abstraction_engine
)

from .concept_units import (
    ConceptDetector, ConceptMemory, ConceptFormation,
    ObjectConcept, SpatialConcept, TemporalConcept, AbstractConcept
)

from .semantic_abstraction import (
    SemanticNetwork, SemanticRelations, ConceptClustering,
    KnowledgeIntegration, SemanticReasoning
)

from .knowledge_graph import (
    ConceptNode, ConceptEdge, KnowledgeGraphManager,
    ConceptHierarchy, KnowledgePropagation
)

from .abstraction_algorithms import (
    ClusteringAbstraction, ProbabilisticAbstraction, NeuralAbstraction,
    HierarchicalAbstraction, SymbolicAbstraction
)

from .abstraction_config import (
    AbstractionConfig, ConceptConfig as ConfigConceptConfig,
    AbstractionType, KnowledgeRepresentationType
)

__all__ = [
    # 核心抽象引擎
    'AbstractionEngine',
    'ConceptUnit',
    'SemanticAbstraction', 
    'KnowledgeGraph',
    'ConceptConfig',
    'ConceptType',
    'AbstractionLevel',
    'create_abstraction_engine',
    
    # 概念单元
    'ConceptDetector',
    'ConceptMemory',
    'ConceptFormation',
    'ObjectConcept',
    'SpatialConcept',
    'TemporalConcept',
    'AbstractConcept',
    
    # 语义抽象
    'SemanticNetwork',
    'SemanticRelations',
    'ConceptClustering',
    'KnowledgeIntegration',
    'SemanticReasoning',
    
    # 知识图谱
    'ConceptNode',
    'ConceptEdge',
    'KnowledgeGraphManager',
    'ConceptHierarchy',
    'KnowledgePropagation',
    
    # 抽象算法
    'ClusteringAbstraction',
    'ProbabilisticAbstraction',
    'NeuralAbstraction',
    'HierarchicalAbstraction',
    'SymbolicAbstraction',
    
    # 配置
    'AbstractionConfig',
    'ConfigConceptConfig',
    'AbstractionType',
    'KnowledgeRepresentationType'
]