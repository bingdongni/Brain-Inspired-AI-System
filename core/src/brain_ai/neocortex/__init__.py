#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
新皮层处理架构模块
================

实现新皮层的层次化处理架构，包括:
- 层次化信息处理
- 注意力机制
- 跨模态整合
- 决策制定
"""

from .neocortex_architecture import NeocortexArchitecture
from .hierarchical_layers.processing_hierarchy import HierarchicalProcessor
from .processing_modules.attention_module import AttentionModule
from .processing_modules.decision_module import DecisionModule
from .processing_modules.crossmodal_module import CrossModalModule
from .sparse_activation import SparseActivation

__all__ = [
    "NeocortexArchitecture",
    "HierarchicalProcessor",
    "AttentionModule",
    "DecisionModule", 
    "CrossModalModule",
    "SparseActivation"
]