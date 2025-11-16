"""
模式分离机制模块

实现基于CA3-CA1通路的重构和连接重塑机制
"""

from .mechanism import (
    PatternSeparationNetwork,
    CA3PatternSeparator,
    InputSpecificityEnhancer,
    SynapticRemodeling,
    HierarchicalPatternSeparation,
    SparseCodingLayer
)

__all__ = [
    "PatternSeparationNetwork",
    "CA3PatternSeparator",
    "InputSpecificityEnhancer", 
    "SynapticRemodeling",
    "HierarchicalPatternSeparation",
    "SparseCodingLayer"
]