"""
快速学习模块

实现非同步激活的快速学习机制
"""

from .rapid_learning import (
    EpisodicLearningSystem,
    RapidEncodingUnit,
    SingleTrialLearner,
    FastAssociativeMemory
)

__all__ = [
    "EpisodicLearningSystem",
    "RapidEncodingUnit",
    "SingleTrialLearner", 
    "FastAssociativeMemory"
]