"""
知识回放和巩固机制模块
实现基于海马体-新皮层回路的记忆巩固和知识整合
"""

from .consolidation_engine import ConsolidationEngine
from .replay_system import ReplaySystem
from .memory_strengthener import MemoryStrengthener
from .knowledge_integrator import KnowledgeIntegrator

__all__ = [
    'ConsolidationEngine',
    'ReplaySystem',
    'MemoryStrengthener',
    'KnowledgeIntegrator'
]