"""
记忆接口主模块
整合所有记忆-计算接口组件
"""

from ..attention_mechanism import AttentionController, AttentionConfig
from ..communication import CommunicationController
from .memory_integrator import MemoryIntegrator, InformationFlowController

from .integrated_memory_system import IntegratedMemorySystem

__all__ = [
    'IntegratedMemorySystem'
]