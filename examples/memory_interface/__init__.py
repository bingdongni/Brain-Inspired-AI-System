"""
记忆-计算接口模块主入口
整合所有记忆系统组件的统一接口
"""

from .attention_mechanism import AttentionController, AttentionConfig
from .communication import CommunicationController
from .consolidation import ConsolidationEngine, ConsolidationConfig
from .integration import IntegratedMemorySystem, SystemConfig
from .control_flow import InformationFlowController, FlowControlConfig

from .memory_interface_core import MemoryInterfaceCore

__all__ = [
    'MemoryInterfaceCore',
    'AttentionController',
    'CommunicationController', 
    'ConsolidationEngine',
    'IntegratedMemorySystem',
    'InformationFlowController'
]