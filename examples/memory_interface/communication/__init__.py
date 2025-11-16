"""
海马体与新皮层通信模块
实现双向通信协议，支持信息在不同脑区间的传输和同步
"""

from .protocol_handler import CommunicationProtocol
from .message_types import MessageType, MemoryMessage, ControlMessage
from .hippocampus_interface import HippocampusInterface
from .neocortex_interface import NeocortexInterface
from .communication_controller import CommunicationController

__all__ = [
    'CommunicationProtocol',
    'MessageType',
    'MemoryMessage', 
    'ControlMessage',
    'HippocampusInterface',
    'NeocortexInterface',
    'CommunicationController'
]