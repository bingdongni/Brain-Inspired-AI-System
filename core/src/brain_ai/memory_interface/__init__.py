#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
记忆接口模块
===========

提供统一的记忆系统接口，包括:
- 注意力机制控制
- 通信控制器
- 巩固引擎
- 记忆整合
"""

from .attention_mechanism.attention_controller import AttentionController
from .attention_mechanism.attention_memory import AttentionMemory
from .communication.communication_controller import CommunicationController
from .communication.hippocampus_interface import HippocampusInterface
from .communication.neocortex_interface import NeocortexInterface
from .consolidation.consolidation_engine import ConsolidationEngine
from .integration.integrated_memory_system import IntegratedMemorySystem

__all__ = [
    "AttentionController",
    "AttentionMemory",
    "CommunicationController", 
    "HippocampusInterface",
    "NeocortexInterface",
    "ConsolidationEngine",
    "IntegratedMemorySystem"
]