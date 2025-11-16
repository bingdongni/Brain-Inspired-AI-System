#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Brain-Inspired AI 核心模块
========================

包含系统的核心抽象类和基础组件。
"""

from .base_module import BaseModule, ModuleState, ModuleType
from .brain_system import BrainSystem
from .interfaces import IModule, INeuralComponent, ITrainingComponent

__all__ = [
    "BaseModule",
    "ModuleState", 
    "ModuleType",
    "BrainSystem",
    "IModule",
    "INeuralComponent", 
    "ITrainingComponent"
]