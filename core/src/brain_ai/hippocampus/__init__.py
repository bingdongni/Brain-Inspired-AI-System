#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
海马体记忆系统模块
=================

实现海马体的核心功能，包括:
- 情景记忆存储与检索
- 快速学习机制
- 模式分离
- 时间序列对齐
"""

from .core.simulator import HippocampusSimulator
from .core.episodic_memory import EpisodicMemory
from .core.fast_learning import FastLearning
from .encoders.transformer_encoder import TransformerEncoder
from .pattern_separation.mechanism import PatternSeparation
from .temporal_alignment.temporal_alignment import TemporalAlignment

__all__ = [
    "HippocampusSimulator",
    "EpisodicMemory", 
    "FastLearning",
    "TransformerEncoder",
    "PatternSeparation",
    "TemporalAlignment"
]