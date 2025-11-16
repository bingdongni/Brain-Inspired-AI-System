#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
海马体核心模块
=============

包含海马体的核心功能实现。
"""

from .simulator import HippocampusSimulator, HippocampusConfig
from .episodic_memory import EpisodicMemory, EpisodeMemory
from .fast_learning import FastLearningSystem, MAMLModule, RapidConsolidation, create_fast_learner

__all__ = [
    "HippocampusSimulator",
    "HippocampusConfig", 
    "EpisodicMemory",
    "EpisodeMemory",
    "FastLearningSystem",
    "MAMLModule",
    "RapidConsolidation",
    "create_fast_learner"
]