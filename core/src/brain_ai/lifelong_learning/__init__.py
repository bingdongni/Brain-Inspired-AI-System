#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
持续学习模块
===========

实现各种持续学习策略，包括:
- 弹性权重巩固 (EWC)
- 生成式重放
- 动态扩展
- 知识转移
"""

from .elastic_weight_consolidation.ewc_trainer import ElasticWeightConsolidation
from .elastic_weight_consolidation.fisher_matrix import FisherMatrix
from .generative_replay.generative_replay_trainer import GenerativeReplay
from .generative_replay.experience_replay import ExperienceReplay
from .dynamic_expansion.dynamic_capacity_growth import DynamicExpansion
from .knowledge_transfer.knowledge_distillation import KnowledgeTransfer
from .knowledge_transfer.learning_without_forgetting import LearningWithoutForgetting

__all__ = [
    "ElasticWeightConsolidation",
    "FisherMatrix",
    "GenerativeReplay",
    "ExperienceReplay",
    "DynamicExpansion",
    "KnowledgeTransfer",
    "LearningWithoutForgetting"
]