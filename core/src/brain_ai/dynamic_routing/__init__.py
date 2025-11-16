#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
动态路由模块
===========

实现智能的动态路由机制，包括:
- 实时路由控制
- 自适应资源分配
- 效率优化
- 强化学习路由
"""

from .realtime_routing_controller import DynamicRoutingController
from .adaptive_allocation.allocation_controller import AdaptiveAllocation
from .adaptive_allocation.load_balancer import LoadBalancer
from .adaptive_allocation.predictive_early_exit import PredictiveEarlyExit
from .efficiency_optimization.power_optimization import PowerOptimization
from .efficiency_optimization.intelligent_path_selector import IntelligentPathSelector
from .reinforcement_routing.actor_critic import ActorCriticRouting
from .reinforcement_routing.routing_environment import RoutingEnvironment

__all__ = [
    "DynamicRoutingController",
    "AdaptiveAllocation",
    "LoadBalancer", 
    "PredictiveEarlyExit",
    "PowerOptimization",
    "IntelligentPathSelector",
    "ActorCriticRouting",
    "RoutingEnvironment"
]