"""
核心模块

海马体模拟器的核心组件
"""

from .simulator import (
    HippocampalSimulator,
    create_hippocampus_simulator,
    get_hippocampus_config,
    quick_hippocampus_demo
)

__all__ = [
    "HippocampalSimulator",
    "create_hippocampus_simulator",
    "get_hippocampus_config",
    "quick_hippocampus_demo"
]