"""
动态网络扩展技术模块

实现动态网络扩展方法，包括渐进式神经网络、动态容量增长和新任务学习。
用于持续学习中通过架构扩展来避免灾难性遗忘。

参考论文:
- Progressive Neural Networks (Rusu et al., 2016)
- Progressive Learning: A Deep Learning Framework (Zhang et al., 2020)
"""

from .progressive_neural_network import ProgressiveNeuralNetwork
from .dynamic_capacity_growth import DynamicCapacityGrowth
from .new_task_learning import NewTaskLearner
from .progressive_trainer import ProgressiveTrainer

__all__ = [
    'ProgressiveNeuralNetwork',
    'DynamicCapacityGrowth',
    'NewTaskLearner',
    'ProgressiveTrainer'
]

__version__ = '1.0.0'