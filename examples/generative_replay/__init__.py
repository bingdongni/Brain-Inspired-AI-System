"""
生成式回放(Generative Replay)模块

实现生成式回放机制，包括生成对抗网络、经验重放和双侧记忆巩固。
用于在持续学习中重演历史经验，避免灾难性遗忘。

基于生成对抗网络的知识重放机制，参考相关研究：
- Generating Classifier and Rehearsal Data for Continual Learning (2019)
- Deep Generative Replay (2018)
"""

from .generative_adversarial_network import GenerativeAdversarialNetwork
from .experience_replay import ExperienceReplayBuffer
from .bilateral_consolidation import BilateralMemoryConsolidation
from .generative_replay_trainer import GenerativeReplayTrainer

__all__ = [
    'GenerativeAdversarialNetwork',
    'ExperienceReplayBuffer', 
    'BilateralMemoryConsolidation',
    'GenerativeReplayTrainer'
]

__version__ = '1.0.0'