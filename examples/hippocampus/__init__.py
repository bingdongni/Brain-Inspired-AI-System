"""
海马体模拟器核心模块
基于神经科学研究成果的生物启发式记忆系统
"""

from .hippocampus_simulator import HippocampusSimulator, create_hippocampus_simulator, get_default_config
from .encoders.transformer_encoder import TransformerMemoryEncoder
from .encoders.attention_mechanism import EnhancedAttention
from .encoders.pattern_completion import PatternCompletionModule
from .encoders.temporal_alignment import TemporalAlignmentModule
from .fast_learning import OneShotLearner
from .episodic_memory import EpisodicMemorySystem

__version__ = "2.0.0"
__author__ = "Hippocampus Simulator Team"
__email__ = "team@hippocampus.ai"

# 模块元信息
__all__ = [
    'HippocampusSimulator',
    'create_hippocampus_simulator',
    'get_default_config',
    'TransformerMemoryEncoder',
    'EnhancedAttention',
    'PatternCompletionModule',
    'TemporalAlignmentModule',
    'OneShotLearner',
    'EpisodicMemorySystem'
]

# 版本信息
VERSION_INFO = {
    'version': __version__,
    'author': __author__,
    'email': __email__,
    'description': '基于神经科学的海马体记忆模拟器',
    'features': [
        'Transformer-based记忆编码',
        '可微分神经字典',
        '模式分离机制',
        '快速一次性学习',
        '情景记忆存储检索'
    ],
    'based_on': '小鼠海马体记忆印迹的突触架构研究 (Science 2025)',
    'citation': 'Synaptic architecture of a memory engram in the mouse hippocampus. Science 2025.'
}