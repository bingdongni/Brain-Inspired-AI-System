"""
弹性权重巩固(Elastic Weight Consolidation, EWC)模块

基于Fisher信息矩阵的重要参数保护机制，用于防止灾难性遗忘。
参考Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting in neural networks"
"""

from .fisher_matrix import FisherInformationMatrix
from .ewc_loss import EWCLossFunction
from .intelligent_protection import IntelligentWeightProtection
from .ewc_trainer import EWCTrainer

__all__ = [
    'FisherInformationMatrix',
    'EWCLossFunction', 
    'IntelligentWeightProtection',
    'EWCTrainer'
]

__version__ = '1.0.0'