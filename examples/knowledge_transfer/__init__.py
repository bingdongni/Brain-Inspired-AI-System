"""
知识迁移和重用机制模块

基于终身学习研究报告的理论基础，实现知识迁移和重用的核心机制：
1. 学习无遗忘(Learning Without Forgetting, LwF) - 保持旧任务知识
2. 元学习(Meta-Learning) - 快速适应新任务的机制  
3. 知识蒸馏(Knowledge Distillation) - 高效的知识传递方法

这些方法共同解决持续学习中的灾难性遗忘问题，实现知识的有效迁移和重用。

参考论文:
- Learning without Forgetting (Li & Hoiem, 2016)
- Model-Agnostic Meta-Learning (Finn et al., 2017)
- Distilling the Knowledge in a Neural Network (Hinton et al., 2015)
- A Comprehensive Survey of Continual Learning (2024)

Author: Lifelong Learning Team
Date: 2025-11-16
"""

# 学习无遗忘(LwF)模块
from .learning_without_forgetting import (
    LearningWithoutForgetting,
    LWFTrainer,
    AdaptiveLWFLoss
)

# 元学习模块
from .meta_learning import (
    MetaLearner,
    MAML,
    Reptile,
    MetaSGD,
    MetaLearningTrainer,
    MAMLClassifier,
    OmniglotNShotTask,
    create_meta_learning_model
)

# 知识蒸馏模块
from .knowledge_distillation import (
    DistillationLoss,
    LogitsDistillation,
    FeatureDistillation,
    AttentionDistillation,
    AdaptiveDistillation,
    TemperatureSoftmax,
    KLDivLoss,
    KnowledgeDistillationTrainer,
    ProgressiveKnowledgeDistillation,
    create_simple_mlp
)

# 版本信息
__version__ = '1.0.0'

# 导出所有核心类
__all__ = [
    # 学习无遗忘
    'LearningWithoutForgetting',
    'LWFTrainer', 
    'AdaptiveLWFLoss',
    
    # 元学习
    'MetaLearner',
    'MAML',
    'Reptile', 
    'MetaSGD',
    'MetaLearningTrainer',
    'MAMLClassifier',
    'OmniglotNShotTask',
    'create_meta_learning_model',
    
    # 知识蒸馏
    'DistillationLoss',
    'LogitsDistillation',
    'FeatureDistillation', 
    'AttentionDistillation',
    'AdaptiveDistillation',
    'TemperatureSoftmax',
    'KLDivLoss',
    'KnowledgeDistillationTrainer',
    'ProgressiveKnowledgeDistillation',
    'create_simple_mlp'
]