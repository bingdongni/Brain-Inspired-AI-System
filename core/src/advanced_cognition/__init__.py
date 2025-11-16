"""
高级认知功能模块
==============

包含系统集成、推理机制、性能调优、类比学习等高级认知功能。

主要模块：
- 端到端训练管道
- 系统性能调优
- 多步推理机制
- 类比学习与创造性问题解决

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入主要组件
from advanced_cognition.end_to_end_pipeline import (
    EndToEndTrainingPipeline,
    PipelineStage
)

from advanced_cognition.performance_optimization import (
    PerformanceOptimizer,
    OptimizationStrategy
)

from advanced_cognition.multi_step_reasoning import (
    MultiStepReasoner,
    ReasoningType
)

from advanced_cognition.analogical_learning import (
    AnalogicalLearner
)

from advanced_cognition.system_integration import (
    CognitiveSystemIntegrator
)

# 导入便利函数
from advanced_cognition.end_to_end_pipeline import (
    create_standard_classification_pipeline,
    create_custom_pipeline
)

from advanced_cognition.performance_optimization import (
    create_neural_network_optimization_config
)

from advanced_cognition.multi_step_reasoning import (
    create_comprehensive_reasoner
)

from advanced_cognition.analogical_learning import (
    create_analogical_learner
)

from advanced_cognition.system_integration import (
    create_cognitive_system_integrator
)

__all__ = [
    # 主要类
    'EndToEndTrainingPipeline',
    'PerformanceOptimizer',
    'MultiStepReasoner',
    'AnalogicalLearner',
    'CognitiveSystemIntegrator',
    'PipelineStage',
    'OptimizationStrategy',
    'ReasoningType',
    
    # 便利函数
    'create_standard_classification_pipeline',
    'create_custom_pipeline',
    'create_neural_network_optimization_config',
    'create_comprehensive_reasoner',
    'create_analogical_learner',
    'create_cognitive_system_integrator'
]

__version__ = "1.0.0"