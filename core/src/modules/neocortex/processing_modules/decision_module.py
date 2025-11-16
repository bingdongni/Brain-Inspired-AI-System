"""
决策模块实现
=============

实现基于群体动力学的决策形成机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .processing_config import DecisionConfig, DecisionMode, ModuleType


class DecisionModule(nn.Module):
    """决策模块基类"""
    
    def __init__(self, config: DecisionConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.feature_dim
        self.output_dim = getattr(config, 'output_dim', 2)  # 默认二分类
    
    def forward(self, features: torch.Tensor, options: Optional[List] = None) -> Dict[str, Any]:
        raise NotImplementedError


class ProbabilisticDecisionEngine(DecisionModule):
    """概率性决策引擎"""
    
    def __init__(self, config: DecisionConfig):
        super().__init__(config)
        
        self.decision_head = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.output_dim),
            nn.Softmax(dim=1)
        )
        
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, features: torch.Tensor, options: Optional[List] = None) -> Dict[str, Any]:
        # 生成决策概率
        decision_probs = self.decision_head(features)
        
        # 估计置信度
        confidence = self.confidence_estimator(features)
        
        # 应用决策阈值
        decisions = (decision_probs > self.config.decision_threshold).float()
        
        return {
            'decisions': decisions,
            'probabilities': decision_probs,
            'confidence': confidence,
            'decision_mode': 'probabilistic'
        }


class NeuralDecisionEngine(DecisionModule):
    """神经网络决策引擎"""
    
    def __init__(self, config: DecisionConfig):
        super().__init__(config)
        
        self.decision_network = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.output_dim)
        )
        
        self.dynamics_simulator = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.Tanh(),
            nn.Linear(self.input_dim // 2, self.input_dim // 2)
        )
    
    def forward(self, features: torch.Tensor, options: Optional[List] = None) -> Dict[str, Any]:
        # 神经网络决策
        decision_logits = self.decision_network(features)
        decisions = torch.argmax(decision_logits, dim=1)
        
        # 模拟决策动力学
        dynamics_state = self.dynamics_simulator(features)
        
        return {
            'decisions': decisions,
            'logits': decision_logits,
            'dynamics_state': dynamics_state,
            'decision_mode': 'neural'
        }


class CognitiveDecisionEngine(DecisionModule):
    """认知决策引擎"""
    
    def __init__(self, config: DecisionConfig):
        super().__init__(config)
        
        self.cognitive_processor = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.output_dim)
        )
        
        self.working_memory = nn.GRU(
            self.input_dim // 2, self.input_dim // 4, batch_first=True
        )
    
    def forward(self, features: torch.Tensor, options: Optional[List] = None) -> Dict[str, Any]:
        # 认知处理
        cognitive_features = self.cognitive_processor(features)
        
        # 工作记忆更新
        memory_output, _ = self.working_memory(cognitive_features.unsqueeze(1))
        
        # 决策生成
        decisions = torch.argmax(cognitive_features, dim=1)
        
        return {
            'decisions': decisions,
            'cognitive_features': cognitive_features,
            'working_memory': memory_output,
            'decision_mode': 'cognitive'
        }


# 工厂函数
def create_decision_module(input_dim: int,
                         output_dim: int,
                         decision_mode: str = "probabilistic",
                         **kwargs):
    """创建决策模块"""
    config = DecisionConfig(
        module_type=ModuleType.DECISION,
        feature_dim=input_dim,
        decision_mode=DecisionMode(decision_mode)
    )
    config.output_dim = output_dim
    
    if decision_mode == "probabilistic":
        return ProbabilisticDecisionEngine(config)
    elif decision_mode == "neural":
        return NeuralDecisionEngine(config)
    elif decision_mode == "cognitive":
        return CognitiveDecisionEngine(config)
    else:
        return ProbabilisticDecisionEngine(config)