#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统接口定义
===========

定义Brain-Inspired AI系统各个组件之间的接口。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import torch

class IModule(ABC):
    """基础模块接口"""
    
    @abstractmethod
    def initialize(self, config: Optional[Dict] = None) -> bool:
        """初始化模块"""
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """前向传播"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        pass

class INeuralComponent(ABC):
    """神经网络组件接口"""
    
    @abstractmethod
    def train(self) -> None:
        """设置为训练模式"""
        pass
    
    @abstractmethod
    def eval(self) -> None:
        """设置为评估模式"""
        pass
    
    @abstractmethod
    def save_state(self, path: str) -> None:
        """保存模型状态"""
        pass
    
    @abstractmethod
    def load_state(self, path: str) -> None:
        """加载模型状态"""
        pass

class ITrainingComponent(ABC):
    """训练组件接口"""
    
    @abstractmethod
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """训练一个epoch"""
        pass
    
    @abstractmethod
    def validate(self, dataloader) -> Dict[str, float]:
        """验证模型"""
        pass
    
    @abstractmethod
    def get_training_history(self) -> Dict[str, List[float]]:
        """获取训练历史"""
        pass

class IHippocampusComponent(IModule, INeuralComponent):
    """海马体组件接口"""
    
    @abstractmethod
    def store_memory(self, key: str, value: Any) -> str:
        """存储记忆"""
        pass
    
    @abstractmethod
    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """检索记忆"""
        pass
    
    @abstractmethod
    def pattern_separation(self, inputs: torch.Tensor) -> torch.Tensor:
        """模式分离"""
        pass

class INeocortexComponent(IModule, INeuralComponent):
    """新皮层组件接口"""
    
    @abstractmethod
    def process_hierarchical(self, input_data: torch.Tensor) -> torch.Tensor:
        """层次化处理"""
        pass
    
    @abstractmethod
    def attention_forward(self, query: torch.Tensor, 
                         key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """注意力前向传播"""
        pass

class IMemoryInterface(ABC):
    """记忆接口"""
    
    @abstractmethod
    def write(self, address: torch.Tensor, data: torch.Tensor) -> None:
        """写入记忆"""
        pass
    
    @abstractmethod
    def read(self, address: torch.Tensor) -> torch.Tensor:
        """读取记忆"""
        pass
    
    @abstractmethod
    def get_capacity(self) -> int:
        """获取记忆容量"""
        pass

class IRoutingController(ABC):
    """路由控制器接口"""
    
    @abstractmethod
    def route(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """路由数据"""
        pass
    
    @abstractmethod
    def update_routes(self, performance_metrics: Dict[str, float]) -> None:
        """更新路由策略"""
        pass

class ILifelongLearningComponent(IModule):
    """持续学习组件接口"""
    
    @abstractmethod
    def learn_new_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """学习新任务"""
        pass
    
    @abstractmethod
    def consolidate_knowledge(self) -> None:
        """巩固知识"""
        pass
    
    @abstractmethod
    def prevent_forgetting(self, old_task_data: Dict[str, Any]) -> float:
        """防止遗忘"""
        pass