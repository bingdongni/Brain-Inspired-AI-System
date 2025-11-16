#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基础模块类
==========

提供所有Brain-Inspired AI组件的基础抽象。
"""

import torch
import torch.nn as nn
from enum import Enum
from typing import Dict, Any, Optional
import logging

class ModuleState(Enum):
    """模块状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing" 
    READY = "ready"
    TRAINING = "training"
    EVALUATING = "evaluating"
    ERROR = "error"

class ModuleType(Enum):
    """模块类型枚举"""
    HIPPOCAMPUS = "hippocampus"
    NEOCORTEX = "neocortex"
    MEMORY = "memory"
    ROUTING = "routing"
    LEARNING = "learning"

class BaseModule(nn.Module):
    """所有Brain-Inspired AI组件的基础模块类"""
    
    def __init__(self, name: str, module_type: ModuleType):
        """
        初始化基础模块
        
        Args:
            name: 模块名称
            module_type: 模块类型
        """
        super().__init__()
        self.name = name
        self.module_type = module_type
        self.state = ModuleState.UNINITIALIZED
        self.logger = logging.getLogger(f"brain_ai.{name}")
        
        # 配置参数
        self.config = {}
        self.metadata = {}
        
        # 性能统计
        self.forward_count = 0
        self.total_forward_time = 0.0
        
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        初始化模块
        
        Args:
            config: 配置参数
            
        Returns:
            初始化是否成功
        """
        try:
            self.state = ModuleState.INITIALIZING
            
            if config:
                self.config.update(config)
            
            self._initialize()
            
            self.state = ModuleState.READY
            self.logger.info(f"模块 {self.name} 初始化完成")
            return True
            
        except Exception as e:
            self.state = ModuleState.ERROR
            self.logger.error(f"模块 {self.name} 初始化失败: {e}")
            return False
    
    def _initialize(self):
        """子类实现具体的初始化逻辑"""
        pass
    
    def forward(self, *args, **kwargs):
        """前向传播，子类必须实现"""
        raise NotImplementedError("子类必须实现forward方法")
    
    def get_info(self) -> Dict[str, Any]:
        """获取模块信息"""
        return {
            "name": self.name,
            "type": self.module_type.value,
            "state": self.state.value,
            "config": self.config,
            "forward_count": self.forward_count,
            "avg_forward_time": self.total_forward_time / max(1, self.forward_count)
        }
    
    def save_state(self, path: str):
        """保存模块状态"""
        state = {
            "name": self.name,
            "module_type": self.module_type.value,
            "state_dict": self.state_dict(),
            "config": self.config,
            "metadata": self.metadata
        }
        torch.save(state, path)
    
    def load_state(self, path: str):
        """加载模块状态"""
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state["state_dict"])
        self.config.update(state.get("config", {}))
        self.metadata.update(state.get("metadata", {}))
    
    def reset(self):
        """重置模块状态"""
        self.forward_count = 0
        self.total_forward_time = 0.0
        self.state = ModuleState.READY