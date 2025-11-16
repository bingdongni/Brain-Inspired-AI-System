"""
基础模块抽象类
=============

定义了所有大脑启发AI系统模块的基类，提供了统一的接口和生命周期管理。
每个模块都继承自BaseModule并实现特定的业务逻辑。

设计原则:
- 模块化: 每个模块职责单一，易于维护
- 可扩展: 支持动态加载和配置
- 可监控: 提供状态和性能监控
- 可测试: 易于单元测试和集成测试

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import logging
import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from threading import Lock
import uuid


class ModuleState(Enum):
    """模块状态枚举"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing" 
    INITIALIZED = "initialized"
    LOADING = "loading"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    DISPOSED = "disposed"


class ModuleType(Enum):
    """模块类型枚举"""
    CORE = "core"
    NEURAL = "neural"
    MEMORY = "memory"
    TRAINING = "training"
    UTILITY = "utility"
    VISUALIZATION = "visualization"


@dataclass
class ModuleConfig:
    """模块配置类"""
    name: str
    version: str = "1.0.0"
    enabled: bool = True
    priority: int = 1
    timeout: float = 30.0
    retry_count: int = 3
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    event_handlers: Dict[str, Callable] = field(default_factory=dict)


class BaseModule(ABC):
    """
    基础模块抽象类
    
    所有系统模块的基类，提供统一的生命周期管理、状态监控、
    错误处理和事件分发机制。
    """
    
    def __init__(self, config: ModuleConfig):
        """
        初始化模块
        
        Args:
            config: 模块配置对象
        """
        self.config = config
        self.module_id = str(uuid.uuid4())
        self._state = ModuleState.UNINITIALIZED  # 先初始化私有属性
        self.logger = logging.getLogger(f"{__name__}.{config.name}")
        self._lock = Lock()
        self._start_time: Optional[float] = None
        self._metrics: Dict[str, Any] = {}
        self._subscribers: List[Callable] = []
        
    @property
    def name(self) -> str:
        """获取模块名称"""
        return self.config.name
    
    @property
    def version(self) -> str:
        """获取模块版本"""
        return self.config.version
    
    @property
    def state(self) -> ModuleState:
        """获取模块当前状态"""
        return self._state
    
    @state.setter
    def state(self, value: ModuleState):
        """设置模块状态"""
        old_state = getattr(self, '_state', None)
        self._state = value
        if old_state is not None and old_state != value:
            self.logger.info(f"状态变更: {old_state.value} -> {value.value}")
            self._notify_state_change(old_state, value)
    
    @property
    def uptime(self) -> float:
        """获取模块运行时间"""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """获取模块性能指标"""
        return self._metrics.copy()
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        初始化模块
        
        子类必须实现此方法来进行特定的初始化逻辑。
        
        Returns:
            bool: 初始化是否成功
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """
        清理资源
        
        子类必须实现此方法来释放资源和清理状态。
        
        Returns:
            bool: 清理是否成功
        """
        pass
    
    def start(self) -> bool:
        """
        启动模块
        
        模板方法，定义了启动的标准流程：
        1. 检查依赖
        2. 初始化
        3. 设置启动时间
        4. 切换到活跃状态
        
        Returns:
            bool: 启动是否成功
        """
        with self._lock:
            try:
                if self.state != ModuleState.INITIALIZED:
                    self.logger.error(f"模块未正确初始化，当前状态: {self.state.value}")
                    return False
                
                self.state = ModuleState.LOADING
                
                # 预启动检查
                if not self._pre_start_checks():
                    self.state = ModuleState.ERROR
                    return False
                
                # 执行启动逻辑
                if not self._do_start():
                    self.state = ModuleState.ERROR
                    return False
                
                self._start_time = time.time()
                self.state = ModuleState.ACTIVE
                
                self.logger.info(f"模块 {self.name} 启动成功")
                return True
                
            except Exception as e:
                self.logger.error(f"模块启动失败: {e}")
                self.state = ModuleState.ERROR
                return False
    
    def stop(self) -> bool:
        """
        停止模块
        
        模板方法，定义了停止的标准流程。
        
        Returns:
            bool: 停止是否成功
        """
        with self._lock:
            try:
                if self.state not in [ModuleState.ACTIVE, ModuleState.PAUSED]:
                    return True
                
                self.state = ModuleState.STOPPING
                
                # 执行停止逻辑
                if not self._do_stop():
                    self.logger.error("模块停止过程中发生错误")
                    return False
                
                self.state = ModuleState.STOPPED
                self._start_time = None
                
                self.logger.info(f"模块 {self.name} 已停止")
                return True
                
            except Exception as e:
                self.logger.error(f"模块停止失败: {e}")
                self.state = ModuleState.ERROR
                return False
    
    def pause(self) -> bool:
        """暂停模块"""
        if self.state == ModuleState.ACTIVE:
            self.state = ModuleState.PAUSED
            return True
        return False
    
    def resume(self) -> bool:
        """恢复模块"""
        if self.state == ModuleState.PAUSED:
            self.state = ModuleState.ACTIVE
            return True
        return False
    
    def subscribe(self, callback: Callable) -> None:
        """
        订阅状态变化事件
        
        Args:
            callback: 状态变化回调函数
        """
        self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable) -> None:
        """
        取消订阅
        
        Args:
            callback: 要取消的回调函数
        """
        if callback in self._subscribers:
            self._subscribers.remove(callback)
    
    def update_metric(self, key: str, value: Any) -> None:
        """
        更新性能指标
        
        Args:
            key: 指标名称
            value: 指标值
        """
        self._metrics[key] = value
        self.logger.debug(f"更新指标 {key}: {value}")
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取模块信息
        
        Returns:
            Dict: 模块详细信息
        """
        return {
            'module_id': self.module_id,
            'name': self.name,
            'version': self.version,
            'type': self.__class__.__name__,
            'state': self.state.value,
            'uptime': self.uptime,
            'metrics': self.metrics.copy(),
            'config': {
                'enabled': self.config.enabled,
                'priority': self.config.priority,
                'timeout': self.config.timeout,
                'retry_count': self.config.retry_count
            }
        }
    
    def _pre_start_checks(self) -> bool:
        """启动前的检查逻辑，子类可以重写"""
        return True
    
    def _do_start(self) -> bool:
        """具体的启动逻辑，子类可以重写"""
        return True
    
    def _do_stop(self) -> bool:
        """具体的停止逻辑，子类可以重写"""
        return True
    
    def _notify_state_change(self, old_state: ModuleState, new_state: ModuleState) -> None:
        """通知状态变化"""
        for callback in self._subscribers:
            try:
                callback(old_state, new_state)
            except Exception as e:
                self.logger.error(f"状态变化通知失败: {e}")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name}, {self.state.value})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ModuleManager:
    """
    模块管理器
    
    负责管理所有模块的生命周期、依赖关系和状态同步。
    """
    
    def __init__(self):
        self.modules: Dict[str, BaseModule] = {}
        self.logger = logging.getLogger(f"{__name__}.ModuleManager")
    
    def register_module(self, module: BaseModule) -> bool:
        """
        注册模块
        
        Args:
            module: 要注册的模块
            
        Returns:
            bool: 注册是否成功
        """
        try:
            self.modules[module.name] = module
            self.logger.info(f"注册模块: {module.name}")
            return True
        except Exception as e:
            self.logger.error(f"模块注册失败: {e}")
            return False
    
    def unregister_module(self, name: str) -> bool:
        """
        注销模块
        
        Args:
            name: 模块名称
            
        Returns:
            bool: 注销是否成功
        """
        if name not in self.modules:
            return False
        
        try:
            module = self.modules[name]
            if module.state == ModuleState.ACTIVE:
                module.stop()
            
            del self.modules[name]
            self.logger.info(f"注销模块: {name}")
            return True
        except Exception as e:
            self.logger.error(f"模块注销失败: {e}")
            return False
    
    def start_all(self) -> Dict[str, bool]:
        """
        启动所有模块
        
        Returns:
            Dict: 各模块启动结果
        """
        results = {}
        for name, module in self.modules.items():
            if module.config.enabled:
                results[name] = module.start()
            else:
                self.logger.info(f"跳过禁用的模块: {name}")
        return results
    
    def stop_all(self) -> Dict[str, bool]:
        """
        停止所有模块
        
        Returns:
            Dict: 各模块停止结果
        """
        results = {}
        for name, module in self.modules.items():
            results[name] = module.stop()
        return results
    
    def get_module(self, name: str) -> Optional[BaseModule]:
        """获取指定名称的模块"""
        return self.modules.get(name)
    
    def get_modules_by_type(self, module_type: ModuleType) -> List[BaseModule]:
        """根据类型获取模块列表"""
        return [m for m in self.modules.values() if isinstance(m, module_type.value)]
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统整体状态"""
        total_modules = len(self.modules)
        active_modules = len([m for m in self.modules.values() if m.state == ModuleState.ACTIVE])
        
        return {
            'total_modules': total_modules,
            'active_modules': active_modules,
            'modules': {name: module.get_info() for name, module in self.modules.items()}
        }