"""
模块化架构设计
============

实现了可扩展的模块化架构系统，支持组件注册、依赖管理、
动态加载和插件机制。提供了完整的系统构建和配置管理功能。

主要特性:
- 组件注册与发现
- 依赖注入与管理
- 动态模块加载
- 配置管理与验证
- 插件系统支持
- 事件驱动架构

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import logging
import importlib
import inspect
from typing import Dict, List, Optional, Any, Type, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import json
import os
import sys

from .base_module import BaseModule, ModuleConfig, ModuleState


class ComponentType(Enum):
    """组件类型枚举"""
    CORE = "core"                     # 核心组件
    NEURAL_NETWORK = "neural_network" # 神经网络组件
    TRAINING = "training"            # 训练组件
    DATA_PROCESSING = "data_processing" # 数据处理组件
    VISUALIZATION = "visualization"  # 可视化组件
    UTILITY = "utility"              # 工具组件
    PLUGIN = "plugin"                # 插件组件


class DependencyType(Enum):
    """依赖类型"""
    HARD = "hard"                    # 硬依赖（必须存在）
    SOFT = "soft"                    # 软依赖（可选）
    OPTIONAL = "optional"            # 可选依赖（增强功能）


@dataclass
class ComponentMetadata:
    """组件元数据"""
    name: str
    type: ComponentType
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    dependencies: Dict[str, DependencyType] = field(default_factory=dict)
    interfaces: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    lifecycle_hooks: Dict[str, str] = field(default_factory=dict)


@dataclass
class ComponentInstance:
    """组件实例"""
    metadata: ComponentMetadata
    instance: BaseModule
    state: ModuleState = ModuleState.UNINITIALIZED
    dependencies: Dict[str, 'ComponentInstance'] = field(default_factory=dict)
    initialized: bool = False
    loaded_at: float = 0.0


class InterfaceRegistry:
    """接口注册表"""
    
    def __init__(self):
        self._interfaces: Dict[str, Type] = {}
        self._implementations: Dict[str, List[str]] = defaultdict(list)
        self.logger = logging.getLogger(f"{__name__}.InterfaceRegistry")
    
    def register_interface(self, interface_name: str, interface_class: Type):
        """注册接口"""
        self._interfaces[interface_name] = interface_class
        self.logger.info(f"注册接口: {interface_name}")
    
    def register_implementation(self, interface_name: str, implementation_name: str):
        """注册接口实现"""
        if interface_name in self._interfaces:
            self._implementations[interface_name].append(implementation_name)
            self.logger.info(f"注册实现: {implementation_name} -> {interface_name}")
    
    def get_interface(self, interface_name: str) -> Optional[Type]:
        """获取接口"""
        return self._interfaces.get(interface_name)
    
    def get_implementations(self, interface_name: str) -> List[str]:
        """获取接口的所有实现"""
        return self._implementations.get(interface_name, [])
    
    def is_implementation_of(self, impl_name: str, interface_name: str) -> bool:
        """检查实现是否实现了指定接口"""
        interface_class = self.get_interface(interface_name)
        if not interface_class:
            return False
        
        # 这里需要更复杂的接口检查逻辑
        # 简化版本：检查实现是否继承自接口类
        return issubclass(type(impl_name), interface_class)


class ComponentRegistry:
    """组件注册表"""
    
    def __init__(self, interface_registry: InterfaceRegistry = None):
        self._components: Dict[str, ComponentMetadata] = {}
        self._factories: Dict[str, Callable] = {}
        self._loaded_components: Dict[str, ComponentInstance] = {}
        self.interface_registry = interface_registry or InterfaceRegistry()
        self.logger = logging.getLogger(f"{__name__}.ComponentRegistry")
        self._lock = threading.Lock()
    
    def register_component(self, metadata: ComponentMetadata, factory: Callable = None):
        """注册组件"""
        with self._lock:
            self._components[metadata.name] = metadata
            
            if factory:
                self._factories[metadata.name] = factory
            
            # 注册接口
            for interface_name in metadata.interfaces:
                self.interface_registry.register_interface(interface_name, type)
                self.interface_registry.register_implementation(interface_name, metadata.name)
            
            self.logger.info(f"注册组件: {metadata.name} (类型: {metadata.type.value})")
    
    def register_module(self, module_class: Type, config: Dict[str, Any] = None):
        """注册模块类"""
        # 从模块类提取元数据
        metadata = self._extract_metadata_from_class(module_class, config or {})
        
        # 创建工厂函数
        def factory() -> BaseModule:
            config_obj = ModuleConfig(
                name=metadata.name,
                version=metadata.version,
                parameters=metadata.config_schema
            )
            return module_class(config_obj)
        
        self.register_component(metadata, factory)
    
    def _extract_metadata_from_class(self, module_class: Type, config: Dict[str, Any]) -> ComponentMetadata:
        """从模块类提取元数据"""
        # 默认值
        name = config.get('name', module_class.__name__.lower())
        component_type = ComponentType.CORE
        version = config.get('version', "1.0.0")
        description = config.get('description', module_class.__doc__ or "")
        author = config.get('author', "Unknown")
        
        # 分析依赖关系
        dependencies = {}
        if hasattr(module_class, '__init__'):
            signature = inspect.signature(module_class.__init__)
            for param_name, param in signature.parameters.items():
                if param_name != 'self':
                    dependencies[param_name] = DependencyType.SOFT
        
        return ComponentMetadata(
            name=name,
            type=component_type,
            version=version,
            description=description,
            author=author,
            dependencies=dependencies
        )
    
    def get_component_metadata(self, name: str) -> Optional[ComponentMetadata]:
        """获取组件元数据"""
        return self._components.get(name)
    
    def list_components(self, component_type: ComponentType = None) -> List[ComponentMetadata]:
        """列出组件"""
        components = list(self._components.values())
        
        if component_type:
            components = [c for c in components if c.type == component_type]
        
        return components
    
    def create_component(self, name: str, config: ModuleConfig = None) -> Optional[BaseModule]:
        """创建组件实例"""
        with self._lock:
            if name not in self._factories:
                self.logger.error(f"组件工厂未找到: {name}")
                return None
            
            try:
                factory = self._factories[name]
                if config is None:
                    config = ModuleConfig(name)
                
                instance = factory()
                
                # 创建组件实例记录
                component_instance = ComponentInstance(
                    metadata=self._components[name],
                    instance=instance
                )
                
                self._loaded_components[name] = component_instance
                
                self.logger.info(f"创建组件实例: {name}")
                return instance
                
            except Exception as e:
                self.logger.error(f"组件创建失败 {name}: {e}")
                return None
    
    def get_component_instance(self, name: str) -> Optional[ComponentInstance]:
        """获取组件实例"""
        return self._loaded_components.get(name)


class DependencyGraph:
    """依赖图管理"""
    
    def __init__(self, registry: ComponentRegistry):
        self.registry = registry
        self._graph: Dict[str, List[str]] = defaultdict(list)
        self._reverse_graph: Dict[str, List[str]] = defaultdict(list)
        self.logger = logging.getLogger(f"{__name__}.DependencyGraph")
    
    def add_dependency(self, component_name: str, depends_on: str):
        """添加依赖关系"""
        self._graph[component_name].append(depends_on)
        self._reverse_graph[depends_on].append(component_name)
    
    def build_graph(self, component_names: List[str]):
        """从组件列表构建依赖图"""
        for component_name in component_names:
            metadata = self.registry.get_component_metadata(component_name)
            if metadata:
                for dep_name, dep_type in metadata.dependencies.items():
                    if dep_type != DependencyType.OPTIONAL:
                        self.add_dependency(component_name, dep_name)
    
    def get_loading_order(self, component_names: List[str]) -> List[str]:
        """获取组件加载顺序（拓扑排序）"""
        # 构建完整的依赖图
        self.build_graph(component_names)
        
        # 计算入度
        in_degree = {name: 0 for name in component_names}
        for component, deps in self._graph.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[component] += 1
        
        # 拓扑排序
        queue = deque([name for name, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            component = queue.popleft()
            result.append(component)
            
            # 更新相关组件的入度
            for dependent in self._reverse_graph[component]:
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        # 检查循环依赖
        if len(result) != len(component_names):
            remaining = set(component_names) - set(result)
            self.logger.error(f"检测到循环依赖: {remaining}")
            raise RuntimeError(f"循环依赖检测: {remaining}")
        
        return result
    
    def validate_dependencies(self, component_names: List[str]) -> Dict[str, List[str]]:
        """验证依赖关系"""
        missing_dependencies = {}
        
        for component_name in component_names:
            metadata = self.registry.get_component_metadata(component_name)
            if not metadata:
                continue
            
            missing = []
            for dep_name, dep_type in metadata.dependencies.items():
                if dep_type == DependencyType.HARD:
                    if dep_name not in component_names:
                        missing.append(f"{dep_name} (硬依赖)")
                elif dep_type == DependencyType.SOFT:
                    if dep_name not in component_names:
                        missing.append(f"{dep_name} (软依赖)")
            
            if missing:
                missing_dependencies[component_name] = missing
        
        return missing_dependencies


class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(f"{__name__}.ConfigurationManager")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """从文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 验证配置结构
            validated_config = self._validate_config(config)
            self.configs.update(validated_config)
            
            self.logger.info(f"配置已加载: {config_path}")
            return validated_config
            
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            return {}
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置结构"""
        # 基本验证逻辑
        validated = {}
        
        for key, value in config.items():
            if isinstance(value, dict):
                validated[key] = self._validate_config(value)
            else:
                validated[key] = value
        
        return validated
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.configs
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.configs
        
        # 创建嵌套字典结构
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, config_path: str = None):
        """保存配置到文件"""
        path = config_path or self.config_file
        if not path:
            raise ValueError("配置文件路径未指定")
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.configs, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"配置已保存: {path}")
            
        except Exception as e:
            self.logger.error(f"配置保存失败: {e}")
    
    def create_component_config(self, component_name: str, 
                               base_config: Dict[str, Any] = None) -> ModuleConfig:
        """为组件创建配置对象"""
        config_dict = self.get_config(f"components.{component_name}", {})
        
        if base_config:
            config_dict.update(base_config)
        
        return ModuleConfig(
            name=config_dict.get('name', component_name),
            version=config_dict.get('version', '1.0.0'),
            enabled=config_dict.get('enabled', True),
            priority=config_dict.get('priority', 1),
            parameters=config_dict.get('parameters', {})
        )


class EventManager:
    """事件管理器"""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = defaultdict(list)
        self._event_history: deque = deque(maxlen=1000)
        self.logger = logging.getLogger(f"{__name__}.EventManager")
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: str, callback: Callable):
        """订阅事件"""
        with self._lock:
            self._listeners[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """取消订阅"""
        with self._lock:
            if callback in self._listeners[event_type]:
                self._listeners[event_type].remove(callback)
    
    def publish(self, event_type: str, data: Any = None):
        """发布事件"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        
        with self._lock:
            # 记录历史
            self._event_history.append(event)
            
            # 通知监听器
            listeners = self._listeners[event_type].copy()
            for callback in listeners:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"事件处理器执行失败: {e}")


class ModularArchitecture(BaseModule):
    """模块化架构管理器"""
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.interface_registry = InterfaceRegistry()
        self.component_registry = ComponentRegistry(self.interface_registry)
        self.dependency_graph = DependencyGraph(self.component_registry)
        self.config_manager = ConfigurationManager()
        self.event_manager = EventManager()
        
        self._active_components: Dict[str, ComponentInstance] = {}
        self._component_lifecycle_handlers: Dict[str, Callable] = {}
        
    def load_plugin(self, plugin_path: str, plugin_config: Dict[str, Any] = None):
        """加载插件"""
        try:
            # 动态导入插件模块
            spec = importlib.util.spec_from_file_location("plugin", plugin_path)
            if spec is None:
                raise ImportError(f"无法加载插件: {plugin_path}")
            
            plugin_module = importlib.util.module_from_spec(spec)
            sys.modules["plugin"] = plugin_module
            spec.loader.exec_module(plugin_module)
            
            # 注册插件中的组件
            if hasattr(plugin_module, 'register_components'):
                plugin_module.register_components(self)
            
            self.event_manager.publish('plugin_loaded', {
                'path': plugin_path,
                'config': plugin_config
            })
            
            self.logger.info(f"插件已加载: {plugin_path}")
            
        except Exception as e:
            self.logger.error(f"插件加载失败 {plugin_path}: {e}")
            raise
    
    def build_system(self, component_list: List[str], 
                    config_path: str = None) -> bool:
        """构建系统"""
        try:
            # 加载配置
            if config_path:
                self.config_manager.load_config(config_path)
            
            # 验证依赖关系
            missing_deps = self.dependency_graph.validate_dependencies(component_list)
            if missing_deps:
                self.logger.error(f"缺少依赖: {missing_deps}")
                return False
            
            # 获取加载顺序
            loading_order = self.dependency_graph.get_loading_order(component_list)
            
            # 初始化组件
            for component_name in loading_order:
                if not self._initialize_component(component_name):
                    self.logger.error(f"组件初始化失败: {component_name}")
                    return False
            
            self.event_manager.publish('system_built', {
                'components': component_list,
                'loading_order': loading_order
            })
            
            self.logger.info(f"系统构建完成，共加载 {len(component_list)} 个组件")
            return True
            
        except Exception as e:
            self.logger.error(f"系统构建失败: {e}")
            return False
    
    def _initialize_component(self, component_name: str) -> bool:
        """初始化单个组件"""
        try:
            # 创建配置
            component_config = self.config_manager.create_component_config(component_name)
            
            # 创建组件实例
            instance = self.component_registry.create_component(component_name, component_config)
            if instance is None:
                return False
            
            # 记录组件实例
            component_instance = self.component_registry.get_component_instance(component_name)
            self._active_components[component_name] = component_instance
            
            # 初始化组件
            if not instance.initialize():
                self.logger.error(f"组件初始化失败: {component_name}")
                return False
            
            component_instance.state = ModuleState.INITIALIZED
            component_instance.initialized = True
            component_instance.loaded_at = time.time()
            
            self.event_manager.publish('component_initialized', {
                'name': component_name,
                'instance': instance
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"组件初始化异常 {component_name}: {e}")
            return False
    
    def start_system(self) -> Dict[str, bool]:
        """启动系统"""
        results = {}
        
        for component_name, component_instance in self._active_components.items():
            try:
                success = component_instance.instance.start()
                results[component_name] = success
                
                if success:
                    component_instance.state = ModuleState.ACTIVE
                else:
                    component_instance.state = ModuleState.ERROR
                    
            except Exception as e:
                self.logger.error(f"组件启动失败 {component_name}: {e}")
                results[component_name] = False
        
        active_count = sum(1 for success in results.values() if success)
        
        self.event_manager.publish('system_started', {
            'total_components': len(results),
            'active_components': active_count
        })
        
        self.logger.info(f"系统启动完成: {active_count}/{len(results)} 组件活跃")
        return results
    
    def stop_system(self) -> Dict[str, bool]:
        """停止系统"""
        results = {}
        
        # 按加载顺序的逆序停止组件
        component_names = list(self._active_components.keys())
        component_names.reverse()
        
        for component_name in component_names:
            component_instance = self._active_components[component_name]
            try:
                success = component_instance.instance.stop()
                results[component_name] = success
                
                if success:
                    component_instance.state = ModuleState.STOPPED
                    
            except Exception as e:
                self.logger.error(f"组件停止失败 {component_name}: {e}")
                results[component_name] = False
        
        self.event_manager.publish('system_stopped', results)
        
        return results
    
    def get_component(self, name: str) -> Optional[BaseModule]:
        """获取组件实例"""
        component_instance = self._active_components.get(name)
        return component_instance.instance if component_instance else None
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        component_status = {}
        for name, instance in self._active_components.items():
            component_status[name] = {
                'state': instance.state.value,
                'uptime': time.time() - instance.loaded_at,
                'info': instance.instance.get_info()
            }
        
        return {
            'total_components': len(self._active_components),
            'active_components': sum(1 for i in component_status.values() 
                                   if i['state'] == 'active'),
            'components': component_status
        }
    
    def get_registry_info(self) -> Dict[str, Any]:
        """获取注册表信息"""
        return {
            'total_registered_components': len(self.component_registry._components),
            'component_types': {
                comp_type.value: len(self.component_registry.list_components(comp_type))
                for comp_type in ComponentType
            },
            'interfaces': list(self.interface_registry._interfaces.keys())
        }
    
    def initialize(self) -> bool:
        """初始化架构管理器"""
        self.state = ModuleState.INITIALIZED
        
        # 订阅核心事件
        self.event_manager.subscribe('component_error', self._handle_component_error)
        self.event_manager.subscribe('system_shutdown', self._handle_system_shutdown)
        
        return True
    
    def cleanup(self) -> bool:
        """清理架构管理器"""
        # 停止所有组件
        self.stop_system()
        
        # 清理注册表
        self._active_components.clear()
        
        return True
    
    def _handle_component_error(self, data: Dict[str, Any]):
        """处理组件错误事件"""
        component_name = data.get('name', 'unknown')
        error = data.get('error', 'unknown error')
        self.logger.error(f"组件错误处理: {component_name} - {error}")
    
    def _handle_system_shutdown(self, data: Dict[str, Any]):
        """处理系统关闭事件"""
        self.logger.info("系统关闭事件已处理")


# 全局架构管理器实例
_global_architecture = None

def get_architecture_manager() -> ModularArchitecture:
    """获取全局架构管理器实例"""
    global _global_architecture
    if _global_architecture is None:
        config = ModuleConfig("modular_architecture", version="1.0.0")
        _global_architecture = ModularArchitecture(config)
        _global_architecture.initialize()
    return _global_architecture