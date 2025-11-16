#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理工具
===========

提供统一的配置管理功能，支持:
- YAML/JSON配置文件
- 环境变量覆盖
- 配置验证
- 动态配置更新
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """模型配置"""
    type: str = "brain_system"
    hidden_size: int = 512
    num_layers: int = 6
    dropout: float = 0.1
    activation: str = "relu"
    
@dataclass
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"
    scheduler: str = "cosine"
    early_stopping: bool = True
    patience: int = 10
    
@dataclass
class SystemConfig:
    """系统配置"""
    device: str = "cpu"
    num_workers: int = 4
    seed: int = 42
    debug: bool = False
    logging_level: str = "INFO"
    save_frequency: int = 10
    
class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._original_config: Dict[str, Any] = {}
        
        # 加载配置
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
        else:
            self._load_default_config()
    
    def _load_default_config(self):
        """加载默认配置"""
        self._config = {
            "model": asdict(ModelConfig()),
            "training": asdict(TrainingConfig()),
            "system": asdict(SystemConfig()),
            "paths": {
                "data_dir": "./data",
                "model_dir": "./models", 
                "log_dir": "./logs",
                "output_dir": "./output"
            }
        }
        self._original_config = self._config.copy()
        logger.info("已加载默认配置")
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        从文件加载配置
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    loaded_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
            
            # 合并配置
            self._update_config(loaded_config)
            self.config_path = config_path
            
            logger.info(f"配置文件加载成功: {config_path}")
            return self._config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def _update_config(self, new_config: Dict[str, Any]):
        """递归更新配置"""
        def update_dict(base: Dict, update: Dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    update_dict(base[key], value)
                else:
                    base[key] = value
        
        update_dict(self._config, new_config)
    
    def save_config(self, output_path: Union[str, Path], format: str = 'yaml') -> None:
        """
        保存配置到文件
        
        Args:
            output_path: 输出路径
            format: 文件格式 ('yaml' 或 'json')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(self._config, f, default_flow_style=False, 
                             allow_unicode=True, indent=2)
                elif format.lower() == 'json':
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的格式: {format}")
            
            logger.info(f"配置已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._config.copy()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键，支持点号分隔的嵌套键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        # 导航到最后一级的父级
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # 设置值
        config[keys[-1]] = value
        logger.debug(f"配置已更新: {key} = {value}")
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        批量更新配置
        
        Args:
            config_dict: 配置字典
        """
        self._update_config(config_dict)
        logger.info("配置已批量更新")
    
    def reset(self) -> None:
        """重置为默认配置"""
        self._config = self._original_config.copy()
        logger.info("配置已重置")
    
    def validate(self) -> bool:
        """
        验证配置的有效性
        
        Returns:
            配置是否有效
        """
        try:
            # 验证必需的配置项
            required_keys = ['model', 'training', 'system']
            for key in required_keys:
                if key not in self._config:
                    logger.error(f"缺少必需的配置项: {key}")
                    return False
            
            # 验证数据类型
            model_config = self._config.get('model', {})
            if not isinstance(model_config.get('hidden_size'), int) or model_config.get('hidden_size', 0) <= 0:
                logger.error("模型隐藏层大小必须是正整数")
                return False
            
            training_config = self._config.get('training', {})
            if not isinstance(training_config.get('learning_rate'), (int, float)) or training_config.get('learning_rate', 0) <= 0:
                logger.error("学习率必须是正数")
                return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False
    
    def apply_environment_overrides(self) -> None:
        """应用环境变量覆盖"""
        env_prefix = "BRAIN_AI_"
        
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace('_', '.')
                
                # 类型转换
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif self._is_float(value):
                    value = float(value)
                
                self.set(config_key, value)
                logger.debug(f"环境变量覆盖: {config_key} = {value}")
    
    @staticmethod
    def _is_float(value: str) -> bool:
        """检查字符串是否为浮点数"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def __repr__(self) -> str:
        return f"ConfigManager(config_path={self.config_path})"
    
    def __str__(self) -> str:
        return f"ConfigManager with {len(self._config)} sections"

# 全局配置管理器实例
_global_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    return _global_config_manager

def set_global_config_manager(config_manager: ConfigManager):
    """设置全局配置管理器"""
    global _global_config_manager
    _global_config_manager = config_manager