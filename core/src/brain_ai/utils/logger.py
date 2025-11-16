#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日志系统工具
===========

提供统一的日志管理功能，包括:
- 多级别日志记录
- 文件和控制台输出
- 日志轮转
- 格式化输出
- 性能日志
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json

class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 添加颜色
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, '')
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)

class JSONFormatter(logging.Formatter):
    """JSON日志格式化器"""
    
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # 添加异常信息
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # 添加额外字段
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)

class Logger:
    """增强的日志记录器"""
    
    def __init__(self, name: str = "brain_ai", level: str = "INFO"):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 避免重复添加处理器
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "brain_ai.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 错误文件处理器
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / "brain_ai_errors.log",
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
    
    def debug(self, msg: str, **kwargs):
        """调试级别日志"""
        self.logger.debug(msg, extra=kwargs)
    
    def info(self, msg: str, **kwargs):
        """信息级别日志"""
        self.logger.info(msg, extra=kwargs)
    
    def warning(self, msg: str, **kwargs):
        """警告级别日志"""
        self.logger.warning(msg, extra=kwargs)
    
    def warn(self, msg: str, **kwargs):
        """警告级别日志（别名）"""
        self.warning(msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        """错误级别日志"""
        self.logger.error(msg, extra=kwargs)
    
    def critical(self, msg: str, **kwargs):
        """严重错误级别日志"""
        self.logger.critical(msg, extra=kwargs)
    
    def exception(self, msg: str, **kwargs):
        """异常日志（自动包含堆栈信息）"""
        self.logger.exception(msg, extra=kwargs)
    
    def performance(self, operation: str, duration: float, **kwargs):
        """性能日志"""
        self.info(
            f"性能监控: {operation} 耗时 {duration:.4f}s",
            operation=operation,
            duration=duration,
            **kwargs
        )
    
    def memory(self, operation: str, memory_usage: Dict[str, float], **kwargs):
        """内存使用日志"""
        memory_info = ", ".join([f"{k}: {v:.2f}MB" for k, v in memory_usage.items()])
        self.info(
            f"内存监控: {operation} - {memory_info}",
            operation=operation,
            memory_usage=memory_usage,
            **kwargs
        )
    
    def model_info(self, model_name: str, parameters: Dict[str, Any], **kwargs):
        """模型信息日志"""
        self.info(
            f"模型信息: {model_name}",
            model_name=model_name,
            model_parameters=parameters,
            **kwargs
        )
    
    def training_progress(self, epoch: int, total_epochs: int, 
                         loss: float, metrics: Optional[Dict[str, float]] = None, **kwargs):
        """训练进度日志"""
        base_msg = f"训练进度: Epoch {epoch}/{total_epochs}, Loss: {loss:.4f}"
        if metrics:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            base_msg += f", 指标: {metrics_str}"
        
        self.info(base_msg, epoch=epoch, total_epochs=total_epochs, 
                 loss=loss, metrics=metrics, **kwargs)

class PerformanceLogger:
    """性能监控日志记录器"""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger()
        self.operations = {}
    
    def start_operation(self, operation_name: str):
        """开始记录操作"""
        import time
        self.operations[operation_name] = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage()
        }
    
    def end_operation(self, operation_name: str, **kwargs):
        """结束记录操作"""
        import time
        
        if operation_name not in self.operations:
            self.logger.warning(f"未找到操作记录: {operation_name}")
            return
        
        operation = self.operations[operation_name]
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        duration = end_time - operation['start_time']
        memory_diff = {k: end_memory[k] - operation['start_memory'][k] 
                      for k in end_memory}
        
        self.logger.performance(operation_name, duration, 
                              memory_usage=memory_diff, **kwargs)
        
        del self.operations[operation_name]
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """获取内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024
            }
        except ImportError:
            return {'rss_mb': 0, 'vms_mb': 0}

# 全局日志记录器
_global_logger: Optional[Logger] = None
_performance_logger: Optional[PerformanceLogger] = None

def setup_logging(name: str = "brain_ai", 
                  level: str = "INFO",
                  log_dir: Optional[str] = None,
                  console: bool = True,
                  json_format: bool = False) -> Logger:
    """
    设置全局日志配置
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_dir: 日志目录
        console: 是否输出到控制台
        json_format: 是否使用JSON格式
        
    Returns:
        配置好的日志记录器
    """
    global _global_logger
    
    _global_logger = Logger(name, level)
    
    # 如果需要自定义日志目录
    if log_dir:
        _global_logger.logger.handlers.clear()
        
        # 重新设置处理器
        _global_logger._setup_handlers()
        
        # 更新文件路径
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # 添加自定义处理器
        formatter = JSONFormatter() if json_format else logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / f"{name}.log",
            maxBytes=10*1024*1024,
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        _global_logger.logger.addHandler(file_handler)
    
    return _global_logger

def get_logger(name: Optional[str] = None) -> Logger:
    """获取日志记录器"""
    global _global_logger
    if name:
        return Logger(name)
    if _global_logger is None:
        _global_logger = Logger()
    return _global_logger

def get_performance_logger() -> PerformanceLogger:
    """获取性能监控记录器"""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger(get_logger())
    return _performance_logger

# 装饰器：自动记录函数执行时间
def log_execution_time(logger: Optional[Logger] = None):
    """装饰器：记录函数执行时间"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            perf_logger = get_performance_logger()
            perf_logger.start_operation(func.__name__)
            try:
                result = func(*args, **kwargs)
                perf_logger.end_operation(func.__name__)
                return result
            except Exception as e:
                perf_logger.end_operation(func.__name__)
                raise
        return wrapper
    return decorator

# 上下文管理器：自动记录操作
class LogOperation:
    """操作日志上下文管理器"""
    
    def __init__(self, operation_name: str, logger: Optional[Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or get_logger()
        self.perf_logger = get_performance_logger()
    
    def __enter__(self):
        self.perf_logger.start_operation(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.perf_logger.end_operation(self.operation_name)
        if exc_type:
            self.logger.error(f"操作 {self.operation_name} 失败: {exc_val}")