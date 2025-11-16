#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指标收集器
=========

提供统一的指标监控和收集功能，包括:
- 训练指标监控
- 模型性能评估
- 实时指标记录
- 指标可视化
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
import numpy as np
from datetime import datetime

class MetricsCollector:
    """指标收集器"""
    
    def __init__(self, save_path: Optional[str] = None):
        """
        初始化指标收集器
        
        Args:
            save_path: 指标保存路径
        """
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.timestamps: List[float] = []
        self.save_path = Path(save_path) if save_path else None
        self.start_time = time.time()
        
    def add_metric(self, name: str, value: float, timestamp: Optional[float] = None):
        """
        添加指标值
        
        Args:
            name: 指标名称
            value: 指标值
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()
            
        self.metrics[name].append(value)
        self.timestamps.append(timestamp)
    
    def add_metrics(self, metrics_dict: Dict[str, float], 
                   timestamp: Optional[float] = None):
        """
        批量添加指标
        
        Args:
            metrics_dict: 指标字典
            timestamp: 时间戳
        """
        for name, value in metrics_dict.items():
            self.add_metric(name, value, timestamp)
    
    def get_metric(self, name: str) -> List[float]:
        """获取指定指标的所有值"""
        return self.metrics[name]
    
    def get_latest(self, name: str) -> Optional[float]:
        """获取最新指标值"""
        values = self.metrics[name]
        return values[-1] if values else None
    
    def get_average(self, name: str, window: Optional[int] = None) -> Optional[float]:
        """获取指标平均值"""
        values = self.metrics[name]
        if not values:
            return None
            
        if window is not None:
            values = values[-window:]
            
        return np.mean(values)
    
    def get_statistics(self, name: str) -> Dict[str, float]:
        """获取指标统计信息"""
        values = self.metrics[name]
        if not values:
            return {}
            
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'count': len(values)
        }
    
    def save(self, filepath: Optional[str] = None):
        """保存指标到文件"""
        save_path = Path(filepath) if filepath else self.save_path
        if not save_path:
            return
            
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metrics': dict(self.metrics),
            'timestamps': self.timestamps,
            'start_time': self.start_time,
            'end_time': time.time(),
            'total_duration': time.time() - self.start_time
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str):
        """从文件加载指标"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.metrics = defaultdict(list, data['metrics'])
        self.timestamps = data['timestamps']
        self.start_time = data['start_time']
    
    def reset(self):
        """重置所有指标"""
        self.metrics.clear()
        self.timestamps.clear()
        self.start_time = time.time()

class TrainingMetrics:
    """训练指标监控"""
    
    def __init__(self):
        self.epoch_metrics = []
        self.current_epoch = 0
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """记录一轮训练"""
        epoch_data = {
            'epoch': epoch,
            'timestamp': time.time(),
            'metrics': metrics.copy()
        }
        self.epoch_metrics.append(epoch_data)
        self.current_epoch = epoch
    
    def get_best_epoch(self, metric_name: str, mode: str = 'max') -> Optional[Dict]:
        """获取最佳轮次"""
        if not self.epoch_metrics:
            return None
            
        best_epoch = None
        best_value = float('-inf') if mode == 'max' else float('inf')
        
        for epoch_data in self.epoch_metrics:
            value = epoch_data['metrics'].get(metric_name)
            if value is not None:
                if (mode == 'max' and value > best_value) or \
                   (mode == 'min' and value < best_value):
                    best_value = value
                    best_epoch = epoch_data
        
        return best_epoch
    
    def get_latest_metrics(self) -> Optional[Dict[str, float]]:
        """获取最新训练指标"""
        if not self.epoch_metrics:
            return None
        return self.epoch_metrics[-1]['metrics'].copy()

class ModelMetrics:
    """模型性能指标"""
    
    @staticmethod
    def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算准确率"""
        return float(np.mean(predictions == targets))
    
    @staticmethod
    def precision(predictions: np.ndarray, targets: np.ndarray, 
                  average: str = 'macro') -> float:
        """计算精确率"""
        from sklearn.metrics import precision_score
        return precision_score(targets, predictions, average=average, zero_division=0)
    
    @staticmethod
    def recall(predictions: np.ndarray, targets: np.ndarray, 
               average: str = 'macro') -> float:
        """计算召回率"""
        from sklearn.metrics import recall_score
        return recall_score(targets, predictions, average=average, zero_division=0)
    
    @staticmethod
    def f1_score(predictions: np.ndarray, targets: np.ndarray, 
                 average: str = 'macro') -> float:
        """计算F1分数"""
        from sklearn.metrics import f1_score
        return f1_score(targets, predictions, average=average, zero_division=0)
    
    @staticmethod
    def confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算混淆矩阵"""
        from sklearn.metrics import confusion_matrix
        return confusion_matrix(targets, predictions)
    
    @staticmethod
    def classification_report(predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """生成分类报告"""
        from sklearn.metrics import classification_report
        report = classification_report(targets, predictions, output_dict=True, zero_division=0)
        return report

class MemoryMetrics:
    """内存使用指标"""
    
    def __init__(self):
        self.usage_history = []
    
    def get_current_usage(self) -> Dict[str, float]:
        """获取当前内存使用情况"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent()
            }
        except ImportError:
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0}
    
    def record_usage(self):
        """记录当前内存使用"""
        usage = self.get_current_usage()
        usage['timestamp'] = time.time()
        self.usage_history.append(usage)
    
    def get_peak_usage(self) -> Dict[str, float]:
        """获取峰值内存使用"""
        if not self.usage_history:
            return {}
            
        peak_rss = max(self.usage_history, key=lambda x: x['rss_mb'])
        return peak_rss
    
    def get_average_usage(self) -> Dict[str, float]:
        """获取平均内存使用"""
        if not self.usage_history:
            return {}
            
        avg_rss = np.mean([u['rss_mb'] for u in self.usage_history])
        avg_vms = np.mean([u['vms_mb'] for u in self.usage_history])
        avg_percent = np.mean([u['percent'] for u in self.usage_history])
        
        return {
            'rss_mb': avg_rss,
            'vms_mb': avg_vms,
            'percent': avg_percent
        }

class TimeMetrics:
    """时间性能指标"""
    
    def __init__(self):
        self.timers = {}
        self.history = defaultdict(list)
    
    def start_timer(self, name: str):
        """开始计时"""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """结束计时并记录"""
        if name not in self.timers:
            raise ValueError(f"计时器 {name} 未开始")
            
        duration = time.time() - self.timers[name]
        self.history[name].append(duration)
        del self.timers[name]
        return duration
    
    def get_total_time(self, name: str) -> float:
        """获取总时间"""
        return sum(self.history[name])
    
    def get_average_time(self, name: str) -> float:
        """获取平均时间"""
        times = self.history[name]
        return np.mean(times) if times else 0
    
    def get_speedup(self, name: str, baseline_time: float) -> float:
        """获取加速比"""
        avg_time = self.get_average_time(name)
        return baseline_time / avg_time if avg_time > 0 else float('inf')

# 装饰器：自动记录函数执行时间
def time_function(logger_func: Optional[Callable] = None):
    """装饰器：自动记录函数执行时间"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                if logger_func:
                    logger_func(f"{func.__name__} 执行时间: {duration:.4f}s")
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                if logger_func:
                    logger_func(f"{func.__name__} 执行失败，耗时: {duration:.4f}s, 错误: {e}")
                raise
        return wrapper
    return decorator

# 上下文管理器：自动计时
class Timer:
    """计时上下文管理器"""
    
    def __init__(self, name: str, logger_func: Optional[Callable] = None):
        self.name = name
        self.logger_func = logger_func
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if self.logger_func:
            self.logger_func(f"{self.name} 耗时: {duration:.4f}s")

# 全局指标收集器实例
_global_metrics_collector: Optional[MetricsCollector] = None

def get_global_metrics() -> MetricsCollector:
    """获取全局指标收集器"""
    global _global_metrics_collector
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    return _global_metrics_collector

def set_global_metrics(collector: MetricsCollector):
    """设置全局指标收集器"""
    global _global_metrics_collector
    _global_metrics_collector = collector