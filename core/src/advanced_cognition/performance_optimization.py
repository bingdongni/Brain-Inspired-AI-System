"""
系统性能全面调优功能
==================

实现自适应参数调节和性能监控系统，包括：
- 自适应参数调优器
- 性能监控与指标分析
- 资源管理与优化
- 多策略性能调优

主要特性：
- 实时性能监控
- 自适应参数调节
- 资源使用优化
- 多目标优化
- 自动调参算法

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import numpy as np
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import statistics
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_module import BaseModule, ModuleConfig, ModuleState


class OptimizationStrategy(Enum):
    """优化策略"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    ADAPTIVE_HYPERBAND = "adaptive_hyperband"
    HYPEROPT = "hyperopt"


class PerformanceMetric(Enum):
    """性能指标"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC = "auc"
    LOSS = "loss"
    TRAINING_TIME = "training_time"
    INFERENCE_TIME = "inference_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    GPU_USAGE = "gpu_usage"
    ENERGY_CONSUMPTION = "energy_consumption"


@dataclass
class ParameterSpace:
    """参数空间定义"""
    name: str
    param_type: str  # 'continuous', 'discrete', 'categorical'
    bounds: Tuple[float, float]  # 对于连续参数
    values: List[Any]  # 对于离散或分类参数
    description: str = ""
    
    def sample(self) -> Any:
        """从参数空间采样"""
        if self.param_type == 'continuous':
            return np.random.uniform(self.bounds[0], self.bounds[1])
        elif self.param_type == 'discrete':
            return np.random.choice(self.values)
        elif self.param_type == 'categorical':
            return np.random.choice(self.values)
        else:
            raise ValueError(f"未知参数类型: {self.param_type}")


@dataclass
class PerformanceResult:
    """性能测试结果"""
    config: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: float
    execution_time: float
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class OptimizationConfig:
    """优化配置"""
    strategy: OptimizationStrategy
    parameter_spaces: List[ParameterSpace]
    max_iterations: int = 100
    max_parallel_trials: int = 4
    objective_metric: PerformanceMetric = PerformanceMetric.ACCURACY
    objective_direction: str = "maximize"  # "maximize" or "minimize"
    early_stopping_patience: int = 10
    timeout_per_trial: int = 3600  # 秒
    save_intermediate_results: bool = True
    result_cache_size: int = 1000


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 监控数据存储
        self.metrics_history = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'gpu_usage': deque(maxlen=1000) if self._has_gpu() else None,
            'training_time': deque(maxlen=100),
            'inference_time': deque(maxlen=100),
            'accuracy': deque(maxlen=100),
            'loss': deque(maxlen=100)
        }
        
        # 监控状态
        self.is_monitoring = False
        self.monitor_thread = None
        self.start_time = None
        
        # 系统信息
        self.system_info = self._collect_system_info()
    
    def start_monitoring(self):
        """开始监控"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.start_time = time.time()
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                # 收集系统指标
                metrics = self._collect_metrics()
                
                # 存储指标
                for metric_name, value in metrics.items():
                    if metric_name in self.metrics_history:
                        self.metrics_history[metric_name].append({
                            'value': value,
                            'timestamp': time.time()
                        })
                
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
    
    def _collect_metrics(self) -> Dict[str, float]:
        """收集性能指标"""
        metrics = {}
        
        # CPU使用率
        metrics['cpu_usage'] = psutil.cpu_percent(interval=0.1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent
        
        # GPU使用率（如果可用）
        if self._has_gpu():
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # 使用第一个GPU
                    metrics['gpu_usage'] = gpu.load * 100
            except ImportError:
                metrics['gpu_usage'] = 0.0
        else:
            metrics['gpu_usage'] = 0.0
        
        return metrics
    
    def _collect_system_info(self) -> Dict[str, Any]:
        """收集系统信息"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'has_gpu': self._has_gpu(),
            'platform': psutil.os.name,
            'python_version': psutil.sys.version
        }
    
    def _has_gpu(self) -> bool:
        """检查是否有GPU"""
        try:
            import GPUtil
            return len(GPUtil.getGPUs()) > 0
        except ImportError:
            return False
    
    def get_current_metrics(self) -> Dict[str, float]:
        """获取当前指标"""
        current_metrics = {}
        for metric_name, history in self.metrics_history.items():
            if history and len(history) > 0:
                current_metrics[metric_name] = history[-1]['value']
        return current_metrics
    
    def get_metric_history(self, metric_name: str, window_size: int = 50) -> List[Dict[str, Any]]:
        """获取指标历史"""
        if metric_name not in self.metrics_history:
            return []
        
        history = list(self.metrics_history[metric_name])
        return history[-window_size:] if len(history) > window_size else history
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {
            'monitoring_duration': time.time() - self.start_time if self.start_time else 0,
            'current_metrics': self.get_current_metrics(),
            'system_info': self.system_info,
            'metric_statistics': {}
        }
        
        # 计算各指标统计信息
        for metric_name, history in self.metrics_history.items():
            if history and len(history) > 0:
                values = [item['value'] for item in history]
                summary['metric_statistics'][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1]
                }
        
        return summary


class AdaptiveParameterTuner:
    """自适应参数调优器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 优化状态
        self.trial_history: List[PerformanceResult] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        self.current_iteration = 0
        
        # 自适应参数
        self.exploration_rate = 0.1
        self.adaptation_threshold = 10
        
    def optimize(self, objective_function: Callable[[Dict[str, Any]], PerformanceResult]) -> Dict[str, Any]:
        """执行参数优化"""
        self.logger.info(f"开始参数优化，策略: {self.config.strategy.value}")
        
        if self.config.strategy == OptimizationStrategy.GRID_SEARCH:
            return self._grid_search_optimize(objective_function)
        elif self.config.strategy == OptimizationStrategy.RANDOM_SEARCH:
            return self._random_search_optimize(objective_function)
        elif self.config.strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
            return self._bayesian_optimize(objective_function)
        else:
            return self._random_search_optimize(objective_function)
    
    def _grid_search_optimize(self, objective_function: Callable[[Dict[str, Any]], PerformanceResult]) -> Dict[str, Any]:
        """网格搜索优化"""
        best_result = None
        best_score = float('-inf') if self.config.objective_direction == "maximize" else float('inf')
        
        # 生成网格搜索参数组合
        param_combinations = self._generate_grid_combinations()
        
        for config in param_combinations:
            try:
                result = self._evaluate_config(config, objective_function)
                score = self._get_objective_score(result.metrics)
                
                if self._is_better_score(score, best_score):
                    best_score = score
                    best_result = result
                    self.best_config = config
                    self.best_score = score
                
                self.trial_history.append(result)
                self.current_iteration += 1
                
            except Exception as e:
                self.logger.error(f"配置评估失败 {config}: {e}")
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'best_result': best_result,
            'total_trials': len(self.trial_history),
            'optimization_time': time.time() - (best_result.timestamp if best_result else time.time())
        }
    
    def _random_search_optimize(self, objective_function: Callable[[Dict[str, Any]], PerformanceResult]) -> Dict[str, Any]:
        """随机搜索优化"""
        best_result = None
        best_score = float('-inf') if self.config.objective_direction == "maximize" else float('inf')
        
        for iteration in range(self.config.max_iterations):
            # 生成随机配置
            config = self._generate_random_config()
            
            try:
                result = self._evaluate_config(config, objective_function)
                score = self._get_objective_score(result.metrics)
                
                if self._is_better_score(score, best_score):
                    best_score = score
                    best_result = result
                    self.best_config = config
                    self.best_score = score
                
                self.trial_history.append(result)
                self.current_iteration = iteration + 1
                
                # 自适应调整探索率
                if iteration > 0 and iteration % self.adaptation_threshold == 0:
                    self._adapt_exploration_rate()
                
            except Exception as e:
                self.logger.error(f"配置评估失败 {config}: {e}")
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'best_result': best_result,
            'total_trials': len(self.trial_history),
            'final_exploration_rate': self.exploration_rate
        }
    
    def _bayesian_optimize(self, objective_function: Callable[[Dict[str, Any]], PerformanceResult]) -> Dict[str, Any]:
        """贝叶斯优化（简化版）"""
        # 使用高斯过程进行贝叶斯优化
        # 这里实现一个简化版本，实际应用中可以集成GPyOpt等库
        
        best_result = None
        best_score = float('-inf') if self.config.objective_direction == "maximize" else float('inf')
        
        # 初始随机采样
        initial_samples = min(5, self.config.max_iterations // 4)
        
        for iteration in range(self.config.max_iterations):
            if iteration < initial_samples:
                # 随机采样
                config = self._generate_random_config()
            else:
                # 使用贝叶斯优化选择下一个配置
                config = self._bayesian_select_config()
            
            try:
                result = self._evaluate_config(config, objective_function)
                score = self._get_objective_score(result.metrics)
                
                if self._is_better_score(score, best_score):
                    best_score = score
                    best_result = result
                    self.best_config = config
                    self.best_score = score
                
                self.trial_history.append(result)
                self.current_iteration = iteration + 1
                
            except Exception as e:
                self.logger.error(f"配置评估失败 {config}: {e}")
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'best_result': best_result,
            'total_trials': len(self.trial_history)
        }
    
    def _evaluate_config(self, config: Dict[str, Any], objective_function: Callable[[Dict[str, Any]], PerformanceResult]) -> PerformanceResult:
        """评估单个配置"""
        start_time = time.time()
        
        try:
            result = objective_function(config)
            execution_time = time.time() - start_time
            
            # 创建完整的结果对象
            performance_result = PerformanceResult(
                config=config.copy(),
                metrics=result.metrics,
                timestamp=start_time,
                execution_time=execution_time,
                resource_usage=result.resource_usage if hasattr(result, 'resource_usage') else {}
            )
            
            return performance_result
            
        except Exception as e:
            return PerformanceResult(
                config=config.copy(),
                metrics={},
                timestamp=start_time,
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _generate_grid_combinations(self) -> List[Dict[str, Any]]:
        """生成网格搜索参数组合"""
        import itertools
        
        param_ranges = []
        param_names = []
        
        for param_space in self.config.parameter_spaces:
            if param_space.param_type == 'continuous':
                # 连续参数离散化
                n_samples = 5  # 网格密度
                values = np.linspace(param_space.bounds[0], param_space.bounds[1], n_samples)
            elif param_space.param_type in ['discrete', 'categorical']:
                values = param_space.values
            else:
                continue
            
            param_ranges.append(values)
            param_names.append(param_space.name)
        
        combinations = []
        for combination in itertools.product(*param_ranges):
            config = dict(zip(param_names, combination))
            combinations.append(config)
        
        return combinations
    
    def _generate_random_config(self) -> Dict[str, Any]:
        """生成随机配置"""
        config = {}
        
        for param_space in self.config.parameter_spaces:
            if np.random.random() < self.exploration_rate:
                # 探索：随机采样
                value = param_space.sample()
            else:
                # 利用：基于历史最佳值进行微调
                if self.best_config and param_space.name in self.best_config:
                    base_value = self.best_config[param_space.name]
                    if param_space.param_type == 'continuous':
                        # 在最佳值附近添加小扰动
                        perturbation = np.random.normal(0, (param_space.bounds[1] - param_space.bounds[0]) * 0.05)
                        value = base_value + perturbation
                        value = np.clip(value, param_space.bounds[0], param_space.bounds[1])
                    else:
                        value = base_value
                else:
                    value = param_space.sample()
            
            config[param_space.name] = value
        
        return config
    
    def _bayesian_select_config(self) -> Dict[str, Any]:
        """基于贝叶斯优化选择配置（简化版）"""
        # 简化实现：使用随机选择和历史信息
        return self._generate_random_config()
    
    def _adapt_exploration_rate(self):
        """自适应调整探索率"""
        if len(self.trial_history) >= 2:
            recent_improvements = 0
            for i in range(max(0, len(self.trial_history) - self.adaptation_threshold), len(self.trial_history) - 1):
                if self._is_better_score(
                    self._get_objective_score(self.trial_history[i + 1].metrics),
                    self._get_objective_score(self.trial_history[i].metrics)
                ):
                    recent_improvements += 1
            
            # 如果最近改进较少，增加探索率
            if recent_improvements < self.adaptation_threshold / 4:
                self.exploration_rate = min(0.5, self.exploration_rate * 1.2)
            else:
                # 如果最近改进较多，减少探索率
                self.exploration_rate = max(0.05, self.exploration_rate * 0.9)
    
    def _get_objective_score(self, metrics: Dict[str, float]) -> float:
        """获取目标指标分数"""
        metric_name = self.config.objective_metric.value
        if metric_name not in metrics:
            return 0.0
        
        score = metrics[metric_name]
        
        # 如果需要最小化，则取负值
        if self.config.objective_direction == "minimize":
            score = -score
        
        return score
    
    def _is_better_score(self, score1: float, score2: float) -> bool:
        """判断是否更好"""
        return score1 > score2


class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.resource_limits = {
            'max_memory_gb': 16,
            'max_cpu_percent': 80,
            'max_gpu_memory_gb': 8,
            'max_concurrent_trials': 4
        }
        
        self.current_usage = {
            'memory_gb': 0.0,
            'cpu_percent': 0.0,
            'gpu_memory_gb': 0.0,
            'active_trials': 0
        }
    
    def check_resource_availability(self, required_resources: Dict[str, float] = None) -> bool:
        """检查资源可用性"""
        if required_resources is None:
            required_resources = {'memory_gb': 1.0, 'cpu_percent': 10.0}
        
        current_metrics = self._get_current_metrics()
        
        # 检查各项资源
        checks = []
        
        if 'memory_gb' in required_resources:
            memory_usage = current_metrics.get('memory_gb', 0)
            checks.append(memory_usage + required_resources['memory_gb'] <= self.resource_limits['max_memory_gb'])
        
        if 'cpu_percent' in required_resources:
            cpu_usage = current_metrics.get('cpu_percent', 0)
            checks.append(cpu_usage + required_resources['cpu_percent'] <= self.resource_limits['max_cpu_percent'])
        
        if 'gpu_memory_gb' in required_resources:
            gpu_memory = current_metrics.get('gpu_memory_gb', 0)
            checks.append(gpu_memory + required_resources['gpu_memory_gb'] <= self.resource_limits['max_gpu_memory_gb'])
        
        return all(checks)
    
    def allocate_resources(self, required_resources: Dict[str, float]) -> str:
        """分配资源"""
        resource_id = f"resource_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        if self.check_resource_availability(required_resources):
            # 更新当前使用量
            self.current_usage['memory_gb'] += required_resources.get('memory_gb', 0)
            self.current_usage['cpu_percent'] += required_resources.get('cpu_percent', 0)
            self.current_usage['gpu_memory_gb'] += required_resources.get('gpu_memory_gb', 0)
            self.current_usage['active_trials'] += 1
            
            self.logger.info(f"分配资源 {resource_id}: {required_resources}")
            return resource_id
        else:
            raise ResourceError("资源不足，无法分配")
    
    def release_resources(self, resource_id: str, required_resources: Dict[str, float]):
        """释放资源"""
        # 更新当前使用量
        self.current_usage['memory_gb'] -= required_resources.get('memory_gb', 0)
        self.current_usage['cpu_percent'] -= required_resources.get('cpu_percent', 0)
        self.current_usage['gpu_memory_gb'] -= required_resources.get('gpu_memory_gb', 0)
        self.current_usage['active_trials'] -= 1
        
        self.logger.info(f"释放资源 {resource_id}")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """获取资源状态"""
        current_metrics = self._get_current_metrics()
        
        return {
            'current_usage': self.current_usage,
            'resource_limits': self.resource_limits,
            'current_metrics': current_metrics,
            'utilization_rates': {
                'memory': self.current_usage['memory_gb'] / self.resource_limits['max_memory_gb'],
                'cpu': self.current_usage['cpu_percent'] / self.resource_limits['max_cpu_percent'],
                'gpu_memory': self.current_usage['gpu_memory_gb'] / self.resource_limits['max_gpu_memory_gb']
            }
        }
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """获取当前系统指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # 内存使用
        memory = psutil.virtual_memory()
        memory_gb = (memory.total - memory.available) / (1024**3)  # 转换为GB
        
        # GPU内存
        gpu_memory_gb = 0.0
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_memory_gb = gpu.memoryUsed / 1024  # MB转GB
        except ImportError:
            pass
        
        return {
            'cpu_percent': cpu_percent,
            'memory_gb': memory_gb,
            'gpu_memory_gb': gpu_memory_gb
        }


class PerformanceOptimizer(BaseModule):
    """性能优化器"""
    
    def __init__(self, config: OptimizationConfig):
        module_config = ModuleConfig("performance_optimizer", version="1.0")
        super().__init__(module_config)
        self.config = config
        
        # 组件初始化
        self.monitor = PerformanceMonitor()
        self.tuner = AdaptiveParameterTuner(config)
        self.resource_manager = ResourceManager()
        
        # 优化状态
        self.optimization_results = []
        self.current_optimization = None
        
    def start_optimization(self, objective_function: Callable[[Dict[str, Any]], PerformanceResult]) -> Dict[str, Any]:
        """开始性能优化"""
        self.state = ModuleState.ACTIVE
        self.logger.info("开始性能优化")
        
        try:
            # 开始监控
            self.monitor.start_monitoring()
            
            # 执行优化
            self.current_optimization = self.tuner.optimize(objective_function)
            self.optimization_results.append(self.current_optimization)
            
            # 停止监控
            self.monitor.stop_monitoring()
            
            # 生成优化报告
            optimization_report = self._generate_optimization_report()
            
            self.state = ModuleState.COMPLETED
            self.logger.info("性能优化完成")
            
            return optimization_report
            
        except Exception as e:
            self.logger.error(f"性能优化失败: {e}")
            self.state = ModuleState.ERROR
            raise e
    
    def _generate_optimization_report(self) -> Dict[str, Any]:
        """生成优化报告"""
        if not self.current_optimization:
            return {}
        
        # 获取性能监控摘要
        monitoring_summary = self.monitor.get_performance_summary()
        
        # 获取资源状态
        resource_status = self.resource_manager.get_resource_status()
        
        report = {
            'optimization_summary': self.current_optimization,
            'performance_monitoring': monitoring_summary,
            'resource_utilization': resource_status,
            'optimization_config': {
                'strategy': self.config.strategy.value,
                'objective_metric': self.config.objective_metric.value,
                'max_iterations': self.config.max_iterations,
                'parameter_spaces': [
                    {
                        'name': p.name,
                        'type': p.param_type,
                        'bounds': p.bounds,
                        'values': p.values
                    } for p in self.config.parameter_spaces
                ]
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return self.optimization_results
    
    def get_current_performance(self) -> Dict[str, Any]:
        """获取当前性能状态"""
        return {
            'monitoring_summary': self.monitor.get_performance_summary(),
            'resource_status': self.resource_manager.get_resource_status(),
            'active_optimizations': len(self.optimization_results)
        }
    
    def initialize(self) -> bool:
        """初始化性能优化器"""
        self.state = ModuleState.INITIALIZED
        return True
    
    def cleanup(self) -> bool:
        """清理性能优化器"""
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
        return True


class ResourceError(Exception):
    """资源不足异常"""
    pass


# 便利函数
def create_optimization_config(strategy: OptimizationStrategy,
                             parameter_spaces: List[ParameterSpace],
                             **kwargs) -> OptimizationConfig:
    """创建优化配置"""
    default_config = {
        'strategy': strategy,
        'parameter_spaces': parameter_spaces,
        'max_iterations': 100,
        'objective_metric': PerformanceMetric.ACCURACY,
        'objective_direction': 'maximize'
    }
    
    default_config.update(kwargs)
    return OptimizationConfig(**default_config)


def create_neural_network_optimization_config() -> OptimizationConfig:
    """创建神经网络优化配置"""
    parameter_spaces = [
        ParameterSpace("learning_rate", "continuous", (0.0001, 0.1), [], "学习率"),
        ParameterSpace("batch_size", "discrete", (0, 0), [16, 32, 64, 128, 256], "批次大小"),
        ParameterSpace("hidden_layers", "discrete", (0, 0), [2, 3, 4, 5], "隐藏层数量"),
        ParameterSpace("neurons_per_layer", "discrete", (0, 0), [64, 128, 256, 512], "每层神经元数"),
        ParameterSpace("dropout_rate", "continuous", (0.0, 0.5), [], "Dropout率"),
        ParameterSpace("activation_function", "categorical", (0, 0), ['relu', 'tanh', 'sigmoid'], "激活函数")
    ]
    
    return create_optimization_config(
        strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
        parameter_spaces=parameter_spaces,
        max_iterations=50,
        objective_metric=PerformanceMetric.ACCURACY
    )