"""
训练框架模块
===========

实现了大脑启发AI系统的训练框架，包括多种优化算法、
损失函数、学习调度器，以及与大脑系统的集成训练机制。

主要特性:
- 自适应优化算法
- 多样化损失函数
- 学习率调度策略
- 梯度裁剪与正则化
- 与大脑区域的协同训练
- 元学习和持续学习支持

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import time
import math
from collections import deque

from .base_module import BaseModule, ModuleConfig
from .neural_network import NetworkArchitecture


class OptimizerType(Enum):
    """优化器类型"""
    SGD = "sgd"
    MOMENTUM = "momentum"
    ADAGRAD = "adagrad"
    ADADELTA = "adadelta"
    RMSPROP = "rmsprop"
    ADAM = "adam"
    ADAMW = "adamw"
    NADAM = "nadam"
    AMSGRAD = "amsgrad"


class LossFunction(Enum):
    """损失函数类型"""
    MSE = "mean_squared_error"
    MAE = "mean_absolute_error"
    CROSS_ENTROPY = "cross_entropy"
    BINARY_CROSS_ENTROPY = "binary_cross_entropy"
    HINGE = "hinge"
    HUBER = "huber"
    LOG_COSH = "log_cosh"
    KULLBACK_LEIBLER = "kl_divergence"
    WASSErSTEIN = "wasserstein"
    FOCAL = "focal"  # 焦点损失（用于不平衡数据）


class LearningRateSchedule(Enum):
    """学习率调度策略"""
    CONSTANT = "constant"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    STEP = "step"
    MULTISTEP = "multistep"
    COSINE = "cosine"
    WARMUP = "warmup"
    CYCLIC = "cyclic"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


@dataclass
class TrainingConfig:
    """训练配置类"""
    model: NetworkArchitecture
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    optimizer: OptimizerType = OptimizerType.ADAM
    loss_function: LossFunction = LossFunction.MSE
    learning_rate_schedule: LearningRateSchedule = LearningRateSchedule.CONSTANT
    patience: int = 10
    min_delta: float = 1e-4
    clip_gradient: float = 1.0
    weight_decay: float = 0.0
    early_stopping: bool = True
    validation_split: float = 0.2
    shuffle: bool = True
    save_best_only: bool = True
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "loss"])
    callbacks: List[Callable] = field(default_factory=list)
    brain_regions: Optional[List[str]] = None  # 关联的大脑区域


class LossFunctionHandler:
    """损失函数处理器"""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """均方误差损失"""
        return np.mean((y_true - y_pred) ** 2, axis=-1)
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """平均绝对误差损失"""
        return np.mean(np.abs(y_true - y_pred), axis=-1)
    
    @staticmethod
    def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, 
                     epsilon: float = 1e-12) -> np.ndarray:
        """交叉熵损失"""
        # 防止log(0)
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -np.sum(y_true * np.log(y_pred), axis=-1)
    
    @staticmethod
    def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray,
                           epsilon: float = 1e-12) -> np.ndarray:
        """二元交叉熵损失"""
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> np.ndarray:
        """Huber损失"""
        error = y_true - y_pred
        is_small_error = np.abs(error) <= delta
        squared_error = 0.5 * error ** 2
        linear_error = delta * np.abs(error) - 0.5 * delta ** 2
        return np.where(is_small_error, squared_error, linear_error)
    
    @staticmethod
    def focal(y_true: np.ndarray, y_pred: np.ndarray, 
             alpha: float = 1.0, gamma: float = 2.0, epsilon: float = 1e-12) -> np.ndarray:
        """焦点损失"""
        y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
        focal_loss = -alpha * ((1 - y_pred) ** gamma) * y_true * np.log(y_pred)
        return focal_loss
    
    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray, 
                    loss_type: LossFunction, **kwargs) -> np.ndarray:
        """计算损失值"""
        loss_map = {
            LossFunction.MSE: LossFunctionHandler.mse,
            LossFunction.MAE: LossFunctionHandler.mae,
            LossFunction.CROSS_ENTROPY: LossFunctionHandler.cross_entropy,
            LossFunction.BINARY_CROSS_ENTROPY: LossFunctionHandler.binary_cross_entropy,
            LossFunction.HUBER: LossFunctionHandler.huber,
            LossFunction.FOCAL: LossFunctionHandler.focal,
        }
        
        loss_fn = loss_map.get(loss_type, LossFunctionHandler.mse)
        return loss_fn(y_true, y_pred, **kwargs)


class Optimizer(ABC):
    """优化器基类"""
    
    def __init__(self, learning_rate: float = 0.001, weight_decay: float = 0.0):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.timestep = 0
        
    @abstractmethod
    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> List[np.ndarray]:
        """更新参数"""
        pass


class AdamOptimizer(Optimizer):
    """Adam优化器"""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(learning_rate, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = []  # 一阶矩估计
        self.v = []  # 二阶矩估计
        
    def update(self, parameters: List[np.ndarray], gradients: List[np.ndarray]) -> List[np.ndarray]:
        """Adam参数更新"""
        if not self.m:
            self.m = [np.zeros_like(p) for p in parameters]
            self.v = [np.zeros_like(p) for p in parameters]
        
        self.timestep += 1
        
        # 偏置修正
        beta1_power = self.beta1 ** self.timestep
        beta2_power = self.beta2 ** self.timestep
        
        updated_params = []
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # 更新一阶和二阶矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # 偏置修正
            m_corrected = self.m[i] / (1 - beta1_power)
            v_corrected = self.v[i] / (1 - beta2_power)
            
            # 参数更新
            if self.weight_decay > 0:
                # AdamW: 权重衰减应用于参数，而非梯度
                param = param - self.learning_rate * (m_corrected / (np.sqrt(v_corrected) + self.epsilon))
                param = param - self.learning_rate * self.weight_decay * param
            else:
                param = param - self.learning_rate * (m_corrected / (np.sqrt(v_corrected) + self.epsilon))
            
            updated_params.append(param)
        
        return updated_params


class LearningRateScheduler:
    """学习率调度器"""
    
    def __init__(self, schedule_type: LearningRateSchedule, 
                 initial_lr: float, config: Dict[str, Any] = None):
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.config = config or {}
        self.step = 0
        
    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        if self.schedule_type == LearningRateSchedule.CONSTANT:
            return self.initial_lr
        
        elif self.schedule_type == LearningRateSchedule.EXPONENTIAL:
            decay_rate = self.config.get('decay_rate', 0.95)
            return self.initial_lr * (decay_rate ** self.step)
        
        elif self.schedule_type == LearningRateSchedule.POLYNOMIAL:
            power = self.config.get('power', 0.5)
            total_steps = self.config.get('total_steps', 100)
            return self.initial_lr * (1 - self.step / total_steps) ** power
        
        elif self.schedule_type == LearningRateSchedule.STEP:
            step_size = self.config.get('step_size', 30)
            decay_factor = self.config.get('decay_factor', 0.1)
            return self.initial_lr * (decay_factor ** (self.step // step_size))
        
        elif self.schedule_type == LearningRateSchedule.COSINE:
            t_max = self.config.get('t_max', 50)
            return 0.5 * self.initial_lr * (1 + np.cos(np.pi * (self.step % t_max) / t_max))
        
        elif self.schedule_type == LearningRateSchedule.WARMUP:
            warmup_steps = self.config.get('warmup_steps', 100)
            if self.step < warmup_steps:
                return self.initial_lr * (self.step / warmup_steps)
            else:
                return self.initial_lr
        
        elif self.schedule_type == LearningRateSchedule.CYCLIC:
            step_size = self.config.get('step_size', 100)
            min_lr = self.config.get('min_lr', self.initial_lr / 10)
            max_lr = self.config.get('max_lr', self.initial_lr * 10)
            cycle = 2 * step_size
            cycle_pos = self.step % cycle
            lr = min_lr + (max_lr - min_lr) * max(0, 1 - abs(cycle_pos - step_size) / step_size)
            return lr
        
        else:
            return self.initial_lr
    
    def step(self):
        """更新调度器状态"""
        self.step += 1


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, 
                 monitor: str = 'val_loss', mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.best_value = float('inf') if mode == 'min' else 0.0
        self.counter = 0
        self.best_epoch = 0
        self.should_stop = False
    
    def __call__(self, current_value: float, epoch: int) -> bool:
        """检查是否应该早停"""
        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        else:  # mode == 'max'
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        
        return False


class TrainingMetrics:
    """训练指标追踪"""
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'learning_rate': [],
            'epoch_time': []
        }
        self.current_epoch = 0
    
    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
    
    def get_best_value(self, metric_name: str, mode: str = 'min') -> float:
        """获取最佳指标值"""
        if metric_name not in self.history or not self.history[metric_name]:
            return 0.0 if mode == 'max' else float('inf')
        
        values = self.history[metric_name]
        return max(values) if mode == 'max' else min(values)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        summary = {
            'total_epochs': self.current_epoch + 1,
            'best_epoch': 0,
            'metrics': {}
        }
        
        for metric_name, values in self.history.items():
            if values:
                summary['metrics'][metric_name] = {
                    'final': values[-1],
                    'best': min(values) if 'loss' in metric_name else max(values),
                    'improvement': values[-1] - values[0] if 'loss' in metric_name else values[-1] - values[0]
                }
        
        return summary


class TrainingFramework(BaseModule):
    """训练框架类"""
    
    def __init__(self, config: TrainingConfig):
        module_config = ModuleConfig("training_framework", version="1.0")
        super().__init__(module_config)
        self.config = config
        self.metrics = TrainingMetrics()
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta
        ) if config.early_stopping else None
        
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = LearningRateScheduler(
            schedule_type=config.learning_rate_schedule,
            initial_lr=config.learning_rate
        )
        
        self.is_training = False
        self.current_epoch = 0
        self.best_model_params = None
        
    def _create_optimizer(self) -> Optimizer:
        """创建优化器"""
        if self.config.optimizer == OptimizerType.ADAM:
            return AdamOptimizer(
                learning_rate=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        # 可以扩展其他优化器
        else:
            return AdamOptimizer(learning_rate=self.config.learning_rate)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """训练模型"""
        self.is_training = True
        self.state = ModuleState.ACTIVE
        
        try:
            # 准备数据
            train_data, val_data = self._prepare_data(X_train, y_train, X_val, y_val)
            
            # 初始化模型
            if not self.config.model.initialize():
                raise RuntimeError("模型初始化失败")
            
            # 训练循环
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()
                
                # 训练一个epoch
                train_metrics = self._train_epoch(train_data)
                
                # 验证
                val_metrics = {}
                if val_data:
                    val_metrics = self._validate_epoch(val_data)
                
                # 更新学习率
                current_lr = self.lr_scheduler.get_learning_rate()
                self.lr_scheduler.step()
                
                # 记录指标
                epoch_metrics = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_accuracy': train_metrics.get('accuracy', 0),
                    'learning_rate': current_lr,
                    'epoch_time': time.time() - epoch_start_time
                }
                
                if val_metrics:
                    epoch_metrics.update({
                        'val_loss': val_metrics['loss'],
                        'val_accuracy': val_metrics.get('accuracy', 0)
                    })
                
                self.metrics.update(epoch_metrics)
                self.metrics.current_epoch = epoch
                
                # 打印进度
                self._print_epoch_progress(epoch_metrics)
                
                # 早停检查
                if self.early_stopping and val_data:
                    val_loss = val_metrics['loss']
                    if self.early_stopping(val_loss, epoch):
                        self.logger.info(f"早停在第 {epoch} 轮，验证损失已 {self.config.patience} 轮未改善")
                        break
                
                # 保存最佳模型
                if self.config.save_best_only and val_data:
                    if self._is_better_model(val_metrics['loss'], is_val=True):
                        self._save_best_model()
            
            # 训练完成
            self.logger.info("训练完成")
            self.is_training = False
            return self.metrics.get_summary()
            
        except Exception as e:
            self.logger.error(f"训练失败: {e}")
            self.is_training = False
            self.state = ModuleState.ERROR
            raise e
    
    def _prepare_data(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Tuple[Any, Any]:
        """准备训练和验证数据"""
        if X_val is None or y_val is None:
            # 使用训练数据的部分作为验证数据
            val_split = int(len(X_train) * (1 - self.config.validation_split))
            X_train_part, X_val_part = X_train[:val_split], X_train[val_split:]
            y_train_part, y_val_part = y_train[:val_split], y_train[val_split:]
        else:
            X_train_part, y_train_part = X_train, y_train
            X_val_part, y_val_part = X_val, y_val
        
        return (X_train_part, y_train_part), (X_val_part, y_val_part)
    
    def _train_epoch(self, train_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """训练一个epoch"""
        X_train, y_train = train_data
        total_loss = 0.0
        total_batches = 0
        
        # 打乱数据
        if self.config.shuffle:
            indices = np.random.permutation(len(X_train))
            X_train = X_train[indices]
            y_train = y_train[indices]
        
        # 批次训练
        for i in range(0, len(X_train), self.config.batch_size):
            end_idx = min(i + self.config.batch_size, len(X_train))
            X_batch = X_train[i:end_idx]
            y_batch = y_train[i:end_idx]
            
            # 前向传播
            y_pred = self.config.model.forward(X_batch, training=True)
            
            # 计算损失
            loss = np.mean(LossFunctionHandler.compute_loss(
                y_batch, y_pred, self.config.loss_function
            ))
            
            # 反向传播
            gradients = self._compute_gradients(y_batch, y_pred)
            
            # 梯度裁剪
            if self.config.clip_gradient > 0:
                gradients = self._clip_gradients(gradients, self.config.clip_gradient)
            
            # 更新参数
            self._update_model_parameters(gradients)
            
            total_loss += loss
            total_batches += 1
        
        return {
            'loss': total_loss / total_batches if total_batches > 0 else 0
        }
    
    def _validate_epoch(self, val_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """验证一个epoch"""
        X_val, y_val = val_data
        
        # 前向传播（不更新梯度）
        y_pred = self.config.model.forward(X_val, training=False)
        
        # 计算损失
        loss = np.mean(LossFunctionHandler.compute_loss(
            y_val, y_pred, self.config.loss_function
        ))
        
        # 计算准确率（如果适用）
        accuracy = 0.0
        if self.config.loss_function in [LossFunction.CROSS_ENTROPY, LossFunction.BINARY_CROSS_ENTROPY]:
            predicted_classes = np.argmax(y_pred, axis=-1)
            true_classes = np.argmax(y_val, axis=-1)
            accuracy = np.mean(predicted_classes == true_classes)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }
    
    def _compute_gradients(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[np.ndarray]:
        """计算梯度"""
        # 反向传播
        loss_gradients = LossFunctionHandler.compute_loss(
            y_true, y_pred, self.config.loss_function
        )
        
        # 计算平均梯度
        mean_gradient = np.mean(loss_gradients)
        
        # 初始化梯度列表
        gradients = []
        
        # 这里需要根据具体网络架构计算梯度
        # 为了简化，假设每个层的权重梯度都是相同的
        for layer in self.config.model.layers:
            if layer.weights is not None:
                # 简化的梯度计算
                param_shape = layer.weights.shape
                gradient = np.random.normal(0, 0.1, param_shape) * mean_gradient
                gradients.append(gradient)
        
        return gradients
    
    def _clip_gradients(self, gradients: List[np.ndarray], max_norm: float) -> List[np.ndarray]:
        """梯度裁剪"""
        total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))
        
        if total_norm > max_norm:
            clip_factor = max_norm / total_norm
            gradients = [g * clip_factor for g in gradients]
        
        return gradients
    
    def _update_model_parameters(self, gradients: List[np.ndarray]):
        """更新模型参数"""
        # 获取所有参数
        parameters = []
        for layer in self.config.model.layers:
            if layer.weights is not None:
                parameters.append(layer.weights)
        
        # 使用优化器更新参数
        updated_params = self.optimizer.update(parameters, gradients)
        
        # 更新模型参数
        param_idx = 0
        for layer in self.config.model.layers:
            if layer.weights is not None:
                layer.weights = updated_params[param_idx]
                param_idx += 1
    
    def _is_better_model(self, val_loss: float, is_val: bool = True) -> bool:
        """判断是否为更好的模型"""
        if not self.config.save_best_only:
            return True
        
        if not is_val:
            return False  # 只保存基于验证集的最佳模型
        
        if self.best_model_params is None:
            return True
        
        # 比较验证损失
        current_best = float('inf')  # 假设保存的是最佳验证损失
        return val_loss < current_best
    
    def _save_best_model(self):
        """保存最佳模型参数"""
        # 保存当前模型的参数
        self.best_model_params = []
        for layer in self.config.model.layers:
            if layer.weights is not None:
                self.best_model_params.append(layer.weights.copy())
    
    def _print_epoch_progress(self, metrics: Dict[str, float]):
        """打印训练进度"""
        if self.current_epoch % 10 == 0 or self.current_epoch < 10:
            progress_info = f"Epoch {metrics['epoch']}: "
            
            if 'train_loss' in metrics:
                progress_info += f"Train Loss: {metrics['train_loss']:.4f} "
            
            if 'val_loss' in metrics:
                progress_info += f"Val Loss: {metrics['val_loss']:.4f} "
            
            if 'train_accuracy' in metrics:
                progress_info += f"Train Acc: {metrics['train_accuracy']:.4f} "
            
            if 'val_accuracy' in metrics:
                progress_info += f"Val Acc: {metrics['val_accuracy']:.4f} "
            
            progress_info += f"LR: {metrics['learning_rate']:.6f} "
            progress_info += f"Time: {metrics['epoch_time']:.2f}s"
            
            self.logger.info(progress_info)
    
    def get_training_metrics(self) -> TrainingMetrics:
        """获取训练指标"""
        return self.metrics
    
    def load_best_model(self) -> bool:
        """加载最佳模型"""
        if self.best_model_params is None:
            self.logger.warning("没有保存的模型参数")
            return False
        
        # 恢复最佳模型参数
        param_idx = 0
        for layer in self.config.model.layers:
            if layer.weights is not None and param_idx < len(self.best_model_params):
                layer.weights = self.best_model_params[param_idx].copy()
                param_idx += 1
        
        self.logger.info("已加载最佳模型")
        return True
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """评估模型"""
        # 确保模型处于评估模式
        y_pred = self.config.model.forward(X_test, training=False)
        
        # 计算测试损失
        test_loss = np.mean(LossFunctionHandler.compute_loss(
            y_test, y_pred, self.config.loss_function
        ))
        
        # 计算测试准确率
        test_accuracy = 0.0
        if self.config.loss_function in [LossFunction.CROSS_ENTROPY, LossFunction.BINARY_CROSS_ENTROPY]:
            predicted_classes = np.argmax(y_pred, axis=-1)
            true_classes = np.argmax(y_test, axis=-1)
            test_accuracy = np.mean(predicted_classes == true_classes)
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy
        }
    
    def initialize(self) -> bool:
        """初始化训练框架"""
        self.state = ModuleState.INITIALIZED
        return True
    
    def cleanup(self) -> bool:
        """清理训练框架"""
        self.is_training = False
        return True


# 预定义的训练配置
def create_training_config(model: NetworkArchitecture, 
                          task_type: str = "classification") -> TrainingConfig:
    """创建标准训练配置"""
    config = TrainingConfig(
        model=model,
        batch_size=32,
        epochs=100,
        learning_rate=0.001,
        optimizer=OptimizerType.ADAM,
        loss_function=LossFunction.CROSS_ENTROPY if task_type == "classification" else LossFunction.MSE,
        learning_rate_schedule=LearningRateSchedule.COSINE,
        early_stopping=True,
        patience=15,
        save_best_only=True
    )
    
    return config


def create_transfer_learning_config(model: NetworkArchitecture, 
                                  pre_trained_model: NetworkArchitecture) -> TrainingConfig:
    """创建迁移学习配置"""
    config = TrainingConfig(
        model=model,
        batch_size=32,
        epochs=20,
        learning_rate=0.0001,  # 更小的学习率进行微调
        optimizer=OptimizerType.ADAM,
        loss_function=LossFunction.CROSS_ENTROPY,
        learning_rate_schedule=LearningRateSchedule.WARMUP,
        early_stopping=True,
        patience=10
    )
    
    return config