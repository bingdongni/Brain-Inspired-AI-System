"""
渐进式训练器模块

整合渐进式神经网络、动态容量增长和新任务学习的完整训练系统。
实现端到端的动态网络扩展训练流程。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from collections import defaultdict, deque
import time
import os

from .progressive_neural_network import ProgressiveNeuralNetwork, AdaptiveProgressiveNetwork
from .dynamic_capacity_growth import DynamicCapacityGrowth
from .new_task_learning import NewTaskLearner

logger = logging.getLogger(__name__)


class ProgressiveTrainer:
    """
    渐进式训练器
    
    完整的动态网络扩展训练系统，集成了所有渐进式学习组件。
    """
    
    def __init__(self,
                 base_network: nn.Module,
                 network_type: str = 'progressive',
                 trainer_config: Optional[Dict] = None):
        """
        初始化渐进式训练器
        
        Args:
            base_network: 基础网络架构
            network_type: 网络类型 ('progressive', 'adaptive_progressive')
            trainer_config: 训练器配置
        """
        self.base_network = base_network
        self.network_type = network_type
        self.trainer_config = trainer_config or {}
        
        # 默认配置
        default_config = {
            'max_tasks': 10,
            'capacity_growth_enabled': True,
            'task_learning_enabled': True,
            'auto_expansion': True,
            'expansion_threshold': 0.1,
            'save_checkpoints': True,
            'evaluation_frequency': 5
        }
        default_config.update(self.trainer_config)
        self.trainer_config = default_config
        
        # 初始化网络架构
        if network_type == 'progressive':
            self.network = ProgressiveNeuralNetwork(
                base_network=base_network,
                max_tasks=default_config['max_tasks']
            )
        elif network_type == 'adaptive_progressive':
            self.network = AdaptiveProgressiveNetwork(
                base_network=base_network,
                max_tasks=default_config['max_tasks']
            )
        else:
            raise ValueError(f"不支持的网络类型: {network_type}")
        
        # 初始化容量增长系统
        self.capacity_growth = None
        if default_config['capacity_growth_enabled']:
            self.capacity_growth = DynamicCapacityGrowth(base_network)
        
        # 初始化新任务学习器
        self.task_learner = None
        if default_config['task_learning_enabled']:
            self.task_learner = NewTaskLearner(
                network=self.network,
                config={
                    'enable_fast_adaptation': True,
                    'enable_transfer_learning': True,
                    'enable_curriculum_learning': True
                }
            )
        
        # 训练状态
        self.current_task = 0
        self.completed_tasks = []
        self.task_history = []
        self.training_stats = defaultdict(list)
        
        # 性能监控
        self.performance_history = deque(maxlen=100)
        self.capacity_history = deque(maxlen=100)
        self.expansion_history = []
        
        # 优化器
        self.optimizer = None
        self.scheduler = None
        
        logger.info(f"渐进式训练器初始化完成，网络类型: {network_type}")
    
    def setup_optimizer(self,
                       optimizer_type: str = 'adam',
                       learning_rate: float = 0.001,
                       weight_decay: float = 1e-4,
                       scheduler_type: Optional[str] = None) -> None:
        """
        设置优化器
        
        Args:
            optimizer_type: 优化器类型
            learning_rate: 学习率
            weight_decay: 权重衰减
            scheduler_type: 调度器类型
        """
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
        elif optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
        
        # 设置学习率调度器
        if scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        elif scheduler_type == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)
        
        logger.info(f"设置优化器: {optimizer_type}, 学习率: {learning_rate}")
    
    def add_task(self,
                task_data: Dict[str, DataLoader],
                task_id: Optional[int] = None,
                learning_mode: str = 'auto') -> int:
        """
        添加新任务
        
        Args:
            task_data: 任务数据字典
            task_id: 任务ID，None表示自动分配
            learning_mode: 学习模式
            
        Returns:
            任务ID
        """
        if task_id is None:
            task_id = self.current_task
        
        logger.info(f"添加任务 {task_id}，学习模式: {learning_mode}")
        
        # 添加新的网络列（如果是渐进式网络）
        if isinstance(self.network, ProgressiveNeuralNetwork):
            # 推断输出维度
            output_dim = self._infer_output_dim(task_data)
            task_column_id = self.network.add_new_task(output_dim=output_dim)
            assert task_column_id == task_id, f"任务ID不匹配: {task_column_id} != {task_id}"
        
        # 设置学习器任务列表
        if self.task_learner and hasattr(self.task_learner, 'curriculum_learner'):
            # 设置课程学习器的任务列表
            tasks = []
            for name, dataloader in task_data.items():
                task_info = {
                    'name': name,
                    'dataloader': dataloader,
                    'sample_complexity': 1.0  # 可根据任务调整
                }
                tasks.append(task_info)
            
            self.task_learner.curriculum_learner.tasks = tasks
        
        self.current_task += 1
        return task_id
    
    def _infer_output_dim(self, task_data: Dict[str, DataLoader]) -> int:
        """推断输出维度"""
        # 尝试从数据加载器推断
        for name, dataloader in task_data.items():
            if hasattr(dataloader, 'dataset') and hasattr(dataloader.dataset, 'targets'):
                # 分类任务
                targets = dataloader.dataset.targets
                if isinstance(targets, torch.Tensor):
                    return len(torch.unique(targets))
                else:
                    return len(set(targets))
        
        # 默认值
        return 10
    
    def train_task(self,
                  task_id: int,
                  train_loader: DataLoader,
                  val_loader: Optional[DataLoader] = None,
                  num_epochs: int = 50,
                  learning_mode: str = 'auto',
                  save_intermediate: bool = True) -> Dict[str, Any]:
        """
        训练单个任务
        
        Args:
            task_id: 任务ID
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮次
            learning_mode: 学习模式
            save_intermediate: 是否保存中间结果
            
        Returns:
            训练结果字典
        """
        logger.info(f"开始训练任务 {task_id}，轮次: {num_epochs}")
        
        training_start = time.time()
        
        # 获取任务数据
        task_data = {'train': train_loader, 'val': val_loader}
        
        # 准备标签（这里简化处理）
        # 实际实现中需要根据具体数据格式调整
        sample_batch = next(iter(train_loader))
        if isinstance(sample_batch, (list, tuple)):
            sample_data, sample_labels = sample_batch
        else:
            sample_data = sample_batch
            sample_labels = torch.zeros(sample_data.size(0), dtype=torch.long)
        
        # 选择学习模式
        if learning_mode == 'auto':
            learning_mode = self._select_learning_mode(task_id)
        
        # 执行训练
        training_results = {}
        
        if learning_mode == 'progressive' and isinstance(self.network, ProgressiveNeuralNetwork):
            # 渐进式训练
            training_results = self._train_progressive_task(
                task_id, train_loader, val_loader, num_epochs
            )
        elif learning_mode == 'capacity_growth' and self.capacity_growth:
            # 容量增长训练
            training_results = self._train_capacity_growth_task(
                task_id, train_loader, val_loader, num_epochs
            )
        elif learning_mode == 'fast_adaptation' and self.task_learner:
            # 快速适应训练
            training_results = self._train_fast_adaptation_task(
                task_id, task_data, sample_labels
            )
        else:
            # 标准训练
            training_results = self._train_standard_task(
                task_id, train_loader, val_loader, num_epochs
            )
        
        training_time = time.time() - training_start
        
        # 评估任务性能
        if val_loader is not None:
            final_performance = self._evaluate_task(task_id, val_loader)
        else:
            final_performance = training_results.get('final_accuracy', 0.0)
        
        # 检查是否需要容量扩展
        expansion_decision = None
        if self.trainer_config['auto_expansion'] and self.capacity_growth:
            expansion_decision = self.capacity_growth.adapt_capacity(
                current_performance=final_performance,
                task_complexity=1.0  # 可根据任务调整
            )
            
            if expansion_decision.get('needs_expansion', False):
                logger.info("触发容量扩展")
                self.network = expansion_decision['expanded_network']
        
        # 记录任务历史
        task_record = {
            'task_id': task_id,
            'learning_mode': learning_mode,
            'training_results': training_results,
            'final_performance': final_performance,
            'training_time': training_time,
            'expansion_decision': expansion_decision,
            'num_epochs': num_epochs,
            'timestamp': time.time()
        }
        
        self.task_history.append(task_record)
        self.completed_tasks.append(task_id)
        
        # 更新性能历史
        self.performance_history.append(final_performance)
        
        # 更新容量历史
        if hasattr(self.network, 'get_memory_usage'):
            memory_usage = self.network.get_memory_usage()
            total_params = memory_usage['total_parameters']
            self.capacity_history.append(total_params)
        
        logger.info(f"任务 {task_id} 训练完成，耗时: {training_time:.2f}s，性能: {final_performance:.4f}")
        
        return task_record
    
    def _select_learning_mode(self, task_id: int) -> str:
        """选择学习模式"""
        if task_id == 0:
            return 'standard'  # 第一个任务使用标准训练
        elif isinstance(self.network, ProgressiveNeuralNetwork):
            return 'progressive'  # 使用渐进式训练
        elif self.capacity_growth:
            return 'capacity_growth'  # 使用容量增长
        elif self.task_learner:
            return 'fast_adaptation'  # 使用快速适应
        else:
            return 'standard'
    
    def _train_progressive_task(self,
                              task_id: int,
                              train_loader: DataLoader,
                              val_loader: Optional[DataLoader],
                              num_epochs: int) -> Dict[str, Any]:
        """
        训练渐进式任务
        
        Args:
            task_id: 任务ID
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮次
            
        Returns:
            训练结果
        """
        if not isinstance(self.network, ProgressiveNeuralNetwork):
            raise ValueError("网络不是渐进式网络")
        
        # 确保优化器已设置
        if self.optimizer is None:
            self.setup_optimizer()
        
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # 训练阶段
            train_metrics = self._train_progressive_epoch(task_id, train_loader)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_acc'].append(train_metrics['accuracy'])
            
            # 验证阶段
            if val_loader is not None:
                val_metrics = self._validate_progressive_epoch(task_id, val_loader)
                training_history['val_loss'].append(val_metrics['loss'])
                training_history['val_acc'].append(val_metrics['accuracy'])
                
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            if epoch % 10 == 0:
                log_msg = f"Progressive Task {task_id}, Epoch {epoch}/{num_epochs} - "
                log_msg += f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}"
                
                if val_loader is not None:
                    log_msg += f", Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
                
                log_msg += f", Time: {epoch_time:.2f}s"
                
                logger.info(log_msg)
        
        return {
            'training_history': training_history,
            'best_val_accuracy': best_val_acc,
            'final_accuracy': best_val_acc if val_loader is not None else training_history['train_acc'][-1]
        }
    
    def _train_progressive_epoch(self, task_id: int, train_loader: DataLoader) -> Dict[str, float]:
        """训练渐进式网络轮次"""
        self.network.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to('cpu'), targets.to('cpu')  # 简化处理
            
            self.optimizer.zero_grad()
            
            # 渐进式前向传播
            outputs = self.network(inputs, task_id=task_id)
            
            # 计算损失
            loss = nn.functional.cross_entropy(outputs, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples
        }
    
    def _validate_progressive_epoch(self, task_id: int, val_loader: DataLoader) -> Dict[str, float]:
        """验证渐进式网络轮次"""
        self.network.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to('cpu'), targets.to('cpu')
                
                outputs = self.network(inputs, task_id=task_id)
                loss = nn.functional.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
        
        self.network.train()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': total_correct / total_samples
        }
    
    def _train_capacity_growth_task(self,
                                   task_id: int,
                                   train_loader: DataLoader,
                                   val_loader: Optional[DataLoader],
                                   num_epochs: int) -> Dict[str, Any]:
        """训练容量增长任务"""
        if self.capacity_growth is None:
            raise ValueError("容量增长系统未初始化")
        
        # 使用容量增长系统进行训练
        training_results = {
            'expansion_history': [],
            'final_network_size': 0,
            'performance_progression': []
        }
        
        # 简化的训练过程
        for epoch in range(num_epochs):
            # 训练网络
            self.network.train()
            epoch_loss = 0.0
            
            for inputs, targets in train_loader:
                self.optimizer.zero_grad()
                outputs = self.network(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            # 评估性能
            if val_loader is not None:
                accuracy = self._evaluate_task(task_id, val_loader)
                training_results['performance_progression'].append(accuracy)
                
                # 检查是否需要扩展
                adaptation_result = self.capacity_growth.adapt_capacity(accuracy)
                if adaptation_result.get('needs_expansion', False):
                    training_results['expansion_history'].append(adaptation_result)
        
        training_results['final_network_size'] = sum(p.numel() for p in self.network.parameters())
        
        return training_results
    
    def _train_fast_adaptation_task(self,
                                   task_id: int,
                                   task_data: Dict[str, Any],
                                   task_labels: torch.Tensor) -> Dict[str, Any]:
        """训练快速适应任务"""
        if self.task_learner is None:
            raise ValueError("任务学习器未初始化")
        
        # 准备数据格式
        formatted_data = {
            'train': task_data['train'],
            'test': task_data.get('val', task_data['train'])
        }
        
        # 执行快速适应学习
        result = self.task_learner.learn_new_task(
            formatted_data, task_labels, task_id, learning_mode='fast_adaptation'
        )
        
        return result['learning_results']
    
    def _train_standard_task(self,
                           task_id: int,
                           train_loader: DataLoader,
                           val_loader: Optional[DataLoader],
                           num_epochs: int) -> Dict[str, Any]:
        """训练标准任务"""
        if self.optimizer is None:
            self.setup_optimizer()
        
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # 训练
            train_metrics = self._train_standard_epoch(train_loader)
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['train_acc'].append(train_metrics['accuracy'])
            
            # 验证
            if val_loader is not None:
                val_metrics = self._validate_standard_epoch(val_loader)
                training_history['val_loss'].append(val_metrics['loss'])
                training_history['val_acc'].append(val_metrics['accuracy'])
                
                best_val_acc = max(best_val_acc, val_metrics['accuracy'])
        
        return {
            'training_history': training_history,
            'best_val_accuracy': best_val_acc,
            'final_accuracy': best_val_acc if val_loader is not None else training_history['train_acc'][-1]
        }
    
    def _train_standard_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练标准网络轮次"""
        self.network.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to('cpu'), targets.to('cpu')
            
            self.optimizer.zero_grad()
            outputs = self.network(inputs)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total_correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': total_correct / total_samples
        }
    
    def _validate_standard_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证标准网络轮次"""
        self.network.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to('cpu'), targets.to('cpu')
                
                outputs = self.network(inputs)
                loss = nn.functional.cross_entropy(outputs, targets)
                
                total_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
        
        self.network.train()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': total_correct / total_samples
        }
    
    def _evaluate_task(self, task_id: int, data_loader: DataLoader) -> float:
        """评估任务性能"""
        self.network.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to('cpu'), targets.to('cpu')
                
                if isinstance(self.network, ProgressiveNeuralNetwork):
                    outputs = self.network(inputs, task_id=task_id)
                else:
                    outputs = self.network(inputs)
                
                predicted = outputs.argmax(dim=1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
        
        self.network.train()
        
        return total_correct / max(total_samples, 1)
    
    def evaluate_all_tasks(self, task_data_loaders: Dict[int, DataLoader]) -> Dict[str, float]:
        """评估所有任务"""
        results = {}
        
        for task_id, data_loader in task_data_loaders.items():
            accuracy = self._evaluate_task(task_id, data_loader)
            results[f'task_{task_id}'] = accuracy
        
        # 计算总体指标
        all_accuracies = list(results.values())
        results['average_accuracy'] = np.mean(all_accuracies)
        results['accuracy_std'] = np.std(all_accuracies)
        results['forgetting_measure'] = self._calculate_forgetting_measure()
        
        return results
    
    def _calculate_forgetting_measure(self) -> float:
        """计算遗忘度量"""
        if len(self.performance_history) < 2:
            return 0.0
        
        # 简化的遗忘度量：性能下降程度
        recent_performance = list(self.performance_history)[-5:]
        peak_performance = max(self.performance_history)
        
        forgetting = peak_performance - np.mean(recent_performance)
        return max(0.0, forgetting)
    
    def train_continual_learning_sequence(self,
                                        task_sequence: List[Dict],
                                        save_directory: Optional[str] = None) -> Dict[str, Any]:
        """
        训练持续学习序列
        
        Args:
            task_sequence: 任务序列
            save_directory: 保存目录
            
        Returns:
            完整训练结果
        """
        logger.info(f"开始持续学习序列，共 {len(task_sequence)} 个任务")
        
        sequence_start = time.time()
        sequence_results = {
            'task_results': [],
            'final_evaluation': {},
            'training_statistics': {},
            'sequence_time': 0.0
        }
        
        for task_idx, task_info in enumerate(task_sequence):
            task_id = task_info['task_id']
            train_loader = task_info['train_loader']
            val_loader = task_info.get('val_loader')
            num_epochs = task_info.get('num_epochs', 50)
            
            # 添加任务
            self.add_task({'train': train_loader, 'val': val_loader}, task_id)
            
            # 训练任务
            task_result = self.train_task(
                task_id, train_loader, val_loader, num_epochs
            )
            
            sequence_results['task_results'].append(task_result)
            
            # 中间评估
            if task_idx % self.trainer_config['evaluation_frequency'] == 0:
                logger.info(f"执行中间评估，任务 {task_idx + 1}")
        
        # 最终评估
        task_data_loaders = {}
        for task_info in task_sequence:
            if 'val_loader' in task_info and task_info['val_loader']:
                task_data_loaders[task_info['task_id']] = task_info['val_loader']
            else:
                task_data_loaders[task_info['task_id']] = task_info['train_loader']
        
        final_evaluation = self.evaluate_all_tasks(task_data_loaders)
        sequence_results['final_evaluation'] = final_evaluation
        
        # 训练统计
        training_stats = self.get_training_statistics()
        sequence_results['training_statistics'] = training_stats
        
        sequence_results['sequence_time'] = time.time() - sequence_start
        
        # 保存结果
        if save_directory:
            self.save_training_results(sequence_results, save_directory)
        
        logger.info(f"持续学习序列完成，总耗时: {sequence_results['sequence_time']:.2f}s")
        
        return sequence_results
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """获取训练统计信息"""
        stats = {
            'completed_tasks': len(self.completed_tasks),
            'current_task': self.current_task,
            'network_type': self.network_type,
            'total_training_time': sum(h.get('training_time', 0) for h in self.task_history),
            'average_task_time': np.mean([h.get('training_time', 0) for h in self.task_history]) if self.task_history else 0.0,
            'performance_progression': list(self.performance_history),
            'capacity_progression': list(self.capacity_history),
            'expansion_history': self.expansion_history,
            'forgetting_measure': self._calculate_forgetting_measure(),
            'network_statistics': self._get_network_statistics()
        }
        
        # 添加组件统计
        if self.capacity_growth:
            stats['capacity_growth'] = self.capacity_growth.get_growth_statistics()
        
        if self.task_learner:
            stats['task_learning'] = self.task_learner.get_learning_statistics()
        
        # 添加网络内存使用情况
        if hasattr(self.network, 'get_memory_usage'):
            stats['memory_usage'] = self.network.get_memory_usage()
        
        return stats
    
    def _get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        network_stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'parameter_efficiency': trainable_params / max(total_params, 1)
        }
        
        if isinstance(self.network, ProgressiveNeuralNetwork):
            network_stats.update({
                'num_columns': self.network.num_columns,
                'lateral_connections': len(self.network.lateral_connections),
                'task_specific_heads': len(self.network.task_heads)
            })
        
        return network_stats
    
    def save_training_results(self, results: Dict[str, Any], save_dir: str) -> None:
        """保存训练结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存网络模型
        model_path = os.path.join(save_dir, 'progressive_network.pt')
        if hasattr(self.network, 'save_checkpoint'):
            self.network.save_checkpoint(model_path)
        else:
            torch.save(self.network.state_dict(), model_path)
        
        # 保存训练结果
        results_path = os.path.join(save_dir, 'training_results.pt')
        torch.save(results, results_path)
        
        # 保存配置
        config_path = os.path.join(save_dir, 'training_config.pt')
        config_data = {
            'trainer_config': self.trainer_config,
            'network_type': self.network_type,
            'base_network_config': str(self.base_network)
        }
        torch.save(config_data, config_path)
        
        logger.info(f"训练结果已保存到: {save_dir}")
    
    def load_training_results(self, save_dir: str) -> None:
        """加载训练结果"""
        model_path = os.path.join(save_dir, 'progressive_network.pt')
        
        if hasattr(self.network, 'load_checkpoint'):
            self.network.load_checkpoint(model_path)
        else:
            self.network.load_state_dict(torch.load(model_path))
        
        logger.info(f"训练结果已从 {save_dir} 加载")
    
    def get_continual_learning_metrics(self) -> Dict[str, float]:
        """获取持续学习指标"""
        if not self.task_history:
            return {}
        
        # 计算各种持续学习指标
        final_accuracies = [h.get('final_performance', 0) for h in self.task_history]
        
        # 平均精度
        avg_accuracy = np.mean(final_accuracies)
        
        # 最终精度
        final_accuracy = final_accuracies[-1] if final_accuracies else 0.0
        
        # 前向迁移
        forward_transfer = 0.0
        if len(final_accuracies) > 1:
            early_performance = np.mean(final_accuracies[:3]) if len(final_accuracies) >= 3 else final_accuracies[0]
            later_performance = np.mean(final_accuracies[3:]) if len(final_accuracies) > 3 else final_accuracies[-1]
            forward_transfer = later_performance - early_performance
        
        # 遗忘度量
        forgetting = self._calculate_forgetting_measure()
        
        # 学习曲线平滑度
        smoothness = np.std(np.diff(final_accuracies)) if len(final_accuracies) > 1 else 0.0
        
        # 样本效率
        sample_efficiency = avg_accuracy / max(sum(h.get('training_time', 1) for h in self.task_history), 1)
        
        return {
            'average_accuracy': avg_accuracy,
            'final_accuracy': final_accuracy,
            'forward_transfer': forward_transfer,
            'backward_transfer': -forgetting,  # 负的遗忘就是后向迁移
            'forgetting_measure': forgetting,
            'learning_stability': 1.0 / (1.0 + smoothness),
            'sample_efficiency': sample_efficiency,
            'total_tasks': len(self.task_history)
        }