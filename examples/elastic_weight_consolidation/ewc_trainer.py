"""
EWC训练器模块

整合Fisher信息矩阵计算、EWC损失函数和智能权重保护的完整训练系统。
实现端到端的持续学习训练流程。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from collections import defaultdict
import time
import os

from .fisher_matrix import FisherInformationMatrix, EfficientFisherComputation
from .ewc_loss import EWCLossFunction, AdaptiveEWC, ProportionalEWC
from .intelligent_protection import IntelligentWeightProtection, HierarchicalProtection

logger = logging.getLogger(__name__)


class EWCTrainer:
    """
    弹性权重巩固训练器
    
    完整的EWC训练系统，集成了Fisher计算、损失函数和智能保护功能。
    支持多种训练模式和自适应策略。
    """
    
    def __init__(self,
                 model: nn.Module,
                 base_criterion: Optional[nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 device: Optional[torch.device] = None,
                 fisher_mode: str = 'diagonal',
                 ewc_variant: str = 'standard',
                 use_intelligent_protection: bool = True):
        """
        初始化EWC训练器
        
        Args:
            model: 神经网络模型
            base_criterion: 基础损失函数
            optimizer: 优化器
            device: 计算设备
            fisher_mode: Fisher矩阵计算模式
            ewc_variant: EWC变体 ('standard', 'adaptive', 'proportional')
            use_intelligent_protection: 是否使用智能保护
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 基础组件
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(model.parameters(), lr=0.001)
        
        # Fisher计算器
        self.fisher_computer = FisherInformationMatrix(
            model, mode=fisher_mode, stable_numeric=True)
        
        # EWC损失函数
        if ewc_variant == 'standard':
            self.ewc_loss = EWCLossFunction(self.base_criterion)
        elif ewc_variant == 'adaptive':
            self.ewc_loss = AdaptiveEWC(self.base_criterion)
        elif ewc_variant == 'proportional':
            self.ewc_loss = ProportionalEWC(self.base_criterion)
        else:
            raise ValueError(f"不支持的EWC变体: {ewc_variant}")
        
        # 智能保护系统
        if use_intelligent_protection:
            self.intelligent_protection = IntelligentWeightProtection(model)
            self.hierarchical_protection = HierarchicalProtection(self.intelligent_protection)
        else:
            self.intelligent_protection = None
            self.hierarchical_protection = None
        
        # 训练状态
        self.current_task = 0
        self.task_history = []
        self.best_performance = {}
        
        # 统计信息
        self.training_stats = defaultdict(list)
        self.memory_usage = defaultdict(float)
        
        logger.info(f"初始化EWC训练器，设备: {self.device}, EWC变体: {ewc_variant}")
    
    def compute_fisher_matrix(self,
                            data_loader: DataLoader,
                            num_samples: Optional[int] = None,
                            method: str = 'empirical') -> Dict[str, torch.Tensor]:
        """
        计算Fisher信息矩阵
        
        Args:
            data_loader: 数据加载器
            num_samples: 使用的样本数量
            method: 计算方法 ('empirical', 'diagonal', 'monte_carlo')
            
        Returns:
            Fisher信息矩阵字典
        """
        logger.info(f"开始计算Fisher矩阵，方法: {method}")
        
        start_time = time.time()
        
        if method == 'empirical':
            fisher_dict = self.fisher_computer.compute_fisher_empirical(
                data_loader, num_samples, self.device)
        elif method == 'diagonal':
            fisher_dict = self.fisher_computer.compute_fisher_diagonal(
                data_loader, num_samples, self.device)
        elif method == 'monte_carlo':
            efficient_computer = EfficientFisherComputation(self.model)
            fisher_dict = efficient_computer.monte_carlo_fisher(
                data_loader, num_mc_samples=10)
        else:
            raise ValueError(f"不支持的Fisher计算方法: {method}")
        
        computation_time = time.time() - start_time
        
        # 更新统计信息
        self.training_stats['fisher_computation_time'].append(computation_time)
        self.training_stats['fisher_memory_usage'].append(self._estimate_memory_usage(fisher_dict))
        
        # 保存当前任务的Fisher矩阵
        self.task_history.append({
            'task_id': self.current_task,
            'fisher_dict': fisher_dict,
            'computation_method': method,
            'num_samples': num_samples,
            'computation_time': computation_time
        })
        
        logger.info(f"Fisher矩阵计算完成，耗时: {computation_time:.2f}s")
        return fisher_dict
    
    def train_task(self,
                  train_loader: DataLoader,
                  val_loader: Optional[DataLoader] = None,
                  task_id: Optional[int] = None,
                  num_epochs: int = 50,
                  lambda_ewc: float = 1000.0,
                  save_checkpoints: bool = True,
                  checkpoint_dir: Optional[str] = None) -> Dict[str, List[float]]:
        """
        训练单个任务
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            task_id: 任务ID
            num_epochs: 训练轮次
            lambda_ewc: EWC权重系数
            save_checkpoints: 是否保存检查点
            checkpoint_dir: 检查点保存目录
            
        Returns:
            训练历史字典
        """
        if task_id is not None:
            self.current_task = task_id
        
        logger.info(f"开始训练任务 {self.current_task}, 轮次: {num_epochs}")
        
        # 获取旧参数（如果有的话）
        old_params = self._get_old_parameters()
        
        # 计算Fisher矩阵（除了第一个任务）
        if self.current_task > 0:
            fisher_dict = self.compute_fisher_matrix(train_loader)
            self._setup_ewc_loss(fisher_dict, old_params, lambda_ewc)
        else:
            fisher_dict = None
        
        # 训练循环
        train_history = self._training_loop(
            train_loader, val_loader, num_epochs, fisher_dict, old_params)
        
        # 更新任务状态
        self._update_task_state(train_history, val_loader)
        
        # 保存检查点
        if save_checkpoints:
            self._save_checkpoint(checkpoint_dir)
        
        # 完成当前任务
        self.current_task += 1
        
        logger.info(f"任务 {self.current_task-1} 训练完成")
        return train_history
    
    def _training_loop(self,
                      train_loader: DataLoader,
                      val_loader: Optional[DataLoader],
                      num_epochs: int,
                      fisher_dict: Optional[Dict[str, torch.Tensor]],
                      old_params: Optional[Dict[str, torch.Tensor]]) -> Dict[str, List[float]]:
        """
        训练循环
        """
        self.model.train()
        
        train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'ewc_loss': [],
            'base_loss': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # 训练阶段
            train_metrics = self._train_epoch(train_loader, fisher_dict, old_params, epoch, num_epochs)
            
            # 验证阶段
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader)
                val_acc = val_metrics['accuracy']
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                train_history['val_loss'].append(val_metrics['loss'])
                train_history['val_acc'].append(val_acc)
            
            # 记录训练指标
            train_history['train_loss'].append(train_metrics['loss'])
            train_history['train_acc'].append(train_metrics['accuracy'])
            train_history['ewc_loss'].append(train_metrics.get('ewc_loss', 0.0))
            train_history['base_loss'].append(train_metrics.get('base_loss', 0.0))
            
            # 计算训练时间
            epoch_time = time.time() - epoch_start
            self.training_stats['epoch_time'].append(epoch_time)
            
            # 打印进度
            if epoch % 10 == 0:
                log_msg = f"Epoch {epoch}/{num_epochs} - "
                log_msg += f"Train Loss: {train_metrics['loss']:.4f}, "
                log_msg += f"Train Acc: {train_metrics['accuracy']:.4f}"
                
                if val_loader is not None:
                    log_msg += f", Val Loss: {val_metrics['loss']:.4f}, "
                    log_msg += f"Val Acc: {val_metrics['accuracy']:.4f}"
                
                if 'ewc_loss' in train_metrics:
                    log_msg += f", EWC Loss: {train_metrics['ewc_loss']:.4f}"
                
                log_msg += f", Time: {epoch_time:.2f}s"
                
                logger.info(log_msg)
        
        return train_history
    
    def _train_epoch(self,
                    train_loader: DataLoader,
                    fisher_dict: Optional[Dict[str, torch.Tensor]],
                    old_params: Optional[Dict[str, torch.Tensor]],
                    epoch: int,
                    num_epochs: int) -> Dict[str, float]:
        """
        单个训练轮次
        """
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        ewc_loss_total = 0.0
        base_loss_total = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失
            if fisher_dict and old_params and hasattr(self.ewc_loss, 'forward'):
                # EWC训练
                if isinstance(self.ewc_loss, AdaptiveEWC):
                    loss, loss_info = self.ewc_loss(outputs, targets, self._get_current_parameters())
                    ewc_loss_total += loss_info.get('ewc_loss', 0.0)
                    base_loss_total += loss_info.get('base_loss', 0.0)
                elif isinstance(self.ewc_loss, ProportionalEWC):
                    loss, loss_info = self.ewc_loss(
                        outputs, targets, self._get_current_parameters(), old_params, fisher_dict)
                    ewc_loss_total += loss_info.get('ewc_loss', 0.0)
                    base_loss_total += loss_info.get('base_loss', 0.0)
                else:
                    # 标准EWC
                    current_params = self._get_current_parameters()
                    loss, loss_info = self.ewc_loss(outputs, targets, current_params, epoch)
                    ewc_loss_total += loss_info.get('ewc_loss', 0.0)
                    base_loss_total += loss_info.get('base_loss', 0.0)
            else:
                # 标准训练
                loss = self.base_criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            total_correct += pred.eq(targets.view_as(pred)).sum().item()
            total_samples += targets.size(0)
        
        # 计算平均指标
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        
        metrics = {
            'loss': avg_loss,
            'accuracy': avg_acc
        }
        
        if ewc_loss_total > 0:
            metrics['ewc_loss'] = ewc_loss_total / len(train_loader)
            metrics['base_loss'] = base_loss_total / len(train_loader)
        
        return metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证轮次
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.base_criterion(outputs, targets)
                
                total_loss += loss.item()
                pred = outputs.argmax(dim=1, keepdim=True)
                total_correct += pred.eq(targets.view_as(pred)).sum().item()
                total_samples += targets.size(0)
        
        self.model.train()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': total_correct / total_samples
        }
    
    def evaluate_all_tasks(self,
                          task_data_loaders: Dict[int, DataLoader],
                          compute_forgetting: bool = True) -> Dict[str, float]:
        """
        评估所有任务的性能
        
        Args:
            task_data_loaders: 各任务的数据加载器字典
            compute_forgetting: 是否计算遗忘度量
            
        Returns:
            评估结果字典
        """
        results = {}
        self.model.eval()
        
        for task_id, data_loader in task_data_loaders.items():
            metrics = self._validate_epoch(data_loader)
            results[f'task_{task_id}'] = metrics
            
            # 保存最佳性能
            if task_id not in self.best_performance or metrics['accuracy'] > self.best_performance[task_id]:
                self.best_performance[task_id] = metrics['accuracy']
        
        # 计算遗忘度量
        if compute_forgetting and len(self.best_performance) > 1:
            forgetting = self._compute_forgetting_metric(results)
            results['forgetting'] = forgetting
        
        # 计算平均性能
        all_accuracies = [metrics['accuracy'] for metrics in results.values() if isinstance(metrics, dict)]
        results['average_accuracy'] = np.mean(all_accuracies)
        
        return results
    
    def _compute_forgetting_metric(self, current_results: Dict[str, float]) -> float:
        """
        计算遗忘度量
        """
        total_forgetting = 0.0
        num_tasks = 0
        
        for task_id in range(len(current_results) - 1):  # 减去遗忘度量本身
            task_key = f'task_{task_id}'
            if task_key in current_results and task_id in self.best_performance:
                current_acc = current_results[task_key]['accuracy']
                best_acc = self.best_performance[task_id]
                forgetting = best_acc - current_acc
                total_forgetting += forgetting
                num_tasks += 1
        
        return total_forgetting / max(num_tasks, 1)
    
    def _get_old_parameters(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        获取上一个任务的参数
        """
        if not self.task_history:
            return None
        
        # 获取最后一个任务的参数
        last_task = self.task_history[-1]
        old_state_dict = last_task.get('model_state', None)
        
        if old_state_dict:
            old_params = {}
            for name, param in old_state_dict.items():
                old_params[name] = param.clone()
            return old_params
        
        return None
    
    def _get_current_parameters(self) -> Dict[str, torch.Tensor]:
        """
        获取当前模型参数
        """
        current_params = {}
        for name, param in self.model.named_parameters():
            current_params[name] = param.detach().clone()
        return current_params
    
    def _setup_ewc_loss(self,
                       fisher_dict: Dict[str, torch.Tensor],
                       old_params: Optional[Dict[str, torch.Tensor]],
                       lambda_ewc: float) -> None:
        """
        设置EWC损失函数
        """
        if hasattr(self.ewc_loss, 'set_fisher_matrix'):
            self.ewc_loss.set_fisher_matrix(fisher_dict)
        
        if hasattr(self.ewc_loss, 'set_old_parameters') and old_params:
            self.ewc_loss.set_old_parameters(old_params)
        
        if hasattr(self.ewc_loss, 'lambda_ewc'):
            self.ewc_loss.lambda_ewc = lambda_ewc
    
    def _update_task_state(self,
                          train_history: Dict[str, List[float]],
                          val_loader: Optional[DataLoader]) -> None:
        """
        更新任务状态
        """
        # 保存模型状态
        if self.task_history:
            self.task_history[-1]['model_state'] = {
                name: param.cpu().clone() for name, param in self.model.named_parameters()
            }
        
        # 更新智能保护系统
        if self.intelligent_protection:
            final_train_acc = train_history['train_acc'][-1]
            self.intelligent_protection.performance_history.append(final_train_acc)
        
        # 更新统计信息
        self.training_stats['task_completion_time'].append(time.time())
    
    def _estimate_memory_usage(self, fisher_dict: Dict[str, torch.Tensor]) -> float:
        """
        估计内存使用量
        """
        total_memory = 0
        for fisher in fisher_dict.values():
            total_memory += fisher.numel() * 4  # 假设float32，4字节
        return total_memory / (1024 * 1024)  # 转换为MB
    
    def _save_checkpoint(self, checkpoint_dir: Optional[str]) -> None:
        """
        保存检查点
        """
        if checkpoint_dir is None:
            return
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_task': self.current_task,
            'task_history': self.task_history,
            'training_stats': dict(self.training_stats),
            'best_performance': self.best_performance
        }
        
        checkpoint_path = os.path.join(
            checkpoint_dir, f'ewc_checkpoint_task_{self.current_task}.pt')
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_task = checkpoint['current_task']
        self.task_history = checkpoint['task_history']
        self.training_stats = defaultdict(list, checkpoint['training_stats'])
        self.best_performance = checkpoint['best_performance']
        
        logger.info(f"检查点已加载: {checkpoint_path}")
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型和EWC状态
        
        Args:
            filepath: 保存路径
        """
        save_data = {
            'model_state_dict': self.model.state_dict(),
            'fisher_matrices': [task.get('fisher_dict', {}) for task in self.task_history],
            'task_history': self.task_history,
            'training_config': {
                'fisher_mode': self.fisher_computer.mode,
                'ewc_variant': type(self.ewc_loss).__name__
            }
        }
        
        torch.save(save_data, filepath)
        logger.info(f"模型已保存: {filepath}")
    
    def get_training_statistics(self) -> Dict[str, Union[float, int, List]]:
        """
        获取训练统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_tasks_completed': self.current_task,
            'total_training_time': sum(self.training_stats.get('epoch_time', [])),
            'average_epoch_time': np.mean(self.training_stats.get('epoch_time', [0])),
            'total_forgetting': self._compute_forgetting_metric(
                {f'task_{i}': {'accuracy': acc} for i, acc in self.best_performance.items()}),
            'average_accuracy': np.mean(list(self.best_performance.values())) if self.best_performance else 0.0,
            'task_history_length': len(self.task_history),
            'fisher_computation_overhead': np.mean(self.training_stats.get('fisher_computation_time', [0])),
            'training_stats': dict(self.training_stats),
            'best_performance': self.best_performance
        }
        
        if self.intelligent_protection:
            protection_stats = self.intelligent_protection.get_protection_statistics()
            stats['intelligent_protection'] = protection_stats
        
        return stats


class ContinualLearner:
    """
    持续学习器（高级接口）
    
    提供简化的持续学习接口，自动处理任务管理和EWC训练。
    """
    
    def __init__(self,
                 model: nn.Module,
                 tasks: List[Dict],
                 base_criterion: Optional[nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 device: Optional[torch.device] = None,
                 ewc_config: Optional[Dict] = None):
        """
        初始化持续学习器
        
        Args:
            model: 神经网络模型
            tasks: 任务列表，每个任务包含数据加载器等信息
            base_criterion: 基础损失函数
            optimizer: 优化器
            device: 计算设备
            ewc_config: EWC配置字典
        """
        self.model = model
        self.tasks = tasks
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # EWC配置
        default_ewc_config = {
            'fisher_mode': 'diagonal',
            'ewc_variant': 'standard',
            'use_intelligent_protection': True,
            'lambda_ewc': 1000.0
        }
        default_ewc_config.update(ewc_config or {})
        
        # 创建EWC训练器
        self.trainer = EWCTrainer(
            model=model,
            base_criterion=base_criterion,
            optimizer=optimizer,
            device=device,
            **default_ewc_config
        )
        
        self.results = {}
    
    def learn_all_tasks(self,
                       eval_all_tasks: bool = True,
                       save_checkpoints: bool = True) -> Dict[str, Dict]:
        """
        学习所有任务
        
        Args:
            eval_all_tasks: 是否在每个任务后评估所有任务
            save_checkpoints: 是否保存检查点
            
        Returns:
            所有任务的训练结果
        """
        logger.info(f"开始持续学习，共 {len(self.tasks)} 个任务")
        
        for task_idx, task_config in enumerate(self.tasks):
            logger.info(f"学习任务 {task_idx + 1}/{len(self.tasks)}")
            
            # 训练当前任务
            train_history = self.trainer.train_task(
                train_loader=task_config['train_loader'],
                val_loader=task_config.get('val_loader'),
                task_id=task_idx,
                num_epochs=task_config.get('num_epochs', 50),
                lambda_ewc=task_config.get('lambda_ewc', 1000.0),
                save_checkpoints=save_checkpoints
            )
            
            # 评估所有任务（如果需要）
            if eval_all_tasks:
                # 构建所有任务的数据加载器
                all_data_loaders = {}
                for prev_idx in range(task_idx + 1):
                    all_data_loaders[prev_idx] = self.tasks[prev_idx]['test_loader']
                
                # 评估
                eval_results = self.trainer.evaluate_all_tasks(all_data_loaders)
                self.results[task_idx] = eval_results
                
                logger.info(f"任务 {task_idx} 评估完成，平均准确率: {eval_results['average_accuracy']:.4f}")
            else:
                # 只评估当前任务
                current_test_results = self.trainer._validate_epoch(
                    task_config['test_loader'])
                self.results[task_idx] = current_test_results
        
        logger.info("持续学习完成")
        return self.results
    
    def get_final_results(self) -> Dict[str, float]:
        """
        获取最终结果
        
        Returns:
            最终结果字典
        """
        if not self.results:
            return {}
        
        # 计算最终统计
        final_accuracies = []
        forgetting_scores = []
        
        for task_results in self.results.values():
            if isinstance(task_results, dict):
                if 'average_accuracy' in task_results:
                    final_accuracies.append(task_results['average_accuracy'])
                elif 'accuracy' in task_results:
                    final_accuracies.append(task_results['accuracy'])
                
                if 'forgetting' in task_results:
                    forgetting_scores.append(task_results['forgetting'])
        
        final_results = {
            'final_average_accuracy': np.mean(final_accuracies) if final_accuracies else 0.0,
            'accuracy_std': np.std(final_accuracies) if final_accuracies else 0.0,
            'total_forgetting': np.mean(forgetting_scores) if forgetting_scores else 0.0,
            'task_count': len(self.results),
            'training_statistics': self.trainer.get_training_statistics()
        }
        
        return final_results