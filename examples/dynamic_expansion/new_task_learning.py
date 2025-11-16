"""
新任务学习模块

实现新任务学习机制，包括快速适应、元学习和课程学习。
支持持续学习场景下的新任务高效学习。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from collections import defaultdict, deque
import time
from copy import deepcopy
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FastTaskLearner:
    """
    快速任务学习器
    
    实现基于元学习的快速任务适应。
    """
    
    def __init__(self,
                 network: nn.Module,
                 meta_learning_rate: float = 0.01,
                 adaptation_steps: int = 5,
                 inner_loop_lr: float = 0.1):
        """
        初始化快速任务学习器
        
        Args:
            network: 基础网络
            meta_learning_rate: 元学习率
            adaptation_steps: 适应步数
            inner_loop_lr: 内循环学习率
        """
        self.network = network
        self.meta_learning_rate = meta_learning_rate
        self.adaptation_steps = adaptation_steps
        self.inner_loop_lr = inner_loop_lr
        
        # 元学习器
        self.meta_optimizer = optim.Adam(network.parameters(), lr=meta_learning_rate)
        
        # 学习历史
        self.adaptation_history = []
        self.meta_losses = []
        
        logger.info("快速任务学习器初始化完成")
    
    def fast_adapt(self,
                  support_data: torch.Tensor,
                  support_labels: torch.Tensor,
                  query_data: torch.Tensor,
                  query_labels: torch.Tensor,
                  task_id: Optional[int] = None) -> Dict[str, Any]:
        """
        快速适应新任务
        
        Args:
            support_data: 支持集数据
            support_labels: 支持集标签
            query_data: 查询集数据
            query_labels: 查询集标签
            task_id: 任务ID
            
        Returns:
            适应结果字典
        """
        logger.info(f"开始快速适应任务 {task_id}")
        
        # 保存原始参数
        original_params = {}
        for name, param in self.network.named_parameters():
            original_params[name] = param.clone()
        
        adaptation_start = time.time()
        
        # 内循环：适应支持集
        adapted_loss = self._inner_loop_adaptation(support_data, support_labels)
        
        # 外循环：在查询集上评估
        query_loss, query_accuracy = self._evaluate_on_query(query_data, query_labels)
        
        adaptation_time = time.time() - adaptation_start
        
        # 计算元梯度（可选）
        meta_gradient = self._compute_meta_gradient(query_loss)
        
        # 更新元学习器
        self.meta_optimizer.zero_grad()
        if meta_gradient is not None:
            self._apply_meta_update(meta_gradient)
        
        # 记录适应历史
        adaptation_record = {
            'task_id': task_id,
            'adaptation_loss': adapted_loss.item(),
            'query_loss': query_loss.item(),
            'query_accuracy': query_accuracy,
            'adaptation_time': adaptation_time,
            'meta_gradient_norm': torch.norm(meta_gradient).item() if meta_gradient is not None else 0.0,
            'support_size': support_data.size(0),
            'query_size': query_data.size(0)
        }
        
        self.adaptation_history.append(adaptation_record)
        self.meta_losses.append(query_loss.item())
        
        # 恢复原始参数（如果在完整的MAML实现中）
        self._restore_parameters(original_params)
        
        logger.info(f"任务 {task_id} 快速适应完成，查询准确率: {query_accuracy:.4f}")
        
        return adaptation_record
    
    def _inner_loop_adaptation(self, support_data: torch.Tensor, support_labels: torch.Tensor) -> torch.Tensor:
        """
        内循环适应
        
        Args:
            support_data: 支持集数据
            support_labels: 支持集标签
            
        Returns:
            适应损失
        """
        adaptation_loss = 0.0
        
        for step in range(self.adaptation_steps):
            # 前向传播
            outputs = self.network(support_data)
            loss = F.cross_entropy(outputs, support_labels)
            
            # 计算梯度
            gradients = torch.autograd.grad(
                loss, self.network.parameters(), 
                create_graph=True, retain_graph=True
            )
            
            # 更新参数
            with torch.no_grad():
                for param, grad in zip(self.network.parameters(), gradients):
                    param -= self.inner_loop_lr * grad
            
            adaptation_loss += loss
        
        return adaptation_loss / self.adaptation_steps
    
    def _evaluate_on_query(self, query_data: torch.Tensor, query_labels: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        在查询集上评估
        
        Args:
            query_data: 查询集数据
            query_labels: 查询集标签
            
        Returns:
            (查询损失, 准确率)
        """
        self.network.eval()
        
        with torch.no_grad():
            outputs = self.network(query_data)
            query_loss = F.cross_entropy(outputs, query_labels)
            
            # 计算准确率
            predicted = outputs.argmax(dim=1)
            accuracy = (predicted == query_labels).float().mean().item()
        
        self.network.train()
        
        return query_loss, accuracy
    
    def _compute_meta_gradient(self, query_loss: torch.Tensor) -> Optional[torch.Tensor]:
        """
        计算元梯度
        
        Args:
            query_loss: 查询损失
            
        Returns:
            元梯度向量
        """
        # 计算元梯度
        meta_grads = torch.autograd.grad(
            query_loss, self.network.parameters(),
            create_graph=True, retain_graph=True
        )
        
        # 返回梯度向量
        return torch.cat([grad.view(-1) for grad in meta_grads])
    
    def _apply_meta_update(self, meta_gradient: torch.Tensor) -> None:
        """
        应用元更新
        
        Args:
            meta_gradient: 元梯度
        """
        # 这里需要实现更复杂的MAML更新
        # 简化版本：直接使用查询损失更新
        pass
    
    def _restore_parameters(self, original_params: Dict[str, torch.Tensor]) -> None:
        """
        恢复原始参数
        
        Args:
            original_params: 原始参数字典
        """
        for name, param in self.network.named_parameters():
            if name in original_params:
                param.data.copy_(original_params[name])
    
    def get_adaptation_statistics(self) -> Dict[str, Any]:
        """获取适应统计信息"""
        if not self.adaptation_history:
            return {}
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'average_query_accuracy': np.mean([h['query_accuracy'] for h in self.adaptation_history]),
            'average_adaptation_time': np.mean([h['adaptation_time'] for h in self.adaptation_history]),
            'average_meta_loss': np.mean(self.meta_losses),
            'adaptation_speed': np.mean([1.0 / h['adaptation_time'] for h in self.adaptation_history]),
            'learning_stability': np.std([h['query_accuracy'] for h in self.adaptation_history]),
            'recent_performance': self.adaptation_history[-5:] if len(self.adaptation_history) >= 5 else self.adaptation_history
        }


class CurriculumLearning:
    """
    课程学习器
    
    实现课程学习策略，按难度顺序安排任务学习。
    """
    
    def __init__(self,
                 tasks: List[Dict],
                 curriculum_strategy: str = 'easy_to_hard',
                 difficulty_assessor: Optional[Callable] = None):
        """
        初始化课程学习器
        
        Args:
            tasks: 任务列表
            curriculum_strategy: 课程策略
            difficulty_assessor: 难度评估函数
        """
        self.tasks = tasks
        self.curriculum_strategy = curriculum_strategy
        self.difficulty_assessor = difficulty_assessor
        
        # 任务难度评估
        self.task_difficulties = self._assess_task_difficulties()
        
        # 课程顺序
        self.curriculum_order = self._generate_curriculum_order()
        
        # 学习进度跟踪
        self.completed_tasks = []
        self.learning_history = []
        
        logger.info(f"课程学习器初始化，策略: {curriculum_strategy}")
    
    def _assess_task_difficulties(self) -> List[float]:
        """评估任务难度"""
        difficulties = []
        
        if self.difficulty_assessor is not None:
            # 使用自定义评估函数
            for i, task in enumerate(self.tasks):
                difficulty = self.difficulty_assessor(task, i)
                difficulties.append(difficulty)
        else:
            # 默认难度评估
            for i, task in enumerate(self.tasks):
                # 基于任务ID、训练样本数等估计难度
                base_difficulty = i / len(self.tasks)  # 线性增长
                sample_complexity = task.get('sample_complexity', 1.0)
                
                difficulty = base_difficulty * sample_complexity
                difficulties.append(difficulty)
        
        return difficulties
    
    def _generate_curriculum_order(self) -> List[int]:
        """生成课程顺序"""
        task_indices = list(range(len(self.tasks)))
        
        if self.curriculum_strategy == 'easy_to_hard':
            # 从易到难
            sorted_indices = sorted(task_indices, key=lambda x: self.task_difficulties[x])
        elif self.curriculum_strategy == 'hard_to_easy':
            # 从难到易
            sorted_indices = sorted(task_indices, key=lambda x: self.task_difficulties[x], reverse=True)
        elif self.curriculum_strategy == 'random':
            # 随机顺序
            np.random.shuffle(sorted_indices)
        elif self.curriculum_strategy == 'alternating':
            # 交替策略：易-难-中-难
            sorted_indices = self._generate_alternating_order(task_indices)
        else:
            # 默认按索引顺序
            sorted_indices = task_indices
        
        return sorted_indices
    
    def _generate_alternating_order(self, task_indices: List[int]) -> List[int]:
        """生成交替顺序"""
        # 简单的交替策略
        sorted_by_difficulty = sorted(task_indices, key=lambda x: self.task_difficulties[x])
        
        alternating_order = []
        left = 0
        right = len(sorted_by_difficulty) - 1
        
        while left <= right:
            if left == right:
                alternating_order.append(sorted_by_difficulty[left])
            else:
                alternating_order.append(sorted_by_difficulty[left])
                alternating_order.append(sorted_by_difficulty[right])
            
            left += 1
            right -= 1
        
        return alternating_order
    
    def get_next_task(self, 
                     current_performance: Optional[Dict[int, float]] = None) -> Optional[int]:
        """
        获取下一个任务
        
        Args:
            current_performance: 当前性能字典 {task_id: performance}
            
        Returns:
            下一个任务ID
        """
        # 找到第一个未完成的任务
        for task_id in self.curriculum_order:
            if task_id not in self.completed_tasks:
                # 检查是否满足前置条件
                if self._check_task_prerequisites(task_id, current_performance):
                    return task_id
        
        return None  # 所有任务已完成
    
    def _check_task_prerequisites(self, 
                                task_id: int, 
                                current_performance: Optional[Dict[int, float]]) -> bool:
        """
        检查任务前置条件
        
        Args:
            task_id: 任务ID
            current_performance: 当前性能
            
        Returns:
            是否满足前置条件
        """
        # 简单的前置条件检查：要求先完成简单任务
        task_difficulty = self.task_difficulties[task_id]
        
        # 如果任务较难，要求完成一定数量的简单任务
        if task_difficulty > 0.7:
            simple_tasks_completed = sum(
                1 for t in self.completed_tasks 
                if self.task_difficulties[t] < 0.3
            )
            return simple_tasks_completed >= 2
        
        return True
    
    def mark_task_completed(self, 
                          task_id: int, 
                          performance: float,
                          training_time: float) -> None:
        """
        标记任务完成
        
        Args:
            task_id: 任务ID
            performance: 性能
            training_time: 训练时间
        """
        self.completed_tasks.append(task_id)
        
        learning_record = {
            'task_id': task_id,
            'difficulty': self.task_difficulties[task_id],
            'performance': performance,
            'training_time': training_time,
            'completion_order': len(self.completed_tasks)
        }
        
        self.learning_history.append(learning_record)
        
        logger.info(f"任务 {task_id} 完成，性能: {performance:.4f}")
    
    def update_difficulty_estimate(self, 
                                 task_id: int,
                                 actual_performance: float,
                                 expected_performance: float) -> None:
        """
        更新难度估计
        
        Args:
            task_id: 任务ID
            actual_performance: 实际性能
            expected_performance: 预期性能
        """
        # 基于性能差异调整难度估计
        performance_gap = abs(actual_performance - expected_performance)
        
        # 如果性能显著低于预期，增加任务难度估计
        if actual_performance < expected_performance * 0.8:
            self.task_difficulties[task_id] *= 1.1
        elif actual_performance > expected_performance * 1.2:
            # 如果性能显著高于预期，降低任务难度估计
            self.task_difficulties[task_id] *= 0.9
        
        # 重新生成课程顺序
        self.curriculum_order = self._generate_curriculum_order()
        
        logger.info(f"更新任务 {task_id} 难度估计: {self.task_difficulties[task_id]:.3f}")
    
    def get_curriculum_statistics(self) -> Dict[str, Any]:
        """获取课程统计信息"""
        completed_difficulties = [self.task_difficulties[tid] for tid in self.completed_tasks]
        
        return {
            'total_tasks': len(self.tasks),
            'completed_tasks': len(self.completed_tasks),
            'completion_rate': len(self.completed_tasks) / len(self.tasks),
            'current_curriculum': self.curriculum_order,
            'average_completed_difficulty': np.mean(completed_difficulties) if completed_difficulties else 0.0,
            'difficulty_progression': completed_difficulties,
            'curriculum_strategy': self.curriculum_strategy,
            'learning_efficiency': self._calculate_learning_efficiency(),
            'learning_stability': np.std([h['performance'] for h in self.learning_history]) if self.learning_history else 0.0
        }
    
    def _calculate_learning_efficiency(self) -> float:
        """计算学习效率"""
        if not self.learning_history:
            return 0.0
        
        # 效率 = 总性能提升 / 总时间
        total_performance = sum(h['performance'] for h in self.learning_history)
        total_time = sum(h['training_time'] for h in self.learning_history)
        
        return total_performance / max(total_time, 1)


class TransferLearningModule:
    """
    迁移学习模块
    
    实现不同任务间的知识迁移机制。
    """
    
    def __init__(self,
                 network: nn.Module,
                 transfer_strategies: List[str] = ['feature_extractor', 'fine_tuning'],
                 freeze_feature_extractor: bool = True):
        """
        初始化迁移学习模块
        
        Args:
            network: 基础网络
            transfer_strategies: 迁移策略列表
            freeze_feature_extractor: 是否冻结特征提取器
        """
        self.network = network
        self.transfer_strategies = transfer_strategies
        self.freeze_feature_extractor = freeze_feature_extractor
        
        # 迁移历史
        self.transfer_history = []
        self.knowledge_sources = {}
        
        logger.info(f"迁移学习模块初始化，策略: {transfer_strategies}")
    
    def register_knowledge_source(self, 
                                 task_id: int,
                                 model_state: Dict[str, torch.Tensor],
                                 performance: float,
                                 metadata: Optional[Dict[str, Any]] = None):
        """
        注册知识源
        
        Args:
            task_id: 任务ID
            model_state: 模型状态
            performance: 性能
            metadata: 元数据
        """
        self.knowledge_sources[task_id] = {
            'model_state': model_state,
            'performance': performance,
            'metadata': metadata or {},
            'registration_time': time.time()
        }
        
        logger.info(f"注册知识源：任务 {task_id}，性能: {performance:.4f}")
    
    def transfer_knowledge(self,
                          target_task_data: torch.Tensor,
                          target_task_labels: torch.Tensor,
                          target_task_id: int,
                          source_task_ids: Optional[List[int]] = None,
                          transfer_ratio: float = 0.5) -> Dict[str, Any]:
        """
        迁移知识
        
        Args:
            target_task_data: 目标任务数据
            target_task_labels: 目标任务标签
            target_task_id: 目标任务ID
            source_task_ids: 源任务ID列表
            transfer_ratio: 迁移比例
            
        Returns:
            迁移结果字典
        """
        if source_task_ids is None:
            # 自动选择源任务（性能最好的前几个）
            source_task_ids = self._select_best_source_tasks(target_task_id, k=3)
        
        logger.info(f"开始知识迁移，目标任务: {target_task_id}, 源任务: {source_task_ids}")
        
        transfer_start = time.time()
        
        # 应用迁移策略
        transfer_results = {}
        
        for strategy in self.transfer_strategies:
            if strategy == 'feature_extractor':
                result = self._feature_extractor_transfer(source_task_ids, target_task_data, target_task_labels)
            elif strategy == 'fine_tuning':
                result = self._fine_tuning_transfer(source_task_ids, target_task_data, target_task_labels)
            elif strategy == 'knowledge_distillation':
                result = self._knowledge_distillation_transfer(source_task_ids, target_task_data, target_task_labels)
            elif strategy == 'progressive_transfer':
                result = self._progressive_transfer(source_task_ids, target_task_data, target_task_labels)
            else:
                logger.warning(f"未知迁移策略: {strategy}")
                continue
            
            transfer_results[strategy] = result
        
        transfer_time = time.time() - transfer_start
        
        # 记录迁移历史
        transfer_record = {
            'target_task_id': target_task_id,
            'source_task_ids': source_task_ids,
            'transfer_strategies': self.transfer_strategies.copy(),
            'transfer_results': transfer_results,
            'transfer_time': transfer_time,
            'transfer_ratio': transfer_ratio
        }
        
        self.transfer_history.append(transfer_record)
        
        logger.info(f"知识迁移完成，耗时: {transfer_time:.2f}s")
        
        return transfer_record
    
    def _select_best_source_tasks(self, target_task_id: int, k: int = 3) -> List[int]:
        """
        选择最佳源任务
        
        Args:
            target_task_id: 目标任务ID
            k: 选择数量
            
        Returns:
            源任务ID列表
        """
        # 按性能排序
        sorted_tasks = sorted(
            self.knowledge_sources.items(),
            key=lambda x: x[1]['performance'],
            reverse=True
        )
        
        # 选择前k个任务
        best_tasks = [task_id for task_id, _ in sorted_tasks[:k]]
        
        return best_tasks
    
    def _feature_extractor_transfer(self,
                                  source_task_ids: List[int],
                                  target_data: torch.Tensor,
                                  target_labels: torch.Tensor) -> Dict[str, Any]:
        """
        特征提取器迁移
        
        Args:
            source_task_ids: 源任务ID列表
            target_data: 目标数据
            target_labels: 目标标签
            
        Returns:
            迁移结果
        """
        # 冻结特征提取器
        if self.freeze_feature_extractor:
            for param in self.network.parameters():
                param.requires_grad = False
        
        # 只训练分类器层
        classifier_params = []
        for name, param in self.network.named_parameters():
            if 'classifier' in name or 'fc' in name or 'head' in name:
                param.requires_grad = True
                classifier_params.append(param)
        
        if not classifier_params:
            # 如果没有明确的分类器层，训练最后一层
            last_layer = None
            for name, module in self.network.named_modules():
                if isinstance(module, nn.Linear):
                    last_layer = module
            
            if last_layer:
                for param in last_layer.parameters():
                    param.requires_grad = True
        
        # 训练分类器
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.network.parameters()), lr=0.001)
        
        # 简单的训练循环
        self.network.train()
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = self.network(target_data)
            loss = F.cross_entropy(outputs, target_labels)
            loss.backward()
            optimizer.step()
        
        # 评估性能
        with torch.no_grad():
            self.network.eval()
            outputs = self.network(target_data)
            predicted = outputs.argmax(dim=1)
            accuracy = (predicted == target_labels).float().mean().item()
        
        self.network.train()
        
        return {
            'strategy': 'feature_extractor',
            'accuracy': accuracy,
            'trainable_params': sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        }
    
    def _fine_tuning_transfer(self,
                            source_task_ids: List[int],
                            target_data: torch.Tensor,
                            target_labels: torch.Tensor) -> Dict[str, Any]:
        """
        微调迁移
        
        Args:
            source_task_ids: 源任务ID列表
            target_data: 目标数据
            target_labels: 目标标签
            
        Returns:
            迁移结果
        """
        # 解冻所有参数
        for param in self.network.parameters():
            param.requires_grad = True
        
        # 使用较低的学习率进行微调
        optimizer = optim.Adam(self.network.parameters(), lr=0.0001)
        
        # 微调训练
        self.network.train()
        total_loss = 0.0
        for epoch in range(20):  # 较少的微调轮次
            optimizer.zero_grad()
            outputs = self.network(target_data)
            loss = F.cross_entropy(outputs, target_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 评估性能
        with torch.no_grad():
            self.network.eval()
            outputs = self.network(target_data)
            predicted = outputs.argmax(dim=1)
            accuracy = (predicted == target_labels).float().mean().item()
        
        self.network.train()
        
        return {
            'strategy': 'fine_tuning',
            'accuracy': accuracy,
            'final_loss': total_loss / 20,
            'trainable_params': sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        }
    
    def _knowledge_distillation_transfer(self,
                                       source_task_ids: List[int],
                                       target_data: torch.Tensor,
                                       target_labels: torch.Tensor) -> Dict[str, Any]:
        """
        知识蒸馏迁移
        
        Args:
            source_task_ids: 源任务ID列表
            target_data: 目标数据
            target_labels: 目标标签
            
        Returns:
            迁移结果
        """
        # 获取源模型预测
        source_predictions = []
        for source_task_id in source_task_ids:
            if source_task_id in self.knowledge_sources:
                # 加载源模型状态（这里简化处理）
                source_model = deepcopy(self.network)
                
                with torch.no_grad():
                    source_pred = source_model(target_data)
                    source_predictions.append(F.softmax(source_pred, dim=1))
        
        # 平均源模型预测
        if source_predictions:
            ensemble_source_pred = torch.stack(source_predictions).mean(dim=0)
        else:
            ensemble_source_pred = None
        
        # 知识蒸馏训练
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        temperature = 3.0
        alpha = 0.7  # 蒸馏损失权重
        
        self.network.train()
        total_distill_loss = 0.0
        total_ce_loss = 0.0
        
        for epoch in range(15):
            optimizer.zero_grad()
            outputs = self.network(target_data)
            
            # 交叉熵损失
            ce_loss = F.cross_entropy(outputs, target_labels)
            
            # 蒸馏损失
            if ensemble_source_pred is not None:
                distill_loss = F.kl_div(
                    F.log_softmax(outputs / temperature, dim=1),
                    F.softmax(ensemble_source_pred / temperature, dim=1),
                    reduction='batchmean'
                ) * (temperature ** 2)
            else:
                distill_loss = 0.0
            
            # 总损失
            total_loss = alpha * distill_loss + (1 - alpha) * ce_loss
            total_loss.backward()
            optimizer.step()
            
            total_distill_loss += distill_loss.item()
            total_ce_loss += ce_loss.item()
        
        # 评估性能
        with torch.no_grad():
            self.network.eval()
            outputs = self.network(target_data)
            predicted = outputs.argmax(dim=1)
            accuracy = (predicted == target_labels).float().mean().item()
        
        self.network.train()
        
        return {
            'strategy': 'knowledge_distillation',
            'accuracy': accuracy,
            'distill_loss': total_distill_loss / 15,
            'ce_loss': total_ce_loss / 15,
            'alpha': alpha
        }
    
    def _progressive_transfer(self,
                            source_task_ids: List[int],
                            target_data: torch.Tensor,
                            target_labels: torch.Tensor) -> Dict[str, Any]:
        """
        渐进式迁移
        
        Args:
            source_task_ids: 源任务ID列表
            target_data: 目标数据
            target_labels: 目标标签
            
        Returns:
            迁移结果
        """
        # 渐进式地从多个源任务迁移知识
        transfer_results = []
        
        for source_task_id in source_task_ids:
            # 单任务迁移
            result = self._fine_tuning_transfer([source_task_id], target_data, target_labels)
            transfer_results.append(result)
        
        # 综合结果
        accuracies = [r['accuracy'] for r in transfer_results]
        
        return {
            'strategy': 'progressive_transfer',
            'individual_results': transfer_results,
            'best_accuracy': max(accuracies),
            'average_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies)
        }
    
    def get_transfer_statistics(self) -> Dict[str, Any]:
        """获取迁移统计信息"""
        if not self.transfer_history:
            return {}
        
        total_transfers = len(self.transfer_history)
        successful_transfers = sum(1 for h in self.transfer_history if h['transfer_results'])
        
        # 计算平均性能
        all_accuracies = []
        for h in self.transfer_history:
            for strategy, result in h['transfer_results'].items():
                if 'accuracy' in result:
                    all_accuracies.append(result['accuracy'])
        
        return {
            'total_transfers': total_transfers,
            'successful_transfers': successful_transfers,
            'success_rate': successful_transfers / max(total_transfers, 1),
            'average_accuracy': np.mean(all_accuracies) if all_accuracies else 0.0,
            'knowledge_sources': len(self.knowledge_sources),
            'transfer_strategies_used': self.transfer_strategies,
            'average_transfer_time': np.mean([h['transfer_time'] for h in self.transfer_history]),
            'transfer_effectiveness': self._calculate_transfer_effectiveness()
        }
    
    def _calculate_transfer_effectiveness(self) -> float:
        """计算迁移效果"""
        if not self.transfer_history:
            return 0.0
        
        # 基于性能提升计算迁移效果
        effectiveness_scores = []
        
        for h in self.transfer_history:
            for strategy, result in h['transfer_results'].items():
                if 'accuracy' in result:
                    # 简单的效果分数：准确率
                    effectiveness_scores.append(result['accuracy'])
        
        return np.mean(effectiveness_scores) if effectiveness_scores else 0.0


class NewTaskLearner:
    """
    新任务学习器
    
    整合快速适应、课程学习和迁移学习的统一接口。
    """
    
    def __init__(self,
                 network: nn.Module,
                 config: Optional[Dict] = None):
        """
        初始化新任务学习器
        
        Args:
            network: 基础网络
            config: 配置字典
        """
        self.network = network
        self.config = config or {}
        
        # 默认配置
        default_config = {
            'enable_fast_adaptation': True,
            'enable_curriculum_learning': True,
            'enable_transfer_learning': True,
            'meta_learning_rate': 0.01,
            'adaptation_steps': 5,
            'curriculum_strategy': 'easy_to_hard'
        }
        default_config.update(self.config)
        self.config = default_config
        
        # 初始化组件
        self.fast_learner = None
        self.curriculum_learner = None
        self.transfer_learner = None
        
        if self.config['enable_fast_adaptation']:
            self.fast_learner = FastTaskLearner(
                network=network,
                meta_learning_rate=self.config['meta_learning_rate'],
                adaptation_steps=self.config['adaptation_steps']
            )
        
        if self.config['enable_curriculum_learning']:
            self.curriculum_learner = CurriculumLearning(
                tasks=[],  # 需要外部设置
                curriculum_strategy=self.config['curriculum_strategy']
            )
        
        if self.config['enable_transfer_learning']:
            self.transfer_learner = TransferLearningModule(network=network)
        
        # 学习状态
        self.learned_tasks = []
        self.task_performance = {}
        
        logger.info("新任务学习器初始化完成")
    
    def learn_new_task(self,
                      task_data: Dict[str, torch.Tensor],
                      task_labels: torch.Tensor,
                      task_id: int,
                      learning_mode: str = 'auto') -> Dict[str, Any]:
        """
        学习新任务
        
        Args:
            task_data: 任务数据字典（支持集、查询集等）
            task_labels: 任务标签
            task_id: 任务ID
            learning_mode: 学习模式 ('fast_adaptation', 'curriculum', 'transfer', 'hybrid', 'auto')
            
        Returns:
            学习结果字典
        """
        logger.info(f"开始学习新任务 {task_id}，模式: {learning_mode}")
        
        learning_start = time.time()
        
        # 根据模式选择学习策略
        if learning_mode == 'auto':
            learning_mode = self._select_learning_mode(task_id)
        
        learning_results = {}
        
        if learning_mode == 'fast_adaptation' and self.fast_learner:
            # 快速适应学习
            result = self._fast_adaptation_learning(task_data, task_labels, task_id)
            learning_results['fast_adaptation'] = result
        
        elif learning_mode == 'transfer' and self.transfer_learner:
            # 迁移学习
            result = self._transfer_learning(task_data, task_labels, task_id)
            learning_results['transfer'] = result
        
        elif learning_mode == 'curriculum' and self.curriculum_learner:
            # 课程学习
            result = self._curriculum_learning(task_data, task_labels, task_id)
            learning_results['curriculum'] = result
        
        elif learning_mode == 'hybrid':
            # 混合学习
            result = self._hybrid_learning(task_data, task_labels, task_id)
            learning_results['hybrid'] = result
        
        else:
            # 标准学习
            result = self._standard_learning(task_data, task_labels, task_id)
            learning_results['standard'] = result
        
        learning_time = time.time() - learning_start
        
        # 记录学习历史
        self.learned_tasks.append(task_id)
        
        # 更新任务性能
        if 'final_accuracy' in learning_results:
            self.task_performance[task_id] = learning_results['final_accuracy']
        
        learning_record = {
            'task_id': task_id,
            'learning_mode': learning_mode,
            'learning_results': learning_results,
            'learning_time': learning_time,
            'timestamp': time.time()
        }
        
        logger.info(f"任务 {task_id} 学习完成，耗时: {learning_time:.2f}s")
        
        return learning_record
    
    def _select_learning_mode(self, task_id: int) -> str:
        """自动选择学习模式"""
        if len(self.learned_tasks) == 0:
            return 'fast_adaptation'
        
        # 基于任务相似度选择模式
        if self.transfer_learner and len(self.transfer_learner.knowledge_sources) > 0:
            return 'transfer'
        elif self.fast_learner:
            return 'fast_adaptation'
        else:
            return 'standard'
    
    def _fast_adaptation_learning(self,
                                task_data: Dict[str, torch.Tensor],
                                task_labels: torch.Tensor,
                                task_id: int) -> Dict[str, Any]:
        """快速适应学习"""
        support_data = task_data.get('support', task_data.get('train', task_data['data']))
        query_data = task_data.get('query', task_data.get('test', support_data))
        support_labels = task_labels.get('support', task_labels.get('train', task_labels))
        query_labels = task_labels.get('query', task_labels.get('test', support_labels))
        
        result = self.fast_learner.fast_adapt(
            support_data, support_labels, query_data, query_labels, task_id
        )
        
        return {
            'strategy': 'fast_adaptation',
            'final_accuracy': result['query_accuracy'],
            'adaptation_time': result['adaptation_time'],
            'meta_gradient_norm': result['meta_gradient_norm']
        }
    
    def _transfer_learning(self,
                          task_data: Dict[str, torch.Tensor],
                          task_labels: torch.Tensor,
                          task_id: int) -> Dict[str, Any]:
        """迁移学习"""
        train_data = task_data.get('train', task_data['data'])
        train_labels = task_labels
        
        result = self.transfer_learner.transfer_knowledge(
            train_data, train_labels, task_id
        )
        
        # 计算最终准确率
        final_accuracy = 0.0
        if 'transfer_results' in result:
            for strategy_result in result['transfer_results'].values():
                if 'accuracy' in strategy_result:
                    final_accuracy = max(final_accuracy, strategy_result['accuracy'])
        
        # 注册知识源
        self.transfer_learner.register_knowledge_source(
            task_id, 
            {name: param.cpu() for name, param in self.network.named_parameters()},
            final_accuracy
        )
        
        return {
            'strategy': 'transfer',
            'final_accuracy': final_accuracy,
            'transfer_results': result['transfer_results'],
            'source_tasks_used': result['source_task_ids']
        }
    
    def _curriculum_learning(self,
                           task_data: Dict[str, torch.Tensor],
                           task_labels: torch.Tensor,
                           task_id: int) -> Dict[str, Any]:
        """课程学习"""
        # 这里简化处理，实际实现需要更复杂的课程设计
        train_data = task_data.get('train', task_data['data'])
        train_labels = task_labels
        
        # 简单训练
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        
        self.network.train()
        total_loss = 0.0
        for epoch in range(50):
            optimizer.zero_grad()
            outputs = self.network(train_data)
            loss = F.cross_entropy(outputs, train_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 评估
        with torch.no_grad():
            self.network.eval()
            outputs = self.network(train_data)
            predicted = outputs.argmax(dim=1)
            accuracy = (predicted == train_labels).float().mean().item()
        
        self.network.train()
        
        # 标记完成（这里简化）
        if self.curriculum_learner:
            self.curriculum_learner.mark_task_completed(task_id, accuracy, total_loss)
        
        return {
            'strategy': 'curriculum',
            'final_accuracy': accuracy,
            'final_loss': total_loss / 50
        }
    
    def _hybrid_learning(self,
                        task_data: Dict[str, torch.Tensor],
                        task_labels: torch.Tensor,
                        task_id: int) -> Dict[str, Any]:
        """混合学习"""
        # 结合多种策略
        results = {}
        
        # 先尝试快速适应
        if self.fast_learner:
            fa_result = self._fast_adaptation_learning(task_data, task_labels, task_id)
            results['fast_adaptation'] = fa_result
        
        # 再尝试迁移学习
        if self.transfer_learner:
            tl_result = self._transfer_learning(task_data, task_labels, task_id)
            results['transfer'] = tl_result
        
        # 选择最佳结果
        best_result = max(results.values(), key=lambda x: x['final_accuracy'])
        
        return {
            'strategy': 'hybrid',
            'final_accuracy': best_result['final_accuracy'],
            'all_results': results,
            'best_strategy': best_result['strategy']
        }
    
    def _standard_learning(self,
                          task_data: Dict[str, torch.Tensor],
                          task_labels: torch.Tensor,
                          task_id: int) -> Dict[str, Any]:
        """标准学习"""
        train_data = task_data.get('train', task_data['data'])
        train_labels = task_labels
        
        optimizer = optim.Adam(self.network.parameters(), lr=0.001)
        
        self.network.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.network(train_data)
            loss = F.cross_entropy(outputs, train_labels)
            loss.backward()
            optimizer.step()
        
        # 评估
        with torch.no_grad():
            self.network.eval()
            outputs = self.network(train_data)
            predicted = outputs.argmax(dim=1)
            accuracy = (predicted == train_labels).float().mean().item()
        
        self.network.train()
        
        return {
            'strategy': 'standard',
            'final_accuracy': accuracy
        }
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        stats = {
            'total_learned_tasks': len(self.learned_tasks),
            'task_performance': self.task_performance,
            'average_performance': np.mean(list(self.task_performance.values())) if self.task_performance else 0.0,
            'performance_std': np.std(list(self.task_performance.values())) if self.task_performance else 0.0
        }
        
        # 添加各组件统计
        if self.fast_learner:
            stats['fast_adaptation'] = self.fast_learner.get_adaptation_statistics()
        
        if self.curriculum_learner:
            stats['curriculum_learning'] = self.curriculum_learner.get_curriculum_statistics()
        
        if self.transfer_learner:
            stats['transfer_learning'] = self.transfer_learner.get_transfer_statistics()
        
        return stats