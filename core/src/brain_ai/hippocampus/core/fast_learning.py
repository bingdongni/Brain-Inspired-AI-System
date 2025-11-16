#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速学习模块
===========

实现海马体的快速学习机制，包括单样本学习和少样本学习。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

class FastLearningConfig:
    """快速学习配置"""
    
    def __init__(self):
        self.learning_rate = 0.1
        self.meta_learning_rate = 0.01
        self.adaptation_steps = 5
        self.inner_loop_steps = 1
        self.inner_loop_lr = 0.05
        self.task_batch_size = 4
        self.num_tasks = 100
        self.hidden_size = 128

class MAMLModule(nn.Module):
    """Model-Agnostic Meta-Learning (MAML) 模块"""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 128):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        # 神经网络层
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x: torch.Tensor, adapt: bool = False) -> torch.Tensor:
        return self.layers(x)
    
    def adapt(self, support_data: torch.Tensor, support_labels: torch.Tensor, 
              steps: int = 1, lr: float = 0.01) -> 'MAMLModule':
        """适应新任务"""
        adapted_model = MAMLModule(self.input_size, self.output_size, self.hidden_size)
        adapted_model.load_state_dict(self.state_dict())
        adapted_model.train()
        
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            output = adapted_model(support_data)
            loss = F.mse_loss(output, support_labels)
            loss.backward()
            optimizer.step()
        
        return adapted_model

class FastLearningSystem:
    """快速学习系统"""
    
    def __init__(self, input_size: int, output_size: int, config: Optional[FastLearningConfig] = None):
        """
        初始化快速学习系统
        
        Args:
            input_size: 输入维度
            output_size: 输出维度
            config: 配置对象
        """
        if config is None:
            config = FastLearningConfig()
        
        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 基础学习器
        self.base_learner = MAMLModule(input_size, output_size, config.hidden_size).to(self.device)
        
        # 元学习器
        self.meta_learner = MAMLModule(input_size, output_size, config.hidden_size).to(self.device)
        
        # 任务学习历史
        self.task_history = []
        self.performance_history = []
        
    def few_shot_learn(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                      query_data: torch.Tensor, num_shots: int = 1, num_ways: int = 2) -> Dict[str, torch.Tensor]:
        """
        少样本学习
        
        Args:
            support_data: 支持集数据 [num_shots*num_ways, input_size]
            support_labels: 支持集标签 [num_shots*num_ways, output_size]
            query_data: 查询数据 [query_size, input_size]
            num_shots: 每类样本数
            num_ways: 类别数
            
        Returns:
            预测结果字典
        """
        self.base_learner.train()
        
        # 适应支持集
        adapted_learner = self.base_learner.adapt(
            support_data, support_labels, 
            steps=self.config.adaptation_steps,
            lr=self.config.inner_loop_lr
        )
        
        # 在查询集上预测
        with torch.no_grad():
            query_predictions = adapted_learner(query_data)
        
        return {
            'predictions': query_predictions,
            'adapted_learner': adapted_learner,
            'support_loss': self._compute_adaptation_loss(support_data, support_labels, adapted_learner)
        }
    
    def _compute_adaptation_loss(self, data: torch.Tensor, labels: torch.Tensor, 
                               learner: MAMLModule) -> torch.Tensor:
        """计算适应损失"""
        output = learner(data)
        return F.mse_loss(output, labels)
    
    def meta_train_step(self, task_batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """
        元训练一步
        
        Args:
            task_batch: 任务批次 [(support_data, support_labels, query_data, query_labels), ...]
            
        Returns:
            训练损失
        """
        self.meta_learner.train()
        meta_optimizer = torch.optim.Adam(self.meta_learner.parameters(), lr=self.config.meta_learning_rate)
        
        total_meta_loss = 0.0
        
        for support_data, support_labels, query_data, query_labels in task_batch:
            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            query_data = query_data.to(self.device)
            query_labels = query_labels.to(self.device)
            
            # 内循环：适应支持集
            adapted_learner = self.base_learner.adapt(
                support_data, support_labels,
                steps=self.config.inner_loop_steps,
                lr=self.config.inner_loop_lr
            )
            
            # 外循环：在查询集上计算元损失
            query_output = adapted_learner(query_data)
            meta_loss = F.mse_loss(query_output, query_labels)
            total_meta_loss += meta_loss
        
        # 更新元学习器
        meta_optimizer.zero_grad()
        total_meta_loss.backward()
        meta_optimizer.step()
        
        return {
            'meta_loss': total_meta_loss.item() / len(task_batch)
        }
    
    def learn_from_single_example(self, input_data: torch.Tensor, target_data: torch.Tensor,
                                num_adaptations: int = 10) -> Dict[str, Any]:
        """
        从单个示例学习
        
        Args:
            input_data: 输入数据
            target_data: 目标数据
            num_adaptations: 适应步数
            
        Returns:
            学习结果
        """
        input_data = input_data.to(self.device)
        target_data = target_data.to(self.device)
        
        # 创建适配的学习器
        adapted_learner = MAMLModule(self.input_size, self.output_size, self.config.hidden_size)
        adapted_learner.load_state_dict(self.base_learner.state_dict())
        adapted_learner = adapted_learner.to(self.device)
        adapted_learner.train()
        
        optimizer = torch.optim.SGD(adapted_learner.parameters(), lr=self.config.learning_rate)
        
        losses = []
        
        for step in range(num_adaptations):
            optimizer.zero_grad()
            output = adapted_learner(input_data.unsqueeze(0))
            loss = F.mse_loss(output, target_data.unsqueeze(0))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # 评估学习效果
        with torch.no_grad():
            final_output = adapted_learner(input_data.unsqueeze(0))
            final_loss = F.mse_loss(final_output, target_data.unsqueeze(0))
        
        return {
            'adapted_learner': adapted_learner,
            'learning_curve': losses,
            'final_loss': final_loss.item(),
            'improvement': losses[0] - final_loss.item()
        }
    
    def generate_task_batch(self, num_tasks: int = None) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        生成任务批次用于元学习
        
        Args:
            num_tasks: 任务数量
            
        Returns:
            任务批次列表
        """
        if num_tasks is None:
            num_tasks = self.config.task_batch_size
        
        tasks = []
        
        for _ in range(num_tasks):
            # 生成随机线性变换作为任务
            true_params = torch.randn(self.input_size, self.output_size)
            
            # 生成支持集和查询集
            support_size = self.config.num_shots * self.config.num_ways
            
            support_data = torch.randn(support_size, self.input_size)
            support_labels = torch.mm(support_data, true_params)
            
            query_data = torch.randn(10, self.input_size)  # 10个查询样本
            query_labels = torch.mm(query_data, true_params)
            
            tasks.append((support_data, support_labels, query_data, query_labels))
        
        return tasks
    
    def evaluate_few_shot_performance(self, test_tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """
        评估少样本学习性能
        
        Args:
            test_tasks: 测试任务列表
            
        Returns:
            性能指标
        """
        self.meta_learner.eval()
        
        total_accuracy = 0.0
        total_loss = 0.0
        
        with torch.no_grad():
            for support_data, support_labels, query_data, query_labels in test_tasks:
                support_data = support_data.to(self.device)
                support_labels = support_labels.to(self.device)
                query_data = query_data.to(self.device)
                query_labels = query_labels.to(self.device)
                
                # 适应支持集
                adapted_learner = self.meta_learner.adapt(
                    support_data, support_labels,
                    steps=self.config.adaptation_steps
                )
                
                # 在查询集上预测
                query_output = adapted_learner(query_data)
                
                # 计算损失和准确率
                loss = F.mse_loss(query_output, query_labels)
                accuracy = 1.0 / (1.0 + loss.item())  # 简单的准确率计算
                
                total_loss += loss.item()
                total_accuracy += accuracy
        
        num_tasks = len(test_tasks)
        return {
            'average_loss': total_loss / num_tasks,
            'average_accuracy': total_accuracy / num_tasks,
            'num_test_tasks': num_tasks
        }

class RapidConsolidation:
    """快速巩固机制"""
    
    def __init__(self, memory_capacity: int = 1000):
        self.memory_capacity = memory_capacity
        self.consolidated_memories = {}
        self.memory_strengths = {}
        
    def consolidate_memory(self, memory_id: str, memory_content: Any, 
                          importance_score: float = 1.0) -> bool:
        """
        快速巩固记忆
        
        Args:
            memory_id: 记忆ID
            memory_content: 记忆内容
            importance_score: 重要性分数
            
        Returns:
            是否成功巩固
        """
        if len(self.consolidated_memories) >= self.memory_capacity:
            # 删除最不重要的记忆
            self._remove_least_important()
        
        # 根据重要性调整巩固强度
        consolidation_strength = min(importance_score, 2.0)
        
        self.consolidated_memories[memory_id] = {
            'content': memory_content,
            'consolidated_at': len(self.consolidated_memories),
            'importance': importance_score,
            'strength': consolidation_strength,
            'access_count': 0
        }
        
        self.memory_strengths[memory_id] = consolidation_strength
        return True
    
    def _remove_least_important(self):
        """删除最不重要的记忆"""
        if not self.consolidated_memories:
            return
        
        # 找到重要性和强度最低的记忆
        least_important_id = min(self.memory_strengths.keys(), 
                               key=lambda x: self.memory_strengths[x] * 
                               self.consolidated_memories[x]['importance'])
        
        del self.consolidated_memories[least_important_id]
        del self.memory_strengths[least_important_id]
    
    def retrieve_consolidated(self, memory_id: str) -> Optional[Any]:
        """检索巩固后的记忆"""
        if memory_id not in self.consolidated_memories:
            return None
        
        memory = self.consolidated_memories[memory_id]
        memory['access_count'] += 1
        memory['strength'] *= 1.05  # 强化访问的记忆
        
        return memory['content']
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """获取巩固统计信息"""
        if not self.consolidated_memories:
            return {'consolidated_count': 0}
        
        importances = [mem['importance'] for mem in self.consolidated_memories.values()]
        strengths = [mem['strength'] for mem in self.consolidated_memories.values()]
        access_counts = [mem['access_count'] for mem in self.consolidated_memories.values()]
        
        return {
            'consolidated_count': len(self.consolidated_memories),
            'capacity_usage': len(self.consolidated_memories) / self.memory_capacity,
            'average_importance': np.mean(importances),
            'average_strength': np.mean(strengths),
            'total_accesses': sum(access_counts),
            'max_strength': max(strengths),
            'min_strength': min(strengths)
        }

# 工厂函数
def create_fast_learner(input_size: int, output_size: int, 
                       learning_type: str = 'maml') -> Any:
    """
    创建快速学习器
    
    Args:
        input_size: 输入维度
        output_size: 输出维度
        learning_type: 学习类型 ('maml', 'prototypical', 'matching')
        
    Returns:
        快速学习器实例
    """
    if learning_type.lower() == 'maml':
        return FastLearningSystem(input_size, output_size)
    else:
        raise ValueError(f"不支持的学习类型: {learning_type}")

if __name__ == "__main__":
    # 测试快速学习系统
    input_size, output_size = 10, 5
    
    fast_learner = create_fast_learner(input_size, output_size, 'maml')
    
    # 生成测试任务
    support_data = torch.randn(4, input_size)  # 4个支持样本
    support_labels = torch.randn(4, output_size)
    query_data = torch.randn(2, input_size)   # 2个查询样本
    
    # 执行少样本学习
    result = fast_learner.few_shot_learn(support_data, support_labels, query_data)
    print(f"少样本学习结果形状: {result['predictions'].shape}")
    
    # 单样本学习测试
    single_input = torch.randn(input_size)
    single_target = torch.randn(output_size)
    
    single_result = fast_learner.learn_from_single_example(single_input, single_target)
    print(f"单样本学习改进: {single_result['improvement']:.4f}")
    
    # 巩固测试
    consolidator = RapidConsolidation()
    consolidator.consolidate_memory("test_memory", "测试记忆内容", importance_score=0.8)
    
    stats = consolidator.get_consolidation_stats()
    print(f"巩固统计: {stats}")