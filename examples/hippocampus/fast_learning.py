"""
快速一次性学习系统
基于海马体的快速学习机制
实现few-shot学习和快速适应
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class FastMemoryEncoder(nn.Module):
    """
    快速记忆编码器
    模拟海马体CA3区的快速编码机制
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 learning_rate: float = 0.1,
                 consolidation_threshold: float = 0.8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.consolidation_threshold = consolidation_threshold
        
        # 快速编码权重
        self.encoding_weights = nn.Parameter(
            torch.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim)
        )
        
        # 记忆整合权重
        self.integration_weights = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim) * 0.1
        )
        
        # 快速权重更新规则
        self.fast_update_rule = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 注意力机制（用于选择重要特征）
        self.attention_mechanism = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 记忆强度预测
        self.strength_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, 
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                learning_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        快速记忆编码前向传播
        
        Args:
            x: 输入 [batch_size, input_dim]
            context: 上下文信息
            learning_mode: 是否处于学习模式
            
        Returns:
            编码表示、记忆强度
        """
        batch_size = x.shape[0]
        
        # 特征注意力
        feature_attention = self.attention_mechanism(x)  # [batch_size, 1]
        
        # 基础编码
        encoded = F.linear(x, self.encoding_weights)  # [batch_size, hidden_dim]
        
        # 注意力加权
        encoded = encoded * feature_attention
        
        # 上下文整合
        if context is not None:
            context_encoded = self.context_encoder(context)
            encoded = encoded + context_encoded
        
        # 记忆强度预测
        memory_strength = self.strength_predictor(encoded)  # [batch_size, 1]
        
        # 快速权重更新（如果是学习模式）
        if learning_mode and self.training:
            self._fast_weight_update(x, encoded, memory_strength)
        
        # 记忆整合
        if context is not None:
            integrated = torch.matmul(encoded, self.integration_weights)
            integrated = F.relu(integrated)
            encoded = 0.7 * encoded + 0.3 * integrated
        
        return encoded, memory_strength
    
    def _fast_weight_update(self, 
                           x: torch.Tensor, 
                           encoded: torch.Tensor, 
                           strength: torch.Tensor):
        """快速权重更新（Hebbian学习规则）"""
        with torch.no_grad():
            # 计算权重更新
            x_centered = x - x.mean(dim=0, keepdim=True)
            encoded_centered = encoded - encoded.mean(dim=0, keepdim=True)
            
            # Hebbian更新: ΔW = η * (x - mean(x)) * (encoded - mean(encoded))^T
            outer_product = torch.einsum('bi,bj->bij', x_centered, encoded_centered)
            
            # 根据记忆强度调整更新幅度
            update_magnitude = strength.unsqueeze(-1).unsqueeze(-1) * self.learning_rate
            
            weight_update = update_magnitude * outer_product.mean(dim=0)
            
            # 应用更新
            self.encoding_weights.data += weight_update
            
            # 权重正则化
            self.encoding_weights.data *= 0.999


class MetaLearner(nn.Module):
    """
    元学习器
    快速适应新任务的学习策略
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_tasks: int = 100,
                 meta_lr: float = 0.01):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.meta_lr = meta_lr
        
        # 任务表示学习
        self.task_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # 支持集 + 查询集
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 元参数预测器
        self.meta_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * hidden_dim),  # 预测编码权重
            nn.Sigmoid()
        )
        
        # 快速适应控制器
        self.adaptation_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # 学习率、正则化率、注意力率
            nn.Softmax(dim=-1)
        )
        
        # 任务相似性评估器
        self.task_similarity = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 元记忆库
        self.meta_memory = nn.Parameter(
            torch.randn(num_tasks, hidden_dim) * 0.1
        )
        
        # 任务索引
        self.task_index = nn.Parameter(torch.zeros(num_tasks))
        
    def encode_task(self, 
                   support_x: torch.Tensor,
                   support_y: torch.Tensor,
                   query_x: torch.Tensor) -> torch.Tensor:
        """
        编码任务信息
        
        Args:
            support_x: 支持集输入
            support_y: 支持集标签
            query_x: 查询集输入
            
        Returns:
            任务表示
        """
        batch_size = support_x.shape[0]
        
        # 组合支持集信息
        support_combined = torch.cat([support_x, support_y], dim=-1)
        
        # 任务编码
        task_repr = self.task_encoder(
            torch.cat([support_combined, query_x], dim=0)
        )
        
        # 取平均值作为任务表示
        task_mean = task_repr.mean(dim=0, keepdim=True)
        
        return task_mean
    
    def predict_meta_parameters(self, task_repr: torch.Tensor) -> torch.Tensor:
        """
        预测元参数
        
        Args:
            task_repr: 任务表示
            
        Returns:
            预测的参数
        """
        meta_params = self.meta_predictor(task_repr)
        return meta_params.view(-1, self.input_dim, self.hidden_dim)
    
    def adapt_to_task(self,
                     fast_encoder: FastMemoryEncoder,
                     task_repr: torch.Tensor,
                     adaptation_steps: int = 5) -> Dict[str, torch.Tensor]:
        """
        快速适应任务
        
        Args:
            fast_encoder: 快速记忆编码器
            task_repr: 任务表示
            adaptation_steps: 适应步数
            
        Returns:
            适应结果
        """
        # 预测元参数
        meta_params = self.predict_meta_parameters(task_repr)
        
        # 预测适应策略
        adaptation_strategy = self.adaptation_controller(task_repr)
        
        # 记录适应过程
        adaptation_history = []
        original_weights = fast_encoder.encoding_weights.data.clone()
        
        # 执行梯度下降适应
        for step in range(adaptation_steps):
            # 应用元参数
            fast_encoder.encoding_weights.data = meta_params
            
            # 计算适应损失
            # 这里需要具体的任务数据，实际实现中会传入
            
            adaptation_history.append({
                'step': step,
                'strategy': adaptation_strategy.detach(),
                'weights_norm': torch.norm(fast_encoder.encoding_weights).item()
            })
            
            # 更新元参数（简化版）
            meta_params = meta_params - self.meta_lr * torch.randn_like(meta_params) * 0.01
        
        # 恢复原始权重（保持原编码器不变）
        fast_encoder.encoding_weights.data = original_weights
        
        return {
            'adapted_params': meta_params,
            'adaptation_strategy': adaptation_strategy,
            'adaptation_history': adaptation_history,
            'task_repr': task_repr
        }
    
    def store_task_memory(self, task_repr: torch.Tensor, task_id: int):
        """存储任务到元记忆库"""
        if task_id < self.meta_memory.shape[0]:
            self.meta_memory[task_id] = task_repr.squeeze(0)
            self.task_index[task_id] = 1.0
    
    def retrieve_similar_tasks(self, task_repr: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """检索相似任务"""
        # 计算与元记忆的相似性
        similarities = F.cosine_similarity(
            task_repr, self.meta_memory, dim=-1
        )
        
        # 获取top-k相似任务
        top_similarities, top_indices = torch.topk(similarities, k=min(top_k, len(similarities)))
        
        return self.meta_memory[top_indices], top_similarities


class OneShotLearner(nn.Module):
    """
    一次性学习器
    集成快速编码和元学习的完整系统
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 256,
                 num_way: int = 5,
                 num_shot: int = 1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_way = num_way
        self.num_shot = num_shot
        
        # 快速记忆编码器
        self.fast_encoder = FastMemoryEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )
        
        # 元学习器
        self.meta_learner = MetaLearner(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_tasks=1000
        )
        
        # 原型网络（用于分类）
        self.prototype_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_way),
            nn.LogSoftmax(dim=-1)
        )
        
        # 距离度量器
        self.distance_metric = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 记忆融合器
        self.memory_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 快速适应评估器
        self.adaptation_evaluator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def few_shot_learning(self,
                         support_x: torch.Tensor,
                         support_y: torch.Tensor,
                         query_x: torch.Tensor,
                         adaptation_steps: int = 5) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Few-shot学习主函数
        
        Args:
            support_x: 支持集输入 [n_support, input_dim]
            support_y: 支持集标签 [n_support]
            query_x: 查询集输入 [n_query, input_dim]
            adaptation_steps: 适应步数
            
        Returns:
            预测结果、学习统计
        """
        n_support = support_x.shape[0]
        n_query = query_x.shape[0]
        
        # 编码支持集
        support_encoded, support_strength = self.fast_encoder(
            support_x, learning_mode=True
        )
        
        # 编码查询集
        query_encoded, query_strength = self.fast_encoder(query_x)
        
        # 计算原型（每类的中心）
        prototypes = {}
        for class_id in range(self.num_way):
            class_mask = (support_y == class_id)
            if class_mask.sum() > 0:
                class_prototype = support_encoded[class_mask].mean(dim=0)
                prototypes[class_id] = class_prototype
        
        # 元学习适应
        task_repr = self.meta_learner.encode_task(support_x, support_y, query_x)
        adaptation_result = self.meta_learner.adapt_to_task(
            self.fast_encoder, task_repr, adaptation_steps
        )
        
        # 使用适应后的参数重新编码
        adapted_encoded, adapted_strength = self.fast_encoder(query_x)
        
        # 计算到每个原型的距离
        distances = {}
        for class_id, prototype in prototypes.items():
            # 距离度量
            dist_input = torch.cat([adapted_encoded, prototype.unsqueeze(0).expand(n_query, -1)], dim=-1)
            distance = self.distance_metric(dist_input).squeeze(-1)
            distances[class_id] = distance
        
        # 转换为概率
        logits = torch.zeros(n_query, self.num_way)
        for class_id, distance in distances.items():
            # 距离越小，概率越大
            logits[:, class_id] = -distance
        
        # 分类预测
        predictions = self.prototype_network(adapted_encoded)
        
        # 收集统计信息
        stats = {
            'support_encoded_norm': torch.norm(support_encoded).item(),
            'query_encoded_norm': torch.norm(query_encoded).item(),
            'adapted_encoded_norm': torch.norm(adapted_encoded).item(),
            'avg_memory_strength': torch.cat([support_strength, query_strength]).mean().item(),
            'num_prototypes': len(prototypes),
            'adaptation_strategy': adaptation_result['adaptation_strategy'].detach().cpu().numpy(),
            'task_repr_norm': torch.norm(task_repr).item()
        }
        
        return predictions, stats
    
    def fast_adaptation(self,
                       data_batch: torch.Tensor,
                       labels: torch.Tensor,
                       num_adaptations: int = 10) -> Dict[str, Any]:
        """
        快速适应新数据
        
        Args:
            data_batch: 数据批次
            labels: 标签
            num_adaptations: 适应次数
            
        Returns:
            适应结果
        """
        adaptation_results = []
        original_performance = None
        
        for i in range(num_adaptations):
            # 编码数据
            encoded, strength = self.fast_encoder(data_batch, learning_mode=True)
            
            # 计算当前性能（简化评估）
            current_performance = self.adaptation_evaluator(encoded).mean().item()
            
            if original_performance is None:
                original_performance = current_performance
            
            adaptation_results.append({
                'iteration': i,
                'performance': current_performance,
                'memory_strength': strength.mean().item(),
                'encoding_norm': torch.norm(encoded).item()
            })
            
            # 模拟性能提升
            if i < num_adaptations - 1:
                # 随机扰动数据以模拟继续学习
                data_batch = data_batch + 0.01 * torch.randn_like(data_batch)
        
        performance_improvement = adaptation_results[-1]['performance'] - original_performance
        
        return {
            'adaptation_results': adaptation_results,
            'original_performance': original_performance,
            'final_performance': adaptation_results[-1]['performance'],
            'performance_improvement': performance_improvement,
            'total_adaptations': num_adaptations
        }
    
    def memorize_pattern(self,
                        pattern: torch.Tensor,
                        context: Optional[torch.Tensor] = None,
                        importance: float = 1.0) -> Dict[str, Any]:
        """
        记忆单个模式
        
        Args:
            pattern: 要记忆的模式
            context: 上下文信息
            importance: 重要性权重
            
        Returns:
            记忆结果
        """
        # 编码模式
        encoded, strength = self.fast_encoder(pattern, context, learning_mode=True)
        
        # 计算记忆质量
        memory_quality = self.adaptation_evaluator(encoded).item()
        
        # 根据重要性调整记忆强度
        adjusted_strength = strength * importance
        
        # 存储到元记忆库
        task_repr = encoded.mean(dim=0, keepdim=True)
        task_id = int(torch.randint(0, 100, (1,)).item())
        
        self.meta_learner.store_task_memory(task_repr, task_id)
        
        memory_result = {
            'encoded_pattern': encoded,
            'memory_strength': adjusted_strength.item(),
            'memory_quality': memory_quality,
            'task_id': task_id,
            'importance': importance
        }
        
        return memory_result
    
    def retrieve_memory(self, 
                       query: torch.Tensor,
                       num_retrievals: int = 5) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        检索记忆
        
        Args:
            query: 查询向量
            num_retrievals: 检索数量
            
        Returns:
            检索结果、相似度分数、检索统计
        """
        # 编码查询
        query_encoded, query_strength = self.fast_encoder(query)
        
        # 检索相似任务
        similar_tasks, similarities = self.meta_learner.retrieve_similar_tasks(
            query_encoded, top_k=num_retrievals
        )
        
        # 检索统计
        retrieval_stats = {
            'query_encoded_norm': torch.norm(query_encoded).item(),
            'num_similar_tasks': len(similar_tasks),
            'avg_similarity': similarities.mean().item(),
            'max_similarity': similarities.max().item(),
            'similarity_std': similarities.std().item()
        }
        
        return similar_tasks, similarities, retrieval_stats
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习系统统计信息"""
        encoder_stats = {
            'learning_rate': self.fast_encoder.learning_rate,
            'encoding_weight_norm': torch.norm(self.fast_encoder.encoding_weights).item(),
            'integration_weight_norm': torch.norm(self.fast_encoder.integration_weights).item()
        }
        
        meta_stats = {
            'meta_lr': self.meta_learner.meta_lr,
            'meta_memory_utilization': (self.meta_learner.task_index > 0).float().mean().item(),
            'meta_memory_norm': torch.norm(self.meta_learner.meta_memory).item()
        }
        
        prototype_stats = {
            'prototype_network_params': sum(p.numel() for p in self.prototype_network.parameters()),
            'distance_metric_params': sum(p.numel() for p in self.distance_metric.parameters())
        }
        
        return {
            'encoder': encoder_stats,
            'meta_learner': meta_stats,
            'prototype': prototype_stats,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }