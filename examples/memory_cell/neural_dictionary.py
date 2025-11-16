"""
可微分神经字典
基于海马体CA1区的记忆存储和检索机制
实现高效的神经网络记忆库
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class SynapticMemoryCell(nn.Module):
    """
    突触记忆细胞
    模拟单个突触的记忆存储单元
    """
    
    def __init__(self,
                 key_dim: int,
                 value_dim: int,
                 capacity: int = 1000,
                 temperature: float = 1.0):
        super().__init__()
        
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity
        self.temperature = temperature
        
        # 记忆键值存储
        self.memory_keys = nn.Parameter(
            torch.randn(capacity, key_dim) * 0.1
        )
        self.memory_values = nn.Parameter(
            torch.randn(capacity, value_dim) * 0.1
        )
        
        # 记忆强度(模拟突触权重)
        self.memory_strengths = nn.Parameter(
            torch.ones(capacity) * 0.5
        )
        
        # 记忆重要性权重(基于使用频率)
        self.importance_weights = nn.Parameter(
            torch.ones(capacity) * 0.1
        )
        
        # 记忆容量管理
        self.current_size = 0
        self.write_position = 0
        
        # 记忆检索网络
        self.retrieval_network = nn.Sequential(
            nn.Linear(key_dim + value_dim, key_dim),
            nn.ReLU(),
            nn.Linear(key_dim, 1),
            nn.Sigmoid()
        )
        
        # 记忆强度预测器
        self.strength_predictor = nn.Sequential(
            nn.Linear(key_dim + value_dim, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, 1),
            nn.Sigmoid()
        )
        
    def write_memory(self, 
                    key: torch.Tensor, 
                    value: torch.Tensor,
                    strength: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        写入记忆到字典
        
        Args:
            key: 记忆键 [batch_size, key_dim]
            value: 记忆值 [batch_size, value_dim]
            strength: 记忆强度
            
        Returns:
            写入位置信息
        """
        batch_size = key.shape[0]
        
        # 计算记忆强度
        if strength is None:
            combined_input = torch.cat([key, value], dim=-1)
            strength = self.strength_predictor(combined_input)
        
        # 选择写入位置（LRU策略）
        if self.current_size < self.capacity:
            write_positions = torch.arange(self.current_size, 
                                         self.current_size + batch_size, 
                                         device=key.device)
            self.current_size += batch_size
        else:
            # 替换最不重要的记忆
            importance_scores = self.importance_weights * self.memory_strengths
            _, write_positions = torch.topk(importance_scores, k=batch_size, largest=False)
        
        # 写入记忆（处理维度不匹配）
        for i in range(min(batch_size, key.shape[0], self.capacity)):
            pos = write_positions[i].item() % self.capacity
            # 避免对参数进行原位操作
            with torch.no_grad():
                # 确保维度匹配
                key_slice = key[i][:self.key_dim] if key[i].shape[0] > self.key_dim else key[i]
                value_slice = value[i][:self.value_dim] if value[i].shape[0] > self.value_dim else value[i]
                strength_val = strength[i].item() if strength[i].numel() == 1 else strength[i][:1].item()
                
                self.memory_keys[pos] = F.pad(key_slice, (0, max(0, self.key_dim - key_slice.shape[0])))
                self.memory_values[pos] = F.pad(value_slice, (0, max(0, self.value_dim - value_slice.shape[0])))
                self.memory_strengths[pos] = strength_val
                # 衰减重要性权重（避免原位操作）
                self.importance_weights[pos] = self.importance_weights[pos] * 0.99
        
        # 更新重要性权重
        importance_input = torch.cat([key, value, strength], dim=-1)
        with torch.no_grad():
            new_importance = self.retrieval_network(importance_input).squeeze(-1)
            self.importance_weights[write_positions] = torch.max(
                self.importance_weights[write_positions], new_importance
            )
        
        return write_positions.float()
    
    def retrieve_memory(self, 
                       query: torch.Tensor, 
                       top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从字典检索记忆
        
        Args:
            query: 查询向量 [batch_size, key_dim]
            top_k: 返回前k个记忆
            
        Returns:
            检索的键、值、相似度分数
        """
        batch_size = query.shape[0]
        
        if self.current_size == 0:
            # 空字典，返回零值
            empty_keys = torch.zeros_like(query)
            empty_values = torch.zeros(batch_size, top_k, self.value_dim, device=query.device)
            empty_scores = torch.zeros(batch_size, top_k, device=query.device)
            return empty_keys, empty_values, empty_scores
        
        # 计算查询与记忆键的相似度
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(self.memory_keys[:self.current_size], dim=-1)
        
        similarities = torch.matmul(query_norm, keys_norm.transpose(0, 1))  # [batch_size, current_size]
        
        # 应用温度缩放
        similarities = similarities / self.temperature
        
        # 获取top-k相似记忆
        top_similarities, top_indices = torch.topk(similarities, k=min(top_k, self.current_size), dim=-1)
        
        # 应用记忆强度加权
        weighted_similarities = top_similarities * self.memory_strengths[top_indices]
        
        # 检索对应的值
        retrieved_keys = self.memory_keys[top_indices]  # [batch_size, top_k, key_dim]
        retrieved_values = self.memory_values[top_indices]  # [batch_size, top_k, value_dim]
        retrieval_scores = F.softmax(weighted_similarities, dim=-1)  # [batch_size, top_k]
        
        return retrieved_keys, retrieved_values, retrieval_scores
    
    def update_memory(self, 
                     position: torch.Tensor,
                     new_value: torch.Tensor,
                     update_strength: float = 0.1) -> None:
        """
        更新现有记忆
        
        Args:
            position: 记忆位置
            new_value: 新值
            update_strength: 更新强度
        """
        positions = position.long()
        
        # 指数移动平均更新
        self.memory_values[positions] = (1 - update_strength) * self.memory_values[positions] + \
                                       update_strength * new_value
        
        # 增加重要性权重
        self.importance_weights[positions] = torch.min(
            self.importance_weights[positions] + 0.01, 
            torch.ones_like(self.importance_weights[positions])
        )
    
    def forget_weak_memories(self, threshold: float = 0.01) -> int:
        """
        遗忘弱记忆
        
        Args:
            threshold: 遗忘阈值
            
        Returns:
            遗忘的记忆数量
        """
        weak_mask = (self.memory_strengths * self.importance_weights) < threshold
        
        # 重置弱记忆
        self.memory_keys[weak_mask] = torch.randn_like(self.memory_keys[weak_mask]) * 0.1
        self.memory_values[weak_mask] = torch.randn_like(self.memory_values[weak_mask]) * 0.1
        self.memory_strengths[weak_mask] = 0.5
        self.importance_weights[weak_mask] = 0.1
        
        return weak_mask.sum().item()
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        if self.current_size == 0:
            return {
                'current_size': 0,
                'capacity_utilization': 0.0,
                'avg_strength': 0.0,
                'avg_importance': 0.0,
                'strongest_memory': 0.0,
                'weakest_memory': 0.0
            }
        
        active_strengths = self.memory_strengths[:self.current_size]
        active_importance = self.importance_weights[:self.current_size]
        
        return {
            'current_size': self.current_size,
            'capacity_utilization': self.current_size / self.capacity,
            'avg_strength': active_strengths.mean().item(),
            'avg_importance': active_importance.mean().item(),
            'strongest_memory': active_strengths.max().item(),
            'weakest_memory': active_strengths.min().item(),
            'strength_variance': active_strengths.var().item()
        }


class DifferentiableNeuralDictionary(nn.Module):
    """
    可微分神经字典
    集成多个突触记忆细胞的层次化记忆系统
    """
    
    def __init__(self,
                 key_dim: int,
                 value_dim: int,
                 num_cells: int = 8,
                 capacity_per_cell: int = 1000,
                 temperature: float = 1.0,
                 hierarchical_levels: int = 2):
        super().__init__()
        
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_cells = num_cells
        self.hierarchical_levels = hierarchical_levels
        
        # 创建多个突触记忆细胞
        self.memory_cells = nn.ModuleList([
            SynapticMemoryCell(
                key_dim=key_dim,
                value_dim=value_dim,
                capacity=capacity_per_cell,
                temperature=temperature * (1.0 + 0.1 * i)
            ) for i in range(num_cells)
        ])
        
        # 层次化索引网络
        self.hierarchical_indexing = nn.ModuleDict({
            f'level_{i}': nn.Sequential(
                nn.Linear(key_dim, key_dim // 2),
                nn.ReLU(),
                nn.Linear(key_dim // 2, num_cells),
                nn.Softmax(dim=-1)
            ) for i in range(hierarchical_levels)
        })
        
        # 记忆融合网络
        self.memory_fusion = nn.Sequential(
            nn.Linear(value_dim * num_cells, value_dim),
            nn.ReLU(),
            nn.Linear(value_dim, value_dim),
            nn.LayerNorm(value_dim)
        )
        
        # 注意力权重生成器
        self.attention_weights = nn.Sequential(
            nn.Linear(key_dim + value_dim * num_cells, num_cells),
            nn.Softmax(dim=-1)
        )
        
        # 记忆压缩器
        self.memory_compressor = nn.Sequential(
            nn.Linear(value_dim, value_dim // 2),
            nn.ReLU(),
            nn.Linear(value_dim // 2, value_dim // 4),
            nn.ReLU(),
            nn.Linear(value_dim // 4, value_dim),
            nn.Tanh()
        )
        
        # 全局记忆统计
        self.global_memory_stats = {
            'total_writes': 0,
            'total_retrievals': 0,
            'cache_hits': 0
        }
    
    def write_memory(self, 
                    key: torch.Tensor, 
                    value: torch.Tensor,
                    hierarchical_routing: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        层次化写入记忆
        
        Args:
            key: 记忆键 [batch_size, key_dim]
            value: 记忆值 [batch_size, value_dim]
            hierarchical_routing: 层次化路由权重
            
        Returns:
            写入结果统计
        """
        batch_size = key.shape[0]
        
        # 计算层次化路由
        if hierarchical_routing is None:
            hierarchical_routing = {}
            for level, indexing_net in self.hierarchical_indexing.items():
                routing_weight = indexing_net(key)
                hierarchical_routing[level] = routing_weight
        
        write_results = {}
        total_writes = 0
        
        # 在每个记忆细胞中写入
        for i, cell in enumerate(self.memory_cells):
            # 选择此细胞写入的记忆
            cell_mask = hierarchical_routing[f'level_{i % self.hierarchical_levels}'][:, i]
            write_mask = cell_mask > 0.1
            
            if write_mask.sum() > 0:
                # 筛选需要写入的数据
                cell_keys = key[write_mask]
                cell_values = value[write_mask]
                
                # 计算写入强度
                cell_strength = cell_mask[write_mask].unsqueeze(-1)
                
                # 写入记忆
                write_positions = cell.write_memory(cell_keys, cell_values, cell_strength)
                
                write_results[f'cell_{i}_writes'] = write_positions
                total_writes += write_mask.sum().item()
        
        # 更新全局统计
        self.global_memory_stats['total_writes'] += total_writes
        
        return {
            'write_results': write_results,
            'total_writes': total_writes,
            'hierarchical_routing': hierarchical_routing
        }
    
    def retrieve_memory(self, 
                       query: torch.Tensor,
                       top_k: int = 5,
                       fusion_method: str = 'attention') -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        层次化检索记忆
        
        Args:
            query: 查询向量 [batch_size, key_dim]
            top_k: 返回前k个记忆
            fusion_method: 融合方法 ('attention', 'weighted', 'concatenate')
            
        Returns:
            融合的检索结果、检索统计信息
        """
        batch_size = query.shape[0]
        
        # 层次化检索
        all_retrieved_keys = []
        all_retrieved_values = []
        all_retrieval_scores = []
        
        cell_stats = {}
        
        for i, cell in enumerate(self.memory_cells):
            # 从每个细胞检索
            retrieved_keys, retrieved_values, retrieval_scores = cell.retrieve_memory(
                query, top_k=top_k
            )
            
            all_retrieved_keys.append(retrieved_keys)
            all_retrieved_values.append(retrieved_values)
            all_retrieval_scores.append(retrieval_scores)
            
            cell_stats[f'cell_{i}'] = cell.get_memory_statistics()
        
        # 融合检索结果
        if fusion_method == 'attention':
            # 注意力加权融合
            fused_result = self._attention_fusion(all_retrieved_values, all_retrieval_scores, query)
        elif fusion_method == 'weighted':
            # 加权平均融合
            fused_result = self._weighted_fusion(all_retrieved_values, all_retrieval_scores)
        else:  # concatenate
            # 拼接融合
            fused_result = self._concatenate_fusion(all_retrieved_values)
        
        # 更新全局统计
        self.global_memory_stats['total_retrievals'] += batch_size
        
        retrieval_stats = {
            'cell_stats': cell_stats,
            'fusion_method': fusion_method,
            'top_k': top_k,
            'batch_size': batch_size
        }
        
        return fused_result, retrieval_stats
    
    def _attention_fusion(self, 
                         all_values: List[torch.Tensor],
                         all_scores: List[torch.Tensor],
                         query: torch.Tensor) -> torch.Tensor:
        """注意力加权融合"""
        batch_size, top_k, value_dim = all_values[0].shape
        
        # 计算每个细胞的注意力权重
        flattened_values = torch.cat(all_values, dim=1)  # [batch_size, num_cells*top_k, value_dim]
        attention_input = torch.cat([query.unsqueeze(1).expand(-1, flattened_values.shape[1], -1), 
                                   flattened_values], dim=-1)
        
        attention_weights = self.attention_weights(attention_input)  # [batch_size, num_cells*top_k, num_cells]
        
        # 应用注意力权重
        weighted_values = flattened_values * attention_weights.mean(dim=-1).unsqueeze(-1)
        
        # 重新reshape
        weighted_values = weighted_values.view(batch_size, len(all_values), top_k, value_dim)
        
        # 加权求和
        fused_values = (weighted_values * torch.stack(all_scores, dim=1).unsqueeze(-1)).sum(dim=1)
        
        # 最终融合
        fused_result = self.memory_fusion(fused_values.view(batch_size, -1))
        
        return fused_result
    
    def _weighted_fusion(self, 
                        all_values: List[torch.Tensor], 
                        all_scores: List[torch.Tensor]) -> torch.Tensor:
        """加权平均融合"""
        batch_size, top_k, value_dim = all_values[0].shape
        num_cells = len(all_values)
        
        # 计算每个细胞的平均分数
        cell_avg_scores = [scores.mean(dim=1, keepdim=True) for scores in all_scores]
        cell_weights = F.softmax(torch.cat(cell_avg_scores, dim=1), dim=1)
        
        # 加权融合
        weighted_sum = torch.zeros(batch_size, top_k, value_dim, device=all_values[0].device)
        weight_sum = torch.zeros(batch_size, top_k, 1, device=all_values[0].device)
        
        for i, (values, scores) in enumerate(zip(all_values, all_scores)):
            weighted_sum += values * scores.unsqueeze(-1) * cell_weights[:, i:i+1, :]
            weight_sum += scores.unsqueeze(-1) * cell_weights[:, i:i+1, :]
        
        fused_result = weighted_sum / (weight_sum + 1e-8)
        fused_result = fused_result.mean(dim=1)  # 平均top-k结果
        
        return fused_result
    
    def _concatenate_fusion(self, all_values: List[torch.Tensor]) -> torch.Tensor:
        """拼接融合"""
        batch_size, top_k, value_dim = all_values[0].shape
        concatenated = torch.cat(all_values, dim=1)  # [batch_size, num_cells*top_k, value_dim]
        
        # 压缩
        compressed = self.memory_compressor(concatenated.mean(dim=1))
        
        return compressed
    
    def compress_memories(self, compression_ratio: float = 0.8) -> Dict[str, int]:
        """
        压缩记忆（遗忘机制）
        
        Args:
            compression_ratio: 压缩比例
            
        Returns:
            压缩统计
        """
        forgotten_counts = {}
        
        for i, cell in enumerate(self.memory_cells):
            # 基于重要性遗忘
            importance_threshold = 1.0 - compression_ratio
            forgotten_count = cell.forget_weak_memories(threshold=importance_threshold)
            forgotten_counts[f'cell_{i}'] = forgotten_count
        
        return forgotten_counts
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """
        获取全局记忆统计信息
        """
        # 汇总各细胞的统计
        cell_stats = [cell.get_memory_statistics() for cell in self.memory_cells]
        
        global_stats = {
            'global_memory_stats': self.global_memory_stats,
            'cell_statistics': cell_stats,
            'total_capacity': self.num_cells * self.memory_cells[0].capacity,
            'total_current_size': sum(stats['current_size'] for stats in cell_stats),
            'average_utilization': np.mean([stats['capacity_utilization'] for stats in cell_stats]),
            'memory_diversity': self._calculate_memory_diversity()
        }
        
        return global_stats
    
    def _calculate_memory_diversity(self) -> float:
        """计算记忆多样性"""
        if self.memory_cells[0].current_size == 0:
            return 0.0
        
        # 计算所有活跃记忆的方差
        all_strengths = []
        all_importance = []
        
        for cell in self.memory_cells:
            if cell.current_size > 0:
                active_strengths = cell.memory_strengths[:cell.current_size]
                active_importance = cell.importance_weights[:cell.current_size]
                all_strengths.extend(active_strengths.detach().cpu().numpy())
                all_importance.extend(active_importance.detach().cpu().numpy())
        
        if not all_strengths:
            return 0.0
        
        strength_diversity = np.var(all_strengths)
        importance_diversity = np.var(all_importance)
        
        return (strength_diversity + importance_diversity) / 2
    
    def clear_memory(self):
        """清空所有记忆"""
        for cell in self.memory_cells:
            cell.current_size = 0
            cell.write_position = 0
            cell.memory_keys = nn.Parameter(torch.randn_like(cell.memory_keys) * 0.1)
            cell.memory_values = nn.Parameter(torch.randn_like(cell.memory_values) * 0.1)
            cell.memory_strengths = nn.Parameter(torch.ones_like(cell.memory_strengths) * 0.5)
            cell.importance_weights = nn.Parameter(torch.ones_like(cell.importance_weights) * 0.1)
        
        # 重置全局统计
        self.global_memory_stats = {
            'total_writes': 0,
            'total_retrievals': 0,
            'cache_hits': 0
        }