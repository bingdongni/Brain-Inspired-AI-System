"""
可微分神经字典
实现情景记忆的存储和检索系统
基于海马体纳米级分辨率的突触结构存储机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import math
from collections import defaultdict


class SynapticStorage(nn.Module):
    """突触存储单元
    
    基于多突触末梢(MSBs)的结构复杂性存储机制
    """
    
    def __init__(self, memory_dim: int, storage_capacity: int = 10000):
        super().__init__()
        self.memory_dim = memory_dim
        self.storage_capacity = storage_capacity
        
        # 记忆模式存储矩阵
        self.memory_patterns = nn.Parameter(
            torch.randn(storage_capacity, memory_dim) * 0.1
        )
        
        # 突触权重矩阵
        self.synaptic_weights = nn.Parameter(
            torch.randn(storage_capacity, memory_dim, memory_dim) * 0.01
        )
        
        # 突触强度门控
        self.synaptic_gates = nn.Parameter(
            torch.ones(storage_capacity, memory_dim)
        )
        
        # 存储使用计数器
        self.usage_counts = torch.zeros(storage_capacity)
        self.usage_temperature = 1.0
        
        # 突触结构复杂性权重
        self.structural_complexity = nn.Parameter(
            torch.ones(storage_capacity) * 0.1
        )
        
    def store_memory(self, memory_vector: torch.Tensor, 
                    target_similarity: float = 0.8) -> Dict[str, Any]:
        """存储记忆到最合适的突触位置"""
        
        batch_size, memory_dim = memory_vector.shape
        
        # 计算与现有记忆的相似度
        similarities = F.cosine_similarity(
            memory_vector.unsqueeze(1), 
            self.memory_patterns.unsqueeze(0), 
            dim=-1
        )  # [batch_size, storage_capacity]
        
        # 选择存储位置（相似度较低的位置）
        target_similarities = similarities - target_similarity
        storage_candidates = torch.where(
            target_similarities < 0, 
            torch.zeros_like(similarities), 
            torch.full_like(similarities, -1e9)
        )
        
        # 使用频率调整（基于LRU）
        lru_weights = 1.0 / (1.0 + self.usage_counts.unsqueeze(0))
        
        # 综合选择权重
        selection_weights = storage_candidates + lru_weights * 0.1
        
        # 选择最佳存储位置
        storage_indices = torch.argmax(selection_weights, dim=-1)  # [batch_size]
        
        # 更新存储内容
        with torch.no_grad():
            for i, idx in enumerate(storage_indices):
                self.memory_patterns[idx] = memory_vector[i].clone()
                
                # 更新突触权重（基于输入相关性）
                correlation = torch.outer(memory_vector[i], memory_vector[i])
                self.synaptic_weights[idx] = correlation * self.structural_complexity[idx]
                
                # 更新突触门控
                self.synaptic_gates[idx] = F.sigmoid(
                    torch.sum(memory_vector[i] ** 2)
                )
            
        # 更新使用计数
        self.usage_counts[storage_indices] += 1
        
        return {
            'storage_indices': storage_indices,
            'similarities': torch.gather(similarities, -1, storage_indices.unsqueeze(-1)),
            'storage_load': self.get_storage_utilization()
        }
    
    def retrieve_memory(self, query_vector: torch.Tensor, 
                       top_k: int = 5) -> Dict[str, torch.Tensor]:
        """从突触存储中检索记忆"""
        
        # 计算查询与存储模式的相似度
        similarities = F.cosine_similarity(
            query_vector.unsqueeze(1), 
            self.memory_patterns.unsqueeze(0), 
            dim=-1
        )  # [batch_size, storage_capacity]
        
        # 获取top-k相似的结果
        top_similarities, top_indices = torch.topk(similarities, top_k, dim=-1)
        
        # 基于突触结构复杂性加权
        complexity_weights = self.structural_complexity[top_indices]
        weighted_similarities = top_similarities * (1.0 + complexity_weights)
        
        # 计算检索结果
        retrieved_patterns = self.memory_patterns[top_indices]  # [batch_size, top_k, memory_dim]
        retrieval_weights = F.softmax(weighted_similarities, dim=-1)
        
        # 加权合并检索结果
        weighted_retrieval = torch.einsum(
            'bki,bk->bi', retrieved_patterns, retrieval_weights
        )
        
        return {
            'retrieved_memory': weighted_retrieval,
            'similarity_scores': top_similarities,
            'storage_indices': top_indices,
            'retrieval_confidence': torch.max(top_similarities, dim=-1)[0]
        }
    
    def get_storage_utilization(self) -> float:
        """获取存储利用率"""
        return float(torch.sum(self.usage_counts > 0) / self.storage_capacity)


class DifferentiableMemoryKey(nn.Module):
    """可微分记忆键
    
    实现基于内容可寻址的检索机制
    """
    
    def __init__(self, key_dim: int, memory_dim: int):
        super().__init__()
        self.key_dim = key_dim
        self.memory_dim = memory_dim
        
        # 键编码器
        self.key_encoder = nn.Sequential(
            nn.Linear(memory_dim, key_dim * 2),
            nn.ReLU(),
            nn.Linear(key_dim * 2, key_dim),
            nn.Tanh()
        )
        
        # 内容哈希函数
        self.content_hash = nn.Sequential(
            nn.Linear(memory_dim, key_dim),
            nn.Tanh()
        )
        
        # 地址生成器
        self.address_generator = nn.Sequential(
            nn.Linear(key_dim * 2, key_dim * 4),
            nn.ReLU(),
            nn.Linear(key_dim * 4, key_dim),
            nn.Sigmoid()
        )
        
    def generate_key(self, memory_content: torch.Tensor) -> torch.Tensor:
        """生成记忆键"""
        
        # 内容特征
        content_features = self.key_encoder(memory_content)
        
        # 内容哈希
        content_hash = self.content_hash(memory_content)
        
        # 组合键
        combined_key = torch.cat([content_features, content_hash], dim=-1)
        
        # 生成地址
        address = self.address_generator(combined_key)
        
        return address


class SynapticConsolidation(nn.Module):
    """突触巩固机制
    
    实现长时记忆巩固和转移
    """
    
    def __init__(self, memory_dim: int):
        super().__init__()
        self.memory_dim = memory_dim
        
        # 短期记忆缓冲
        self.short_term_buffer = nn.GRU(
            memory_dim, memory_dim, batch_first=True, num_layers=2
        )
        
        # 巩固网络
        self.consolidation_network = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(memory_dim * 4, memory_dim * 2),
            nn.ReLU(),
            nn.Linear(memory_dim * 2, memory_dim)
        )
        
        # 记忆强度评估
        self.memory_strength = nn.Sequential(
            nn.Linear(memory_dim, memory_dim // 2),
            nn.ReLU(),
            nn.Linear(memory_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 巩固门控
        self.consolidation_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.Sigmoid()
        )
        
    def consolidate_memory(
        self, 
        short_term_memories: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """巩固短期记忆到长期记忆"""
        
        # 短期记忆处理
        lstm_output, hidden_states = self.short_term_buffer(short_term_memories)
        
        # 结合上下文信息
        if context is not None:
            contextual_input = torch.cat([lstm_output, context.unsqueeze(1).expand(-1, lstm_output.size(1), -1)], dim=-1)
        else:
            contextual_input = lstm_output
        
        # 巩固处理
        consolidated = self.consolidation_network(contextual_input)
        
        # 评估记忆强度
        memory_strength_scores = self.memory_strength(consolidated)
        
        # 巩固门控
        consolidation_gate = self.consolidation_gate(contextual_input)
        
        # 应用巩固
        final_memory = consolidated * consolidation_gate
        
        return {
            'consolidated_memory': final_memory,
            'memory_strength': memory_strength_scores,
            'consolidation_gate': consolidation_gate,
            'hidden_states': hidden_states
        }


class DifferentiableMemoryDictionary(nn.Module):
    """可微分神经字典
    
    整合所有组件的主存储检索系统
    """
    
    def __init__(
        self,
        memory_dim: int,
        storage_capacity: int = 10000,
        key_dim: int = 256,
        consolidation_threshold: float = 0.7
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.storage_capacity = storage_capacity
        self.consolidation_threshold = consolidation_threshold
        
        # 核心组件
        self.synaptic_storage = SynapticStorage(memory_dim, storage_capacity)
        self.memory_key = DifferentiableMemoryKey(key_dim, memory_dim)
        self.synaptic_consolidation = SynapticConsolidation(memory_dim)
        
        # 记忆索引
        self.memory_index = defaultdict(list)
        self.temporal_index = []
        
        # 存储状态
        self.is_training = True
        
    def store_episodic_memory(
        self,
        memory_content: torch.Tensor,
        temporal_context: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """存储情景记忆"""
        
        # 生成记忆键
        memory_key = self.memory_key.generate_key(memory_content)
        
        # 存储到突触系统
        storage_result = self.synaptic_storage.store_memory(memory_content)
        
        # 建立索引
        storage_indices = storage_result['storage_indices']
        for i, idx in enumerate(storage_indices):
            self.memory_index[idx.item()] = {
                'key': memory_key[i],
                'content': memory_content[i],
                'temporal_context': temporal_context[i],
                'metadata': metadata or {}
            }
            self.temporal_index.append(idx.item())
        
        return {
            'storage_indices': storage_indices,
            'memory_keys': memory_key,
            'storage_utilization': storage_result['storage_load']
        }
    
    def retrieve_episodic_memory(
        self,
        query_content: torch.Tensor,
        query_context: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> Dict[str, torch.Tensor]:
        """检索情景记忆"""
        
        # 生成查询键
        query_key = self.memory_key.generate_key(query_content)
        
        # 从突触系统检索
        retrieval_result = self.synaptic_storage.retrieve_memory(query_content, top_k)
        
        # 应用上下文过滤
        if query_context is not None:
            context_filter = self._apply_context_filter(
                retrieval_result, query_context
            )
        else:
            context_filter = retrieval_result
        
        return {
            'retrieved_memories': context_filter['retrieved_memory'],
            'confidence_scores': context_filter['retrieval_confidence'],
            'storage_indices': context_filter['storage_indices'],
            'similarity_scores': context_filter['similarity_scores']
        }
    
    def _apply_context_filter(
        self,
        retrieval_result: Dict[str, torch.Tensor],
        query_context: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """基于上下文过滤检索结果"""
        
        # 简化的上下文过滤实现
        # 实际中可以基于temporal_index进行更精确的过滤
        
        confidence = retrieval_result['retrieval_confidence']
        confidence_filtered = confidence * 0.9  # 应用轻微惩罚
        
        return {
            'retrieved_memory': retrieval_result['retrieved_memory'],
            'retrieval_confidence': confidence_filtered,
            'storage_indices': retrieval_result['storage_indices'],
            'similarity_scores': retrieval_result['similarity_scores']
        }
    
    def consolidate_memories(
        self,
        short_term_memories: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """巩固短期记忆"""
        
        consolidation_result = self.synaptic_consolidation.consolidate_memory(
            short_term_memories, context
        )
        
        # 只有达到阈值的记忆才被巩固
        strong_memories_mask = (
            consolidation_result['memory_strength'] > self.consolidation_threshold
        )
        
        if torch.any(strong_memories_mask):
            # 巩固强记忆
            strong_memory_indices = torch.where(strong_memories_mask)[0]
            
            for idx in strong_memory_indices:
                memory = consolidation_result['consolidated_memory'][idx]
                
                # 重新存储到长期记忆
                storage_result = self.synaptic_storage.store_memory(memory.unsqueeze(0))
                
        return consolidation_result
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        
        return {
            'storage_utilization': self.synaptic_storage.get_storage_utilization(),
            'total_stored_memories': len(self.memory_index),
            'average_synaptic_strength': torch.mean(self.synaptic_storage.synaptic_gates).item(),
            'structural_complexity_mean': torch.mean(self.synaptic_storage.structural_complexity).item()
        }
    
    def clear_memory(self, indices: Optional[List[int]] = None):
        """清除指定记忆"""
        
        if indices is None:
            # 清除所有记忆
            self.synaptic_storage.memory_patterns.data.fill_(0)
            self.synaptic_storage.usage_counts.fill_(0)
            self.memory_index.clear()
            self.temporal_index.clear()
        else:
            # 清除指定记忆
            for idx in indices:
                if idx < self.storage_capacity:
                    self.synaptic_storage.memory_patterns[idx].fill_(0)
                    self.synaptic_storage.usage_counts[idx] = 0
                    if idx in self.memory_index:
                        del self.memory_index[idx]


class MemoryConsolidationScheduler:
    """记忆巩固调度器
    
    管理记忆的自动巩固过程
    """
    
    def __init__(self, dictionary: DifferentiableMemoryDictionary):
        self.dictionary = dictionary
        self.consolidation_schedule = []
        self.consolidation_interval = 100  # 存储100次后触发巩固
        
    def schedule_consolidation(self, memory_id: int, priority: float = 1.0):
        """调度记忆巩固"""
        
        self.consolidation_schedule.append({
            'memory_id': memory_id,
            'priority': priority,
            'timestamp': len(self.temporal_index)
        })
        
        # 按优先级排序
        self.consolidation_schedule.sort(key=lambda x: x['priority'], reverse=True)
    
    def execute_scheduled_consolidation(self):
        """执行调度的巩固任务"""
        
        if len(self.consolidation_schedule) >= self.consolidation_interval:
            # 选择高优先级记忆进行巩固
            top_memories = self.consolidation_schedule[:10]
            
            for memory_info in top_memories:
                memory_id = memory_info['memory_id']
                if memory_id in self.dictionary.memory_index:
                    memory_data = self.dictionary.memory_index[memory_id]
                    
                    # 提取记忆内容
                    memory_content = memory_data['content'].unsqueeze(0)
                    memory_context = memory_data['temporal_context'].unsqueeze(0)
                    
                    # 执行巩固
                    consolidation_result = self.dictionary.consolidate_memories(
                        memory_content, memory_context
                    )
                    
                    print(f"记忆 {memory_id} 巩固完成，强度: {consolidation_result['memory_strength'].item():.3f}")
            
            # 清空调度
            self.consolidation_schedule.clear()


if __name__ == "__main__":
    # 测试代码
    memory_dim = 512
    batch_size = 4
    
    # 创建字典
    memory_dict = DifferentiableMemoryDictionary(memory_dim)
    
    # 生成测试记忆
    memories = torch.randn(batch_size, memory_dim)
    contexts = torch.randn(batch_size, 128)
    
    # 存储记忆
    store_result = memory_dict.store_episodic_memory(memories, contexts)
    print(f"存储结果: {store_result['storage_indices']}")
    
    # 检索记忆
    query = memories[0]
    retrieval_result = memory_dict.retrieve_episodic_memory(query.unsqueeze(0), top_k=3)
    print(f"检索结果形状: {retrieval_result['retrieved_memories'].shape}")
    
    # 获取统计信息
    stats = memory_dict.get_memory_statistics()
    print(f"统计信息: {stats}")