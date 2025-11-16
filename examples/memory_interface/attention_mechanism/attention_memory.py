"""
注意力记忆库 - 统一的记忆读写管理系统
整合读写机制，提供完整的记忆操作接口
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque
import time
from dataclasses import dataclass

from .attention_reader import AttentionReader, AttentionReaderConfig
from .attention_writer import AttentionWriter


@dataclass
class MemoryEntry:
    """记忆条目数据类"""
    content: torch.Tensor
    timestamp: float
    importance: float
    access_count: int
    last_access: float
    memory_type: str
    metadata: Dict[str, Any]
    
    def update_access(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_access = time.time()


class AttentionMemory(nn.Module):
    """
    基于注意力的统一记忆库
    集成读写机制，支持动态记忆管理
    """
    
    def __init__(
        self,
        memory_dim: int = 512,
        max_memories: int = 10000,
        num_heads: int = 8,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        importance_threshold: float = 0.7,
        decay_rate: float = 0.95,
        backup_size: int = 100
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.max_memories = max_memories
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.importance_threshold = importance_threshold
        self.decay_rate = decay_rate
        self.backup_size = backup_size
        
        # 初始化读写组件
        self.reader = AttentionReader(
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_memories=max_memories
        )
        
        self.writer = AttentionWriter(
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_memories=max_memories,
            importance_threshold=importance_threshold,
            decay_rate=decay_rate
        )
        
        # 记忆库存储
        self.register_buffer(
            'memory_bank', 
            torch.zeros(1, max_memories, memory_dim)
        )
        
        # 记忆条目管理
        self.memory_entries = {}
        self.memory_index = deque(maxlen=max_memories)
        self.current_size = 0
        
        # 记忆质量跟踪
        self.quality_history = deque(maxlen=backup_size)
        
        # 记忆压缩状态
        self.compression_ratio = 1.0
        self.is_compressed = False
        
        # 统计信息
        self.stats = {
            'total_reads': 0,
            'total_writes': 0,
            'total_compressions': 0,
            'average_importance': 0.0,
            'memory_utilization': 0.0
        }
        
    def store(
        self,
        content: torch.Tensor,
        memory_type: str = 'episodic',
        importance: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        存储新记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型 (episodic, semantic, procedural, declarative)
            importance: 重要性分数
            metadata: 元数据
            
        Returns:
            记忆ID
        """
        memory_id = f"mem_{self.current_size}_{int(time.time())}"
        
        # 计算重要性（如果未提供）
        if importance is None:
            importance = self._compute_importance(content)
        
        # 创建记忆条目
        entry = MemoryEntry(
            content=content.clone(),
            timestamp=time.time(),
            importance=importance,
            access_count=0,
            last_access=time.time(),
            memory_type=memory_type,
            metadata=metadata or {}
        )
        
        # 检查记忆库容量
        if self.current_size >= self.max_memories:
            self._evict_memory()
        
        # 存储到记忆库
        position = self.current_size
        self.memory_bank[0, position] = content
        self.memory_entries[memory_id] = entry
        self.memory_index.append(memory_id)
        
        self.current_size += 1
        self.stats['total_writes'] += 1
        self._update_statistics()
        
        return memory_id
    
    def retrieve(
        self,
        query: torch.Tensor,
        memory_types: Optional[List[str]] = None,
        top_k: int = 10,
        threshold: float = 0.5
    ) -> List[Dict]:
        """
        检索记忆
        
        Args:
            query: 查询向量
            memory_types: 记忆类型过滤
            top_k: 返回数量
            threshold: 相关性阈值
            
        Returns:
            检索结果列表
        """
        if self.current_size == 0:
            return []
        
        # 限制检索范围
        memory_subset = self.memory_bank[:, :self.current_size]
        
        # 应用记忆类型过滤
        if memory_types:
            filtered_indices = self._filter_by_type(memory_types)
            if len(filtered_indices) > 0:
                memory_subset = memory_subset[:, filtered_indices]
            else:
                return []
        
        # 执行检索
        retrieved_content, attention_weights = self.reader(
            query.unsqueeze(0), memory_subset
        )
        
        # 获取top-k结果
        top_memories, top_indices, top_scores = self.reader.retrieve_top_k(
            query.unsqueeze(0), memory_subset, k=min(top_k, self.current_size), 
            threshold=threshold
        )
        
        # 构建结果
        results = []
        for i in range(top_memories.size(1)):
            idx = top_indices[0, i].item()
            score = top_scores[0, i].item()
            
            # 获取对应的记忆ID
            if memory_types:
                memory_id = list(self.memory_entries.keys())[filtered_indices[idx]]
            else:
                memory_id = list(self.memory_entries.keys())[idx]
            
            # 更新访问统计
            if memory_id in self.memory_entries:
                self.memory_entries[memory_id].update_access()
                results.append({
                    'memory_id': memory_id,
                    'content': top_memories[0, i],
                    'relevance_score': score,
                    'memory_type': self.memory_entries[memory_id].memory_type,
                    'importance': self.memory_entries[memory_id].importance,
                    'timestamp': self.memory_entries[memory_id].timestamp,
                    'access_count': self.memory_entries[memory_id].access_count
                })
        
        self.stats['total_reads'] += 1
        return results
    
    def update(
        self,
        memory_id: str,
        new_content: torch.Tensor,
        update_type: str = 'additive'
    ) -> bool:
        """
        更新现有记忆
        
        Args:
            memory_id: 记忆ID
            new_content: 新内容
            update_type: 更新类型 ('additive', 'replace', 'weighted_average')
            
        Returns:
            是否成功更新
        """
        if memory_id not in self.memory_entries:
            return False
        
        # 获取位置信息
        memory_ids = list(self.memory_entries.keys())
        if memory_id not in memory_ids:
            return False
            
        position = memory_ids.index(memory_id)
        
        # 执行更新
        old_content = self.memory_entries[memory_id].content
        
        if update_type == 'additive':
            updated_content = old_content + new_content
        elif update_type == 'replace':
            updated_content = new_content
        elif update_type == 'weighted_average':
            weight = 0.7  # 旧记忆权重
            updated_content = weight * old_content + (1 - weight) * new_content
        else:
            raise ValueError(f"Unknown update type: {update_type}")
        
        # 更新记忆库和条目
        self.memory_bank[0, position] = updated_content
        self.memory_entries[memory_id].content = updated_content
        self.memory_entries[memory_id].timestamp = time.time()
        
        return True
    
    def forget(self, memory_id: str) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否成功删除
        """
        if memory_id not in self.memory_entries:
            return False
        
        # 从记忆索引中移除
        if memory_id in self.memory_index:
            self.memory_index.remove(memory_id)
        
        # 删除条目
        del self.memory_entries[memory_id]
        
        # 重新组织记忆库（可选）
        self._reorganize_memory()
        
        return True
    
    def consolidate(self, consolidation_ratio: float = 0.3) -> Dict:
        """
        记忆巩固 - 优化记忆存储
        
        Args:
            consolidation_ratio: 巩固比例
            
        Returns:
            巩固结果统计
        """
        start_time = time.time()
        
        # 评估记忆重要性
        importance_scores = []
        for entry in self.memory_entries.values():
            importance_scores.append(entry.importance)
        
        importance_scores = torch.tensor(importance_scores)
        
        # 选择要巩固的记忆
        num_to_consolidate = int(len(importance_scores) * consolidation_ratio)
        _, consolidation_indices = torch.topk(importance_scores, num_to_consolidate)
        
        consolidated_count = 0
        memory_ids = list(self.memory_entries.keys())
        
        # 执行巩固
        for idx in consolidation_indices:
            memory_id = memory_ids[idx]
            entry = self.memory_entries[memory_id]
            
            # 强化重要记忆
            enhancement_factor = 1.1
            enhanced_content = entry.content * enhancement_factor
            
            # 更新记忆库
            position = memory_ids.index(memory_id)
            self.memory_bank[0, position] = enhanced_content
            entry.content = enhanced_content
            entry.importance = min(entry.importance * 1.05, 1.0)
            
            consolidated_count += 1
        
        consolidation_time = time.time() - start_time
        
        return {
            'consolidated_count': consolidated_count,
            'consolidation_time': consolidation_time,
            'total_memories': len(self.memory_entries),
            'consolidation_ratio': consolidation_ratio
        }
    
    def compress(self, compression_ratio: float = 0.5) -> Dict:
        """
        记忆压缩
        
        Args:
            compression_ratio: 压缩比例
            
        Returns:
            压缩结果
        """
        if self.current_size == 0:
            return {'status': 'no_memory_to_compress'}
        
        start_time = time.time()
        
        # 使用写入器的压缩网络
        compressed_memories = self.writer.compression_net(
            self.memory_bank[:, :self.current_size]
        )
        
        # 计算压缩后的尺寸
        compressed_size = int(self.current_size * compression_ratio)
        
        # 保留最重要的记忆
        importance_scores = torch.tensor([
            entry.importance for entry in self.memory_entries.values()
        ])
        
        _, top_indices = torch.topk(importance_scores, compressed_size)
        
        # 重组记忆库
        self.memory_bank[0, :compressed_size] = compressed_memories[:, top_indices]
        
        # 更新记忆条目
        memory_ids = list(self.memory_entries.keys())
        new_entries = {}
        for i, idx in enumerate(top_indices):
            memory_id = memory_ids[idx]
            new_entries[memory_id] = self.memory_entries[memory_id]
            new_entries[memory_id].content = compressed_memories[:, i]
        
        self.memory_entries = new_entries
        self.current_size = compressed_size
        self.compression_ratio = compression_ratio
        self.is_compressed = True
        
        self.stats['total_compressions'] += 1
        compression_time = time.time() - start_time
        
        return {
            'compressed_size': compressed_size,
            'original_size': self.current_size,
            'compression_ratio': compression_ratio,
            'compression_time': compression_time,
            'memory_ids_preserved': len(new_entries)
        }
    
    def _compute_importance(self, content: torch.Tensor) -> float:
        """计算记忆重要性"""
        # 基于内容激活度
        activation = torch.norm(content).item()
        
        # 基于内容复杂度
        complexity = torch.var(content).item()
        
        # 归一化组合
        importance = min(1.0, 0.6 * activation / 10 + 0.4 * complexity)
        
        return importance
    
    def _filter_by_type(self, memory_types: List[str]) -> List[int]:
        """按记忆类型过滤"""
        filtered_indices = []
        for i, entry in enumerate(self.memory_entries.values()):
            if entry.memory_type in memory_types:
                filtered_indices.append(i)
        return filtered_indices
    
    def _evict_memory(self):
        """记忆驱逐 - 删除低重要性记忆"""
        if not self.memory_entries:
            return
        
        # 计算驱逐分数 (重要性 + 使用频率)
        eviction_scores = {}
        current_time = time.time()
        
        for memory_id, entry in self.memory_entries.items():
            usage_frequency = entry.access_count / (current_time - entry.timestamp + 1)
            eviction_score = 0.7 * (1 - entry.importance) + 0.3 * (1 - min(usage_frequency, 1.0))
            eviction_scores[memory_id] = eviction_score
        
        # 驱逐分数最高的记忆
        worst_memory = max(eviction_scores, key=eviction_scores.get)
        self.forget(worst_memory)
    
    def _reorganize_memory(self):
        """重新组织记忆库"""
        memory_ids = list(self.memory_entries.keys())
        self.current_size = len(memory_ids)
        
        # 重新排列记忆库中的内容
        for i, memory_id in enumerate(memory_ids):
            self.memory_bank[0, i] = self.memory_entries[memory_id].content
    
    def _update_statistics(self):
        """更新统计信息"""
        if self.memory_entries:
            total_importance = sum(entry.importance for entry in self.memory_entries.values())
            self.stats['average_importance'] = total_importance / len(self.memory_entries)
        
        self.stats['memory_utilization'] = self.current_size / self.max_memories
    
    def get_statistics(self) -> Dict:
        """获取记忆库统计信息"""
        return {
            **self.stats,
            'current_size': self.current_size,
            'max_size': self.max_memories,
            'compression_status': self.is_compressed,
            'memory_types': {
                memory_type: sum(1 for entry in self.memory_entries.values() 
                               if entry.memory_type == memory_type)
                for memory_type in ['episodic', 'semantic', 'procedural', 'declarative']
            }
        }
    
    def get_memory_analysis(self) -> Dict:
        """深度记忆分析"""
        if not self.memory_entries:
            return {'status': 'no_memories'}
        
        # 重要性分布
        importances = [entry.importance for entry in self.memory_entries.values()]
        
        # 访问模式
        access_times = [entry.access_count for entry in self.memory_entries.values()]
        
        # 时间分布
        timestamps = [entry.timestamp for entry in self.memory_entries.values()]
        age_distribution = [(time.time() - ts) / 3600 for ts in timestamps]  # 小时
        
        return {
            'importance_stats': {
                'mean': np.mean(importances),
                'std': np.std(importances),
                'min': np.min(importances),
                'max': np.max(importances)
            },
            'access_stats': {
                'mean': np.mean(access_times),
                'total': np.sum(access_times)
            },
            'age_stats': {
                'mean_hours': np.mean(age_distribution),
                'std_hours': np.std(age_distribution)
            },
            'memory_quality_distribution': {
                'high_quality': sum(1 for imp in importances if imp > 0.8),
                'medium_quality': sum(1 for imp in importances if 0.3 <= imp <= 0.8),
                'low_quality': sum(1 for imp in importances if imp < 0.3)
            }
        }


if __name__ == "__main__":
    # 测试代码
    memory = AttentionMemory(memory_dim=512, max_memories=1000)
    
    # 创建测试记忆
    test_content1 = torch.randn(512)
    test_content2 = torch.randn(512)
    
    # 存储记忆
    id1 = memory.store(test_content1, memory_type='episodic', importance=0.8)
    id2 = memory.store(test_content2, memory_type='semantic', importance=0.6)
    
    print(f"记忆ID1: {id1}")
    print(f"记忆ID2: {id2}")
    
    # 检索记忆
    query = torch.randn(512)
    results = memory.retrieve(query, top_k=5)
    
    print(f"检索结果数量: {len(results)}")
    for result in results:
        print(f"记忆ID: {result['memory_id']}, 分数: {result['relevance_score']:.3f}")
    
    # 获取统计信息
    stats = memory.get_statistics()
    print(f"记忆库统计: {stats}")
    
    # 巩固测试
    consolidation_result = memory.consolidate(consolidation_ratio=0.5)
    print(f"巩固结果: {consolidation_result}")
    
    # 压缩测试
    compression_result = memory.compress(compression_ratio=0.7)
    print(f"压缩结果: {compression_result}")