#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
情景记忆模块
===========

实现情景记忆的存储、检索和管理功能。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import pickle

class EpisodeMemory:
    """单个情景记忆单元"""
    
    def __init__(self, memory_id: str, content: Any, metadata: Optional[Dict] = None):
        self.id = memory_id
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.retrieval_count = 0
        self.strength = 1.0
        self.associations = []  # 关联记忆ID列表
        self.emotional_valence = 0.0  # 情感价值 [-1, 1]
        self.attention_weight = 1.0   # 注意力权重
        
    def strengthen(self, factor: float = 1.1):
        """强化记忆强度"""
        self.strength *= factor
        self.retrieval_count += 1
    
    def weaken(self, factor: float = 0.95):
        """减弱记忆强度"""
        self.strength *= factor
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'retrieval_count': self.retrieval_count,
            'strength': self.strength,
            'associations': self.associations,
            'emotional_valence': self.emotional_valence,
            'attention_weight': self.attention_weight
        }

class EpisodicMemory:
    """情景记忆系统"""
    
    def __init__(self, max_capacity: int = 10000):
        """
        初始化情景记忆系统
        
        Args:
            max_capacity: 最大记忆容量
        """
        self.max_capacity = max_capacity
        self.memories: Dict[str, EpisodeMemory] = {}
        self.memory_index = 0
        
        # 记忆分类索引
        self.category_index = {}  # category -> set of memory_ids
        self.time_index = {}      # timestamp -> memory_id
        self.association_graph = {}  # memory_id -> set of associated memory_ids
        
        # 记忆统计
        self.total_stored = 0
        self.total_retrieved = 0
        self.average_strength = 0.0
        
    def store_episode(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """
        存储一个情景记忆
        
        Args:
            content: 记忆内容
            metadata: 元数据
            
        Returns:
            记忆ID
        """
        memory_id = f"ep_{self.memory_index}"
        self.memory_index += 1
        
        # 创建记忆对象
        memory = EpisodeMemory(memory_id, content, metadata)
        
        # 如果超过容量，删除最弱的记忆
        if len(self.memories) >= self.max_capacity:
            self._remove_weakest_memory()
        
        # 存储记忆
        self.memories[memory_id] = memory
        self.total_stored += 1
        
        # 更新索引
        self._update_indices(memory)
        
        return memory_id
    
    def retrieve_episode(self, memory_id: str) -> Optional[EpisodeMemory]:
        """
        检索情景记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            记忆对象，如果不存在返回None
        """
        if memory_id not in self.memories:
            return None
        
        memory = self.memories[memory_id]
        memory.strengthen()
        self.total_retrieved += 1
        
        return memory
    
    def search_by_content(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """
        基于内容搜索记忆
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            (记忆ID, 相似度分数)列表，按相似度降序排列
        """
        results = []
        query_lower = query.lower()
        
        for memory_id, memory in self.memories.items():
            if isinstance(memory.content, str):
                # 简单的字符串匹配
                content_lower = memory.content.lower()
                if query_lower in content_lower:
                    similarity = self._calculate_text_similarity(query_lower, content_lower)
                    results.append((memory_id, similarity))
            
            # 检查元数据
            if 'tags' in memory.metadata:
                for tag in memory.metadata['tags']:
                    if query_lower in str(tag).lower():
                        similarity = 0.8
                        results.append((memory_id, similarity))
        
        # 按相似度排序并限制结果数量
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（简单实现）"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def retrieve_by_association(self, memory_id: str, depth: int = 1) -> List[str]:
        """
        通过关联检索记忆
        
        Args:
            memory_id: 起始记忆ID
            depth: 关联深度
            
        Returns:
            关联记忆ID列表
        """
        if memory_id not in self.association_graph:
            return []
        
        associated_ids = set()
        current_level = {memory_id}
        
        for _ in range(depth):
            next_level = set()
            for mem_id in current_level:
                if mem_id in self.association_graph:
                    associated = self.association_graph[mem_id]
                    associated_ids.update(associated)
                    next_level.update(associated)
            current_level = next_level
        
        return list(associated_ids)
    
    def create_association(self, memory_id1: str, memory_id2: str, strength: float = 1.0):
        """
        在两个记忆之间创建关联
        
        Args:
            memory_id1: 第一个记忆ID
            memory_id2: 第二个记忆ID
            strength: 关联强度
        """
        if memory_id1 not in self.memories or memory_id2 not in self.memories:
            return
        
        # 更新关联图
        if memory_id1 not in self.association_graph:
            self.association_graph[memory_id1] = set()
        if memory_id2 not in self.association_graph:
            self.association_graph[memory_id2] = set()
        
        self.association_graph[memory_id1].add(memory_id2)
        self.association_graph[memory_id2].add(memory_id1)
        
        # 更新记忆对象
        self.memories[memory_id1].associations.append(memory_id2)
        self.memories[memory_id2].associations.append(memory_id1)
        
        # 增强关联记忆的强度
        self.memories[memory_id1].strength *= (1 + strength * 0.1)
        self.memories[memory_id2].strength *= (1 + strength * 0.1)
    
    def get_recent_memories(self, count: int = 10) -> List[EpisodeMemory]:
        """获取最近的记忆"""
        memories = sorted(self.memories.values(), 
                         key=lambda x: x.timestamp, 
                         reverse=True)
        return memories[:count]
    
    def get_strongest_memories(self, count: int = 10) -> List[EpisodeMemory]:
        """获取最强的记忆"""
        memories = sorted(self.memories.values(), 
                         key=lambda x: x.strength, 
                         reverse=True)
        return memories[:count]
    
    def get_memories_by_category(self, category: str) -> List[EpisodeMemory]:
        """根据类别获取记忆"""
        if category not in self.category_index:
            return []
        
        memory_ids = self.category_index[category]
        return [self.memories[mem_id] for mem_id in memory_ids if mem_id in self.memories]
    
    def _update_indices(self, memory: EpisodeMemory):
        """更新各种索引"""
        # 时间索引
        time_key = memory.timestamp.strftime('%Y-%m-%d %H:%M')
        self.time_index[time_key] = memory.id
        
        # 类别索引
        if 'category' in memory.metadata:
            category = memory.metadata['category']
            if category not in self.category_index:
                self.category_index[category] = set()
            self.category_index[category].add(memory.id)
    
    def _remove_weakest_memory(self):
        """删除最弱的记忆"""
        if not self.memories:
            return
        
        # 找到最弱的记忆
        weakest_id = min(self.memories.keys(), 
                        key=lambda x: self.memories[x].strength)
        
        # 从索引中删除
        memory = self.memories[weakest_id]
        
        # 从类别索引中删除
        if 'category' in memory.metadata:
            category = memory.metadata['category']
            if category in self.category_index:
                self.category_index[category].discard(weakest_id)
                if not self.category_index[category]:
                    del self.category_index[category]
        
        # 从关联图中删除
        if weakest_id in self.association_graph:
            for associated_id in self.association_graph[weakest_id]:
                if associated_id in self.association_graph:
                    self.association_graph[associated_id].discard(weakest_id)
            del self.association_graph[weakest_id]
        
        # 删除记忆
        del self.memories[weakest_id]
    
    def consolidate_memories(self, criteria: str = 'strength') -> int:
        """
        巩固记忆
        
        Args:
            criteria: 巩固标准 ('strength', 'recency', 'frequency')
            
        Returns:
            巩固的记忆数量
        """
        consolidated_count = 0
        
        if criteria == 'strength':
            # 巩固强度较高的记忆
            for memory in self.memories.values():
                if memory.strength > 2.0:
                    memory.strength *= 1.5
                    memory.metadata['consolidated'] = True
                    consolidated_count += 1
        
        elif criteria == 'recency':
            # 巩固最近的记忆
            recent_cutoff = datetime.now().timestamp() - 3600  # 1小时前
            for memory in self.memories.values():
                if memory.timestamp.timestamp() > recent_cutoff:
                    memory.strength *= 1.2
                    consolidated_count += 1
        
        elif criteria == 'frequency':
            # 巩固频繁访问的记忆
            for memory in self.memories.values():
                if memory.retrieval_count > 5:
                    memory.strength *= 1.3
                    consolidated_count += 1
        
        return consolidated_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取记忆系统统计信息"""
        if not self.memories:
            return {'total_memories': 0}
        
        strengths = [memory.strength for memory in self.memories.values()]
        retrieval_counts = [memory.retrieval_count for memory in self.memories.values()]
        
        return {
            'total_memories': len(self.memories),
            'total_stored': self.total_stored,
            'total_retrieved': self.total_retrieved,
            'average_strength': np.mean(strengths),
            'max_strength': np.max(strengths),
            'min_strength': np.min(strengths),
            'average_retrieval_count': np.mean(retrieval_counts),
            'max_retrieval_count': np.max(retrieval_counts),
            'categories': len(self.category_index),
            'associations': sum(len(assoc) for assoc in self.association_graph.values()) // 2
        }
    
    def save_to_file(self, filepath: str):
        """保存记忆系统到文件"""
        data = {
            'memories': {mid: memory.to_dict() for mid, memory in self.memories.items()},
            'max_capacity': self.max_capacity,
            'memory_index': self.memory_index,
            'category_index': {cat: list(mem_ids) for cat, mem_ids in self.category_index.items()},
            'time_index': self.time_index,
            'association_graph': {mid: list(assoc_ids) for mid, assoc_ids in self.association_graph.items()},
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    
    def load_from_file(self, filepath: str):
        """从文件加载记忆系统"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 重建记忆对象
        self.memories = {}
        for mid, mem_data in data['memories'].items():
            memory = EpisodeMemory(mid, mem_data['content'], mem_data['metadata'])
            memory.retrieval_count = mem_data['retrieval_count']
            memory.strength = mem_data['strength']
            memory.associations = mem_data['associations']
            memory.emotional_valence = mem_data['emotional_valence']
            memory.attention_weight = mem_data['attention_weight']
            self.memories[mid] = memory
        
        # 重建索引
        self.category_index = {cat: set(mem_ids) for cat, mem_ids in data['category_index'].items()}
        self.time_index = data['time_index']
        self.association_graph = {mid: set(assoc_ids) for mid, assoc_ids in data['association_graph'].items()}
        
        self.max_capacity = data['max_capacity']
        self.memory_index = data['memory_index']