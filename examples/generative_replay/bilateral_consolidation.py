"""
双侧记忆巩固模块

实现双侧记忆巩固机制，用于长期记忆的存储和巩固。
基于海马体-新皮层的记忆巩固理论，模拟生物记忆的巩固过程。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from collections import defaultdict, deque
import time
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """
    记忆项数据结构
    """
    content: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: float
    consolidation_level: float = 0.0
    access_count: int = 0
    last_access: float = 0.0
    strength: float = 1.0
    memory_type: str = 'episodic'  # episodic, semantic, procedural


@dataclass
class ConsolidationEvent:
    """
    巩固事件数据结构
    """
    memory_id: str
    source_level: str  # hippocampus, cortex
    target_level: str  # hippocampus, cortex
    timestamp: float
    success_rate: float
    consolidation_factor: float


class HippocampalMemory:
    """
    海马体记忆系统
    
    负责短期记忆存储和快速学习。
    """
    
    def __init__(self, 
                 capacity: int = 10000,
                 decay_rate: float = 0.01,
                 consolidation_threshold: float = 0.8):
        """
        初始化海马体记忆系统
        
        Args:
            capacity: 记忆容量
            decay_rate: 记忆衰减率
            consolidation_threshold: 巩固阈值
        """
        self.capacity = capacity
        self.decay_rate = decay_rate
        self.consolidation_threshold = consolidation_threshold
        
        # 记忆存储
        self.memories: Dict[str, MemoryItem] = {}
        self.memory_queue = deque(maxlen=capacity)
        
        # 统计信息
        self.total_stored = 0
        self.total_consolidated = 0
        self.consolidation_history = []
        
        logger.info(f"海马体记忆系统初始化，容量: {capacity}")
    
    def store_memory(self, 
                    content: torch.Tensor,
                    metadata: Optional[Dict[str, Any]] = None,
                    memory_id: Optional[str] = None) -> str:
        """
        存储记忆
        
        Args:
            content: 记忆内容
            metadata: 元数据
            memory_id: 记忆ID
            
        Returns:
            记忆ID
        """
        if memory_id is None:
            memory_id = f"hippo_{self.total_stored}_{int(time.time() * 1000)}"
        
        if metadata is None:
            metadata = {}
        
        memory = MemoryItem(
            content=content,
            metadata=metadata,
            timestamp=time.time(),
            memory_type=metadata.get('type', 'episodic')
        )
        
        self.memories[memory_id] = memory
        self.memory_queue.append(memory_id)
        
        # 如果超过容量，移除最旧的记忆
        if len(self.memories) > self.capacity:
            oldest_id = self.memory_queue.popleft()
            if oldest_id in self.memories:
                del self.memories[oldest_id]
        
        self.total_stored += 1
        
        return memory_id
    
    def retrieve_memory(self, memory_id: str, access_factor: float = 1.0) -> Optional[MemoryItem]:
        """
        检索记忆
        
        Args:
            memory_id: 记忆ID
            access_factor: 访问强度因子
            
        Returns:
            记忆项
        """
        if memory_id not in self.memories:
            return None
        
        memory = self.memories[memory_id]
        
        # 更新访问信息
        current_time = time.time()
        time_diff = current_time - memory.last_access
        memory.last_access = current_time
        memory.access_count += 1
        
        # 增强记忆强度
        memory.strength = min(1.0, memory.strength + access_factor * 0.1)
        
        # 自然衰减
        memory.strength *= math.exp(-self.decay_rate * time_diff / 3600)  # 按小时衰减
        
        return memory
    
    def get_consolidation_candidates(self, 
                                   limit: Optional[int] = None) -> List[Tuple[str, MemoryItem]]:
        """
        获取需要巩固的记忆候选
        
        Args:
            limit: 限制数量
            
        Returns:
            候选记忆列表
        """
        candidates = []
        
        for memory_id, memory in self.memories.items():
            # 计算巩固分数
            consolidation_score = self._compute_consolidation_score(memory)
            
            if consolidation_score >= self.consolidation_threshold:
                candidates.append((memory_id, memory))
        
        # 按巩固分数排序
        candidates.sort(key=lambda x: self._compute_consolidation_score(x[1]), reverse=True)
        
        if limit:
            candidates = candidates[:limit]
        
        return candidates
    
    def _compute_consolidation_score(self, memory: MemoryItem) -> float:
        """
        计算记忆的巩固分数
        
        Args:
            memory: 记忆项
            
        Returns:
            巩固分数 [0, 1]
        """
        # 基础分数基于记忆强度
        base_score = memory.strength
        
        # 访问频率加权
        access_bonus = min(0.3, memory.access_count * 0.05)
        
        # 时间衰减惩罚
        time_diff = time.time() - memory.timestamp
        time_penalty = min(0.2, time_diff / (24 * 3600) * 0.05)  # 24小时内的惩罚
        
        # 记忆类型加权
        type_bonus = {
            'episodic': 0.1,
            'semantic': 0.2,
            'procedural': 0.05
        }.get(memory.memory_type, 0.0)
        
        consolidation_score = base_score + access_bonus + type_bonus - time_penalty
        return max(0.0, min(1.0, consolidation_score))
    
    def consolidate_memories(self, 
                           cortex_memory: 'CorticalMemory',
                           batch_size: int = 100) -> List[ConsolidationEvent]:
        """
        巩固记忆到皮层
        
        Args:
            cortex_memory: 皮层记忆系统
            batch_size: 批次大小
            
        Returns:
            巩固事件列表
        """
        candidates = self.get_consolidation_candidates(limit=batch_size)
        consolidation_events = []
        
        for memory_id, memory in candidates:
            try:
                # 模拟巩固过程
                success_rate = self._simulate_consolidation_process(memory)
                consolidation_factor = success_rate * memory.strength
                
                # 创建巩固事件
                event = ConsolidationEvent(
                    memory_id=memory_id,
                    source_level='hippocampus',
                    target_level='cortex',
                    timestamp=time.time(),
                    success_rate=success_rate,
                    consolidation_factor=consolidation_factor
                )
                consolidation_events.append(event)
                
                # 如果巩固成功，将记忆转移到皮层
                if success_rate > 0.7:
                    cortex_memory.store_memory(
                        content=memory.content,
                        metadata=memory.metadata.copy(),
                        strength=consolidation_factor
                    )
                    
                    # 从海马体移除或降低强度
                    if memory_id in self.memories:
                        memory.strength *= 0.1  # 大幅降低强度
                        if memory.strength < 0.01:
                            del self.memories[memory_id]
                
                self.consolidation_history.append(event)
                self.total_consolidated += 1
                
            except Exception as e:
                logger.error(f"记忆巩固失败 {memory_id}: {str(e)}")
        
        logger.info(f"完成记忆巩固，处理 {len(consolidation_events)} 个记忆项")
        return consolidation_events
    
    def _simulate_consolidation_process(self, memory: MemoryItem) -> float:
        """
        模拟巩固过程
        
        Args:
            memory: 记忆项
            
        Returns:
            巩固成功率
        """
        # 基于记忆特征模拟巩固概率
        base_probability = 0.5
        
        # 记忆强度影响
        strength_factor = memory.strength
        
        # 访问频率影响
        access_factor = min(1.0, memory.access_count / 10)
        
        # 时间影响（适中的时间间隔有利）
        time_diff = time.time() - memory.timestamp
        time_factor = 1.0 - abs(time_diff - 3600) / 7200  # 1小时附近最佳
        time_factor = max(0.1, min(1.0, time_factor))
        
        # 记忆类型影响
        type_factors = {
            'episodic': 0.8,
            'semantic': 0.9,
            'procedural': 0.7
        }
        type_factor = type_factors.get(memory.memory_type, 0.6)
        
        # 计算最终概率
        consolidation_probability = (
            base_probability * 0.2 +
            strength_factor * 0.4 +
            access_factor * 0.2 +
            time_factor * 0.1 +
            type_factor * 0.1
        )
        
        return max(0.0, min(1.0, consolidation_probability))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取海马体统计信息
        
        Returns:
            统计信息字典
        """
        active_memories = len(self.memories)
        avg_strength = np.mean([m.strength for m in self.memories.values()]) if self.memories else 0.0
        avg_consolidation_level = np.mean([m.consolidation_level for m in self.memories.values()]) if self.memories else 0.0
        
        return {
            'active_memories': active_memories,
            'total_stored': self.total_stored,
            'total_consolidated': self.total_consolidated,
            'average_strength': avg_strength,
            'average_consolidation_level': avg_consolidation_level,
            'consolidation_rate': self.total_consolidated / max(self.total_stored, 1),
            'capacity_utilization': active_memories / self.capacity
        }


class CorticalMemory:
    """
    皮层记忆系统
    
    负责长期记忆存储和巩固。
    """
    
    def __init__(self, 
                 capacity: int = 100000,
                 consolidation_rate: float = 0.001,
                 interference_threshold: float = 0.8):
        """
        初始化皮层记忆系统
        
        Args:
            capacity: 记忆容量
            consolidation_rate: 巩固速率
            interference_threshold: 干扰阈值
        """
        self.capacity = capacity
        self.consolidation_rate = consolidation_rate
        self.interference_threshold = interference_threshold
        
        # 记忆存储
        self.memories: Dict[str, MemoryItem] = {}
        self.memory_categories = defaultdict(list)
        
        # 统计信息
        self.total_consolidated = 0
        self.interference_events = 0
        self.consolidation_history = []
        
        logger.info(f"皮层记忆系统初始化，容量: {capacity}")
    
    def store_memory(self,
                    content: torch.Tensor,
                    metadata: Optional[Dict[str, Any]] = None,
                    strength: float = 1.0,
                    memory_id: Optional[str] = None) -> str:
        """
        存储记忆
        
        Args:
            content: 记忆内容
            metadata: 元数据
            strength: 记忆强度
            memory_id: 记忆ID
            
        Returns:
            记忆ID
        """
        if memory_id is None:
            memory_id = f"cortex_{self.total_consolidated}_{int(time.time() * 1000)}"
        
        if metadata is None:
            metadata = {}
        
        memory = MemoryItem(
            content=content,
            metadata=metadata,
            timestamp=time.time(),
            strength=strength,
            consolidation_level=1.0,  # 皮层记忆已经完全巩固
            memory_type=metadata.get('type', 'semantic')
        )
        
        # 检查干扰
        self._check_interference(memory)
        
        # 存储记忆
        self.memories[memory_id] = memory
        category = metadata.get('category', 'default')
        self.memory_categories[category].append(memory_id)
        
        # 如果超过容量，移除最弱的记忆
        if len(self.memories) > self.capacity:
            self._remove_weakest_memory()
        
        self.total_consolidated += 1
        
        return memory_id
    
    def _check_interference(self, new_memory: MemoryItem) -> bool:
        """
        检查记忆干扰
        
        Args:
            new_memory: 新记忆
            
        Returns:
            是否发生干扰
        """
        interference_detected = False
        
        for memory_id, existing_memory in self.memories.items():
            # 计算内容相似度
            similarity = self._compute_content_similarity(new_memory.content, existing_memory.content)
            
            if similarity > self.interference_threshold:
                self.interference_events += 1
                interference_detected = True
                
                # 处理干扰：降低两个记忆的强度
                new_memory.strength *= 0.9
                existing_memory.strength *= 0.8
                
                logger.warning(f"检测到记忆干扰: {memory_id}, 相似度: {similarity:.3f}")
        
        return interference_detected
    
    def _compute_content_similarity(self, content1: torch.Tensor, content2: torch.Tensor) -> float:
        """
        计算内容相似度
        
        Args:
            content1: 内容1
            content2: 内容2
            
        Returns:
            相似度分数
        """
        # 使用余弦相似度
        if content1.shape != content2.shape:
            # 调整形状
            min_size = min(content1.numel(), content2.numel())
            content1_flat = content1.flatten()[:min_size]
            content2_flat = content2.flatten()[:min_size]
        else:
            content1_flat = content1.flatten()
            content2_flat = content2.flatten()
        
        similarity = F.cosine_similarity(content1_flat.unsqueeze(0), 
                                       content2_flat.unsqueeze(0))
        return similarity.item()
    
    def _remove_weakest_memory(self) -> None:
        """移除最弱的记忆"""
        if not self.memories:
            return
        
        # 找到最弱的记忆
        weakest_id = min(self.memories.keys(), 
                        key=lambda x: self.memories[x].strength)
        
        # 从类别中移除
        metadata = self.memories[weakest_id].metadata
        category = metadata.get('category', 'default')
        if weakest_id in self.memory_categories[category]:
            self.memory_categories[category].remove(weakest_id)
        
        # 删除记忆
        del self.memories[weakest_id]
        
        logger.debug(f"移除最弱记忆: {weakest_id}")
    
    def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """
        检索记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            记忆项
        """
        if memory_id not in self.memories:
            return None
        
        memory = self.memories[memory_id]
        
        # 增强记忆强度（通过检索）
        memory.strength = min(1.0, memory.strength + 0.01)
        memory.access_count += 1
        
        return memory
    
    def search_memories(self, 
                       query_content: torch.Tensor,
                       category: Optional[str] = None,
                       similarity_threshold: float = 0.7,
                       max_results: int = 10) -> List[Tuple[str, float]]:
        """
        搜索记忆
        
        Args:
            query_content: 查询内容
            category: 类别过滤
            similarity_threshold: 相似度阈值
            max_results: 最大结果数
            
        Returns:
            (记忆ID, 相似度)列表
        """
        results = []
        
        # 搜索范围
        search_memories = self.memories
        if category:
            memory_ids = self.memory_categories.get(category, [])
            search_memories = {mid: self.memories[mid] for mid in memory_ids if mid in self.memories}
        
        for memory_id, memory in search_memories.items():
            similarity = self._compute_content_similarity(query_content, memory.content)
            
            if similarity >= similarity_threshold:
                results.append((memory_id, similarity))
        
        # 按相似度排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:max_results]
    
    def consolidate_weak_memories(self) -> int:
        """
        巩固弱记忆
        
        Returns:
            巩固的记忆数量
        """
        consolidated_count = 0
        
        weak_memories = [(mid, mem) for mid, mem in self.memories.items() 
                        if mem.strength < 0.1]
        
        for memory_id, memory in weak_memories:
            # 尝试通过重复访问增强记忆
            memory.strength += self.consolidation_rate
            
            if memory.strength >= 0.1:
                consolidated_count += 1
        
        return consolidated_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取皮层统计信息
        
        Returns:
            统计信息字典
        """
        active_memories = len(self.memories)
        avg_strength = np.mean([m.strength for m in self.memories.values()]) if self.memories else 0.0
        
        # 类别分布
        category_distribution = {cat: len(mids) for cat, mids in self.memory_categories.items()}
        
        return {
            'active_memories': active_memories,
            'total_consolidated': self.total_consolidated,
            'interference_events': self.interference_events,
            'average_strength': avg_strength,
            'capacity_utilization': active_memories / self.capacity,
            'category_distribution': category_distribution,
            'interference_rate': self.interference_events / max(self.total_consolidated, 1)
        }


class BilateralMemoryConsolidation:
    """
    双侧记忆巩固系统
    
    整合海马体和皮层记忆系统，实现完整的记忆巩固流程。
    """
    
    def __init__(self,
                 hippocampal_capacity: int = 10000,
                 cortical_capacity: int = 100000,
                 consolidation_interval: int = 3600):  # 1小时
        """
        初始化双侧记忆巩固系统
        
        Args:
            hippocampal_capacity: 海马体容量
            cortical_capacity: 皮层容量
            consolidation_interval: 巩固间隔（秒）
        """
        # 创建记忆系统
        self.hippocampus = HippocampalMemory(capacity=hippocampal_capacity)
        self.cortex = CorticalMemory(capacity=cortical_capacity)
        
        self.consolidation_interval = consolidation_interval
        self.last_consolidation = time.time()
        
        # 统计信息
        self.total_operations = 0
        self.consolidation_cycles = 0
        
        logger.info("双侧记忆巩固系统初始化完成")
    
    def store_episodic_memory(self,
                            content: torch.Tensor,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        存储情景记忆（存储到海马体）
        
        Args:
            content: 记忆内容
            metadata: 元数据
            
        Returns:
            记忆ID
        """
        if metadata is None:
            metadata = {}
        
        metadata['type'] = 'episodic'
        
        memory_id = self.hippocampus.store_memory(content, metadata)
        self.total_operations += 1
        
        return memory_id
    
    def store_semantic_memory(self,
                            content: torch.Tensor,
                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        存储语义记忆（直接存储到皮层）
        
        Args:
            content: 记忆内容
            metadata: 元数据
            
        Returns:
            记忆ID
        """
        if metadata is None:
            metadata = {}
        
        metadata['type'] = 'semantic'
        
        memory_id = self.cortex.store_memory(content, metadata)
        self.total_operations += 1
        
        return memory_id
    
    def retrieve_memory(self, 
                       memory_id: str,
                       source_level: str = 'auto') -> Optional[MemoryItem]:
        """
        检索记忆
        
        Args:
            memory_id: 记忆ID
            source_level: 来源层次 ('hippocampus', 'cortex', 'auto')
            
        Returns:
            记忆项
        """
        if source_level == 'auto':
            # 优先在海马体中查找
            memory = self.hippocampus.retrieve_memory(memory_id)
            if memory is None:
                memory = self.cortex.retrieve_memory(memory_id)
        elif source_level == 'hippocampus':
            memory = self.hippocampus.retrieve_memory(memory_id)
        else:  # cortex
            memory = self.cortex.retrieve_memory(memory_id)
        
        self.total_operations += 1
        return memory
    
    def search_memory(self,
                     query_content: torch.Tensor,
                     source_level: str = 'all',
                     similarity_threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """
        搜索记忆
        
        Args:
            query_content: 查询内容
            source_level: 搜索范围 ('hippocampus', 'cortex', 'all')
            similarity_threshold: 相似度阈值
            
        Returns:
            (记忆ID, 来源, 相似度)列表
        """
        results = []
        
        if source_level in ['hippocampus', 'all']:
            # 在海马体中搜索
            for memory_id, memory in self.hippocampus.memories.items():
                similarity = self.cortex._compute_content_similarity(query_content, memory.content)
                if similarity >= similarity_threshold:
                    results.append((memory_id, 'hippocampus', similarity))
        
        if source_level in ['cortex', 'all']:
            # 在皮层中搜索
            cortex_results = self.cortex.search_memories(
                query_content, similarity_threshold=similarity_threshold)
            for memory_id, similarity in cortex_results:
                results.append((memory_id, 'cortex', similarity))
        
        # 按相似度排序
        results.sort(key=lambda x: x[2], reverse=True)
        
        self.total_operations += 1
        return results
    
    def run_consolidation_cycle(self) -> Dict[str, Any]:
        """
        运行巩固周期
        
        Returns:
            巩固结果统计
        """
        current_time = time.time()
        
        # 检查是否需要巩固
        if current_time - self.last_consolidation < self.consolidation_interval:
            return {'skipped': True, 'reason': '未达到巩固间隔'}
        
        logger.info("开始记忆巩固周期")
        
        # 记录开始时间
        cycle_start = time.time()
        
        # 执行巩固
        consolidation_events = self.hippocampus.consolidate_memories(
            self.cortex, batch_size=1000)
        
        # 巩固皮层弱记忆
        weak_memories_consolidated = self.cortex.consolidate_weak_memories()
        
        # 更新统计
        cycle_duration = time.time() - cycle_start
        self.last_consolidation = current_time
        self.consolidation_cycles += 1
        
        # 统计信息
        results = {
            'cycle_number': self.consolidation_cycles,
            'cycle_duration': cycle_duration,
            'consolidated_memories': len(consolidation_events),
            'weak_memories_consolidated': weak_memories_consolidated,
            'success_rate': np.mean([event.success_rate for event in consolidation_events]) if consolidation_events else 0.0,
            'average_consolidation_factor': np.mean([event.consolidation_factor for event in consolidation_events]) if consolidation_events else 0.0
        }
        
        logger.info(f"记忆巩固周期完成: {results}")
        
        self.total_operations += len(consolidation_events)
        
        return results
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            完整系统统计信息
        """
        hippocampal_stats = self.hippocampus.get_statistics()
        cortical_stats = self.cortex.get_statistics()
        
        # 系统级统计
        time_since_last_consolidation = time.time() - self.last_consolidation
        
        system_stats = {
            'hippocampus': hippocampal_stats,
            'cortex': cortical_stats,
            'system': {
                'total_operations': self.total_operations,
                'consolidation_cycles': self.consolidation_cycles,
                'time_since_last_consolidation': time_since_last_consolidation,
                'consolidation_interval': self.consolidation_interval,
                'next_consolidation_due': time_since_last_consolidation >= self.consolidation_interval
            }
        }
        
        return system_stats
    
    def clear_all_memories(self) -> None:
        """清空所有记忆"""
        self.hippocampus.memories.clear()
        self.hippocampus.memory_queue.clear()
        self.cortex.memories.clear()
        self.cortex.memory_categories.clear()
        
        logger.info("所有记忆已清空")