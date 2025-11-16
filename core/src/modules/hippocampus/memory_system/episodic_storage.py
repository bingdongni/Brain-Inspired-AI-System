"""
情景记忆存储和检索系统
基于海马体纳米级分辨率的突触结构实现完整的情景记忆系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import time
from datetime import datetime, timedelta
import json


class TemporalContextEncoder(nn.Module):
    """时间上下文编码器
    
    编码时间和空间上下文信息
    """
    
    def __init__(self, base_dim: int, temporal_dim: int = 128):
        super().__init__()
        self.base_dim = base_dim
        self.temporal_dim = temporal_dim
        
        # 时间编码
        self.time_encoder = nn.Sequential(
            nn.Linear(1, temporal_dim),
            nn.ReLU(),
            nn.Linear(temporal_dim, temporal_dim),
            nn.Tanh()
        )
        
        # 空间编码
        self.spatial_encoder = nn.Sequential(
            nn.Linear(2, temporal_dim),  # x, y坐标
            nn.ReLU(),
            nn.Linear(temporal_dim, temporal_dim),
            nn.Tanh()
        )
        
        # 上下文整合器
        self.context_integrator = nn.Sequential(
            nn.Linear(temporal_dim * 2, temporal_dim),
            nn.ReLU(),
            nn.Linear(temporal_dim, base_dim),
            nn.Tanh()
        )
        
        # 时间连续性编码器
        self.temporal_continuity = nn.GRU(
            base_dim, base_dim, batch_first=True
        )
        
    def encode_temporal_context(
        self, 
        timestamp: Union[float, torch.Tensor],
        spatial_coords: Optional[Union[Tuple[float, float], torch.Tensor]] = None
    ) -> torch.Tensor:
        """编码时间空间上下文"""
        
        # 时间编码
        if isinstance(timestamp, (int, float)):
            timestamp_tensor = torch.tensor([[timestamp]], dtype=torch.float32)
        else:
            timestamp_tensor = timestamp.unsqueeze(-1) if timestamp.dim() == 1 else timestamp
            
        time_encoding = self.time_encoder(timestamp_tensor)
        
        # 空间编码
        if spatial_coords is not None:
            if isinstance(spatial_coords, (tuple, list)):
                spatial_tensor = torch.tensor([spatial_coords], dtype=torch.float32)
            else:
                spatial_tensor = spatial_coords
            spatial_encoding = self.spatial_encoder(spatial_tensor)
        else:
            spatial_encoding = torch.zeros_like(time_encoding)
        
        # 整合时空信息
        context_features = torch.cat([time_encoding, spatial_encoding], dim=-1)
        integrated_context = self.context_integrator(context_features)
        
        return integrated_context
    
    def encode_temporal_sequence(self, memory_sequence: torch.Tensor) -> torch.Tensor:
        """编码时间序列记忆"""
        
        # 应用时间连续性编码
        sequence_output, _ = self.temporal_continuity(memory_sequence)
        
        # 取最后一个时间步的输出
        final_encoding = sequence_output[:, -1, :]
        
        return final_encoding


class EpisodicMemoryCell(nn.Module):
    """情景记忆单元
    
    存储单个情景记忆的完整信息
    """
    
    def __init__(self, content_dim: int, context_dim: int):
        super().__init__()
        self.content_dim = content_dim
        self.context_dim = context_dim
        
        # 记忆内容编码器
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, content_dim * 2),
            nn.ReLU(),
            nn.Linear(content_dim * 2, content_dim),
            nn.Tanh()
        )
        
        # 记忆强度调制器
        self.intensity_modulator = nn.Sequential(
            nn.Linear(content_dim + context_dim, 1),
            nn.Sigmoid()
        )
        
        # 记忆质量评估器
        self.quality_assessor = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.ReLU(),
            nn.Linear(content_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 记忆持久性预测器
        self.persistence_predictor = nn.Sequential(
            nn.Linear(content_dim + context_dim, content_dim // 4),
            nn.ReLU(),
            nn.Linear(content_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 关联网络
        self.association_network = nn.Sequential(
            nn.Linear(content_dim * 2, content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, content_dim),
            nn.Tanh()
        )
        
    def create_episodic_memory(
        self,
        content: torch.Tensor,
        context: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> 'EpisodicMemory':
        """创建情景记忆实例"""
        
        # 编码内容
        encoded_content = self.content_encoder(content)
        
        # 计算记忆强度
        intensity_input = torch.cat([encoded_content, context], dim=-1)
        memory_intensity = self.intensity_modulator(intensity_input)
        
        # 评估记忆质量
        memory_quality = self.quality_assessor(encoded_content)
        
        # 预测持久性
        persistence_input = torch.cat([encoded_content, context], dim=-1)
        predicted_persistence = self.persistence_predictor(persistence_input)
        
        # 记忆实例
        memory = EpisodicMemory(
            content=encoded_content,
            context=context,
            intensity=memory_intensity,
            quality=memory_quality,
            predicted_persistence=predicted_persistence,
            metadata=metadata or {}
        )
        
        return memory


class EpisodicMemory:
    """情景记忆类
    
    存储和管理单个情景记忆
    """
    
    def __init__(
        self,
        content: torch.Tensor,
        context: torch.Tensor,
        intensity: torch.Tensor,
        quality: torch.Tensor,
        predicted_persistence: torch.Tensor,
        metadata: Dict[str, Any]
    ):
        self.content = content
        self.context = context
        self.intensity = intensity
        self.quality = quality
        self.predicted_persistence = predicted_persistence
        self.metadata = metadata
        
        # 记忆属性
        self.creation_time = time.time()
        self.last_access_time = self.creation_time
        self.access_count = 1
        
        # 记忆状态
        self.is_consolidated = False
        self.consolidation_level = 0.0
        self.forgetting_rate = 0.0
        
    def access(self) -> float:
        """访问记忆，返回访问时间戳"""
        self.last_access_time = time.time()
        self.access_count += 1
        return self.last_access_time
    
    def update_consolidation(self, consolidation_level: float):
        """更新记忆巩固程度"""
        self.consolidation_level = consolidation_level
        if consolidation_level > 0.8:
            self.is_consolidated = True
    
    def calculate_current_strength(self, current_time: Optional[float] = None) -> float:
        """计算当前记忆强度"""
        if current_time is None:
            current_time = time.time()
        
        # 时间衰减
        time_elapsed = current_time - self.creation_time
        time_decay = math.exp(-self.forgetting_rate * time_elapsed)
        
        # 访问频率增强
        access_boost = 1.0 + math.log(1.0 + self.access_count) * 0.1
        
        # 巩固程度增强
        consolidation_boost = 1.0 + self.consolidation_level * 0.5
        
        base_strength = float(self.intensity.item() * self.quality.item())
        current_strength = base_strength * time_decay * access_boost * consolidation_boost
        
        return min(current_strength, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'content': self.content.detach().cpu().numpy().tolist(),
            'context': self.context.detach().cpu().numpy().tolist(),
            'intensity': float(self.intensity.item()),
            'quality': float(self.quality.item()),
            'predicted_persistence': float(self.predicted_persistence.item()),
            'creation_time': self.creation_time,
            'last_access_time': self.last_access_time,
            'access_count': self.access_count,
            'is_consolidated': self.is_consolidated,
            'consolidation_level': float(self.consolidation_level),
            'metadata': self.metadata
        }


class HippocampalIndexing(nn.Module):
    """海马体索引系统
    
    实现高效的情景记忆索引和检索
    """
    
    def __init__(self, content_dim: int, context_dim: int, index_size: int = 10000):
        super().__init__()
        self.content_dim = content_dim
        self.context_dim = context_dim
        self.index_size = index_size
        
        # 内容哈希函数
        self.content_hasher = nn.Sequential(
            nn.Linear(content_dim, content_dim // 2),
            nn.Tanh(),
            nn.Linear(content_dim // 2, content_dim // 4),
            nn.Tanh()
        )
        
        # 上下文哈希函数
        self.context_hasher = nn.Sequential(
            nn.Linear(context_dim, context_dim // 2),
            nn.Tanh(),
            nn.Linear(context_dim // 2, context_dim // 4),
            nn.Tanh()
        )
        
        # 综合索引键生成器
        self.index_key_generator = nn.Sequential(
            nn.Linear(content_dim // 4 + context_dim // 4, content_dim // 8),
            nn.Sigmoid()
        )
        
        # 相似性计算器
        self.similarity_calculator = nn.CosineSimilarity(dim=-1)
        
        # 索引表
        self.index_table = {}
        self.temporal_index = []
        self.spatial_index = {}
        
    def generate_index_key(self, content: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """生成索引键"""
        
        content_hash = self.content_hasher(content)
        context_hash = self.context_hasher(context)
        
        combined_hash = torch.cat([content_hash, context_hash], dim=-1)
        index_key = self.index_key_generator(combined_hash)
        
        return index_key
    
    def add_to_index(self, memory: EpisodicMemory, index_key: torch.Tensor) -> str:
        """添加到索引"""
        
        key_str = str(hash(index_key.cpu().numpy().tobytes()))
        
        # 添加到内容索引
        if key_str not in self.index_table:
            self.index_table[key_str] = []
        self.index_table[key_str].append(memory)
        
        # 添加到时间索引
        self.temporal_index.append((memory.creation_time, key_str))
        
        # 添加到空间索引
        context_data = memory.metadata.get('spatial_coords')
        if context_data:
            spatial_key = f"{context_data[0]:.2f}_{context_data[1]:.2f}"
            if spatial_key not in self.spatial_index:
                self.spatial_index[spatial_key] = []
            self.spatial_index[spatial_key].append(key_str)
        
        return key_str
    
    def search_by_content_similarity(
        self, 
        query_content: torch.Tensor, 
        query_context: torch.Tensor,
        threshold: float = 0.7,
        max_results: int = 10
    ) -> List[Tuple[EpisodicMemory, float]]:
        """基于内容相似性搜索"""
        
        query_key = self.generate_index_key(query_content, query_context)
        
        results = []
        
        # 遍历所有索引项
        for key_str, memories in self.index_table.items():
            for memory in memories:
                memory_key = self.generate_index_key(memory.content, memory.context)
                
                # 计算相似度
                similarity = self.similarity_calculator(query_key.unsqueeze(0), memory_key.unsqueeze(0))
                
                if similarity > threshold:
                    results.append((memory, float(similarity)))
        
        # 按相似度排序并限制结果数量
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def search_by_temporal_context(
        self, 
        time_window: Tuple[float, float],
        max_results: int = 10
    ) -> List[EpisodicMemory]:
        """基于时间窗口搜索"""
        
        start_time, end_time = time_window
        
        # 过滤时间范围内的记忆
        filtered_memories = []
        for timestamp, key_str in self.temporal_index:
            if start_time <= timestamp <= end_time:
                if key_str in self.index_table:
                    filtered_memories.extend(self.index_table[key_str])
        
        # 按访问时间和强度排序
        filtered_memories.sort(
            key=lambda m: (m.last_access_time, m.calculate_current_strength()),
            reverse=True
        )
        
        return filtered_memories[:max_results]
    
    def search_by_spatial_context(
        self, 
        spatial_coords: Tuple[float, float],
        radius: float = 0.5,
        max_results: int = 10
    ) -> List[EpisodicMemory]:
        """基于空间位置搜索"""
        
        results = []
        target_x, target_y = spatial_coords
        
        # 查找空间索引
        for spatial_key, key_strs in self.spatial_index.items():
            coords = spatial_key.split('_')
            if len(coords) == 2:
                memory_x, memory_y = float(coords[0]), float(coords[1])
                
                # 计算空间距离
                distance = math.sqrt((memory_x - target_x)**2 + (memory_y - target_y)**2)
                
                if distance <= radius:
                    for key_str in key_strs:
                        if key_str in self.index_table:
                            results.extend(self.index_table[key_str])
        
        # 去重并排序
        unique_results = list(set(results))
        unique_results.sort(
            key=lambda m: m.calculate_current_strength(),
            reverse=True
        )
        
        return unique_results[:max_results]


class EpisodicMemorySystem(nn.Module):
    """情景记忆系统
    
    整合所有组件的完整情景记忆系统
    """
    
    def __init__(
        self,
        content_dim: int,
        context_dim: int = 128,
        max_memories: int = 100000,
        forgetting_rate: float = 0.01
    ):
        super().__init__()
        self.content_dim = content_dim
        self.context_dim = context_dim
        self.max_memories = max_memories
        self.forgetting_rate = forgetting_rate
        
        # 时间上下文编码器
        self.temporal_encoder = TemporalContextEncoder(content_dim, context_dim)
        
        # 情景记忆单元
        self.memory_cell = EpisodicMemoryCell(content_dim, content_dim)
        
        # 海马体索引系统
        self.hippocampal_index = HippocampalIndexing(content_dim, content_dim)
        
        # 记忆管理器
        self.memory_manager = nn.Sequential(
            nn.Linear(content_dim * 2, content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, 4),  # 存储、检索、巩固、遗忘控制
            nn.Softmax(dim=-1)
        )
        
        # 记忆检索器
        self.memory_retriever = nn.Sequential(
            nn.Linear(content_dim * 2, content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, content_dim),
            nn.Sigmoid()
        )
        
        # 系统状态
        self.stored_memories = []
        self.system_stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'average_strength': 0.0,
            'consolidation_rate': 0.0
        }
        
    def store_episodic_memory(
        self,
        content: torch.Tensor,
        timestamp: float,
        spatial_coords: Optional[Tuple[float, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """存储情景记忆"""
        
        # 编码时间和空间上下文
        temporal_context = self.temporal_encoder.encode_temporal_context(
            timestamp, spatial_coords
        )
        
        # 创建记忆单元
        memory = self.memory_cell.create_episodic_memory(
            content, temporal_context, metadata
        )
        
        # 设置遗忘率
        memory.forgetting_rate = self.forgetting_rate
        
        # 生成索引键并添加到索引
        index_key = self.hippocampal_index.generate_index_key(content, temporal_context)
        memory_id = self.hippocampal_index.add_to_index(memory, index_key)
        
        # 管理记忆存储
        self._manage_memory_storage()
        
        # 更新统计
        self.stored_memories.append(memory)
        self.system_stats['total_stored'] += 1
        
        return memory_id
    
    def retrieve_episodic_memory(
        self,
        query_content: torch.Tensor,
        query_context: Optional[torch.Tensor] = None,
        search_type: str = 'similarity',
        **search_params
    ) -> Dict[str, Any]:
        """检索情景记忆"""
        
        if query_context is None:
            # 使用当前时间作为默认上下文
            current_time = time.time()
            query_context = self.temporal_encoder.encode_temporal_context(current_time)
        
        retrieval_start_time = time.time()
        
        if search_type == 'similarity':
            # 基于相似性搜索
            threshold = search_params.get('threshold', 0.7)
            max_results = search_params.get('max_results', 10)
            
            search_results = self.hippocampal_index.search_by_content_similarity(
                query_content, query_context, threshold, max_results
            )
            
        elif search_type == 'temporal':
            # 基于时间搜索
            time_window = search_params.get('time_window', (0, time.time()))
            max_results = search_params.get('max_results', 10)
            
            memories = self.hippocampal_index.search_by_temporal_context(
                time_window, max_results
            )
            
            search_results = [(memory, memory.calculate_current_strength()) for memory in memories]
            
        elif search_type == 'spatial':
            # 基于空间搜索
            spatial_coords = search_params.get('spatial_coords', (0.0, 0.0))
            radius = search_params.get('radius', 0.5)
            max_results = search_params.get('max_results', 10)
            
            memories = self.hippocampal_index.search_by_spatial_context(
                spatial_coords, radius, max_results
            )
            
            search_results = [(memory, memory.calculate_current_strength()) for memory in memories]
            
        else:
            search_results = []
        
        # 访问检索到的记忆
        for memory, _ in search_results:
            memory.access()
        
        # 计算检索置信度
        if search_results:
            avg_strength = np.mean([strength for _, strength in search_results])
            retrieval_confidence = avg_strength
        else:
            retrieval_confidence = 0.0
        
        # 整合检索结果
        if search_results:
            # 重建记忆内容
            retrieved_contents = torch.stack([memory.content for memory, _ in search_results])
            retrieval_weights = torch.tensor([strength for _, strength in search_results])
            retrieval_weights = F.softmax(retrieval_weights, dim=0)
            
            integrated_retrieval = torch.einsum(
                'nc,n->c', retrieved_contents, retrieval_weights
            ).unsqueeze(0)
            
            final_content = self.memory_retriever(
                torch.cat([query_content, integrated_retrieval], dim=-1)
            )
        else:
            final_content = query_content
        
        retrieval_time = time.time() - retrieval_start_time
        
        # 更新统计
        self.system_stats['total_retrieved'] += 1
        self.system_stats['average_strength'] = (
            (self.system_stats['average_strength'] * (self.system_stats['total_retrieved'] - 1) + retrieval_confidence) /
            self.system_stats['total_retrieved']
        )
        
        return {
            'retrieved_memories': search_results,
            'integrated_content': final_content,
            'retrieval_confidence': retrieval_confidence,
            'retrieval_time': retrieval_time,
            'search_type': search_type,
            'num_results': len(search_results)
        }
    
    def consolidate_memories(self, consolidation_threshold: float = 0.7):
        """巩固重要记忆"""
        
        consolidated_count = 0
        for memory in self.stored_memories:
            current_strength = memory.calculate_current_strength()
            
            if current_strength > consolidation_threshold:
                memory.update_consolidation(min(current_strength, 1.0))
                consolidated_count += 1
        
        consolidation_rate = consolidated_count / max(len(self.stored_memories), 1)
        self.system_stats['consolidation_rate'] = (
            (self.system_stats['consolidation_rate'] * (self.system_stats['total_stored'] - consolidated_count) + consolidation_rate * consolidated_count) /
            max(self.system_stats['total_stored'], 1)
        )
        
        return {
            'consolidated_count': consolidated_count,
            'consolidation_rate': consolidation_rate
        }
    
    def _manage_memory_storage(self):
        """管理记忆存储容量"""
        
        if len(self.stored_memories) > self.max_memories:
            # 移除最弱的记忆
            memory_strengths = [
                (memory, memory.calculate_current_strength()) 
                for memory in self.stored_memories
            ]
            memory_strengths.sort(key=lambda x: x[1])
            
            # 移除10%的最弱记忆
            num_to_remove = max(1, len(memory_strengths) // 10)
            memories_to_remove = [memory for memory, _ in memory_strengths[:num_to_remove]]
            
            for memory in memories_to_remove:
                self.stored_memories.remove(memory)
                
            # 清理索引（简化实现）
            # 实际中需要更复杂的索引维护逻辑
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        
        current_strengths = [memory.calculate_current_strength() for memory in self.stored_memories]
        consolidated_count = sum(1 for memory in self.stored_memories if memory.is_consolidated)
        
        return {
            'total_memories_stored': len(self.stored_memories),
            'system_stats': self.system_stats,
            'average_memory_strength': np.mean(current_strengths) if current_strengths else 0.0,
            'consolidated_memories': consolidated_count,
            'consolidation_percentage': consolidated_count / max(len(self.stored_memories), 1),
            'storage_utilization': len(self.stored_memories) / self.max_memories
        }
    
    def export_memories(self, filepath: str):
        """导出记忆到文件"""
        
        export_data = {
            'system_stats': self.system_stats,
            'memories': [memory.to_dict() for memory in self.stored_memories],
            'export_time': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def clear_system(self):
        """清空系统"""
        
        self.stored_memories.clear()
        self.hippocampal_index.index_table.clear()
        self.hippocampal_index.temporal_index.clear()
        self.hippocampal_index.spatial_index.clear()
        
        self.system_stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'average_strength': 0.0,
            'consolidation_rate': 0.0
        }


if __name__ == "__main__":
    # 测试代码
    content_dim = 512
    context_dim = 128
    
    # 创建情景记忆系统
    memory_system = EpisodicMemorySystem(content_dim, context_dim)
    
    # 生成测试数据
    test_content = torch.randn(1, content_dim)
    test_timestamp = time.time()
    test_coords = (1.0, 2.0)
    
    # 存储记忆
    memory_id = memory_system.store_episodic_memory(
        test_content, test_timestamp, test_coords, 
        metadata={'type': 'test', 'description': 'Test memory'}
    )
    print(f"存储记忆，ID: {memory_id}")
    
    # 检索记忆
    retrieval_result = memory_system.retrieve_episodic_memory(
        test_content, search_type='similarity', threshold=0.5
    )
    print(f"检索结果: {retrieval_result['retrieval_confidence']:.3f}")
    
    # 获取统计信息
    stats = memory_system.get_system_statistics()
    print(f"系统统计: {stats}")
    
    # 巩固记忆
    consolidation_result = memory_system.consolidate_memories()
    print(f"巩固结果: {consolidation_result}")