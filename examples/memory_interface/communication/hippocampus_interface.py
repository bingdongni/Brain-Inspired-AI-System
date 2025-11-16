"""
海马体接口
提供海马体侧的通信接口，处理与新皮层的信息交互
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio
import time
from dataclasses import dataclass
from collections import deque

from .protocol_handler import CommunicationProtocol, NetworkConfig
from .message_types import (
    MessageType, MemoryType, Priority, MemoryMessage, 
    QueryMessage, MessageBuilder, MessageValidator
)
from .message_types import MemoryMessage as CustomMemoryMessage


@dataclass
class HippocampusConfig:
    """海马体配置"""
    memory_capacity: int = 50000
    encoding_dim: int = 512
    working_memory_size: int = 100
    pattern_separation_threshold: float = 0.7
    pattern_completion_enabled: bool = True
    temporal_sequences: bool = True
    context_window: int = 10


class HippocampusInterface:
    """
    海马体接口
    管理海马体记忆编码、存储和检索
    """
    
    def __init__(self, config: HippocampusConfig, protocol: CommunicationProtocol = None):
        self.config = config
        self.protocol = protocol or CommunicationProtocol()
        
        # 记忆存储
        self.episodic_memory = {}  # 情景记忆
        self.working_memory = deque(maxlen=config.working_memory_size)  # 工作记忆
        self.temporal_sequences = {}  # 时间序列
        
        # 编码器和解码器
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, config.encoding_dim),
            nn.Tanh()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(config.encoding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.Tanh()
        )
        
        # 模式分离和完成
        self.pattern_separator = nn.Linear(config.encoding_dim, config.encoding_dim)
        self.pattern_completer = nn.Linear(config.encoding_dim // 2, config.encoding_dim)
        
        # 索引系统
        self.content_index = {}  # 内容索引
        self.temporal_index = {}  # 时间索引
        self.context_index = {}  # 上下文索引
        
        # 统计信息
        self.stats = {
            'episodic_memories': 0,
            'working_memory_accesses': 0,
            'pattern_completions': 0,
            'temporal_sequences_created': 0,
            'compression_ratio': 0.0,
            'retrieval_accuracy': 0.0
        }
        
        # 当前时间步
        self.current_time_step = 0
        
    async def initialize(self) -> bool:
        """初始化海马体接口"""
        try:
            # 启动通信协议
            if not self.protocol.is_running:
                await self.protocol.start()
            
            # 注册消息处理器
            self.protocol.register_handler(MessageType.MEMORY_RETRIEVE, self._handle_retrieve_request)
            self.protocol.register_handler(MessageType.QUERY_PROCESS, self._handle_query_request)
            self.protocol.register_handler(MessageType.SYNC_REQUEST, self._handle_sync_request)
            
            print("海马体接口初始化成功")
            return True
            
        except Exception as e:
            print(f"海马体接口初始化失败: {e}")
            return False
    
    async def encode_experience(
        self,
        experience: torch.Tensor,
        context: torch.Tensor = None,
        importance: float = 0.5
    ) -> str:
        """
        编码经验到记忆
        
        Args:
            experience: 经验内容
            context: 上下文信息
            importance: 重要性分数
            
        Returns:
            记忆ID
        """
        try:
            # 经验编码
            encoded_experience = self.encoder(experience)
            
            # 上下文编码（如果提供）
            encoded_context = None
            if context is not None:
                encoded_context = self.encoder(context)
                encoded_experience = torch.cat([encoded_experience, encoded_context], dim=-1)
            
            # 模式分离
            separated_pattern = self._apply_pattern_separation(encoded_experience)
            
            # 存储到情景记忆
            memory_id = self._store_episodic_memory(
                separated_pattern, encoded_experience, importance
            )
            
            # 更新时间序列
            if self.config.temporal_sequences:
                self._update_temporal_sequence(memory_id, separated_pattern)
            
            # 更新工作记忆
            self.working_memory.append({
                'memory_id': memory_id,
                'content': separated_pattern,
                'timestamp': self.current_time_step
            })
            
            # 更新统计
            self.stats['episodic_memories'] += 1
            
            # 发送存储确认
            await self._send_storage_confirmation(memory_id, encoded_experience)
            
            self.current_time_step += 1
            return memory_id
            
        except Exception as e:
            print(f"经验编码失败: {e}")
            return ""
    
    async def retrieve_memory(
        self,
        query: torch.Tensor,
        memory_type: Optional[MemoryType] = MemoryType.EPISODIC,
        top_k: int = 5,
        context: torch.Tensor = None
    ) -> List[Dict]:
        """
        检索记忆
        
        Args:
            query: 查询向量
            memory_type: 记忆类型
            top_k: 返回数量
            context: 上下文
            
        Returns:
            检索结果列表
        """
        try:
            # 编码查询
            encoded_query = self.encoder(query)
            
            # 如果有上下文，融合查询
            if context is not None:
                encoded_context = self.encoder(context)
                encoded_query = torch.cat([encoded_query, encoded_context], dim=-1)
            
            # 在情景记忆中搜索
            episodic_results = self._search_episodic_memory(encoded_query, top_k)
            
            # 模式完成
            if self.config.pattern_completion_enabled and episodic_results:
                completed_results = self._apply_pattern_completion(episodic_results)
                episodic_results = completed_results
            
            # 在工作记忆中搜索
            working_results = self._search_working_memory(encoded_query, top_k // 2)
            
            # 合并结果
            all_results = episodic_results + working_results
            
            # 按相关性排序
            all_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # 更新统计
            if all_results:
                self.stats['retrieval_accuracy'] = self._calculate_retrieval_accuracy(
                    all_results[:top_k]
                )
            
            return all_results[:top_k]
            
        except Exception as e:
            print(f"记忆检索失败: {e}")
            return []
    
    async def retrieve_sequence(
        self,
        partial_sequence: torch.Tensor,
        sequence_length: int = 5
    ) -> torch.Tensor:
        """
        检索时间序列
        
        Args:
            partial_sequence: 部分序列
            sequence_length: 序列长度
            
        Returns:
            完整序列
        """
        try:
            # 编码部分序列
            encoded_partial = self.encoder(partial_sequence)
            
            # 寻找匹配的序列
            matched_sequences = self._find_matching_sequences(encoded_partial)
            
            if matched_sequences:
                # 返回最匹配的序列
                best_sequence = matched_sequences[0]['sequence']
                return best_sequence[:sequence_length]
            else:
                # 如果没有找到匹配，生成预测序列
                predicted_sequence = self._generate_sequence_prediction(
                    encoded_partial, sequence_length
                )
                return predicted_sequence
            
        except Exception as e:
            print(f"序列检索失败: {e}")
            return partial_sequence
    
    async def consolidate_memories(
        self,
        time_window: int = 100,
        consolidation_threshold: float = 0.7
    ) -> Dict:
        """
        巩固记忆
        
        Args:
            time_window: 时间窗口
            consolidation_threshold: 巩固阈值
            
        Returns:
            巩固结果
        """
        start_time = time.time()
        
        try:
            consolidated_count = 0
            
            # 分析时间窗口内的记忆
            current_step = self.current_time_step
            window_start = max(0, current_step - time_window)
            
            for memory_id, memory_data in self.episodic_memory.items():
                if (memory_data.get('timestamp', 0) >= window_start and 
                    memory_data.get('importance', 0) >= consolidation_threshold):
                    
                    # 强化记忆
                    memory_data['strength'] = memory_data.get('strength', 0.5) * 1.1
                    memory_data['last_consolidated'] = current_step
                    
                    # 压缩重复模式
                    compressed_pattern = self._compress_memory_pattern(
                        memory_data['pattern']
                    )
                    memory_data['pattern'] = compressed_pattern
                    
                    consolidated_count += 1
            
            consolidation_time = time.time() - start_time
            
            # 更新统计
            self.stats['compression_ratio'] = self._calculate_compression_ratio()
            
            result = {
                'consolidated_memories': consolidated_count,
                'consolidation_time': consolidation_time,
                'total_memories': len(self.episodic_memory),
                'compression_ratio': self.stats['compression_ratio']
            }
            
            print(f"记忆巩固完成: {consolidated_count}个记忆被巩固")
            return result
            
        except Exception as e:
            print(f"记忆巩固失败: {e}")
            return {'error': str(e)}
    
    def get_working_memory_state(self) -> Dict:
        """获取工作记忆状态"""
        working_memory_items = list(self.working_memory)
        
        return {
            'current_size': len(working_memory_items),
            'max_size': self.config.working_memory_size,
            'recent_memories': working_memory_items[-5:],  # 最近5个记忆
            'time_range': {
                'oldest': min([item['timestamp'] for item in working_memory_items]) if working_memory_items else 0,
                'newest': max([item['timestamp'] for item in working_memory_items]) if working_memory_items else 0
            }
        }
    
    def get_statistics(self) -> Dict:
        """获取海马体统计信息"""
        return {
            **self.stats,
            'episodic_memory_size': len(self.episodic_memory),
            'working_memory_size': len(self.working_memory),
            'temporal_sequences_count': len(self.temporal_sequences),
            'current_time_step': self.current_time_step,
            'encoding_dim': self.config.encoding_dim,
            'pattern_separation_threshold': self.config.pattern_separation_threshold
        }
    
    def _store_episodic_memory(
        self,
        pattern: torch.Tensor,
        full_encoded: torch.Tensor,
        importance: float
    ) -> str:
        """存储情景记忆"""
        memory_id = f"episodic_{int(time.time() * 1000)}_{len(self.episodic_memory)}"
        
        memory_data = {
            'pattern': pattern.clone(),
            'full_encoded': full_encoded.clone(),
            'importance': importance,
            'timestamp': self.current_time_step,
            'strength': importance,
            'access_count': 0,
            'last_accessed': self.current_time_step
        }
        
        self.episodic_memory[memory_id] = memory_data
        
        # 更新索引
        self._update_indices(memory_id, pattern)
        
        return memory_id
    
    def _search_episodic_memory(
        self,
        query: torch.Tensor,
        top_k: int
    ) -> List[Dict]:
        """在情景记忆中搜索"""
        similarities = []
        
        for memory_id, memory_data in self.episodic_memory.items():
            # 计算余弦相似度
            similarity = torch.cosine_similarity(
                query.unsqueeze(0), 
                memory_data['pattern'].unsqueeze(0)
            ).item()
            
            # 应用模式分离阈值
            if similarity >= self.config.pattern_separation_threshold:
                similarities.append({
                    'memory_id': memory_id,
                    'relevance_score': similarity,
                    'memory_data': memory_data,
                    'memory_type': 'episodic'
                })
        
        # 按相似度排序
        similarities.sort(key=lambda x: x['relevance_score'], reverse=True)
        return similarities[:top_k]
    
    def _search_working_memory(
        self,
        query: torch.Tensor,
        top_k: int
    ) -> List[Dict]:
        """在工作记忆中搜索"""
        results = []
        
        for item in self.working_memory:
            similarity = torch.cosine_similarity(
                query.unsqueeze(0), 
                item['content'].unsqueeze(0)
            ).item()
            
            results.append({
                'memory_id': item['memory_id'],
                'relevance_score': similarity,
                'memory_data': item,
                'memory_type': 'working'
            })
        
        # 按相似度排序
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
    
    def _apply_pattern_separation(self, encoded_pattern: torch.Tensor) -> torch.Tensor:
        """应用模式分离"""
        # 使用随机投影增加模式分离
        random_projection = torch.randn_like(encoded_pattern) * 0.1
        separated = encoded_pattern + random_projection
        
        # 归一化
        separated = torch.nn.functional.normalize(separated, p=2, dim=-1)
        
        return separated
    
    def _apply_pattern_completion(self, memories: List[Dict]) -> List[Dict]:
        """应用模式完成"""
        completed_memories = []
        
        for memory in memories:
            # 提取部分模式
            pattern = memory['memory_data']['pattern']
            partial_pattern = pattern[:len(pattern)//2]
            
            # 使用模式完成器预测完整模式
            completed_pattern = self.pattern_completer(partial_pattern)
            
            # 计算完成置信度
            completion_confidence = torch.cosine_similarity(
                pattern.unsqueeze(0),
                completed_pattern.unsqueeze(0)
            ).item()
            
            memory['completed_pattern'] = completed_pattern
            memory['completion_confidence'] = completion_confidence
            memory['relevance_score'] *= (1 + completion_confidence * 0.1)
            
            completed_memories.append(memory)
            
            # 更新统计
            self.stats['pattern_completions'] += 1
        
        return completed_memories
    
    def _update_temporal_sequence(self, memory_id: str, pattern: torch.Tensor):
        """更新时间序列"""
        if self.current_time_step not in self.temporal_sequences:
            self.temporal_sequences[self.current_time_step] = []
        
        self.temporal_sequences[self.current_time_step].append({
            'memory_id': memory_id,
            'pattern': pattern.clone(),
            'sequence_position': len(self.temporal_sequences[self.current_time_step])
        })
        
        self.stats['temporal_sequences_created'] += 1
    
    def _find_matching_sequences(self, query_pattern: torch.Tensor) -> List[Dict]:
        """寻找匹配的序列"""
        matches = []
        
        for time_step, sequence in self.temporal_sequences.items():
            for item in sequence:
                similarity = torch.cosine_similarity(
                    query_pattern.unsqueeze(0),
                    item['pattern'].unsqueeze(0)
                ).item()
                
                if similarity > 0.6:  # 序列匹配阈值
                    matches.append({
                        'time_step': time_step,
                        'sequence': [s['pattern'] for s in sequence],
                        'match_score': similarity,
                        'match_position': item['sequence_position']
                    })
        
        return matches
    
    def _generate_sequence_prediction(
        self,
        partial_pattern: torch.Tensor,
        sequence_length: int
    ) -> torch.Tensor:
        """生成序列预测"""
        # 简单的基于模式的序列预测
        # 在实际应用中可以使用更复杂的序列预测模型
        predictions = []
        
        # 寻找最相似的模式
        similar_patterns = []
        for time_step, sequence in self.temporal_sequences.items():
            for item in sequence:
                similarity = torch.cosine_similarity(
                    partial_pattern.unsqueeze(0),
                    item['pattern'].unsqueeze(0)
                ).item()
                if similarity > 0.5:
                    similar_patterns.append((item['pattern'], similarity))
        
        # 如果找到了相似模式，生成预测
        if similar_patterns:
            similar_patterns.sort(key=lambda x: x[1], reverse=True)
            base_pattern = similar_patterns[0][0]
            
            # 生成时间相关的模式变化
            for i in range(sequence_length):
                # 添加时间变化趋势
                time_variation = torch.sin(torch.tensor(i * 0.1)) * 0.1
                predicted_pattern = base_pattern + time_variation
                predictions.append(predicted_pattern)
        else:
            # 如果没有找到相似模式，返回重复的查询模式
            predictions = [partial_pattern] * sequence_length
        
        return torch.stack(predictions)
    
    def _calculate_retrieval_accuracy(self, results: List[Dict]) -> float:
        """计算检索准确率"""
        if not results:
            return 0.0
        
        # 基于相关性分数计算准确率
        total_score = sum(result['relevance_score'] for result in results)
        max_possible_score = len(results) * 1.0  # 假设最高分数为1.0
        
        return total_score / max_possible_score if max_possible_score > 0 else 0.0
    
    def _calculate_compression_ratio(self) -> float:
        """计算压缩比"""
        if not self.episodic_memory:
            return 0.0
        
        original_size = len(self.episodic_memory) * self.config.encoding_dim
        compressed_size = sum(
            len(str(memory.get('pattern', torch.tensor([]))))
            for memory in self.episodic_memory.values()
        )
        
        return (original_size - compressed_size) / original_size if original_size > 0 else 0.0
    
    def _compress_memory_pattern(self, pattern: torch.Tensor) -> torch.Tensor:
        """压缩记忆模式"""
        # 简单的压缩：保留主要成分
        compressed = pattern * 0.8  # 衰减权重
        
        return compressed
    
    def _update_indices(self, memory_id: str, pattern: torch.Tensor):
        """更新索引"""
        # 内容索引
        pattern_hash = str(hash(pattern.detach().cpu().numpy().tobytes()))
        if pattern_hash not in self.content_index:
            self.content_index[pattern_hash] = []
        self.content_index[pattern_hash].append(memory_id)
        
        # 时间索引
        if self.current_time_step not in self.temporal_index:
            self.temporal_index[self.current_time_step] = []
        self.temporal_index[self.current_time_step].append(memory_id)
    
    async def _send_storage_confirmation(self, memory_id: str, encoded_content: torch.Tensor):
        """发送存储确认"""
        confirmation_msg = MessageBuilder.create_control_message(
            command="memory_stored",
            parameters={
                "memory_id": memory_id,
                "encoded_size": encoded_content.shape[-1],
                "timestamp": time.time()
            },
            sender="hippocampus",
            receiver="neocortex"
        )
        
        await self.protocol.send_message(confirmation_msg)
    
    async def _handle_retrieve_request(self, message: MemoryMessage) -> Optional[MemoryMessage]:
        """处理检索请求"""
        try:
            if message.query_vector is not None:
                results = await self.retrieve_memory(
                    message.query_vector,
                    memory_type=message.memory_type,
                    top_k=message.top_k
                )
                
                # 创建响应消息
                response = MessageBuilder.create_memory_retrieve(
                    query_vector=message.query_vector,
                    results=results,
                    success=True,
                    sender="hippocampus",
                    receiver=message.sender
                )
                
                return response
        except Exception as e:
            print(f"处理检索请求失败: {e}")
            return None
    
    async def _handle_query_request(self, message: QueryMessage) -> Optional[QueryMessage]:
        """处理查询请求"""
        try:
            if message.query_vector is not None:
                # 执行查询
                results = await self.retrieve_memory(
                    message.query_vector,
                    top_k=message.max_results
                )
                
                response = MessageBuilder.create_query_message(
                    query_vector=message.query_vector,
                    results=results,
                    execution_time=0.1,
                    sender="hippocampus",
                    receiver=message.sender
                )
                
                return response
        except Exception as e:
            print(f"处理查询请求失败: {e}")
            return None
    
    async def _handle_sync_request(self, message) -> Optional:
        """处理同步请求"""
        try:
            # 返回海马体当前状态
            sync_data = {
                'episodic_memory_size': len(self.episodic_memory),
                'working_memory_size': len(self.working_memory),
                'current_time_step': self.current_time_step,
                'statistics': self.get_statistics()
            }
            
            response = MessageBuilder.create_sync_message(
                sync_data=sync_data,
                sender="hippocampus",
                receiver=message.sender,
                sync_status="completed"
            )
            
            return response
        except Exception as e:
            print(f"处理同步请求失败: {e}")
            return None


if __name__ == "__main__":
    import asyncio
    
    # 创建配置
    config = HippocampusConfig(
        memory_capacity=10000,
        encoding_dim=512
    )
    
    # 创建接口
    hippocampus = HippocampusInterface(config)
    
    # 初始化
    asyncio.run(hippocampus.initialize())
    
    # 测试编码
    test_experience = torch.randn(512)
    memory_id = asyncio.run(hippocampus.encode_experience(
        test_experience, importance=0.8
    ))
    
    print(f"编码的记忆ID: {memory_id}")
    
    # 测试检索
    test_query = torch.randn(512)
    results = asyncio.run(hippocampus.retrieve_memory(test_query, top_k=5))
    
    print(f"检索结果数量: {len(results)}")
    for result in results:
        print(f"  记忆ID: {result['memory_id']}, 相关性: {result['relevance_score']:.3f}")
    
    # 获取统计信息
    stats = hippocampus.get_statistics()
    print(f"海马体统计: {stats}")
    
    # 获取工作记忆状态
    working_memory_state = hippocampus.get_working_memory_state()
    print(f"工作记忆状态: {working_memory_state}")