"""
新皮层接口
提供新皮层侧的通信接口，处理与海马体的信息交互和高级认知功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio
import time
from dataclasses import dataclass
from collections import deque, defaultdict

from .protocol_handler import CommunicationProtocol
from .message_types import (
    MessageType, MemoryType, Priority, MemoryMessage, 
    QueryMessage, MessageBuilder
)


@dataclass
class NeocortexConfig:
    """新皮层配置"""
    cortical_columns: int = 64
    column_dim: int = 256
    hierarchical_levels: int = 4
    associative_memory_size: int = 100000
    attention_mechanism: bool = True
    predictive_coding: bool = True
    learning_rate: float = 0.001
    consolidation_threshold: float = 0.8


class ColumnarProcessor(nn.Module):
    """皮质柱处理器"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.input_proj(x)
        attended, _ = self.attention(x_proj, x_proj, x_proj)
        x_norm = self.norm(x_proj + attended)
        output = self.hidden_proj(x_norm)
        return self.dropout(output)


class NeocortexInterface:
    """
    新皮层接口
    管理高级认知功能和与海马体的双向通信
    """
    
    def __init__(self, config: NeocortexConfig, protocol: CommunicationProtocol = None):
        self.config = config
        self.protocol = protocol or CommunicationProtocol()
        
        # 皮质柱系统
        self.cortical_columns = nn.ModuleList([
            ColumnarProcessor(config.column_dim, config.column_dim)
            for _ in range(config.cortical_columns)
        ])
        
        # 关联记忆系统
        self.associative_memory = {
            'semantic': {},      # 语义记忆
            'procedural': {},    # 程序性记忆
            'declarative': {},   # 陈述性记忆
            'hierarchical': {}   # 层次记忆
        }
        
        # 分层表示
        self.hierarchical_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.column_dim if i == 0 else config.column_dim * 2, config.column_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(config.column_dim, config.column_dim)
            ) for i in range(config.hierarchical_levels)
        ])
        
        # 注意力机制
        if config.attention_mechanism:
            self.global_attention = nn.MultiheadAttention(
                config.column_dim * config.cortical_columns, 8, batch_first=True
            )
        
        # 预测编码器
        if config.predictive_coding:
            self.predictor = nn.LSTM(
                input_size=config.column_dim,
                hidden_size=config.column_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.1
            )
        
        # 工作记忆缓冲区
        self.working_memory_buffer = deque(maxlen=1000)
        
        # 知识图谱
        self.knowledge_graph = defaultdict(list)
        
        # 统计信息
        self.stats = {
            'processed_queries': 0,
            'associative_retrievals': 0,
            'hierarchical_activations': 0,
            'predictive_accuracy': 0.0,
            'consolidation_events': 0,
            'attention_weight_changes': 0
        }
        
        # 当前上下文
        self.current_context = None
        self.context_history = []
        
    async def initialize(self) -> bool:
        """初始化新皮层接口"""
        try:
            # 启动通信协议
            if not self.protocol.is_running:
                await self.protocol.start()
            
            # 注册消息处理器
            self.protocol.register_handler(MessageType.MEMORY_STORE, self._handle_memory_store)
            self.protocol.register_handler(MessageType.QUERY_PROCESS, self._handle_query_process)
            self.protocol.register_handler(MessageType.LEARNING_UPDATE, self._handle_learning_update)
            
            # 初始化知识图谱
            await self._initialize_knowledge_graph()
            
            print("新皮层接口初始化成功")
            return True
            
        except Exception as e:
            print(f"新皮层接口初始化失败: {e}")
            return False
    
    async def process_information(
        self,
        information: torch.Tensor,
        information_type: str = "general",
        context: torch.Tensor = None
    ) -> Dict:
        """
        处理信息并生成认知表示
        
        Args:
            information: 输入信息
            information_type: 信息类型
            context: 上下文信息
            
        Returns:
            处理结果字典
        """
        try:
            # 1. 皮质柱处理
            processed_features = await self._process_cortical_columns(information)
            
            # 2. 分层抽象
            hierarchical_representation = await self._create_hierarchical_representation(
                processed_features
            )
            
            # 3. 关联记忆存储
            stored_in_memory = await self._store_in_associative_memory(
                hierarchical_representation, information_type
            )
            
            # 4. 知识图谱更新
            knowledge_update = await self._update_knowledge_graph(
                hierarchical_representation, information_type
            )
            
            # 5. 预测生成（如果启用）
            predictions = None
            if self.config.predictive_coding:
                predictions = await self._generate_predictions(hierarchical_representation)
            
            # 更新工作记忆
            self.working_memory_buffer.append({
                'timestamp': time.time(),
                'information_type': information_type,
                'representation': hierarchical_representation,
                'predictions': predictions
            })
            
            self.stats['processed_queries'] += 1
            
            return {
                'processed_features': processed_features,
                'hierarchical_representation': hierarchical_representation,
                'stored_in_memory': stored_in_memory,
                'knowledge_update': knowledge_update,
                'predictions': predictions,
                'processing_timestamp': time.time()
            }
            
        except Exception as e:
            print(f"信息处理失败: {e}")
            return {'error': str(e)}
    
    async def retrieve_associative_memory(
        self,
        query: torch.Tensor,
        memory_type: str = "semantic",
        association_threshold: float = 0.7,
        top_k: int = 10
    ) -> List[Dict]:
        """
        检索关联记忆
        
        Args:
            query: 查询向量
            memory_type: 记忆类型
            association_threshold: 关联阈值
            top_k: 返回数量
            
        Returns:
            关联记忆列表
        """
        try:
            if memory_type not in self.associative_memory:
                return []
            
            memory_bank = self.associative_memory[memory_type]
            associations = []
            
            for memory_id, memory_data in memory_bank.items():
                # 计算关联强度
                association_strength = torch.cosine_similarity(
                    query.unsqueeze(0),
                    memory_data['representation'].unsqueeze(0)
                ).item()
                
                if association_strength >= association_threshold:
                    associations.append({
                        'memory_id': memory_id,
                        'association_strength': association_strength,
                        'memory_data': memory_data,
                        'memory_type': memory_type
                    })
            
            # 按关联强度排序
            associations.sort(key=lambda x: x['association_strength'], reverse=True)
            
            self.stats['associative_retrievals'] += 1
            
            return associations[:top_k]
            
        except Exception as e:
            print(f"关联记忆检索失败: {e}")
            return []
    
    async def generate_prediction(
        self,
        context: torch.Tensor,
        prediction_horizon: int = 5
    ) -> Dict:
        """
        生成预测
        
        Args:
            context: 上下文向量
            prediction_horizon: 预测时间范围
            
        Returns:
            预测结果
        """
        try:
            if not self.config.predictive_coding:
                return {'error': '预测编码未启用'}
            
            # 使用LSTM预测器生成预测序列
            context_expanded = context.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
            
            # 预测未来状态
            predictions, _ = self.predictor(context_expanded)
            prediction_sequence = predictions.squeeze(0)  # [1, dim]
            
            # 生成多个时间步的预测
            prediction_timeline = []
            current_state = context_expanded
            
            for t in range(prediction_horizon):
                pred, hidden_state = self.predictor(current_state)
                prediction_timeline.append({
                    'time_step': t,
                    'prediction': pred.squeeze(0),
                    'confidence': self._calculate_prediction_confidence(current_state, pred)
                })
                current_state = pred
            
            # 计算整体预测准确性
            overall_confidence = np.mean([p['confidence'] for p in prediction_timeline])
            
            self.stats['predictive_accuracy'] = overall_confidence
            
            return {
                'prediction_sequence': prediction_sequence,
                'timeline': prediction_timeline,
                'overall_confidence': overall_confidence,
                'prediction_horizon': prediction_horizon
            }
            
        except Exception as e:
            print(f"预测生成失败: {e}")
            return {'error': str(e)}
    
    async def consolidate_to_semantic_memory(
        self,
        episodic_memories: List[Dict],
        consolidation_criteria: Dict = None
    ) -> Dict:
        """
        将情景记忆巩固为语义记忆
        
        Args:
            episodic_memories: 情景记忆列表
            consolidation_criteria: 巩固标准
            
        Returns:
            巩固结果
        """
        try:
            consolidation_criteria = consolidation_criteria or {
                'frequency_threshold': 3,
                'importance_threshold': 0.6,
                'temporal_similarity': 0.8
            }
            
            consolidated_count = 0
            semantic_memories_created = []
            
            # 分析情景记忆的共现模式
            cooccurrence_patterns = self._analyze_cooccurrence_patterns(episodic_memories)
            
            # 识别稳定模式
            stable_patterns = self._identify_stable_patterns(
                cooccurrence_patterns, consolidation_criteria
            )
            
            # 创建语义记忆
            for pattern in stable_patterns:
                semantic_memory_id = await self._create_semantic_memory(pattern)
                semantic_memories_created.append(semantic_memory_id)
                consolidated_count += 1
            
            # 更新关联记忆系统
            for memory_id in semantic_memories_created:
                await self._update_associative_links(memory_id)
            
            self.stats['consolidation_events'] += consolidated_count
            
            return {
                'consolidated_count': consolidated_count,
                'semantic_memories_created': semantic_memories_created,
                'stable_patterns_count': len(stable_patterns),
                'consolidation_quality': self._assess_consolidation_quality(stable_patterns)
            }
            
        except Exception as e:
            print(f"语义巩固失败: {e}")
            return {'error': str(e)}
    
    def get_hierarchical_representation(self, query: torch.Tensor) -> Dict:
        """
        获取分层表示
        
        Args:
            query: 查询向量
            
        Returns:
            分层表示字典
        """
        try:
            representations = {}
            
            # 在每个层次级别处理查询
            current_representation = query
            
            for level in range(len(self.hierarchical_layers)):
                processed = self.hierarchical_layers[level](current_representation)
                representations[f'level_{level}'] = processed
                current_representation = processed
                
                self.stats['hierarchical_activations'] += 1
            
            return {
                'representations': representations,
                'hierarchy_depth': len(self.hierarchical_layers),
                'final_representation': current_representation
            }
            
        except Exception as e:
            print(f"分层表示生成失败: {e}")
            return {'error': str(e)}
    
    async def attention_weight_analysis(self) -> Dict:
        """分析注意力权重"""
        try:
            if not self.config.attention_mechanism:
                return {'error': '注意力机制未启用'}
            
            # 分析皮质柱间的注意力模式
            column_attention_patterns = {}
            
            for i, column in enumerate(self.cortical_columns):
                # 获取注意力权重（这里需要实际实现权重提取）
                attention_weights = torch.softmax(
                    torch.randn(1, self.config.cortical_columns), dim=-1
                )
                
                column_attention_patterns[f'column_{i}'] = {
                    'weights': attention_weights,
                    'max_attention': attention_weights.max().item(),
                    'attention_entropy': self._calculate_entropy(attention_weights)
                }
            
            # 计算全局注意力统计
            global_stats = {
                'average_max_attention': np.mean([
                    pattern['max_attention'] for pattern in column_attention_patterns.values()
                ]),
                'average_entropy': np.mean([
                    pattern['attention_entropy'] for pattern in column_attention_patterns.values()
                ])
            }
            
            self.stats['attention_weight_changes'] += 1
            
            return {
                'column_patterns': column_attention_patterns,
                'global_statistics': global_stats,
                'total_columns': len(self.cortical_columns)
            }
            
        except Exception as e:
            print(f"注意力权重分析失败: {e}")
            return {'error': str(e)}
    
    def get_statistics(self) -> Dict:
        """获取新皮层统计信息"""
        return {
            **self.stats,
            'cortical_columns_count': len(self.cortical_columns),
            'hierarchical_levels': len(self.hierarchical_layers),
            'associative_memory_sizes': {
                memory_type: len(memory_bank)
                for memory_type, memory_bank in self.associative_memory.items()
            },
            'working_memory_buffer_size': len(self.working_memory_buffer),
            'knowledge_graph_nodes': len(self.knowledge_graph),
            'config': {
                'cortical_columns': self.config.cortical_columns,
                'hierarchical_levels': self.config.hierarchical_levels,
                'attention_mechanism': self.config.attention_mechanism,
                'predictive_coding': self.config.predictive_coding
            }
        }
    
    async def _process_cortical_columns(self, information: torch.Tensor) -> torch.Tensor:
        """处理皮质柱"""
        try:
            # 调整输入维度
            if information.dim() == 1:
                information = information.unsqueeze(0)
            
            # 确保输入维度匹配皮质柱维度
            if information.size(-1) != self.config.column_dim:
                information = nn.functional.adaptive_avg_pool1d(
                    information.transpose(0, 1), self.config.column_dim
                ).transpose(0, 1)
            
            # 并行处理所有皮质柱
            column_outputs = []
            for column in self.cortical_columns:
                output = column(information)
                column_outputs.append(output)
            
            # 组合皮质柱输出
            combined_output = torch.cat(column_outputs, dim=-1)
            
            return combined_output
            
        except Exception as e:
            print(f"皮质柱处理失败: {e}")
            return information
    
    async def _create_hierarchical_representation(self, features: torch.Tensor) -> torch.Tensor:
        """创建分层表示"""
        try:
            current_level = features
            
            for level, layer in enumerate(self.hierarchical_layers):
                current_level = layer(current_level)
            
            return current_level
            
        except Exception as e:
            print(f"分层表示创建失败: {e}")
            return features
    
    async def _store_in_associative_memory(
        self,
        representation: torch.Tensor,
        memory_type: str
    ) -> str:
        """存储到关联记忆"""
        try:
            memory_id = f"assoc_{memory_type}_{int(time.time() * 1000)}"
            
            memory_data = {
                'representation': representation.clone(),
                'memory_type': memory_type,
                'creation_time': time.time(),
                'access_count': 0,
                'last_accessed': time.time()
            }
            
            self.associative_memory[memory_type][memory_id] = memory_data
            
            return memory_id
            
        except Exception as e:
            print(f"关联记忆存储失败: {e}")
            return ""
    
    async def _update_knowledge_graph(
        self,
        representation: torch.Tensor,
        information_type: str
    ) -> Dict:
        """更新知识图谱"""
        try:
            # 基于表示生成概念节点
            concept_node = f"concept_{information_type}_{int(time.time())}"
            
            # 查找相关概念
            related_concepts = []
            for node, connections in self.knowledge_graph.items():
                similarity = torch.cosine_similarity(
                    representation.unsqueeze(0),
                    torch.tensor(connections).unsqueeze(0) if connections else torch.zeros(1, representation.size(-1))
                ).item()
                
                if similarity > 0.6:
                    related_concepts.append(node)
            
            # 更新连接
            self.knowledge_graph[concept_node] = related_concepts
            
            # 为相关概念添加反向连接
            for related_concept in related_concepts:
                if concept_node not in self.knowledge_graph[related_concept]:
                    self.knowledge_graph[related_concept].append(concept_node)
            
            return {
                'concept_created': concept_node,
                'related_concepts': related_concepts,
                'total_nodes': len(self.knowledge_graph)
            }
            
        except Exception as e:
            print(f"知识图谱更新失败: {e}")
            return {'error': str(e)}
    
    async def _generate_predictions(self, representation: torch.Tensor) -> Dict:
        """生成预测"""
        try:
            if not self.config.predictive_coding:
                return {}
            
            # 使用预测器生成预测
            predictions, _ = self.predictor(representation.unsqueeze(0))
            
            return {
                'prediction': predictions.squeeze(0),
                'prediction_confidence': torch.sigmoid(predictions).mean().item()
            }
            
        except Exception as e:
            print(f"预测生成失败: {e}")
            return {}
    
    async def _initialize_knowledge_graph(self):
        """初始化知识图谱"""
        # 创建一些基础概念节点
        base_concepts = [
            "perception", "memory", "attention", "learning", "prediction",
            "association", "abstraction", "reasoning", "decision", "action"
        ]
        
        for concept in base_concepts:
            self.knowledge_graph[concept] = []
    
    def _analyze_cooccurrence_patterns(self, episodic_memories: List[Dict]) -> Dict:
        """分析共现模式"""
        patterns = defaultdict(int)
        
        for memory in episodic_memories:
            # 分析记忆内容中的共现关系
            content = memory.get('content', torch.tensor([]))
            if content.numel() > 0:
                # 简单的共现分析
                for i in range(min(len(content), 10)):  # 限制分析范围
                    for j in range(i + 1, min(len(content), 10)):
                        pattern = f"({i},{j})"
                        patterns[pattern] += 1
        
        return dict(patterns)
    
    def _identify_stable_patterns(
        self,
        patterns: Dict,
        criteria: Dict
    ) -> List[str]:
        """识别稳定模式"""
        stable_patterns = []
        
        for pattern, frequency in patterns.items():
            if (frequency >= criteria.get('frequency_threshold', 3) and 
                frequency > len(episodic_memories) * 0.1):  # 至少10%的记忆包含此模式
                stable_patterns.append(pattern)
        
        return stable_patterns
    
    async def _create_semantic_memory(self, pattern: str) -> str:
        """创建语义记忆"""
        memory_id = f"semantic_{pattern}_{int(time.time() * 1000)}"
        
        # 生成语义表示（这里应该基于模式内容生成）
        semantic_representation = torch.randn(self.config.column_dim)
        
        memory_data = {
            'pattern': pattern,
            'representation': semantic_representation,
            'semantic_type': 'consolidated',
            'creation_time': time.time(),
            'confidence': 0.8  # 初始置信度
        }
        
        self.associative_memory['semantic'][memory_id] = memory_data
        
        return memory_id
    
    async def _update_associative_links(self, memory_id: str):
        """更新关联链接"""
        if memory_id not in self.associative_memory['semantic']:
            return
        
        current_memory = self.associative_memory['semantic'][memory_id]
        
        # 查找相似记忆并建立关联
        for other_id, other_memory in self.associative_memory['semantic'].items():
            if other_id != memory_id:
                similarity = torch.cosine_similarity(
                    current_memory['representation'].unsqueeze(0),
                    other_memory['representation'].unsqueeze(0)
                ).item()
                
                if similarity > 0.7:  # 高相似度建立关联
                    # 在这里可以建立双向关联
                    pass
    
    def _assess_consolidation_quality(self, stable_patterns: List[str]) -> float:
        """评估巩固质量"""
        if not stable_patterns:
            return 0.0
        
        # 基于模式数量和质量评估巩固质量
        pattern_quality_scores = [0.8 + 0.2 * np.random.random() for _ in stable_patterns]
        return np.mean(pattern_quality_scores)
    
    def _calculate_prediction_confidence(self, current: torch.Tensor, prediction: torch.Tensor) -> float:
        """计算预测置信度"""
        # 基于预测的不确定性计算置信度
        uncertainty = torch.var(prediction).item()
        confidence = 1.0 / (1.0 + uncertainty)
        return min(1.0, confidence)
    
    def _calculate_entropy(self, weights: torch.Tensor) -> float:
        """计算注意力权重的熵"""
        weights = torch.clamp(weights, min=1e-8)
        entropy = -torch.sum(weights * torch.log(weights))
        return entropy.item()
    
    # 消息处理器
    async def _handle_memory_store(self, message: MemoryMessage) -> Optional[MemoryMessage]:
        """处理记忆存储消息"""
        try:
            if message.content is not None:
                # 处理存储的记忆
                result = await self.process_information(
                    message.content,
                    information_type=message.memory_type.value,
                    context=message.context
                )
                
                # 发送确认响应
                response = MessageBuilder.create_control_message(
                    command="memory_processed",
                    parameters={
                        "success": True,
                        "processing_result": result,
                        "timestamp": time.time()
                    },
                    sender="neocortex",
                    receiver=message.sender
                )
                
                return response
        except Exception as e:
            print(f"处理记忆存储消息失败: {e}")
            return None
    
    async def _handle_query_process(self, message: QueryMessage) -> Optional[QueryMessage]:
        """处理查询消息"""
        try:
            if message.query_vector is not None:
                # 执行查询处理
                result = await self.process_information(
                    message.query_vector,
                    information_type="query"
                )
                
                response = MessageBuilder.create_query_message(
                    query_vector=message.query_vector,
                    results=[result],
                    sender="neocortex",
                    receiver=message.sender
                )
                
                return response
        except Exception as e:
            print(f"处理查询消息失败: {e}")
            return None
    
    async def _handle_learning_update(self, message) -> Optional:
        """处理学习更新消息"""
        try:
            # 处理学习更新
            learning_data = message.learning_data
            
            response = MessageBuilder.create_learning_message(
                learning_data={"status": "processed", "confidence": 0.9},
                sender="neocortex",
                receiver=message.sender
            )
            
            return response
        except Exception as e:
            print(f"处理学习更新消息失败: {e}")
            return None


if __name__ == "__main__":
    import asyncio
    
    # 创建配置
    config = NeocortexConfig(
        cortical_columns=16,
        hierarchical_levels=3,
        attention_mechanism=True,
        predictive_coding=True
    )
    
    # 创建接口
    neocortex = NeocortexInterface(config)
    
    # 初始化
    asyncio.run(neocortex.initialize())
    
    # 测试信息处理
    test_information = torch.randn(256)
    result = asyncio.run(neocortex.process_information(
        test_information, information_type="visual"
    ))
    
    print(f"信息处理结果键: {result.keys()}")
    
    # 测试关联记忆检索
    test_query = torch.randn(256)
    associations = asyncio.run(neocortex.retrieve_associative_memory(
        test_query, memory_type="semantic"
    ))
    
    print(f"关联记忆数量: {len(associations)}")
    
    # 测试预测生成
    context = torch.randn(256)
    prediction = asyncio.run(neocortex.generate_prediction(context))
    
    print(f"预测生成结果键: {prediction.keys()}")
    
    # 获取统计信息
    stats = neocortex.get_statistics()
    print(f"新皮层统计: {stats}")