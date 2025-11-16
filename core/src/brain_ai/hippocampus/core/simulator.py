#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
HippocampusSimulator - 海马体记忆系统
==================================

实现海马体的核心功能，包括:
- 情景记忆的存储与检索
- 快速学习机制
- 模式分离与模式完成
- 时间序列编码
- 记忆巩固

基于生物海马体的以下关键区域:
- CA3: 内容可寻址记忆网络
- CA1: 模式完成和记忆提取
- Dentate Gyrus: 模式分离
- Subiculum: 输出层

作者: Brain-Inspired AI Team
版本: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
import pickle

class HippocampusConfig:
    """海马体配置类"""
    
    def __init__(self):
        # CA3网络参数
        self.ca3_hidden_size = 512
        self.ca3_num_layers = 3
        self.ca3_activation = 'relu'
        
        # CA1网络参数  
        self.ca1_hidden_size = 512
        self.ca1_num_layers = 2
        self.ca1_activation = 'tanh'
        
        # DG模式分离参数
        self.dg_hidden_size = 256
        self.dg_num_layers = 2
        
        # 记忆参数
        self.memory_capacity = 10000
        self.max_sequence_length = 512
        self.retrieval_threshold = 0.7
        
        # 学习参数
        self.fast_learning_rate = 0.1
        self.slow_learning_rate = 0.01
        self.consolidation_threshold = 0.8
        
        # 注意力机制
        self.attention_dim = 128
        self.num_attention_heads = 8

class CA3Network(nn.Module):
    """CA3区域：内容可寻址记忆网络"""
    
    def __init__(self, input_size: int, config: HippocampusConfig):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.ca3_hidden_size
        self.num_layers = config.ca3_num_layers
        
        # 编码层
        self.encoder = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU() if config.ca3_activation == 'relu' else nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU() if config.ca3_activation == 'relu' else nn.Tanh(),
        )
        
        # 递归层（模拟CA3的循环连接）
        self.recurrent_layers = nn.ModuleList([
            nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
            for _ in range(self.num_layers)
        ])
        
        # 记忆网络
        self.memory_networks = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size)
            for _ in range(8)  # 多个记忆槽
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.shape
        
        # 编码输入
        x = x.view(-1, input_size)  # flatten sequence
        encoded = self.encoder(x)
        encoded = encoded.view(batch_size, seq_len, -1)
        
        # 循环处理
        for recurrent_layer in self.recurrent_layers:
            encoded, _ = recurrent_layer(encoded)
        
        # 记忆检索和整合
        memory_outputs = []
        for memory_net in self.memory_networks:
            memory_output = torch.sigmoid(memory_net(encoded))
            memory_outputs.append(memory_output)
        
        # 整合所有记忆输出
        integrated = torch.stack(memory_outputs, dim=-1)
        integrated = integrated.mean(dim=-1)
        
        return integrated

class DGNetwork(nn.Module):
    """Dentate Gyrus：模式分离网络"""
    
    def __init__(self, input_size: int, config: HippocampusConfig):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.dg_hidden_size
        self.num_layers = config.dg_num_layers
        
        # 模式分离编码器
        self.pattern_separator = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),  # DG通常使用tanh激活
            nn.Dropout(0.2),
        )
        
        # 稀疏化层（模拟DG的稀疏激活）
        self.sparse_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Softmax(dim=-1)  # 归一化
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        separated = self.pattern_separator(x)
        sparse_output = self.sparse_layer(separated)
        return sparse_output

class CA1Network(nn.Module):
    """CA1区域：模式完成网络"""
    
    def __init__(self, ca3_output_size: int, dg_output_size: int, config: HippocampusConfig):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.ca1_hidden_size
        
        # CA3和DG输入融合
        self.input_fusion = nn.Linear(ca3_output_size + dg_output_size, self.hidden_size)
        
        # 模式完成网络
        self.pattern_completion = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh() if config.ca1_activation == 'tanh' else nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh() if config.ca1_activation == 'tanh' else nn.ReLU(),
        )
        
        # 输出层
        self.output_layer = nn.Linear(self.hidden_size, self.hidden_size)
        
    def forward(self, ca3_output: torch.Tensor, dg_output: torch.Tensor) -> torch.Tensor:
        # 融合CA3和DG的输出
        combined = torch.cat([ca3_output, dg_output], dim=-1)
        fused = self.input_fusion(combined)
        
        # 模式完成
        completed = self.pattern_completion(fused)
        output = self.output_layer(completed)
        
        return output

class HippocampusSimulator(nn.Module):
    """海马体模拟器主类"""
    
    def __init__(self, input_size: int = 512, config: Optional[HippocampusConfig] = None):
        """
        初始化海马体模拟器
        
        Args:
            input_size: 输入特征维度
            config: 配置对象
        """
        super().__init__()
        
        if config is None:
            config = HippocampusConfig()
        
        self.config = config
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建子网络
        self.ca3 = CA3Network(input_size, config)
        self.dg = DGNetwork(input_size, config)
        self.ca1 = CA1Network(config.ca3_hidden_size, config.dg_hidden_size, config)
        
        # 记忆存储
        self.episodic_memory = {}
        self.memory_count = 0
        self.max_memory = config.memory_capacity
        
        # 学习状态
        self.learning_state = 'fast'
        
        # 初始化网络
        self.to(self.device)
        
    def store(self, memory_data: Any, metadata: Optional[Dict] = None) -> str:
        """
        存储记忆到海马体
        
        Args:
            memory_data: 要存储的记忆数据
            metadata: 记忆元数据
            
        Returns:
            记忆ID
        """
        memory_id = f"memory_{self.memory_count}"
        self.memory_count += 1
        
        memory_entry = {
            'id': memory_id,
            'data': memory_data,
            'metadata': metadata or {},
            'timestamp': self.memory_count,
            'retrieval_count': 0,
            'strength': 1.0
        }
        
        # 如果超过容量，删除最弱的记忆
        if len(self.episodic_memory) >= self.max_memory:
            weakest_id = min(self.episodic_memory.keys(), 
                           key=lambda x: self.episodic_memory[x]['strength'])
            del self.episodic_memory[weakest_id]
        
        self.episodic_memory[memory_id] = memory_entry
        return memory_id
    
    def retrieve(self, memory_id: str) -> Optional[Any]:
        """
        从海马体检索记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            检索到的记忆数据，如果不存在返回None
        """
        if memory_id not in self.episodic_memory:
            return None
        
        memory_entry = self.episodic_memory[memory_id]
        
        # 更新检索统计
        memory_entry['retrieval_count'] += 1
        memory_entry['strength'] *= 1.1  # 强化记忆强度
        
        return memory_entry['data']
    
    def retrieve_by_similarity(self, query_data: Any, 
                             threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        基于相似性检索记忆
        
        Args:
            query_data: 查询数据
            threshold: 相似度阈值
            
        Returns:
            (记忆ID, 相似度分数)的列表
        """
        if threshold is None:
            threshold = self.config.retrieval_threshold
        
        query_embedding = self._encode_memory(query_data)
        similarities = []
        
        for memory_id, memory_entry in self.episodic_memory.items():
            memory_embedding = self._encode_memory(memory_entry['data'])
            
            # 计算余弦相似度
            similarity = F.cosine_similarity(query_embedding, memory_embedding, dim=0)
            
            if similarity > threshold:
                similarities.append((memory_id, similarity.item()))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def _encode_memory(self, memory_data: Any) -> torch.Tensor:
        """编码记忆数据为向量表示"""
        if isinstance(memory_data, torch.Tensor):
            return memory_data.squeeze().to(self.device)
        elif isinstance(memory_data, (list, tuple)):
            return torch.tensor(memory_data, dtype=torch.float32).to(self.device)
        elif isinstance(memory_data, (int, float)):
            return torch.tensor([memory_data], dtype=torch.float32).to(self.device)
        elif isinstance(memory_data, str):
            # 简单的字符串编码（实际应用中应使用更复杂的编码）
            return torch.tensor([ord(c) for c in memory_data[:self.input_size]], 
                              dtype=torch.float32).to(self.device)
        else:
            # 默认编码
            return torch.randn(self.input_size).to(self.device)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        海马体前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, input_size]
            
        Returns:
            各层的输出字典
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 添加序列维度
        
        # 通过DG进行模式分离
        dg_output = self.dg(x)
        
        # 通过CA3进行记忆检索
        ca3_output = self.ca3(x)
        
        # 通过CA1进行模式完成
        ca1_output = self.ca1(ca3_output, dg_output)
        
        return {
            'dg_output': dg_output,
            'ca3_output': ca3_output, 
            'ca1_output': ca1_output,
            'final_output': ca1_output
        }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        if not self.episodic_memory:
            return {'total_memories': 0}
        
        retrieval_counts = [m['retrieval_count'] for m in self.episodic_memory.values()]
        strengths = [m['strength'] for m in self.episodic_memory.values()]
        
        return {
            'total_memories': len(self.episodic_memory),
            'avg_retrieval_count': np.mean(retrieval_counts),
            'avg_strength': np.mean(strengths),
            'max_retrieval_count': max(retrieval_counts),
            'min_strength': min(strengths)
        }
    
    def consolidate_memories(self) -> int:
        """
        巩固记忆（模拟睡眠期间的记忆巩固过程）
        
        Returns:
            巩固的记忆数量
        """
        consolidated_count = 0
        
        # 找到需要巩固的记忆
        memories_to_consolidate = []
        for memory_id, memory_entry in self.episodic_memory.items():
            if memory_entry['retrieval_count'] >= 3 and memory_entry['strength'] > self.config.consolidation_threshold:
                memories_to_consolidate.append(memory_id)
        
        # 模拟巩固过程
        for memory_id in memories_to_consolidate:
            memory_entry = self.episodic_memory[memory_id]
            
            # 强化长期记忆
            memory_entry['strength'] *= 1.5
            memory_entry['metadata']['consolidated'] = True
            memory_entry['metadata']['consolidation_time'] = self.memory_count
            
            consolidated_count += 1
        
        return consolidated_count
    
    def save_memories(self, filepath: str):
        """保存记忆到文件"""
        memory_data = {
            'episodic_memory': self.episodic_memory,
            'memory_count': self.memory_count,
            'config': self.config.__dict__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(memory_data, f)
    
    def load_memories(self, filepath: str):
        """从文件加载记忆"""
        with open(filepath, 'rb') as f:
            memory_data = pickle.load(f)
        
        self.episodic_memory = memory_data['episodic_memory']
        self.memory_count = memory_data['memory_count']

# 快速学习机制
class FastLearningMechanism:
    """快速学习机制（模拟海马体的快速学习能力）"""
    
    def __init__(self, hippocampus: HippocampusSimulator):
        self.hippocampus = hippocampus
        self.learning_rate = hippocampus.config.fast_learning_rate
    
    def learn_single_example(self, input_data: Any, target_data: Any) -> Dict[str, Any]:
        """
        单样本快速学习
        
        Args:
            input_data: 输入数据
            target_data: 目标数据
            
        Returns:
            学习结果
        """
        # 存储记忆
        memory_id = self.hippocampus.store(input_data)
        
        # 创建输入-输出对
        learning_pair = {
            'input': input_data,
            'target': target_data,
            'memory_id': memory_id
        }
        
        # 存储学习对
        self.hippocampus.store(learning_pair)
        
        return {
            'memory_id': memory_id,
            'learning_successful': True,
            'learning_rate': self.learning_rate
        }

# 模式分离机制
class PatternSeparationMechanism:
    """模式分离机制"""
    
    def __init__(self, hippocampus: HippocampusSimulator):
        self.hippocampus = hippocampus
    
    def separate_patterns(self, similar_inputs: List[Any]) -> List[torch.Tensor]:
        """
        对相似输入进行模式分离
        
        Args:
            similar_inputs: 相似输入列表
            
        Returns:
            分离后的表示列表
        """
        separated_patterns = []
        
        for input_data in similar_inputs:
            # 编码输入
            encoded = self.hippocampus._encode_memory(input_data)
            
            # 通过DG进行模式分离
            separated = self.hippocampus.dg(encoded.unsqueeze(0))
            separated_patterns.append(separated.squeeze(0))
        
        return separated_patterns

if __name__ == "__main__":
    # 测试海马体模拟器
    hippocampus = HippocampusSimulator(input_size=256)
    
    # 存储一些记忆
    memories = [
        {"text": "第一次使用brain-ai", "emotion": "兴奋"},
        {"text": "学习人工智能", "emotion": "好奇"},
        {"text": "深度学习很复杂", "emotion": "困惑"}
    ]
    
    memory_ids = []
    for memory in memories:
        memory_id = hippocampus.store(memory)
        memory_ids.append(memory_id)
    
    # 检索记忆
    for memory_id in memory_ids[:2]:
        retrieved = hippocampus.retrieve(memory_id)
        print(f"检索记忆 {memory_id}: {retrieved}")
    
    # 基于相似性检索
    query = {"text": "机器学习很有趣", "emotion": "兴趣"}
    similar_memories = hippocampus.retrieve_by_similarity(query)
    print(f"相似记忆: {similar_memories}")
    
    # 记忆统计
    stats = hippocampus.get_memory_stats()
    print(f"记忆统计: {stats}")
    
    # 输出网络信息
    output = hippocampus.forward(torch.randn(1, 10, 256))
    print(f"网络输出形状: {output['ca1_output'].shape}")