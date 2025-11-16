"""
注意力读取器 - 基于查询向量检索相关记忆
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional


class AttentionReader(nn.Module):
    """
    基于注意力的记忆读取器
    使用查询向量通过注意力机制检索相关记忆内容
    """
    
    def __init__(
        self,
        memory_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_memories: int = 10000
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_memories = max_memories
        
        # 注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 查询和记忆投影
        self.query_proj = nn.Linear(memory_dim, hidden_dim)
        self.memory_proj = nn.Linear(memory_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, memory_dim)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 记忆权重网络
        self.memory_weights = nn.Linear(memory_dim, 1)
        
        # 记忆质量评估
        self.quality_estimator = nn.Linear(memory_dim, 1)
        
    def forward(
        self,
        query: torch.Tensor,
        memory_bank: torch.Tensor,
        memory_meta: Optional[Dict] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            query: 查询向量 [batch_size, query_len, memory_dim]
            memory_bank: 记忆库 [batch_size, max_memories, memory_dim]
            memory_meta: 记忆元数据字典
            mask: 注意力掩码
            
        Returns:
            retrieved_memories: 检索到的记忆
            attention_weights: 注意力权重
        """
        batch_size, seq_len, _ = query.shape
        
        # 投影查询和记忆
        query_proj = self.query_proj(query)  # [batch_size, seq_len, hidden_dim]
        memory_proj = self.memory_proj(memory_bank)  # [batch_size, max_memories, hidden_dim]
        
        # 计算记忆质量权重
        memory_quality = self.quality_estimator(memory_bank)  # [batch_size, max_memories, 1]
        quality_weights = torch.sigmoid(memory_quality).squeeze(-1)  # [batch_size, max_memories]
        
        # 多头自注意力
        # 首先进行记忆到记忆的注意力，构建记忆相关性
        memory_attended, memory_weights = self.attention(
            memory_proj, memory_proj, memory_proj,
            key_padding_mask=mask
        )
        
        # 应用记忆质量权重
        memory_attended = memory_attended * quality_weights.unsqueeze(-1)
        
        # 查询-记忆注意力
        query_attended, attention_weights = self.attention(
            query_proj, memory_attended, memory_attended,
            key_padding_mask=mask
        )
        
        # 残差连接和层归一化
        query_attended = self.norm1(query_proj + query_attended)
        
        # 前馈网络
        output = self.norm2(query_attended + self.ffn(query_attended))
        
        # 输出投影
        retrieved_memories = self.output_proj(output)
        
        return retrieved_memories, attention_weights
    
    def retrieve_top_k(
        self,
        query: torch.Tensor,
        memory_bank: torch.Tensor,
        k: int = 10,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        检索前k个最相关的记忆
        
        Args:
            query: 查询向量
            memory_bank: 记忆库
            k: 检索数量
            threshold: 相似度阈值
            
        Returns:
            top_memories: 前k个记忆
            top_indices: 记忆索引
            top_scores: 相似度分数
        """
        batch_size = query.size(0)
        
        # 计算注意力权重
        _, attention_weights = self.forward(query, memory_bank)
        
        # 获取每个查询的最相关记忆索引
        top_weights, top_indices = torch.topk(attention_weights, k, dim=-1)
        
        # 应用阈值过滤
        mask = top_weights > threshold
        
        # 收集top-k记忆
        batch_size, query_len, _ = query.shape
        top_memories = []
        top_scores = []
        
        for b in range(batch_size):
            batch_memories = []
            batch_scores = []
            
            for q in range(query_len):
                idx = top_indices[b, q]
                score = top_weights[b, q]
                memory = memory_bank[b, idx]
                batch_memories.append(memory)
                batch_scores.append(score)
            
            top_memories.append(torch.stack(batch_memories))
            top_scores.append(torch.stack(batch_scores))
        
        return (
            torch.stack(top_memories),
            top_indices,
            torch.stack(top_scores)
        )
    
    def compute_relevance_scores(
        self,
        query: torch.Tensor,
        memory_bank: torch.Tensor
    ) -> torch.Tensor:
        """
        计算查询与记忆库的相关性分数
        
        Args:
            query: 查询向量
            memory_bank: 记忆库
            
        Returns:
            relevance_scores: 相关性分数
        """
        # 余弦相似度
        query_norm = F.normalize(query, p=2, dim=-1)
        memory_norm = F.normalize(memory_bank, p=2, dim=-1)
        
        # 计算相似度矩阵
        similarity = torch.bmm(
            query_norm,
            memory_norm.transpose(-2, -1)
        )
        
        return similarity
    
    def attention_visualization(
        self,
        query: torch.Tensor,
        memory_bank: torch.Tensor,
        memory_labels: Optional[List[str]] = None
    ) -> Dict:
        """
        生成注意力可视化数据
        
        Args:
            query: 查询向量
            memory_bank: 记忆库
            memory_labels: 记忆标签列表
            
        Returns:
            可视化字典
        """
        _, attention_weights = self.forward(query, memory_bank)
        
        viz_data = {
            'attention_weights': attention_weights.detach().cpu().numpy(),
            'memory_bank_shape': memory_bank.shape,
            'query_shape': query.shape
        }
        
        if memory_labels:
            viz_data['memory_labels'] = memory_labels
            
        return viz_data


class AttentionReaderConfig:
    """注意力读取器配置类"""
    
    def __init__(
        self,
        memory_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_memories: int = 10000
    ):
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_memories = max_memories
    
    def create_model(self) -> AttentionReader:
        """创建注意力读取器模型"""
        return AttentionReader(
            memory_dim=self.memory_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            max_memories=self.max_memories
        )


if __name__ == "__main__":
    # 测试代码
    config = AttentionReaderConfig()
    model = config.create_model()
    
    # 创建测试数据
    batch_size = 2
    seq_len = 5
    memory_dim = 512
    max_memories = 1000
    
    query = torch.randn(batch_size, seq_len, memory_dim)
    memory_bank = torch.randn(batch_size, max_memories, memory_dim)
    
    # 前向传播测试
    output, attention_weights = model(query, memory_bank)
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 检索测试
    top_k = 10
    top_memories, top_indices, top_scores = model.retrieve_top_k(
        query, memory_bank, k=top_k, threshold=0.1
    )
    print(f"Top-k记忆形状: {top_memories.shape}")
    print(f"Top-k索引形状: {top_indices.shape}")
    print(f"Top-k分数形状: {top_scores.shape}")