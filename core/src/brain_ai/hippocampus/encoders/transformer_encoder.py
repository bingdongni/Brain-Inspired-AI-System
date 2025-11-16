#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Transformer编码器
===============

基于Transformer架构的编码器，用于海马体记忆编码。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # 线性变换并重塑为多头
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)
        
        # 重塑并输出
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(context)

class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接 + 层归一化
        attention_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    
    def __init__(self, input_size: int, d_model: int = 512, num_heads: int = 8,
                 num_layers: int = 6, d_ff: int = 2048, max_len: int = 512,
                 dropout: float = 0.1):
        """
        初始化Transformer编码器
        
        Args:
            input_size: 输入维度
            d_model: 模型维度
            num_heads: 注意力头数
            num_layers: 编码器层数
            d_ff: 前馈网络维度
            max_len: 最大序列长度
            dropout: Dropout率
        """
        super().__init__()
        
        self.d_model = d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, input_size]
            mask: 注意力掩码
            
        Returns:
            编码输出 [batch_size, seq_len, d_model]
        """
        # 输入投影
        x = self.input_proj(x)
        
        # 添加位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过编码器层
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, mask)
        
        # 输出投影
        output = self.output_proj(x)
        
        return output
    
    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        """
        编码整个序列
        
        Args:
            sequence: 输入序列
            
        Returns:
            序列编码表示
        """
        # 获取序列长度
        seq_len = sequence.size(1)
        
        # 编码
        encoded = self.forward(sequence)
        
        # 池化（使用平均池化）
        pooled = torch.mean(encoded, dim=1)  # [batch_size, d_model]
        
        return pooled

class HippocampusTransformerEncoder(TransformerEncoder):
    """海马体专用Transformer编码器"""
    
    def __init__(self, input_size: int, **kwargs):
        super().__init__(input_size, **kwargs)
        
        # 添加特殊的记忆门控机制
        self.memory_gate = nn.Linear(self.d_model, self.d_model)
        self.memory_norm = nn.LayerNorm(self.d_model)
        
    def forward(self, x: torch.Tensor, memory_context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播，包含记忆门控"""
        # 基础编码
        encoded = super().forward(x, mask)
        
        # 如果有记忆上下文，应用门控机制
        if memory_context is not None:
            gate_values = torch.sigmoid(self.memory_gate(encoded))
            memory_enhanced = encoded * gate_values
            encoded = self.memory_norm(encoded + memory_enhanced)
        
        return encoded

# 工厂函数
def create_transformer_encoder(input_size: int, 
                              model_size: int = 512,
                              num_layers: int = 6,
                              num_heads: int = 8,
                              max_length: int = 512) -> TransformerEncoder:
    """
    创建Transformer编码器
    
    Args:
        input_size: 输入维度
        model_size: 模型维度
        num_layers: 层数
        num_heads: 注意力头数
        max_length: 最大序列长度
        
    Returns:
        Transformer编码器实例
    """
    return TransformerEncoder(
        input_size=input_size,
        d_model=model_size,
        num_layers=num_layers,
        num_heads=num_heads,
        max_len=max_length
    )

if __name__ == "__main__":
    # 测试编码器
    input_size = 256
    batch_size = 4
    seq_len = 32
    
    encoder = create_transformer_encoder(input_size)
    
    # 创建测试输入
    test_input = torch.randn(batch_size, seq_len, input_size)
    
    # 编码
    encoded_output = encoder(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"编码输出形状: {encoded_output.shape}")
    
    # 测试序列编码
    sequence_encoding = encoder.encode_sequence(test_input)
    print(f"序列编码形状: {sequence_encoding.shape}")