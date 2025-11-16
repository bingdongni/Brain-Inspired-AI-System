"""
Transformer-based记忆编码器
基于海马体记忆印迹的多突触末梢(MSBs)机制实现非同步激活的记忆编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
import numpy as np


class MultiSynapticEngram(nn.Module):
    """多突触末梢记忆印迹单元
    
    基于Science研究发现：长时记忆形成与多突触末梢的选择性增加密切相关
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_synapses: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_synapses = num_synapses
        
        # 每个突触末梢的独立权重矩阵
        self.synapse_weights = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_synapses)
        ])
        
        # 突触前结构复杂性增强
        self.structural_complexity = nn.Parameter(
            torch.randn(num_synapses) * 0.1 + 1.0  # 初始复杂度权重
        )
        
        # 突触选择性门控机制
        self.synaptic_gate = nn.Sequential(
            nn.Linear(input_dim, num_synapses),
            nn.Sigmoid()
        )
        
        # 非同步激活机制
        self.asynchronous_activation = nn.Sequential(
            nn.Linear(input_dim, num_synapses * hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # 计算突触选择性
        gate_values = self.synaptic_gate(x)  # [batch, seq_len, num_synapses]
        
        # 非同步激活计算
        async_features = self.asynchronous_activation(x)  # [batch, seq_len, num_synapses*hidden_dim]
        
        outputs = []
        for i in range(self.num_synapses):
            # 每个突触末梢的独立处理
            synapse_input = x * gate_values[:, :, i:i+1]
            synapse_output = F.relu(self.synapse_weights[i](synapse_input))
            
            # 应用结构复杂性权重
            complexity_weight = self.structural_complexity[i]
            synapse_output = synapse_output * complexity_weight
            
            outputs.append(synapse_output)
        
        # 合并多突触输出
        multi_synapse_output = torch.stack(outputs, dim=-1)  # [batch, seq_len, hidden_dim, num_synapses]
        
        # 聚合多突触信息
        aggregated = torch.mean(multi_synapse_output, dim=-1)
        
        # 添加异步激活特征
        async_features_reshaped = async_features.view(batch_size, seq_len, self.num_synapses, self.hidden_dim)
        async_features_mean = torch.mean(async_features_reshaped, dim=-2)
        
        return aggregated + async_features_mean


class PositionalEncoding(nn.Module):
    """位置编码 - 基于海马体空间定位机制"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class AttentionMechanism(nn.Module):
    """海马体式注意力机制
    
    基于输入特异性增强和空间受限的单个突触放大现象
    """
    
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        # 输入特异性增强
        self.input_specificity = nn.Parameter(torch.randn(n_heads, d_model))
        
        # 空间受限机制
        self.spatial_constraint = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.size()
        
        # 计算Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 应用输入特异性增强
        Q = Q * self.input_specificity.unsqueeze(0).unsqueeze(-1)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用空间约束
        attn_weights = attn_weights * self.spatial_constraint(V).unsqueeze(1)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(context)


class TransformerMemoryEncoder(nn.Module):
    """基于Transformer的海马体记忆编码器
    
    实现快速一次性学习和非同步激活记忆编码
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_synapses: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 512
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # 多突触记忆印迹层
        self.memory_engram = MultiSynapticEngram(hidden_dim, hidden_dim, num_synapses)
        
        # Transformer层
        self.transformer_layers = nn.ModuleList([
            AttentionMechanism(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # 前馈网络
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_layers)
        ])
        
        # 层归一化
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers * 2)
        ])
        
        # 快速一次性学习机制
        self.one_shot_learning = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 记忆印迹门控
        self.engram_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        engram_activation: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        # 输入投影
        x = self.input_projection(x)
        
        # 添加位置编码
        x = self.pos_encoding(x)
        
        # 多突触记忆印迹编码
        memory_engram = self.memory_engram(x)
        
        # Transformer层处理
        for i, (attention, feed_forward) in enumerate(zip(self.transformer_layers, self.feed_forward)):
            # 自注意力
            attn_out = attention(x, mask)
            x = self.layer_norm[i * 2](x + self.dropout(attn_out))
            
            # 前馈网络
            ff_out = feed_forward(x)
            x = self.layer_norm[i * 2 + 1](x + self.dropout(ff_out))
            
            # 记忆印迹门控
            if engram_activation is not None:
                gate_value = self.engram_gate(torch.cat([x, engram_activation], dim=-1))
                x = x * gate_value
        
        # 快速一次性学习
        x = self.one_shot_learning(x)
        
        # 输出多突触信息
        output = {
            'encoded_memory': x,
            'memory_engram': memory_engram,
            'synaptic_complexity': self.memory_engram.structural_complexity,
            'attention_weights': self._get_last_attention_weights()
        }
        
        return output
    
    def _get_last_attention_weights(self) -> torch.Tensor:
        """获取最后的注意力权重用于分析"""
        # 这里简化实现，实际中需要从attention模块中提取
        return torch.zeros(1, 1, 1)  # 占位符


class EpisodicMemoryEncoder(TransformerMemoryEncoder):
    """情景记忆编码器
    
    专门用于情景记忆的快速编码和存储
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 情景记忆特定组件
        self.temporal_binding = nn.GRU(
            self.hidden_dim, self.hidden_dim, batch_first=True, bidirectional=True
        )
        
        self.spatial_encoding = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        self.context_integration = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
    def encode_episodic_memory(
        self,
        content: torch.Tensor,
        temporal_context: torch.Tensor,
        spatial_context: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """编码情景记忆"""
        
        # 内容编码
        content_encoding = self.forward(content)
        
        # 时间绑定
        temporal_features, _ = self.temporal_binding(content)
        
        # 空间编码
        spatial_features = self.spatial_encoding(spatial_context)
        
        # 整合时空信息
        combined_features = torch.cat([
            content_encoding['encoded_memory'],
            temporal_features,
            spatial_features
        ], dim=-1)
        
        episodic_memory = self.context_integration(combined_features)
        
        return {
            'episodic_memory': episodic_memory,
            'content_engram': content_encoding['memory_engram'],
            'temporal_binding': temporal_features,
            'spatial_encoding': spatial_features,
            'synaptic_weights': content_encoding['synaptic_complexity']
        }


def create_memory_encoder(
    input_dim: int,
    model_type: str = "transformer",
    **kwargs
) -> TransformerMemoryEncoder:
    """创建记忆编码器实例"""
    
    if model_type == "transformer":
        return TransformerMemoryEncoder(input_dim, **kwargs)
    elif model_type == "episodic":
        return EpisodicMemoryEncoder(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # 测试代码
    batch_size, seq_len, input_dim = 4, 32, 256
    x = torch.randn(batch_size, seq_len, input_dim)
    
    encoder = create_memory_encoder(input_dim, model_type="transformer")
    output = encoder(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output['encoded_memory'].shape}")
    print(f"记忆印迹形状: {output['memory_engram'].shape}")