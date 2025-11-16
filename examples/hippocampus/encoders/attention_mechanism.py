"""
增强注意力机制
基于海马体突触结构的非同步激活机制
实现输入特异性增强的注意力计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, Any


class EnhancedAttention(nn.Module):
    """
    基于神经科学发现的增强注意力机制
    
    核心原理：
    1. 非同步激活 - 不依赖同步激活机制
    2. 输入特异性增强 - 基于多突触末梢(MSBs)的结构复杂性
    3. 纳米级精确性 - 模拟突触级精度
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 msb_threshold: float = 0.7,
                 synaptic_plasticity: float = 0.05):
        """
        初始化增强注意力机制
        
        Args:
            hidden_dim: 隐藏层维度
            num_heads: 注意力头数
            dropout: Dropout概率
            msb_threshold: 多突触末梢激活阈值
            synaptic_plasticity: 突触可塑性强度
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.msb_threshold = msb_threshold
        self.synaptic_plasticity = synaptic_plasticity
        
        # 确保维度可整除
        assert hidden_dim % num_heads == 0
        
        # 查询、键、值投影层
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 多突触末梢结构权重(模拟MSBs)
        self.msb_weights = nn.Parameter(
            torch.ones(num_heads, self.head_dim, self.head_dim) * 0.5
        )
        
        # 输入特异性增强模块
        self.input_specificity_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 突触结构可塑性参数
        self.structural_plasticity = nn.Parameter(
            torch.ones(hidden_dim) * 0.1
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor, 
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            query: 查询张量 [batch_size, seq_len, hidden_dim]
            key: 键张量 [batch_size, seq_len, hidden_dim]
            value: 值张量 [batch_size, seq_len, hidden_dim]
            mask: 注意力掩码
            return_attention: 是否返回注意力权重
            
        Returns:
            输出张量和可选的注意力权重
        """
        batch_size, seq_len, hidden_dim = query.shape
        
        # 计算查询、键、值
        Q = self.query_proj(query)  # [batch_size, seq_len, hidden_dim]
        K = self.key_proj(key)
        V = self.value_proj(value)
        
        # 多头注意力
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数 (scaled dot-product)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # 应用多突触末梢结构增强
        # 使用每个头的平均权重来增强注意力分数
        msb_weights_per_head = self.msb_weights.mean(dim=(1,2))  # [num_heads]
        msb_weights_broadcast = msb_weights_per_head.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, num_heads, 1, 1]
        
        msb_enhancement = torch.sigmoid(attention_scores * msb_weights_broadcast)
        attention_scores = attention_scores * (1 + msb_enhancement * self.synaptic_plasticity)
        
        # 输入特异性增强
        # 为每个注意力头和序列位置应用特异性权重
        input_specificity = self.input_specificity_enhancer(query.mean(dim=1))  # [batch_size, hidden_dim]
        # 将特异性权重调整为适合attention_scores的形状
        input_specificity = input_specificity.mean(dim=-1, keepdim=True)  # [batch_size, 1]
        attention_scores = attention_scores * input_specificity.unsqueeze(-1).unsqueeze(-1)
        
        # 应用掩码
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        
        # 重新reshape并连接
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_dim
        )
        
        # 输出投影
        output = self.output_proj(context)
        
        # 层归一化
        output = self.layer_norm(output + query)  # 残差连接
        
        # 突触可塑性更新
        self.update_synaptic_weights(attention_weights)
        
        if return_attention:
            return output, attention_weights.mean(dim=1)
        else:
            return output, None
            
    def update_synaptic_weights(self, attention_weights: torch.Tensor):
        """模拟突触可塑性更新"""
        # 基于注意力模式更新结构权重
        with torch.no_grad():
            # 计算平均激活强度
            activation_level = attention_weights.mean()  # 标量
            # 为每个head添加微小的随机扰动来模拟可塑性
            update_factor = 1 + self.synaptic_plasticity * activation_level * torch.randn_like(self.msb_weights) * 0.1
            self.msb_weights.data *= update_factor
            
    def get_msb_statistics(self) -> Dict[str, float]:
        """获取多突触末梢统计信息"""
        with torch.no_grad():
            msb_activation = torch.sigmoid(self.msb_weights).mean().item()
            specificity = self.input_specificity_enhancer[3].weight.mean().item()
            
        return {
            'msb_activation_level': msb_activation,
            'input_specificity': specificity,
            'structural_complexity': self.msb_weights.std().item()
        }
        
    def compute_synaptic_contrast(self, 
                                  input1: torch.Tensor, 
                                  input2: torch.Tensor) -> torch.Tensor:
        """
        计算突触对比度，模拟非同步激活机制
        
        Args:
            input1: 第一组输入
            input2: 第二组输入
            
        Returns:
            突触对比度张量
        """
        with torch.no_grad():
            # 提取各自的特征表示
            feat1 = self.query_proj(input1)
            feat2 = self.query_proj(input2)
            
            # 计算相似性
            similarity = F.cosine_similarity(feat1, feat2, dim=-1)
            
            # 非同步激活计算（基于差异而非相似性）
            contrast = 1 - similarity
            
            # 应用多突触末梢结构增强
            contrast_enhanced = contrast * torch.sigmoid(self.msb_weights.mean(dim=0)).mean()
            
        return contrast_enhanced