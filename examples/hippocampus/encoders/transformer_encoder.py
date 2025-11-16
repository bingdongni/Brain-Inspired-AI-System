"""
Transformer记忆编码器
基于海马体CA3-CA1通路重构机制的非同步激活记忆编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from .attention_mechanism import EnhancedAttention
from .pattern_completion import PatternCompletionModule
from .temporal_alignment import TemporalAlignmentModule


class MemoryEncodingBlock(nn.Module):
    """
    记忆编码模块
    基于海马体CA3-CA1神经回路的非同步激活机制
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 msb_enhancement: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.msb_enhancement = msb_enhancement
        
        # 增强注意力机制
        self.attention = EnhancedAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            msb_threshold=0.7 if msb_enhancement else 0.0,
            synaptic_plasticity=0.05 if msb_enhancement else 0.0
        )
        
        # 记忆路由网络(模拟CA3到CA1的投射)
        self.memory_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 突触前结构整合(模拟突触前变化)
        self.presynaptic_integration = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 突触后重新分布(模拟树突棘重新分布)
        self.postsynaptic_redistribution = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 多突触末梢增强系数
        self.msb_enhancement_coef = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, 
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                return_stats: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
            context: 上下文信息
            return_stats: 是否返回统计信息
            
        Returns:
            编码后的张量和统计信息
        """
        stats = {}
        
        # 第一步：增强注意力机制
        if context is not None:
            # CA3到CA1的路由机制
            context_routed = self.memory_router(context)
            x_enhanced = x + context_routed
        else:
            x_enhanced = x
            
        # 注意力计算
        attn_output, attention_weights = self.attention(
            x_enhanced, x_enhanced, x_enhanced, return_attention=True
        )
        
        # 第一层残差连接和层归一化
        x = self.norm1(x_enhanced + attn_output)
        
        # 突触前结构整合
        presyn_integrated = self.presynaptic_integration(x)
        
        # 突触后重新分布
        postsyn_distributed = self.postsynaptic_redistribution(x)
        
        # 多突触末梢增强(如果启用)
        if self.msb_enhancement:
            msb_enhanced = self.msb_enhancement_coef * (presyn_integrated + postsyn_distributed)
            x = x + msb_enhanced
        
        # 前馈网络
        ffn_output = self.ffn(x)
        
        # 第二层残差连接和层归一化
        x = self.norm2(x + ffn_output)
        
        # 收集统计信息
        if return_stats:
            attention_variance = 0.0
            if attention_weights is not None:
                attention_variance = torch.var(attention_weights).item()
            
            stats.update({
                'attention_variance': attention_variance,
                'presynaptic_strength': torch.mean(presyn_integrated).item(),
                'postsynaptic_strength': torch.mean(postsyn_distributed).item(),
                'msb_enhancement_level': self.msb_enhancement_coef.item(),
                'output_magnitude': torch.norm(x).item()
            })
            
        return x, stats if return_stats else None


class TransformerMemoryEncoder(nn.Module):
    """
    基于Transformer的海马体记忆编码器
    
    核心特性：
    1. 非同步激活机制 - 不依赖突触前后神经元同步
    2. 多突触末梢增强 - 基于MSBs结构复杂性
    3. 输入特异性增强 - 纳米级精确性
    4. 模式完成能力 - 基于CA3-CA1通路
    5. 时间序列对齐 - 情景记忆时序编码
    """
    
    def __init__(self,
                 vocab_size: int,
                 hidden_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 max_seq_len: int = 2048,
                 dropout: float = 0.1,
                 msb_enhancement: bool = True,
                 pattern_completion: bool = True,
                 temporal_alignment: bool = True):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.msb_enhancement = msb_enhancement
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # 记忆编码层堆叠
        self.encoding_layers = nn.ModuleList([
            MemoryEncodingBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                msb_enhancement=msb_enhancement,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # 模式完成模块
        if pattern_completion:
            self.pattern_completion = PatternCompletionModule(
                hidden_dim=hidden_dim,
                num_heads=num_heads
            )
        
        # 时间序列对齐模块
        if temporal_alignment:
            self.temporal_alignment = TemporalAlignmentModule(
                hidden_dim=hidden_dim
            )
            
        # 记忆存储模块(模拟海马体CA1区)
        self.memory_storage = nn.ModuleDict({
            'episodic_memory': nn.Linear(hidden_dim, hidden_dim * 2),
            'semantic_memory': nn.Linear(hidden_dim, hidden_dim * 2),
            'procedural_memory': nn.Linear(hidden_dim, hidden_dim * 2)
        })
        
        # 初始化参数
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化模型参数"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_ca3_ca1_network(self, x: torch.Tensor) -> torch.Tensor:
        """
        创建CA3-CA1神经网络通路
        
        基于研究发现的CA3到CA1投射重构机制
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # CA3区：模式分离和完成
        ca3_output = x
        
        # CA1区：记忆存储和检索
        ca1_memory = torch.zeros_like(ca3_output)
        
        # 模拟Schaffer侧支投射
        for i in range(0, seq_len, 4):  # 每4个位置一个连接
            end_idx = min(i + 4, seq_len)
            ca3_slice = ca3_output[:, i:end_idx]
            
            # 投射到CA1
            connection_strength = torch.sigmoid(
                torch.randn(1, device=x.device) * 0.1 + 0.5
            )
            ca1_slice = ca3_slice * connection_strength
            
            ca1_memory[:, i:end_idx] = ca1_slice
            
        return ca1_memory
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                memory_type: str = 'episodic',
                return_stats: bool = False) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        前向传播
        
        Args:
            input_ids: 输入序列 [batch_size, seq_len]
            attention_mask: 注意力掩码
            context: 上下文信息
            memory_type: 记忆类型 ('episodic', 'semantic', 'procedural')
            return_stats: 是否返回统计信息
            
        Returns:
            编码结果和统计信息
        """
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        token_embeds = self.token_embedding(input_ids)
        
        # 位置嵌入
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(batch_size, 1)
        position_embeds = self.position_embedding(position_ids)
        
        # 组合嵌入
        x = token_embeds + position_embeds
        
        # 应用注意力掩码
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)
        
        # 创建CA3-CA1网络通路
        ca3_output = self.create_ca3_ca1_network(x)
        
        # 逐层编码
        all_layer_outputs = []
        all_layer_stats = []
        
        for i, layer in enumerate(self.encoding_layers):
            x, stats = layer(x, context, return_stats=True)
            all_layer_outputs.append(x)
            all_layer_stats.append(stats)
            
        # 模式完成(如果启用)
        if hasattr(self, 'pattern_completion') and seq_len > 1:
            x = self.pattern_completion(x, ca3_output)
        
        # 时间序列对齐(如果启用)
        if hasattr(self, 'temporal_alignment'):
            x = self.temporal_alignment(x)
        
        # 记忆存储到特定区域
        if memory_type in self.memory_storage:
            x = self.memory_storage[memory_type](x)
        
        # 输出投影
        logits = self.output_projection(x)
        
        # 准备返回结果
        if return_stats:
            stats = {
                'num_layers': len(self.encoding_layers),
                'sequence_length': seq_len,
                'memory_type': memory_type,
                'ca3_ca1_connections': seq_len // 4,
                'layer_stats': all_layer_stats,
                'total_params': sum(p.numel() for p in self.parameters())
            }
            
            # 添加每层统计信息
            for i, layer_stats in enumerate(all_layer_stats):
                for key, value in layer_stats.items():
                    stats[f'layer_{i}_{key}'] = value
        
        return logits, stats if return_stats else None
    
    def encode_memory_pattern(self, 
                            memories: torch.Tensor,
                            context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        专门用于记忆模式编码
        
        Args:
            memories: 记忆输入 [batch_size, memory_dim]
            context: 上下文信息
            
        Returns:
            编码后的记忆模式
        """
        # 重塑为序列格式
        batch_size = memories.shape[0]
        memories_seq = memories.unsqueeze(1)  # [batch_size, 1, memory_dim]
        
        # 编码
        encoded, _ = self.forward(
            input_ids=memories_seq,
            context=context,
            memory_type='episodic',
            return_stats=False
        )
        
        return encoded.squeeze(1)  # [batch_size, hidden_dim]
    
    def retrieve_similar_memories(self, 
                                query: torch.Tensor,
                                memory_bank: torch.Tensor,
                                top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检索相似的记忆
        
        Args:
            query: 查询向量
            memory_bank: 记忆库 [num_memories, hidden_dim]
            top_k: 返回前k个最相似的记忆
            
        Returns:
            相似记忆和相似度分数
        """
        # 计算相似度
        query_norm = F.normalize(query, dim=-1)
        bank_norm = F.normalize(memory_bank, dim=-1)
        
        similarities = torch.matmul(query_norm, bank_norm.transpose(0, 1))
        
        # 获取top-k相似记忆
        top_similarities, top_indices = torch.topk(similarities, k=top_k, dim=-1)
        
        similar_memories = memory_bank[top_indices]
        
        return similar_memories, top_similarities
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """
        获取模型记忆统计信息
        """
        stats = {
            'model_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad),
            'model_size_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / (1024**2),
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'max_seq_len': self.max_seq_len,
            'msb_enhancement_enabled': self.msb_enhancement,
            'ca3_ca1_connections': self.max_seq_len // 4
        }
        
        # 统计各层激活强度
        total_activation = 0
        for layer in self.encoding_layers:
            if hasattr(layer, 'attention'):
                stats['attention_layers'] = stats.get('attention_layers', 0) + 1
        
        return stats