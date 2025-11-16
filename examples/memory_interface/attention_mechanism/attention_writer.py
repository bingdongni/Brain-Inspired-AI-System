"""
注意力写入器 - 基于输入信息选择性地存储到记忆库
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import deque


class AttentionWriter(nn.Module):
    """
    基于注意力的记忆写入器
    使用注意力机制决定新信息的存储位置和重要性
    """
    
    def __init__(
        self,
        memory_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_memories: int = 10000,
        importance_threshold: float = 0.7,
        decay_rate: float = 0.95
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_memories = max_memories
        self.importance_threshold = importance_threshold
        self.decay_rate = decay_rate
        
        # 输入投影
        self.input_proj = nn.Linear(memory_dim, hidden_dim)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 重要性评估网络
        self.importance_evaluator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 记忆更新网络
        self.memory_updater = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 记忆压缩网络
        self.compression_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, memory_dim),
            nn.Tanh()
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 遗忘机制
        self.forgetting_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 记忆质量跟踪
        self.memory_quality_tracker = nn.Linear(memory_dim, 1)
        
        # 记忆索引管理
        self.memory_index = deque(maxlen=max_memories)
        self.importance_scores = []
        
    def forward(
        self,
        new_info: torch.Tensor,
        memory_bank: torch.Tensor,
        memory_meta: Optional[Dict] = None,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播 - 决定如何存储新信息
        
        Args:
            new_info: 新信息 [batch_size, seq_len, memory_dim]
            memory_bank: 当前记忆库
            memory_meta: 记忆元数据
            context: 上下文信息
            
        Returns:
            updated_memory_bank: 更新后的记忆库
            write_weights: 写入权重
            importance_scores: 重要性分数
        """
        batch_size, seq_len, _ = new_info.shape
        
        # 投影输入信息
        info_proj = self.input_proj(new_info)
        
        # 如果有上下文，将其融合到信息中
        if context is not None:
            context_proj = self.input_proj(context)
            info_proj = info_proj + context_proj
        
        # 评估信息重要性
        importance = self.importance_evaluator(info_proj)  # [batch_size, seq_len, 1]
        importance_scores = importance.squeeze(-1)  # [batch_size, seq_len]
        
        # 对记忆库进行注意力计算，确定存储位置
        memory_proj = self.input_proj(memory_bank)
        
        # 新信息对记忆库的注意力
        write_weights, _ = self.attention(
            memory_proj, info_proj, info_proj
        )
        
        # 应用重要性权重
        write_weights = write_weights * importance.unsqueeze(-1)
        
        # 记忆更新策略
        updated_memory_bank = memory_bank.clone()
        
        for b in range(batch_size):
            for s in range(seq_len):
                if importance_scores[b, s] > self.importance_threshold:
                    # 选择写入位置
                    write_position = self._select_write_position(
                        write_weights[b], importance_scores[b, s]
                    )
                    
                    # 更新记忆
                    updated_memory_bank[b, write_position] = self._update_memory(
                        memory_bank[b, write_position],
                        new_info[b, s],
                        write_weights[b, write_position]
                    )
                    
                    # 记录记忆索引
                    if len(self.memory_index) < self.max_memories:
                        self.memory_index.append(write_position)
                    else:
                        self.memory_index.popleft()
                        self.memory_index.append(write_position)
        
        # 遗忘机制 - 对低重要性记忆进行衰减
        updated_memory_bank = self._apply_forgetting(updated_memory_bank)
        
        return updated_memory_bank, write_weights, importance_scores
    
    def _select_write_position(
        self,
        write_weights: torch.Tensor,
        importance_score: float
    ) -> int:
        """
        选择写入位置
        
        Args:
            write_weights: 写入权重
            importance_score: 重要性分数
            
        Returns:
            写入位置索引
        """
        # 基于权重和重要性选择最佳位置
        combined_weights = write_weights * importance_score
        
        # 选择权重最高的位置
        _, max_idx = torch.max(combined_weights, dim=0)
        
        return max_idx.item()
    
    def _update_memory(
        self,
        old_memory: torch.Tensor,
        new_info: torch.Tensor,
        write_weight: torch.Tensor
    ) -> torch.Tensor:
        """
        更新记忆内容
        
        Args:
            old_memory: 旧记忆
            new_info: 新信息
            write_weight: 写入权重
            
        Returns:
            更新后的记忆
        """
        # 组合旧记忆和新信息
        combined = torch.cat([old_memory, new_info], dim=-1)
        
        # 计算更新向量
        update_vector = self.memory_updater(combined)
        
        # 应用写入权重
        weighted_update = update_vector * write_weight
        
        # 新记忆 = 旧记忆 + 加权更新
        new_memory = old_memory + weighted_update
        
        return new_memory
    
    def _apply_forgetting(
        self,
        memory_bank: torch.Tensor,
        memory_age: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        应用遗忘机制
        
        Args:
            memory_bank: 记忆库
            memory_age: 记忆年龄列表
            
        Returns:
            应用遗忘后的记忆库
        """
        updated_bank = memory_bank.clone()
        
        # 计算每个记忆的质量分数
        quality_scores = torch.sigmoid(
            self.memory_quality_tracker(memory_bank)
        ).squeeze(-1)
        
        # 应用遗忘率
        decay_factors = torch.pow(
            self.decay_rate, 
            torch.tensor(memory_age or [0] * memory_bank.size(1))
        ).to(memory_bank.device)
        
        # 应用衰减
        updated_bank = updated_bank * decay_factors.unsqueeze(0).unsqueeze(-1)
        
        return updated_bank
    
    def compress_memory(
        self,
        memory_bank: torch.Tensor,
        compression_ratio: float = 0.5
    ) -> torch.Tensor:
        """
        压缩记忆库
        
        Args:
            memory_bank: 原始记忆库
            compression_ratio: 压缩比例
            
        Returns:
            压缩后的记忆
        """
        # 使用压缩网络进行降维
        compressed = self.compression_net(memory_bank)
        
        return compressed
    
    def smart_storage(
        self,
        new_info: torch.Tensor,
        memory_bank: torch.Tensor,
        storage_budget: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        智能存储策略
        
        Args:
            new_info: 新信息
            memory_bank: 当前记忆库
            storage_budget: 存储预算
            
        Returns:
            存储结果字典
        """
        batch_size, seq_len, _ = new_info.shape
        storage_budget = storage_budget or self.max_memories // 2
        
        # 评估每个信息的重要性
        info_proj = self.input_proj(new_info)
        importance_scores = self.importance_evaluator(info_proj).squeeze(-1)
        
        # 按重要性排序
        importance_order = torch.argsort(importance_scores, dim=-1, descending=True)
        
        # 选择要存储的信息
        selected_count = min(seq_len, storage_budget)
        selected_indices = importance_order[:, :selected_count]
        
        # 提取选中的信息
        selected_info = torch.gather(
            new_info, 1,
            selected_indices.unsqueeze(-1).expand(-1, -1, self.memory_dim)
        )
        selected_importance = torch.gather(importance_scores, 1, selected_indices)
        
        # 执行存储
        updated_bank, write_weights, _ = self.forward(
            selected_info, memory_bank
        )
        
        return {
            'updated_memory_bank': updated_bank,
            'selected_indices': selected_indices,
            'selected_importance': selected_importance,
            'write_weights': write_weights,
            'storage_count': selected_count
        }
    
    def memory_maintenance(self, memory_bank: torch.Tensor) -> torch.Tensor:
        """
        记忆维护 - 定期清理和优化记忆库
        
        Args:
            memory_bank: 当前记忆库
            
        Returns:
            维护后的记忆库
        """
        # 评估记忆质量
        quality_scores = torch.sigmoid(
            self.memory_quality_tracker(memory_bank)
        ).squeeze(-1)
        
        # 识别低质量记忆
        low_quality_mask = quality_scores < 0.3
        
        # 对低质量记忆进行衰减或删除
        if low_quality_mask.any():
            # 应用更强衰减
            decay_factor = 0.1
            memory_bank[low_quality_mask] *= decay_factor
        
        return memory_bank
    
    def get_memory_statistics(self, memory_bank: torch.Tensor) -> Dict:
        """
        获取记忆库统计信息
        
        Args:
            memory_bank: 记忆库
            
        Returns:
            统计信息字典
        """
        # 计算记忆激活度
        activation = torch.norm(memory_bank, dim=-1)
        avg_activation = activation.mean().item()
        max_activation = activation.max().item()
        
        # 计算记忆质量
        quality_scores = torch.sigmoid(
            self.memory_quality_tracker(memory_bank)
        ).squeeze(-1)
        avg_quality = quality_scores.mean().item()
        
        # 计算记忆使用频率
        usage_freq = torch.bincount(
            torch.tensor(self.memory_index),
            minlength=self.max_memories
        )
        avg_usage = usage_freq.float().mean().item()
        
        return {
            'memory_bank_shape': memory_bank.shape,
            'average_activation': avg_activation,
            'max_activation': max_activation.item(),
            'average_quality': avg_quality,
            'average_usage_frequency': avg_usage,
            'memory_index_length': len(self.memory_index),
            'importance_threshold': self.importance_threshold
        }


if __name__ == "__main__":
    # 测试代码
    writer = AttentionWriter(
        memory_dim=512,
        hidden_dim=512,
        num_heads=8,
        max_memories=1000
    )
    
    # 创建测试数据
    batch_size = 2
    seq_len = 5
    memory_dim = 512
    max_memories = 1000
    
    new_info = torch.randn(batch_size, seq_len, memory_dim)
    memory_bank = torch.randn(batch_size, max_memories, memory_dim)
    
    # 执行写入
    updated_bank, write_weights, importance_scores = writer(new_info, memory_bank)
    
    print(f"更新后记忆库形状: {updated_bank.shape}")
    print(f"写入权重形状: {write_weights.shape}")
    print(f"重要性分数形状: {importance_scores.shape}")
    
    # 获取统计信息
    stats = writer.get_memory_statistics(memory_bank)
    print(f"记忆统计: {stats}")
    
    # 智能存储测试
    smart_result = writer.smart_storage(new_info, memory_bank)
    print(f"智能存储结果键: {smart_result.keys()}")