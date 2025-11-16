"""
模式完成模块
基于海马体CA3网络的稀疏编码和模式完成机制
实现记忆的模式分离和完整重构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np


class SparseEncodingLayer(nn.Module):
    """
    稀疏编码层
    模拟CA3区神经元的稀疏激活模式
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 sparsity: float = 0.1,
                 temperature: float = 1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        self.temperature = temperature
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # 稀疏约束层
        self.sparse_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 重构器
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # 稀疏性控制参数
        self.sparse_threshold = nn.Parameter(torch.tensor(0.5))
        self.sparse_gain = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            编码向量、重构输出、稀疏掩码
        """
        # 编码
        encoded = self.encoder(x)
        
        # 计算稀疏门控
        sparse_gate = self.sparse_gate(encoded)
        
        # 应用稀疏约束
        sparse_mask = torch.sigmoid(
            (sparse_gate - self.sparse_threshold) * self.sparse_gain / self.temperature
        )
        
        # 应用稀疏掩码
        sparse_encoded = encoded * sparse_mask
        
        # 重构
        reconstructed = self.decoder(sparse_encoded)
        
        return sparse_encoded, reconstructed, sparse_mask


class PatternCompletionLayer(nn.Module):
    """
    模式完成层
    基于Hebbian学习和稀疏编码的记忆完整化
    """
    
    def __init__(self,
                 hidden_dim: int,
                 completion_rate: float = 0.8,
                 hebbian_learning_rate: float = 0.01):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.completion_rate = completion_rate
        self.hebbian_learning_rate = hebbian_learning_rate
        
        # 记忆权重矩阵 (模拟突触连接)
        self.memory_weights = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim) * 0.1
        )
        
        # 完成网络
        self.completion_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Hebbian更新规则实现
        self.hebbian_update = nn.Parameter(
            torch.ones(hidden_dim, hidden_dim) * hebbian_learning_rate
        )
        
    def forward(self, 
                pattern: torch.Tensor,
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        执行模式完成
        
        Args:
            pattern: 输入模式 [batch_size, hidden_dim]
            context: 上下文信息
            
        Returns:
            完成后的模式
        """
        batch_size = pattern.shape[0]
        
        # 应用记忆权重
        memory_projection = torch.matmul(pattern, self.memory_weights)
        
        # 上下文增强(如果有)
        if context is not None:
            # 如果context是3维张量，取其平均或第一个时间步
            if context.dim() == 3:
                context = context.mean(dim=1)  # [batch_size, hidden_dim]
            context_enhanced = context.expand(-1, self.hidden_dim)
            memory_projection = memory_projection + 0.1 * context_enhanced
        
        # 迭代完成过程
        completed_pattern = pattern.clone()
        
        for iteration in range(int(1 / self.completion_rate)):
            # 计算当前模式的激活
            activation = torch.matmul(completed_pattern, self.memory_weights)
            
            # 应用完成网络
            updated_pattern = self.completion_network(activation)
            
            # 结合原始模式和新模式
            completed_pattern = (1 - self.completion_rate) * completed_pattern + \
                              self.completion_rate * updated_pattern
        
        # Hebbian学习更新
        self.update_hebbian_weights(completed_pattern)
        
        return completed_pattern
    
    def update_hebbian_weights(self, pattern: torch.Tensor):
        """更新Hebbian权重"""
        with torch.no_grad():
            # 如果pattern是2维，取第一个维度
            if pattern.dim() > 1:
                pattern = pattern.mean(dim=0)  # [hidden_dim]
            
            # Hebbian更新: W = W + η * (x - mean(x)) * (y - mean(y))^T
            pattern_centered = pattern - pattern.mean()
            outer_product = torch.outer(pattern_centered, pattern_centered)
            
            self.memory_weights.data += self.hebbian_learning_rate * outer_product
            
            # 权重正则化
            self.memory_weights.data *= 0.99  # 防止权重爆炸


class PatternSeparationLayer(nn.Module):
    """
    模式分离层
    确保相似输入产生不同表示
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 separation_strength: float = 0.8):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.separation_strength = separation_strength
        
        # 分离投影
        self.separation_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # 相似性检测器
        self.similarity_detector = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 分离增强器
        self.separation_enhancer = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        执行模式分离
        
        Args:
            x1: 第一组输入
            x2: 第二组输入
            
        Returns:
            分离后的表示1、表示2、分离程度
        """
        # 计算输入相似性
        similarity_score = self.similarity_detector(
            torch.cat([x1, x2], dim=0)
        ).mean()
        
        # 分离投影
        y1 = self.separation_proj(x1)
        y2 = self.separation_proj(x2)
        
        # 相似性驱动的分离增强
        if similarity_score > 0.5:  # 相似性高时增强分离
            separation_factor = self.separation_strength * similarity_score
            
            # 向不同方向扩展
            noise1 = torch.randn_like(y1) * separation_factor * 0.1
            noise2 = -torch.randn_like(y2) * separation_factor * 0.1
            
            y1 = y1 + noise1
            y2 = y2 + noise2
        
        # 进一步分离增强
        y1 = y1 * (1 + self.separation_strength)
        y2 = y2 * (1 + self.separation_strength)
        
        y1 = self.separation_enhancer(y1)
        y2 = self.separation_enhancer(y2)
        
        # 计算实际分离程度
        separation_degree = torch.mean(
            torch.abs(F.normalize(y1, dim=-1) - F.normalize(y2, dim=-1))
        ).item()
        
        return y1, y2, separation_degree


class PatternCompletionModule(nn.Module):
    """
    完整的模式完成模块
    集成稀疏编码、模式分离和模式完成功能
    """
    
    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 sparsity: float = 0.1,
                 completion_rate: float = 0.8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # 多头稀疏编码
        self.sparse_layers = nn.ModuleList([
            SparseEncodingLayer(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim,
                sparsity=sparsity,
                temperature=1.0 / (1.0 + i * 0.1)  # 递减温度
            ) for i in range(num_heads)
        ])
        
        # 模式完成层
        self.completion_layers = nn.ModuleList([
            PatternCompletionLayer(
                hidden_dim=hidden_dim,
                completion_rate=completion_rate,
                hebbian_learning_rate=0.01 * (1.0 + i * 0.1)
            ) for i in range(num_heads)
        ])
        
        # 模式分离层
        self.separation_layer = PatternSeparationLayer(
            input_dim=hidden_dim,
            output_dim=hidden_dim,
            separation_strength=0.8
        )
        
        # 注意力融合
        self.attention_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_heads, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                x: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                mode: str = 'completion') -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_dim]
            context: 上下文信息
            mode: 操作模式 ('completion', 'separation', 'both')
            
        Returns:
            处理后的张量
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        if mode == 'completion':
            return self._pattern_completion(x, context)
        elif mode == 'separation':
            return self._pattern_separation(x)
        elif mode == 'both':
            return self._combined_processing(x, context)
        else:
            return x
    
    def _pattern_completion(self, x: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """模式完成"""
        batch_size, seq_len, hidden_dim = x.shape
        
        # 重塑为批处理形式
        x_flat = x.view(-1, hidden_dim)
        
        # 多头稀疏编码
        sparse_outputs = []
        for sparse_layer in self.sparse_layers:
            encoded, _, sparse_mask = sparse_layer(x_flat)
            sparse_outputs.append(encoded)
        
        # 多头模式完成
        completed_outputs = []
        for i, completion_layer in enumerate(self.completion_layers):
            completed = completion_layer(sparse_outputs[i], context)
            completed_outputs.append(completed)
        
        # 注意力融合
        combined = torch.cat(completed_outputs, dim=-1)
        output = self.attention_fusion(combined)
        
        # 重新reshape
        output = output.view(batch_size, seq_len, hidden_dim)
        output = self.layer_norm(output + x)  # 残差连接
        
        return output
    
    def _pattern_separation(self, x: torch.Tensor) -> torch.Tensor:
        """模式分离"""
        batch_size, seq_len, hidden_dim = x.shape
        
        separated_outputs = []
        
        # 对序列中的相邻元素进行分离
        for i in range(seq_len - 1):
            y1, y2, separation_degree = self.separation_layer(x[:, i], x[:, i + 1])
            separated_outputs.extend([y1, y2])
        
        # 处理最后一个元素
        if seq_len > 0:
            separated_outputs.append(x[:, -1])
        
        # 组合结果
        if len(separated_outputs) > 0:
            output = torch.stack(separated_outputs, dim=1)
            # 确保维度正确
            if output.shape[1] < seq_len:
                padding = torch.zeros_like(x[:, :seq_len - output.shape[1]])
                output = torch.cat([output, padding], dim=1)
        else:
            output = x
            
        return output
    
    def _combined_processing(self, x: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        """组合处理：先分离再完成"""
        # 第一步：模式分离
        separated = self._pattern_separation(x)
        
        # 第二步：模式完成
        completed = self._pattern_completion(separated, context)
        
        return completed
    
    def compute_separation_metrics(self, 
                                 x1: torch.Tensor, 
                                 x2: torch.Tensor) -> Dict[str, float]:
        """
        计算模式分离指标
        """
        y1, y2, separation_degree = self.separation_layer(x1, x2)
        
        # 计算不同类型的分离指标
        metrics = {
            'separation_degree': separation_degree,
            'cosine_distance': torch.mean(
                1 - F.cosine_similarity(F.normalize(y1), F.normalize(y2))
            ).item(),
            'euclidean_distance': torch.mean(
                torch.norm(y1 - y2, dim=-1)
            ).item(),
            'correlation_difference': torch.mean(
                torch.abs(torch.corrcoef(y1.t())[0, 1] - torch.corrcoef(y2.t())[0, 1])
            ).item() if y1.shape[-1] > 1 else 0.0
        }
        
        return metrics
    
    def get_sparse_statistics(self) -> Dict[str, float]:
        """
        获取稀疏编码统计信息
        """
        total_sparsity = 0.0
        total_temperature = 0.0
        
        for sparse_layer in self.sparse_layers:
            total_sparsity += sparse_layer.sparsity
            total_temperature += sparse_layer.temperature
            
        return {
            'average_sparsity': total_sparsity / len(self.sparse_layers),
            'average_temperature': total_temperature / len(self.sparse_layers),
            'num_sparse_layers': len(self.sparse_layers),
            'num_completion_layers': len(self.completion_layers)
        }