"""
时间序列对齐模块
基于海马体CA1区的时序记忆编码机制
实现情景记忆的时间序列处理和记忆巩固
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class TemporalMemoryCell(nn.Module):
    """
    时间记忆细胞
    模拟海马体CA1区的时序编码机制
    """
    
    def __init__(self,
                 hidden_dim: int,
                 time_decay_rate: float = 0.95,
                 consolidation_rate: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.time_decay_rate = time_decay_rate
        self.consolidation_rate = consolidation_rate
        
        # 记忆状态
        self.memory_state = None
        self.time_step = 0
        
        # 遗忘门控
        self.forgetting_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 记忆巩固门控
        self.consolidation_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # 时间特征提取器
        self.temporal_feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, 
                x: torch.Tensor,
                time_encoding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        时间记忆细胞前向传播
        
        Args:
            x: 当前输入 [batch_size, hidden_dim]
            time_encoding: 时间编码
            
        Returns:
            输出的记忆状态
        """
        batch_size = x.shape[0]
        
        # 初始化记忆状态（如果为空）
        if self.memory_state is None:
            self.memory_state = torch.zeros_like(x)
        
        # 时间特征提取
        temporal_features = self.temporal_feature_extractor(x)
        
        # 遗忘计算
        if time_encoding is not None:
           遗忘_input = torch.cat([temporal_features, time_encoding], dim=-1)
        else:
            遗忘_input = torch.cat([temporal_features, self.memory_state], dim=-1)
            
        forgetting_weights = self.forgetting_gate(遗忘_input)
        
        # 应用遗忘
        decayed_memory = self.memory_state * (forgetting_weights * self.time_decay_rate + (1 - forgetting_weights))
        
        # 记忆巩固
        consolidation_weights = self.consolidation_gate(x)
        consolidated_memory = (1 - consolidation_weights) * decayed_memory + \
                            consolidation_weights * temporal_features
        
        # 更新记忆状态
        self.memory_state = consolidated_memory
        
        # 输出计算
        output = self.output_proj(consolidated_memory)
        
        # 更新时间步
        self.time_step += 1
        
        return output
    
    def reset(self):
        """重置记忆状态"""
        self.memory_state = None
        self.time_step = 0


class TimeSeriesAlignment(nn.Module):
    """
    时间序列对齐模块
    实现动态时间规整(DTW)算法的神经网络近似
    """
    
    def __init__(self,
                 hidden_dim: int,
                 max_time_diff: int = 10,
                 alignment_strength: float = 1.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.max_time_diff = max_time_diff
        self.alignment_strength = alignment_strength
        
        # 对齐评分网络
        self.alignment_scoring = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 对齐路径预测
        self.path_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # match, insert, delete
            nn.LogSoftmax(dim=-1)
        )
        
        # 时间压缩/扩展网络
        self.temporal_compression = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
    def compute_alignment_matrix(self, 
                               seq1: torch.Tensor, 
                               seq2: torch.Tensor) -> torch.Tensor:
        """
        计算对齐矩阵
        
        Args:
            seq1: 序列1 [seq_len1, hidden_dim]
            seq2: 序列2 [seq_len2, hidden_dim]
            
        Returns:
            对齐矩阵
        """
        seq_len1, seq_len2 = seq1.shape[0], seq2.shape[0]
        
        # 扩展序列维度用于批量计算
        seq1_expanded = seq1.unsqueeze(1).expand(-1, seq_len2, -1)  # [seq_len1, seq_len2, hidden_dim]
        seq2_expanded = seq2.unsqueeze(0).expand(seq_len1, -1, -1)  # [seq_len1, seq_len2, hidden_dim]
        
        # 计算对齐分数
        alignment_inputs = torch.cat([seq1_expanded, seq2_expanded], dim=-1)
        alignment_scores = self.alignment_scoring(alignment_inputs).squeeze(-1)
        
        return alignment_scores
    
    def forward(self, 
                x: torch.Tensor,
                reference: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        时间序列对齐前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, hidden_dim]
            reference: 参考序列
            
        Returns:
            对齐后的序列
        """
        if reference is None:
            # 自对齐模式
            reference = x
        
        batch_size, seq_len, hidden_dim = x.shape
        
        aligned_outputs = []
        
        for i in range(batch_size):
            x_seq = x[i]  # [seq_len, hidden_dim]
            ref_seq = reference[i] if reference.shape[0] == batch_size else reference
            
            # 计算对齐矩阵
            alignment_matrix = self.compute_alignment_matrix(x_seq, ref_seq)
            
            # 预测对齐路径
            path_logits = self.path_predictor(x_seq)
            
            # 应用对齐
            aligned_seq = self.apply_alignment(x_seq, ref_seq, alignment_matrix, path_logits)
            aligned_outputs.append(aligned_seq)
        
        # 组合结果
        aligned_output = torch.stack(aligned_outputs, dim=0)
        
        return aligned_output
    
    def apply_alignment(self,
                       x_seq: torch.Tensor,
                       ref_seq: torch.Tensor,
                       alignment_matrix: torch.Tensor,
                       path_logits: torch.Tensor) -> torch.Tensor:
        """
        应用对齐算法
        
        Args:
            x_seq: 原始序列
            ref_seq: 参考序列
            alignment_matrix: 对齐矩阵
            path_logits: 路径预测
            
        Returns:
            对齐后的序列
        """
        seq_len, hidden_dim = x_seq.shape
        
        # DTW算法近似
        aligned_seq = torch.zeros_like(x_seq)
        
        # 计算最优路径
        for t in range(seq_len):
            if t == 0:
                aligned_seq[0] = x_seq[0]
            else:
                # 候选对齐操作
                candidates = []
                
                # 匹配
                if t < len(ref_seq):
                    match_score = alignment_matrix[t, t] * path_logits[t, 0]
                    candidates.append((match_score, t, 'match'))
                
                # 插入
                insert_score = alignment_matrix[t-1, t] * path_logits[t, 1] if t-1 >= 0 else 0
                candidates.append((insert_score, t-1, 'insert'))
                
                # 删除
                delete_score = alignment_matrix[t, t-1] * path_logits[t, 2] if t-1 >= 0 else 0
                candidates.append((delete_score, t, 'delete'))
                
                # 选择最佳操作
                if candidates:
                    best_score, best_idx, best_op = max(candidates, key=lambda x: x[0])
                    
                    if best_op == 'match' and best_idx < len(ref_seq):
                        # 线性插值
                        alpha = self.alignment_strength * alignment_matrix[t, best_idx]
                        aligned_seq[t] = (1 - alpha) * x_seq[t] + alpha * ref_seq[best_idx]
                    elif best_op == 'insert':
                        if best_idx >= 0:
                            # 插入参考序列元素
                            aligned_seq[t] = ref_seq[best_idx + 1] if best_idx + 1 < len(ref_seq) else ref_seq[best_idx]
                        else:
                            aligned_seq[t] = x_seq[t]
                    else:  # delete
                        aligned_seq[t] = x_seq[t-1]
                else:
                    aligned_seq[t] = x_seq[t]
        
        return aligned_seq


class MemoryConsolidation(nn.Module):
    """
    记忆巩固模块
    模拟海马体到新皮层的记忆转移过程
    """
    
    def __init__(self,
                 hidden_dim: int,
                 consolidation_tau: float = 100.0,
                 stability_threshold: float = 0.8):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.consolidation_tau = consolidation_tau
        self.stability_threshold = stability_threshold
        
        # 巩固评估器
        self.consolidation_assessor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 稳定性检测器
        self.stability_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 记忆强度预测器
        self.strength_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                memory_states: List[torch.Tensor],
                current_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        记忆巩固前向传播
        
        Args:
            memory_states: 历史记忆状态列表
            current_input: 当前输入
            
        Returns:
            巩固后的记忆、巩固概率、稳定性分数
        """
        batch_size = current_input.shape[0]
        
        # 计算记忆状态统计
        if memory_states:
            memory_tensor = torch.stack(memory_states, dim=1)  # [batch_size, num_memories, hidden_dim]
            memory_mean = memory_tensor.mean(dim=1)  # [batch_size, hidden_dim]
        else:
            memory_mean = torch.zeros_like(current_input)
        
        # 巩固评估
        consolidation_input = torch.cat([current_input, memory_mean], dim=-1)
        consolidation_prob = self.consolidation_assessor(consolidation_input)
        
        # 稳定性检测
        stability_score = self.stability_detector(current_input)
        
        # 记忆强度预测
        strength_score = self.strength_predictor(current_input)
        
        # 应用巩固
        if memory_states and consolidation_prob.item() > self.stability_threshold:
            # 指数衰减巩固
            weight = torch.exp(-consolidation_prob * self.consolidation_tau)
            consolidated_memory = weight * memory_mean + (1 - weight) * current_input
        else:
            consolidated_memory = current_input
        
        return consolidated_memory, consolidation_prob, stability_score
    
    def should_consolidate(self, stability_score: torch.Tensor) -> bool:
        """判断是否应该进行巩固"""
        return stability_score.item() > self.stability_threshold


class TemporalAlignmentModule(nn.Module):
    """
    完整的时间对齐模块
    集成时间记忆细胞、序列对齐和记忆巩固
    """
    
    def __init__(self,
                 hidden_dim: int,
                 num_memory_cells: int = 4,
                 max_time_diff: int = 10):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_memory_cells = num_memory_cells
        
        # 时间记忆细胞堆叠
        self.memory_cells = nn.ModuleList([
            TemporalMemoryCell(
                hidden_dim=hidden_dim,
                time_decay_rate=0.95 + 0.01 * i,
                consolidation_rate=0.1 + 0.02 * i
            ) for i in range(num_memory_cells)
        ])
        
        # 时间序列对齐
        self.time_alignment = TimeSeriesAlignment(
            hidden_dim=hidden_dim,
            max_time_diff=max_time_diff
        )
        
        # 记忆巩固
        self.memory_consolidation = MemoryConsolidation(
            hidden_dim=hidden_dim
        )
        
        # 记忆状态缓存
        self.memory_cache = []
        self.max_cache_size = 100
        
        # 时间编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        时间对齐模块前向传播
        
        Args:
            x: 输入序列 [batch_size, seq_len, hidden_dim]
            
        Returns:
            时间对齐后的序列
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # 生成时间编码
        time_indices = torch.arange(seq_len, dtype=torch.float32, device=x.device).unsqueeze(0)
        time_encoding = self.time_encoder(time_indices.unsqueeze(-1))  # [1, seq_len, hidden_dim]
        
        # 时间记忆处理
        temporal_outputs = []
        for cell in self.memory_cells:
            cell_output = []
            for t in range(seq_len):
                input_t = x[:, t]  # [batch_size, hidden_dim]
                time_enc_t = time_encoding[:, t]  # [hidden_dim]
                
                output_t = cell(input_t, time_enc_t)
                cell_output.append(output_t)
            
            temporal_outputs.append(torch.stack(cell_output, dim=1))  # [batch_size, seq_len, hidden_dim]
        
        # 合并多个时间记忆细胞的输出
        combined_temporal = torch.stack(temporal_outputs, dim=-1)  # [batch_size, seq_len, hidden_dim, num_cells]
        combined_temporal = combined_temporal.mean(dim=-1)  # 平均池化
        
        # 时间序列对齐
        aligned_output = self.time_alignment(combined_temporal)
        
        # 记忆巩固
        if len(self.memory_cache) > 0:
            memory_tensor = torch.stack(self.memory_cache[-self.max_cache_size:], dim=1)
            memory_mean = memory_tensor.mean(dim=1)
            
            consolidated, consolidation_prob, stability = self.memory_consolidation(
                self.memory_cache, aligned_output.mean(dim=1)
            )
            
            # 更新缓存
            self.update_memory_cache(consolidated)
        else:
            aligned_output = aligned_output
        
        return aligned_output
    
    def update_memory_cache(self, new_memory: torch.Tensor):
        """更新记忆缓存"""
        self.memory_cache.append(new_memory.detach().clone())
        
        # 限制缓存大小
        if len(self.memory_cache) > self.max_cache_size:
            self.memory_cache.pop(0)
    
    def reset_memory_cells(self):
        """重置所有记忆细胞"""
        for cell in self.memory_cells:
            cell.reset()
        self.memory_cache.clear()
    
    def get_temporal_statistics(self) -> Dict[str, Any]:
        """
        获取时间对齐统计信息
        """
        stats = {
            'num_memory_cells': len(self.memory_cells),
            'memory_cache_size': len(self.memory_cache),
            'max_cache_size': self.max_cache_size,
            'time_decay_rates': [cell.time_decay_rate for cell in self.memory_cells],
            'consolidation_rates': [cell.consolidation_rate for cell in self.memory_cells]
        }
        
        # 记忆活跃度统计
        if self.memory_cache:
            recent_memories = torch.stack(self.memory_cache[-10:])
            stats['memory_activity'] = torch.norm(recent_memories).item()
            stats['memory_variance'] = torch.var(recent_memories).item()
        
        return stats
    
    def align_to_reference(self, 
                          query_seq: torch.Tensor, 
                          reference_seq: torch.Tensor) -> torch.Tensor:
        """
        将查询序列对齐到参考序列
        
        Args:
            query_seq: 查询序列 [batch_size, seq_len, hidden_dim]
            reference_seq: 参考序列 [batch_size, ref_len, hidden_dim]
            
        Returns:
            对齐后的序列
        """
        aligned_output = self.time_alignment(query_seq, reference_seq)
        return aligned_output
    
    def compute_temporal_similarity(self,
                                   seq1: torch.Tensor,
                                   seq2: torch.Tensor) -> float:
        """
        计算时间序列相似性
        
        Args:
            seq1: 序列1
            seq2: 序列2
            
        Returns:
            相似性分数
        """
        # 时间对齐
        aligned1 = self.time_alignment(seq1)
        aligned2 = self.time_alignment(seq2)
        
        # 计算相似性
        similarity = F.cosine_similarity(
            aligned1.mean(dim=1), aligned2.mean(dim=1)
        )
        
        return similarity.item()