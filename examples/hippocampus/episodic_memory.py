"""
情景记忆存储和检索系统
基于海马体CA1区的情景记忆处理机制
实现时间序列记忆的存储、检索和巩固
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
from collections import deque


class EpisodicMemoryCell(nn.Module):
    """
    情景记忆细胞
    模拟CA1区单个神经元的情景记忆存储
    """
    
    def __init__(self,
                 content_dim: int,
                 temporal_dim: int,
                 context_dim: int,
                 capacity: int = 100):
        super().__init__()
        
        self.content_dim = content_dim
        self.temporal_dim = temporal_dim
        self.context_dim = context_dim
        self.capacity = capacity
        
        # 记忆内容存储
        self.memory_content = nn.Parameter(
            torch.randn(capacity, content_dim) * 0.1
        )
        
        # 时间戳存储
        self.memory_timestamps = nn.Parameter(
            torch.randn(capacity, temporal_dim) * 0.1
        )
        
        # 上下文存储
        self.memory_contexts = nn.Parameter(
            torch.randn(capacity, context_dim) * 0.1
        )
        
        # 记忆强度（重要性权重）
        self.memory_strengths = nn.Parameter(
            torch.ones(capacity) * 0.5
        )
        
        # 记忆新鲜度（时间衰减）
        self.memory_freshness = nn.Parameter(
            torch.ones(capacity)
        )
        
        # 当前记忆数量
        self.current_size = 0
        self.write_pointer = 0
        
        # 时间编码器
        self.time_encoder = nn.Sequential(
            nn.Linear(1, temporal_dim // 2),
            nn.ReLU(),
            nn.Linear(temporal_dim // 2, temporal_dim),
            nn.Tanh()
        )
        
        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        
        # 记忆强度预测器
        self.strength_predictor = nn.Sequential(
            nn.Linear(content_dim + temporal_dim + context_dim, content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, 1),
            nn.Sigmoid()
        )
        
        # 相似性计算器
        self.similarity_calculator = nn.Sequential(
            nn.Linear(content_dim * 2, content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, 1),
            nn.Sigmoid()
        )
        
    def store_episode(self,
                     content: torch.Tensor,
                     timestamp: Union[float, torch.Tensor],
                     context: torch.Tensor,
                     importance: Optional[torch.Tensor] = None) -> int:
        """
        存储情景记忆
        
        Args:
            content: 记忆内容 [batch_size, content_dim]
            timestamp: 时间戳
            context: 上下文信息 [batch_size, context_dim]
            importance: 重要性权重
            
        Returns:
            存储位置
        """
        batch_size = content.shape[0]
        
        # 编码时间戳
        if isinstance(timestamp, (int, float)):
            timestamp_tensor = torch.tensor([[timestamp]], dtype=torch.float32)
        else:
            timestamp_tensor = timestamp
            
        encoded_timestamp = self.time_encoder(timestamp_tensor)  # [1, temporal_dim]
        
        # 编码上下文
        encoded_context = self.context_encoder(context)  # [batch_size, context_dim]
        
        # 选择存储位置
        if self.current_size < self.capacity:
            storage_positions = torch.arange(self.current_size, 
                                           self.current_size + batch_size,
                                           device=content.device)
            self.current_size += batch_size
        else:
            # LRU策略：替换最不重要的记忆
            importance_scores = self.memory_strengths * self.memory_freshness
            _, storage_positions = torch.topk(importance_scores, k=batch_size, largest=False)
        
        # 存储记忆
        for i in range(batch_size):
            pos = storage_positions[i].item() % self.capacity
            
            # 存储内容
            if i == 0:
                self.memory_content[pos] = content[i]
                self.memory_timestamps[pos] = encoded_timestamp.squeeze(0)
                self.memory_contexts[pos] = encoded_context[i]
            
            # 计算记忆强度
            if importance is None:
                combined_input = torch.cat([content[i], encoded_timestamp.squeeze(0), encoded_context[i]])
                predicted_strength = self.strength_predictor(combined_input.unsqueeze(0))
                memory_strength = predicted_strength.item()
            else:
                memory_strength = importance[i].item()
            
            self.memory_strengths[pos] = memory_strength
            
            # 初始化新鲜度
            self.memory_freshness[pos] = 1.0
        
        return storage_positions.sum().item() // batch_size
    
    def retrieve_episodes(self,
                         query_content: torch.Tensor,
                         query_context: torch.Tensor,
                         time_window: Optional[Tuple[float, float]] = None,
                         top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        检索情景记忆
        
        Args:
            query_content: 查询内容
            query_context: 查询上下文
            time_window: 时间窗口限制
            top_k: 返回top-k结果
            
        Returns:
            检索到的记忆、相似度分数、时间权重
        """
        if self.current_size == 0:
            # 空记忆库
            empty_content = torch.zeros(top_k, self.content_dim, device=query_content.device)
            empty_scores = torch.zeros(top_k, device=query_content.device)
            empty_time_weights = torch.ones(top_k, device=query_content.device)
            return empty_content, empty_scores, empty_time_weights
        
        batch_size = query_content.shape[0]
        active_content = self.memory_content[:self.current_size]
        active_contexts = self.memory_contexts[:self.current_size]
        active_timestamps = self.memory_timestamps[:self.current_size]
        active_strengths = self.memory_strengths[:self.current_size]
        active_freshness = self.memory_freshness[:self.current_size]
        
        # 计算内容相似性
        content_similarities = []
        for i in range(batch_size):
            query = query_content[i]
            similarities = []
            
            for j in range(self.current_size):
                memory = active_content[j]
                
                # 相似性计算
                sim_input = torch.cat([query, memory])
                similarity = self.similarity_calculator(sim_input.unsqueeze(0))
                similarities.append(similarity.item())
            
            content_similarities.append(similarities)
        
        content_similarities = torch.tensor(content_similarities)  # [batch_size, current_size]
        
        # 应用时间窗口过滤
        if time_window is not None:
            start_time, end_time = time_window
            # 简化的时间过滤：基于时间戳的方差
            time_mask = torch.ones(self.current_size, device=query_content.device)
            # 这里可以加入更复杂的时间过滤逻辑
        else:
            time_mask = torch.ones(self.current_size, device=query_content.device)
        
        # 计算时间权重（新鲜度）
        time_weights = active_freshness * time_mask
        
        # 计算上下文相似性
        context_similarities = F.cosine_similarity(
            query_context.unsqueeze(1), 
            active_contexts.unsqueeze(0), 
            dim=-1
        )
        
        # 综合相似性（内容+上下文+时间）
        final_similarities = (
            0.5 * content_similarities + 
            0.3 * context_similarities + 
            0.2 * time_weights.unsqueeze(0)
        )
        
        # 获取top-k结果
        top_similarities, top_indices = torch.topk(final_similarities, k=min(top_k, self.current_size), dim=-1)
        
        # 检索记忆
        retrieved_content = active_content[top_indices]  # [batch_size, top_k, content_dim]
        retrieved_strengths = active_strengths[top_indices]  # [batch_size, top_k]
        retrieved_time_weights = time_weights[top_indices]  # [batch_size, top_k]
        
        return retrieved_content, top_similarities, retrieved_time_weights
    
    def update_memory_freshness(self, decay_rate: float = 0.99):
        """更新记忆新鲜度（时间衰减）"""
        self.memory_freshness.data *= decay_rate
    
    def consolidate_memory(self, consolidation_threshold: float = 0.8):
        """记忆巩固（长期存储）"""
        # 找出需要巩固的记忆
        consolidation_candidates = self.memory_strengths > consolidation_threshold
        
        if consolidation_candidates.any():
            # 加强重要记忆
            self.memory_strengths[consolidation_candidates] = torch.min(
                self.memory_strengths[consolidation_candidates] + 0.1,
                torch.ones_like(self.memory_strengths[consolidation_candidates])
            )
            
            # 减少弱记忆的新鲜度
            weak_memories = ~consolidation_candidates
            self.memory_freshness[weak_memories] *= 0.95
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        if self.current_size == 0:
            return {
                'current_size': 0,
                'capacity_utilization': 0.0,
                'avg_strength': 0.0,
                'avg_freshness': 0.0,
                'strongest_memory': 0.0,
                'memory_diversity': 0.0
            }
        
        active_strengths = self.memory_strengths[:self.current_size]
        active_freshness = self.memory_freshness[:self.current_size]
        
        # 计算记忆多样性
        content_variance = torch.var(self.memory_content[:self.current_size], dim=0).mean().item()
        
        return {
            'current_size': self.current_size,
            'capacity_utilization': self.current_size / self.capacity,
            'avg_strength': active_strengths.mean().item(),
            'avg_freshness': active_freshness.mean().item(),
            'strongest_memory': active_strengths.max().item(),
            'weakest_memory': active_strengths.min().item(),
            'freshest_memory': active_freshness.max().item(),
            'memory_diversity': content_variance,
            'strength_variance': active_strengths.var().item()
        }


class EpisodicMemorySystem(nn.Module):
    """
    完整的情景记忆系统
    集成多个情景记忆细胞的层次化系统
    """
    
    def __init__(self,
                 content_dim: int,
                 temporal_dim: int,
                 context_dim: int,
                 num_cells: int = 8,
                 capacity_per_cell: int = 100):
        super().__init__()
        
        self.content_dim = content_dim
        self.temporal_dim = temporal_dim
        self.context_dim = context_dim
        self.num_cells = num_cells
        
        # 创建多个情景记忆细胞
        self.memory_cells = nn.ModuleList([
            EpisodicMemoryCell(
                content_dim=content_dim,
                temporal_dim=temporal_dim,
                context_dim=context_dim,
                capacity=capacity_per_cell
            ) for _ in range(num_cells)
        ])
        
        # 内容编码器
        self.content_encoder = nn.Sequential(
            nn.Linear(content_dim, content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, content_dim)
        )
        
        # 上下文编码器
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        
        # 时间编码器
        self.global_time_encoder = nn.Sequential(
            nn.Linear(1, temporal_dim),
            nn.Tanh()
        )
        
        # 记忆融合器
        self.memory_fusion = nn.Sequential(
            nn.Linear(content_dim * num_cells, content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, content_dim),
            nn.LayerNorm(content_dim)
        )
        
        # 注意力权重生成器
        self.attention_weights = nn.Sequential(
            nn.Linear(content_dim + context_dim, num_cells),
            nn.Softmax(dim=-1)
        )
        
        # 记忆重要性评估器
        self.importance_evaluator = nn.Sequential(
            nn.Linear(content_dim + temporal_dim + context_dim, content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, 1),
            nn.Sigmoid()
        )
        
        # 情景记忆检索网络
        self.retrieval_network = nn.Sequential(
            nn.Linear(content_dim * 2, content_dim),
            nn.ReLU(),
            nn.Linear(content_dim, 1),
            nn.Sigmoid()
        )
        
        # 时间序列分析器
        self.temporal_analyzer = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim // 2),
            nn.ReLU(),
            nn.Linear(temporal_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 全局记忆管理器（普通属性，不是Module）
        self.global_memory_manager = {
            'short_term_buffer': deque(maxlen=1000),
            'long_term_storage': {},
            'memory_index': {},
            'temporal_sequences': {}
        }
        
    def store_episode(self,
                     content: torch.Tensor,
                     timestamp: float,
                     context: torch.Tensor,
                     episode_id: Optional[str] = None) -> Dict[str, Any]:
        """
        存储情景记忆到系统
        
        Args:
            content: 记忆内容
            timestamp: 时间戳
            context: 上下文信息
            episode_id: 记忆片段ID
            
        Returns:
            存储结果
        """
        batch_size = content.shape[0]
        
        # 编码输入
        encoded_content = self.content_encoder(content)
        encoded_context = self.context_encoder(context)
        encoded_timestamp = self.global_time_encoder(
            torch.tensor([[timestamp]], dtype=torch.float32, device=content.device)
        )
        
        # 评估记忆重要性
        combined_input = torch.cat([content, encoded_timestamp.squeeze(0), context], dim=-1)
        importance_scores = self.importance_evaluator(combined_input)
        
        # 在多个记忆细胞中存储
        storage_results = []
        for i, cell in enumerate(self.memory_cells):
            # 选择部分记忆存储到此细胞
            cell_importance = importance_scores[:, i % num_cells] if i < batch_size else importance_scores.mean(dim=1, keepdim=True)
            
            storage_pos = cell.store_episode(
                content=content,
                timestamp=timestamp,
                context=context,
                importance=cell_importance
            )
            
            storage_results.append({
                'cell_id': i,
                'storage_position': storage_pos,
                'importance_score': cell_importance.mean().item()
            })
        
        # 更新全局记忆管理器
        global_episode_id = episode_id or f"episode_{len(self.global_memory_manager['short_term_buffer'])}"
        
        episode_info = {
            'id': global_episode_id,
            'timestamp': timestamp,
            'content_shape': content.shape,
            'context_shape': context.shape,
            'importance': importance_scores.mean().item(),
            'storage_cells': storage_results
        }
        
        self.global_memory_manager['short_term_buffer'].append(episode_info)
        
        return {
            'storage_results': storage_results,
            'global_episode_id': global_episode_id,
            'importance_score': importance_scores.mean().item(),
            'total_cells_used': len(storage_results)
        }
    
    def retrieve_episodes(self,
                         query_content: torch.Tensor,
                         query_context: torch.Tensor,
                         query_timestamp: Optional[float] = None,
                         time_window: Optional[Tuple[float, float]] = None,
                         retrieval_type: str = 'temporal') -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        检索情景记忆
        
        Args:
            query_content: 查询内容
            query_context: 查询上下文
            query_timestamp: 查询时间戳
            time_window: 时间窗口
            retrieval_type: 检索类型 ('temporal', 'content', 'contextual', 'hybrid')
            
        Returns:
            检索结果、检索统计
        """
        batch_size = query_content.shape[0]
        
        # 从各个记忆细胞检索
        all_retrieved_content = []
        all_retrieval_scores = []
        all_time_weights = []
        
        for cell in self.memory_cells:
            retrieved_content, retrieval_scores, time_weights = cell.retrieve_episodes(
                query_content=query_content,
                query_context=query_context,
                time_window=time_window,
                top_k=5
            )
            
            all_retrieved_content.append(retrieved_content)
            all_retrieval_scores.append(retrieval_scores)
            all_time_weights.append(time_weights)
        
        # 融合检索结果
        if retrieval_type == 'temporal':
            fused_result = self._temporal_fusion(all_retrieved_content, all_time_weights)
        elif retrieval_type == 'content':
            fused_result = self._content_fusion(all_retrieved_content, all_retrieval_scores)
        elif retrieval_type == 'contextual':
            fused_result = self._contextual_fusion(all_retrieved_content, query_context)
        else:  # hybrid
            fused_result = self._hybrid_fusion(
                all_retrieved_content, 
                all_retrieval_scores, 
                all_time_weights,
                query_content,
                query_context
            )
        
        # 检索统计
        retrieval_stats = {
            'num_cells_searched': len(self.memory_cells),
            'retrieval_type': retrieval_type,
            'batch_size': batch_size,
            'avg_retrieval_score': torch.cat(all_retrieval_scores).mean().item(),
            'max_retrieval_score': torch.cat(all_retrieval_scores).max().item(),
            'query_timestamp': query_timestamp,
            'time_window': time_window
        }
        
        return fused_result, retrieval_stats
    
    def _temporal_fusion(self, 
                        retrieved_content: List[torch.Tensor], 
                        time_weights: List[torch.Tensor]) -> torch.Tensor:
        """时间权重融合"""
        batch_size = retrieved_content[0].shape[0]
        
        # 时间加权平均
        weighted_sum = torch.zeros_like(retrieved_content[0])
        weight_sum = torch.zeros(batch_size, 1, device=retrieved_content[0].device)
        
        for content, weights in zip(retrieved_content, time_weights):
            weighted_sum += content * weights.unsqueeze(-1)
            weight_sum += weights.unsqueeze(-1)
        
        temporal_fused = weighted_sum / (weight_sum + 1e-8)
        return temporal_fused
    
    def _content_fusion(self,
                       retrieved_content: List[torch.Tensor],
                       retrieval_scores: List[torch.Tensor]) -> torch.Tensor:
        """内容相似性融合"""
        # 加权平均融合
        weighted_sum = torch.zeros_like(retrieved_content[0])
        weight_sum = torch.zeros(retrieved_content[0].shape[0], 1, device=retrieved_content[0].device)
        
        for content, scores in zip(retrieved_content, retrieval_scores):
            weighted_sum += content * scores.unsqueeze(-1)
            weight_sum += scores.unsqueeze(-1)
        
        content_fused = weighted_sum / (weight_sum + 1e-8)
        return content_fused
    
    def _contextual_fusion(self,
                          retrieved_content: List[torch.Tensor],
                          query_context: torch.Tensor) -> torch.Tensor:
        """上下文融合"""
        batch_size = retrieved_content[0].shape[0]
        
        # 计算上下文注意力权重
        context_weights = []
        for content in retrieved_content:
            # 计算查询上下文与检索内容的相似性
            context_sim = F.cosine_similarity(
                query_context.unsqueeze(1), content, dim=-1
            )
            context_weights.append(context_sim)
        
        context_weights = torch.stack(context_weights, dim=-1)  # [batch_size, top_k, num_cells]
        context_weights = F.softmax(context_weights, dim=-1)
        
        # 加权融合
        fused = torch.zeros_like(retrieved_content[0])
        for i, content in enumerate(retrieved_content):
            weight = context_weights[:, :, i].unsqueeze(-1)
            fused += content * weight
        
        return fused
    
    def _hybrid_fusion(self,
                      retrieved_content: List[torch.Tensor],
                      retrieval_scores: List[torch.Tensor],
                      time_weights: List[torch.Tensor],
                      query_content: torch.Tensor,
                      query_context: torch.Tensor) -> torch.Tensor:
        """混合融合"""
        # 结合多种权重
        combined_weighted_sum = torch.zeros_like(retrieved_content[0])
        total_weights = torch.zeros(retrieved_content[0].shape[0], 1, device=retrieved_content[0].device)
        
        for content, scores, t_weights in zip(retrieved_content, retrieval_scores, time_weights):
            # 综合权重：内容相似性 + 时间权重 + 上下文权重
            context_sim = F.cosine_similarity(
                query_content.unsqueeze(1), content, dim=-1
            )
            
            combined_weight = (0.4 * scores + 0.3 * t_weights + 0.3 * context_sim).unsqueeze(-1)
            
            combined_weighted_sum += content * combined_weight
            total_weights += combined_weight
        
        hybrid_fused = combined_weighted_sum / (total_weights + 1e-8)
        return hybrid_fused
    
    def update_memory_system(self):
        """更新记忆系统（时间衰减、巩固等）"""
        update_stats = {
            'cells_updated': 0,
            'memories_consolidated': 0,
            'memories_forgotten': 0
        }
        
        for cell in self.memory_cells:
            # 更新新鲜度
            cell.update_memory_freshness()
            
            # 巩固重要记忆
            before_consolidation = cell.current_size
            cell.consolidate_memory()
            
            update_stats['cells_updated'] += 1
            update_stats['memories_consolidated'] += (cell.memory_strengths > 0.8).sum().item()
            update_stats['memories_forgotten'] += (cell.memory_strengths < 0.1).sum().item()
        
        return update_stats
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """分析时间模式"""
        pattern_analysis = {
            'total_episodes': len(self.global_memory_manager['short_term_buffer']),
            'temporal_distribution': {},
            'sequence_patterns': {},
            'memory_decay_curve': []
        }
        
        if self.global_memory_manager['short_term_buffer']:
            # 分析时间分布
            timestamps = [episode['timestamp'] for episode in self.global_memory_manager['short_term_buffer']]
            
            if len(timestamps) > 1:
                # 计算时间间隔分布
                time_intervals = np.diff(sorted(timestamps))
                pattern_analysis['temporal_distribution'] = {
                    'mean_interval': np.mean(time_intervals),
                    'std_interval': np.std(time_intervals),
                    'min_interval': np.min(time_intervals),
                    'max_interval': np.max(time_intervals)
                }
            
            # 分析记忆衰减曲线
            recent_episodes = list(self.global_memory_manager['short_term_buffer'])[-100:]
            for episode in recent_episodes:
                importance = episode.get('importance', 0.5)
                pattern_analysis['memory_decay_curve'].append(importance)
        
        return pattern_analysis
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        cell_stats = [cell.get_memory_statistics() for cell in self.memory_cells]
        
        system_stats = {
            'num_memory_cells': len(self.memory_cells),
            'total_capacity': sum(cell.capacity for cell in self.memory_cells),
            'total_current_size': sum(stats['current_size'] for stats in cell_stats),
            'average_utilization': np.mean([stats['capacity_utilization'] for stats in cell_stats]),
            'average_memory_strength': np.mean([stats['avg_strength'] for stats in cell_stats]),
            'memory_diversity': np.mean([stats['memory_diversity'] for stats in cell_stats]),
            'cell_statistics': cell_stats,
            'short_term_buffer_size': len(self.global_memory_manager['short_term_buffer'])
        }
        
        # 添加时间模式分析
        temporal_patterns = self.analyze_temporal_patterns()
        system_stats['temporal_patterns'] = temporal_patterns
        
        return system_stats
    
    def clear_memory_system(self):
        """清空记忆系统"""
        for cell in self.memory_cells:
            cell.current_size = 0
            cell.write_pointer = 0
            cell.memory_content = nn.Parameter(torch.randn_like(cell.memory_content) * 0.1)
            cell.memory_strengths = nn.Parameter(torch.ones_like(cell.memory_strengths) * 0.5)
            cell.memory_freshness = nn.Parameter(torch.ones_like(cell.memory_freshness))
        
        self.global_memory_manager['short_term_buffer'].clear()
        self.global_memory_manager['long_term_storage'].clear()
        self.global_memory_manager['memory_index'].clear()
        self.global_memory_manager['temporal_sequences'].clear()