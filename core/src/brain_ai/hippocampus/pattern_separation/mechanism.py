#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模式分离机制
===========

实现海马体DG区域的模式分离功能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.spatial.distance import cosine

class DentateGyrusLayer(nn.Module):
    """齿状回层（简化实现）"""
    
    def __init__(self, input_size: int, hidden_size: int, sparsity: float = 0.1):
        """
        初始化齿状回层
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            sparsity: 稀疏性参数
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(0.2)
        )
        
        # 稀疏化层
        self.sparsify = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Softmax(dim=-1)
        )
        
        # 模式分离器
        self.pattern_separator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 编码
        encoded = self.encoder(x)
        
        # 稀疏化
        sparse = self.sparsify(encoded)
        
        # 模式分离
        separated = self.pattern_separator(sparse)
        
        # 应用稀疏性约束
        separated = self._apply_sparsity(separated)
        
        return separated
    
    def _apply_sparsity(self, x: torch.Tensor) -> torch.Tensor:
        """应用稀疏性约束"""
        # 只保留最活跃的神经元
        k = int(x.size(-1) * self.sparsity)
        if k < 1:
            return x
        
        # 获取top-k激活
        top_values, top_indices = torch.topk(x, k, dim=-1)
        
        # 创建稀疏输出
        sparse_output = torch.zeros_like(x)
        sparse_output.scatter_(-1, top_indices, top_values)
        
        return sparse_output

class PatternSeparationMechanism:
    """模式分离机制"""
    
    def __init__(self, input_size: int, hidden_size: int = 256, sparsity: float = 0.1):
        """
        初始化模式分离机制
        
        Args:
            input_size: 输入维度
            hidden_size: 隐藏层维度
            sparsity: 稀疏性参数
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建DG层
        self.dg_layer = DentateGyrusLayer(input_size, hidden_size, sparsity).to(self.device)
        
        # 存储已分离的模式
        self.separated_patterns = {}
        self.pattern_statistics = {}
        
    def separate_input(self, input_pattern: torch.Tensor) -> torch.Tensor:
        """
        分离输入模式
        
        Args:
            input_pattern: 输入模式
            
        Returns:
            分离后的模式
        """
        input_tensor = input_pattern.to(self.device)
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(0)  # 添加批次维度
        
        with torch.no_grad():
            separated = self.dg_layer(input_tensor)
        
        # 转换为numpy以便存储
        separated_np = separated.cpu().numpy()
        
        return separated_np[0] if separated_np.shape[0] == 1 else separated_np
    
    def separate_similar_patterns(self, patterns: List[torch.Tensor]) -> List[np.ndarray]:
        """
        分离相似模式
        
        Args:
            patterns: 相似模式列表
            
        Returns:
            分离后的模式列表
        """
        separated_patterns = []
        
        for i, pattern in enumerate(patterns):
            separated = self.separate_input(pattern)
            separated_patterns.append(separated)
            
            # 计算模式统计
            pattern_id = f"pattern_{i}"
            self._update_pattern_statistics(pattern_id, separated)
        
        return separated_patterns
    
    def _update_pattern_statistics(self, pattern_id: str, pattern: np.ndarray):
        """更新模式统计信息"""
        if pattern_id not in self.pattern_statistics:
            self.pattern_statistics[pattern_id] = {
                'activations': [],
                'sparsity_levels': [],
                'pattern_strengths': []
            }
        
        stats = self.pattern_statistics[pattern_id]
        
        # 计算激活度
        activations = np.sum(pattern != 0)
        stats['activations'].append(activations)
        
        # 计算稀疏性水平
        sparsity = 1.0 - (activations / len(pattern))
        stats['sparsity_levels'].append(sparsity)
        
        # 计算模式强度
        strength = np.linalg.norm(pattern)
        stats['pattern_strengths'].append(strength)
    
    def compute_separation_quality(self, original_patterns: List[torch.Tensor],
                                 separated_patterns: List[np.ndarray]) -> Dict[str, float]:
        """
        计算模式分离质量
        
        Args:
            original_patterns: 原始模式
            separated_patterns: 分离后的模式
            
        Returns:
            分离质量指标
        """
        if len(original_patterns) != len(separated_patterns):
            raise ValueError("模式数量不匹配")
        
        qualities = []
        
        for orig, sep in zip(original_patterns, separated_patterns):
            # 计算余弦相似度
            orig_np = orig.cpu().numpy().flatten()
            sep_np = sep.flatten()
            
            if np.linalg.norm(orig_np) > 0 and np.linalg.norm(sep_np) > 0:
                similarity = 1 - cosine(orig_np, sep_np)
                separation_quality = 1 - similarity  # 相似度越低，分离质量越高
                qualities.append(separation_quality)
        
        if not qualities:
            return {'average_separation': 0.0}
        
        return {
            'average_separation': np.mean(qualities),
            'max_separation': np.max(qualities),
            'min_separation': np.min(qualities),
            'separation_std': np.std(qualities)
        }
    
    def compute_clustering_coefficient(self, patterns: List[np.ndarray]) -> float:
        """
        计算模式聚类系数
        
        Args:
            patterns: 模式列表
            
        Returns:
            聚类系数
        """
        if len(patterns) < 2:
            return 0.0
        
        # 计算所有模式对之间的距离
        distances = []
        for i in range(len(patterns)):
            for j in range(i+1, len(patterns)):
                dist = np.linalg.norm(patterns[i] - patterns[j])
                distances.append(dist)
        
        # 聚类系数 = 平均距离的倒数
        if distances:
            average_distance = np.mean(distances)
            clustering_coefficient = 1.0 / (1.0 + average_distance)
        else:
            clustering_coefficient = 0.0
        
        return clustering_coefficient
    
    def find_similar_patterns(self, query_pattern: np.ndarray, 
                            stored_patterns: Optional[Dict[str, np.ndarray]] = None,
                            threshold: float = 0.8) -> List[Tuple[str, float]]:
        """
        查找相似的已分离模式
        
        Args:
            query_pattern: 查询模式
            stored_patterns: 存储的模式
            threshold: 相似度阈值
            
        Returns:
            (模式ID, 相似度)列表
        """
        if stored_patterns is None:
            stored_patterns = self.separated_patterns
        
        similarities = []
        
        for pattern_id, stored_pattern in stored_patterns.items():
            similarity = self._compute_pattern_similarity(query_pattern, stored_pattern)
            if similarity > threshold:
                similarities.append((pattern_id, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def _compute_pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """计算两个模式的相似度"""
        if pattern1.shape != pattern2.shape:
            return 0.0
        
        # 使用余弦相似度
        if np.linalg.norm(pattern1) > 0 and np.linalg.norm(pattern2) > 0:
            similarity = 1 - cosine(pattern1.flatten(), pattern2.flatten())
        else:
            similarity = 0.0
        
        return similarity
    
    def store_separated_pattern(self, pattern_id: str, pattern: np.ndarray):
        """存储分离后的模式"""
        self.separated_patterns[pattern_id] = pattern.copy()
        
        # 更新统计信息
        self._update_pattern_statistics(pattern_id, pattern)
    
    def get_separation_statistics(self) -> Dict[str, Any]:
        """获取分离统计信息"""
        stats = {
            'total_patterns': len(self.separated_patterns),
            'average_sparsity': 0.0,
            'average_activation': 0.0,
            'pattern_diversity': 0.0
        }
        
        if not self.separated_patterns:
            return stats
        
        # 计算平均稀疏性和激活度
        all_sparsities = []
        all_activations = []
        
        for pattern_id, pattern in self.separated_patterns.items():
            if pattern_id in self.pattern_statistics:
                stat_data = self.pattern_statistics[pattern_id]
                if stat_data['sparsity_levels']:
                    all_sparsities.extend(stat_data['sparsity_levels'])
                if stat_data['activations']:
                    all_activations.extend(stat_data['activations'])
        
        if all_sparsities:
            stats['average_sparsity'] = np.mean(all_sparsities)
        if all_activations:
            stats['average_activation'] = np.mean(all_activations)
        
        # 计算模式多样性（平均成对距离）
        if len(self.separated_patterns) > 1:
            pattern_list = list(self.separated_patterns.values())
            diversity = self.compute_clustering_coefficient(pattern_list)
            stats['pattern_diversity'] = 1.0 - diversity  # 转换为多样性度量
        
        return stats
    
    def reset_patterns(self):
        """重置存储的模式"""
        self.separated_patterns.clear()
        self.pattern_statistics.clear()

# 工厂函数
def create_pattern_separator(input_size: int, 
                           hidden_size: int = 256,
                           sparsity: float = 0.1) -> PatternSeparationMechanism:
    """
    创建模式分离器
    
    Args:
        input_size: 输入维度
        hidden_size: 隐藏层维度
        sparsity: 稀疏性参数
        
    Returns:
        模式分离机制实例
    """
    return PatternSeparationMechanism(input_size, hidden_size, sparsity)

if __name__ == "__main__":
    # 测试模式分离机制
    separator = create_pattern_separator(input_size=100, hidden_size=64, sparsity=0.2)
    
    # 创建相似输入模式
    base_pattern = torch.randn(100)
    similar_patterns = []
    
    for i in range(5):
        noise = torch.randn(100) * 0.1  # 添加噪声
        pattern = base_pattern + noise
        similar_patterns.append(pattern)
    
    # 分离模式
    print("分离相似模式...")
    separated = separator.separate_similar_patterns(similar_patterns)
    
    # 存储分离的模式
    for i, pattern in enumerate(separated):
        separator.store_separated_pattern(f"separated_{i}", pattern)
    
    # 计算分离质量
    quality = separator.compute_separation_quality(similar_patterns, separated)
    print(f"分离质量: {quality}")
    
    # 获取统计信息
    stats = separator.get_separation_statistics()
    print(f"分离统计: {stats}")
    
    # 测试相似模式查找
    query = separated[0] + np.random.normal(0, 0.1, separated[0].shape)
    similar = separator.find_similar_patterns(query, threshold=0.7)
    print(f"找到 {len(similar)} 个相似模式")