"""
动态权重延迟比路由算法
基于多目标优化的智能路由决策
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math
from collections import deque


@dataclass
class PathMetrics:
    """路径指标"""
    latency: float
    bandwidth: float
    reliability: float
    cost: float
    energy: float
    
    @property
    def weight_delay_ratio(self) -> float:
        """计算权重延迟比"""
        # 综合权重 = 可靠性 * 带宽 / (延迟 * 成本 * 能耗)
        if self.latency <= 0:
            return float('inf')
        
        weight = self.reliability * self.bandwidth
        denominator = self.latency * self.cost * (1 + self.energy)
        return weight / denominator if denominator > 0 else 0
    
    @property
    def quality_score(self) -> float:
        """计算质量分数"""
        # 加权质量评分
        weights = {'latency': 0.3, 'bandwidth': 0.25, 'reliability': 0.25, 'cost': 0.1, 'energy': 0.1}
        
        # 归一化指标（越小越好的指标需要反转）
        norm_latency = 1.0 / (1.0 + self.latency)
        norm_bandwidth = min(1.0, self.bandwidth / 1000.0)  # 假设带宽上限1000
        norm_reliability = self.reliability
        norm_cost = 1.0 / (1.0 + self.cost)
        norm_energy = 1.0 / (1.0 + self.energy)
        
        score = (weights['latency'] * norm_latency +
                weights['bandwidth'] * norm_bandwidth +
                weights['reliability'] * norm_reliability +
                weights['cost'] * norm_cost +
                weights['energy'] * norm_energy)
        
        return score


class WeightLearningNetwork(nn.Module):
    """权重学习网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 5)  # 5个权重的输出
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        weights = torch.softmax(self.fc3(x), dim=-1)  # 权重归一化
        return weights


class DynamicWeightRouter:
    """动态权重延迟比路由器"""
    
    def __init__(self,
                 num_paths: int = 8,
                 state_dim: int = 32,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        
        self.num_paths = num_paths
        self.state_dim = state_dim
        self.device = device
        
        # 路径信息
        self.paths = self._initialize_paths()
        
        # 权重学习网络
        self.weight_network = WeightLearningNetwork(state_dim).to(device)
        self.optimizer = optim.Adam(self.weight_network.parameters(), lr=learning_rate)
        
        # 历史数据
        self.path_history = deque(maxlen=1000)
        self.performance_history = deque(maxlen=1000)
        
        # 自适应权重
        self.adaptive_weights = {
            'latency': 0.3,
            'bandwidth': 0.25,
            'reliability': 0.25,
            'cost': 0.1,
            'energy': 0.1
        }
        
        # 统计信息
        self.total_requests = 0
        self.successful_routes = 0
        self.average_latency = 0.0
        self.average_cost = 0.0
        
    def _initialize_paths(self) -> List[PathMetrics]:
        """初始化路径"""
        paths = []
        for i in range(self.num_paths):
            # 随机初始化路径参数
            latency = np.random.exponential(0.5)  # 指数分布的延迟
            bandwidth = np.random.uniform(100, 1000)  # 带宽
            reliability = np.random.uniform(0.85, 0.99)  # 可靠性
            cost = np.random.uniform(0.1, 2.0)  # 成本
            energy = np.random.uniform(0.5, 3.0)  # 能耗
            
            paths.append(PathMetrics(
                latency=latency,
                bandwidth=bandwidth,
                reliability=reliability,
                cost=cost,
                energy=energy
            ))
        
        return paths
    
    def select_path(self, 
                   traffic_pattern: str = 'normal',
                   quality_requirements: Dict[str, float] = None) -> int:
        """选择最优路径"""
        if quality_requirements is None:
            quality_requirements = {}
        
        # 更新自适应权重
        self._update_adaptive_weights(quality_requirements)
        
        # 基于权重的路径评分
        path_scores = []
        for i, path in enumerate(self.paths):
            score = self._calculate_weighted_score(path)
            path_scores.append((i, score))
        
        # 选择评分最高的路径
        best_path_idx = max(path_scores, key=lambda x: x[1])[0]
        
        # 记录路由决策
        self._record_routing_decision(best_path_idx, traffic_pattern)
        
        self.total_requests += 1
        return best_path_idx
    
    def _calculate_weighted_score(self, path: PathMetrics) -> float:
        """计算加权评分"""
        # 使用当前权重计算评分
        weights = self.adaptive_weights
        
        # 计算各指标的贡献
        latency_contribution = weights['latency'] * (1.0 / (1.0 + path.latency))
        bandwidth_contribution = weights['bandwidth'] * min(1.0, path.bandwidth / 1000.0)
        reliability_contribution = weights['reliability'] * path.reliability
        cost_contribution = weights['cost'] * (1.0 / (1.0 + path.cost))
        energy_contribution = weights['energy'] * (1.0 / (1.0 + path.energy))
        
        # 考虑负载均衡
        load_balance_bonus = self._calculate_load_balance_bonus()
        
        total_score = (latency_contribution + bandwidth_contribution + 
                      reliability_contribution + cost_contribution + 
                      energy_contribution) + load_balance_bonus
        
        return total_score
    
    def _calculate_load_balance_bonus(self) -> float:
        """计算负载均衡奖励"""
        if not self.path_history:
            return 0.0
        
        # 统计路径使用频率
        path_usage = {}
        for path_idx, _ in self.path_history[-50:]:  # 最近50次使用
            path_usage[path_idx] = path_usage.get(path_idx, 0) + 1
        
        if len(path_usage) < 2:
            return 0.0
        
        # 计算均衡度
        max_usage = max(path_usage.values())
        min_usage = min(path_usage.values())
        balance_factor = 1.0 - (max_usage - min_usage) / max(len(self.paths), 1)
        
        return balance_factor * 0.1  # 负载均衡权重
    
    def _update_adaptive_weights(self, requirements: Dict[str, float]):
        """根据需求更新权重"""
        if 'low_latency' in requirements:
            self.adaptive_weights['latency'] = min(0.6, self.adaptive_weights['latency'] + 0.1)
            self.adaptive_weights['cost'] = max(0.05, self.adaptive_weights['cost'] - 0.05)
        
        if 'high_bandwidth' in requirements:
            self.adaptive_weights['bandwidth'] = min(0.4, self.adaptive_weights['bandwidth'] + 0.1)
            self.adaptive_weights['energy'] = max(0.05, self.adaptive_weights['energy'] - 0.05)
        
        if 'high_reliability' in requirements:
            self.adaptive_weights['reliability'] = min(0.4, self.adaptive_weights['reliability'] + 0.1)
            self.adaptive_weights['latency'] = max(0.15, self.adaptive_weights['latency'] - 0.05)
        
        # 归一化权重
        total_weight = sum(self.adaptive_weights.values())
        if total_weight > 0:
            for key in self.adaptive_weights:
                self.adaptive_weights[key] /= total_weight
    
    def _record_routing_decision(self, path_idx: int, traffic_pattern: str):
        """记录路由决策"""
        decision = {
            'path_idx': path_idx,
            'traffic_pattern': traffic_pattern,
            'timestamp': len(self.path_history),
            'path_metrics': self.paths[path_idx]
        }
        self.path_history.append(decision)
    
    def update_path_metrics(self, path_idx: int, actual_latency: float, 
                           actual_cost: float, success: bool):
        """更新路径指标"""
        if path_idx >= len(self.paths):
            return
        
        # 使用指数移动平均更新指标
        alpha = 0.1
        
        self.paths[path_idx].latency = (alpha * actual_latency + 
                                       (1 - alpha) * self.paths[path_idx].latency)
        self.paths[path_idx].cost = (alpha * actual_cost + 
                                    (1 - alpha) * self.paths[path_idx].cost)
        
        # 成功路由增加可靠性
        if success:
            self.paths[path_idx].reliability = min(0.99, 
                self.paths[path_idx].reliability + 0.001)
        else:
            self.paths[path_idx].reliability = max(0.5, 
                self.paths[path_idx].reliability - 0.01)
        
        # 更新统计信息
        if success:
            self.successful_routes += 1
        
        # 更新平均延迟和成本
        total = self.total_requests
        self.average_latency = (self.average_latency * (total - 1) + actual_latency) / total
        self.average_cost = (self.average_cost * (total - 1) + actual_cost) / total
    
    def learn_weights_from_history(self, batch_size: int = 32):
        """从历史数据学习权重"""
        if len(self.path_history) < batch_size:
            return
        
        # 采样历史数据
        recent_history = list(self.path_history)[-batch_size:]
        
        # 准备训练数据
        # 这里简化实现，实际中需要更复杂的状态表示
        states = torch.randn(batch_size, self.state_dim, device=self.device)
        
        # 目标权重（基于历史性能）
        target_weights = self._calculate_target_weights(recent_history)
        
        # 训练权重网络
        predicted_weights = self.weight_network(states)
        loss = nn.MSELoss()(predicted_weights, target_weights)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _calculate_target_weights(self, history: List) -> torch.Tensor:
        """计算目标权重"""
        # 基于历史性能计算最优权重
        weights = torch.ones(5, device=self.device) / 5  # 默认均匀权重
        
        if not history:
            return weights
        
        # 统计成功路径的性能
        successful_paths = [h for h in history if hasattr(h, 'success') and h['success']]
        if not successful_paths:
            return weights
        
        # 分析成功路径的特征
        latencies = [h['path_metrics'].latency for h in successful_paths]
        bandwidths = [h['path_metrics'].bandwidth for h in successful_paths]
        reliabilities = [h['path_metrics'].reliability for h in successful_paths]
        
        # 计算权重
        if latencies:
            weights[0] = 1.0 / (np.mean(latencies) + 0.1)  # 延迟权重
        if bandwidths:
            weights[1] = np.mean(bandwidths) / 1000.0  # 带宽权重
        if reliabilities:
            weights[2] = np.mean(reliabilities)  # 可靠性权重
        
        # 归一化
        weights = weights / weights.sum()
        return weights.unsqueeze(0)
    
    def get_path_analysis(self, path_idx: int) -> Dict:
        """获取路径分析"""
        if path_idx >= len(self.paths):
            return {}
        
        path = self.paths[path_idx]
        
        # 计算使用频率
        usage_count = sum(1 for h in self.path_history if h['path_idx'] == path_idx)
        usage_frequency = usage_count / max(len(self.path_history), 1)
        
        # 计算相对评分
        other_scores = [self._calculate_weighted_score(p) for i, p in enumerate(self.paths) 
                       if i != path_idx]
        relative_score = path.weight_delay_ratio / (np.mean(other_scores) + 0.001)
        
        return {
            'path_id': path_idx,
            'weight_delay_ratio': path.weight_delay_ratio,
            'quality_score': path.quality_score,
            'usage_frequency': usage_frequency,
            'relative_score': relative_score,
            'metrics': {
                'latency': path.latency,
                'bandwidth': path.bandwidth,
                'reliability': path.reliability,
                'cost': path.cost,
                'energy': path.energy
            }
        }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_requests': self.total_requests,
            'successful_routes': self.successful_routes,
            'success_rate': self.successful_routes / max(self.total_requests, 1),
            'average_latency': self.average_latency,
            'average_cost': self.average_cost,
            'adaptive_weights': self.adaptive_weights.copy(),
            'path_count': self.num_paths,
            'history_size': len(self.path_history)
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'weight_network_state_dict': self.weight_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'adaptive_weights': self.adaptive_weights,
            'paths': self.paths,
            'statistics': self.get_statistics()
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.weight_network.load_state_dict(checkpoint['weight_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.adaptive_weights = checkpoint['adaptive_weights']
        self.paths = checkpoint['paths']