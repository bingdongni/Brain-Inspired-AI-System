"""
智能路径选择器
基于多目标优化的智能路径选择算法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
from scipy.optimize import minimize
import networkx as nx


@dataclass
class PathNode:
    """路径节点"""
    id: str
    x: float
    y: float
    energy_consumption: float
    processing_capacity: float
    current_load: float
    connectivity: List[str] = field(default_factory=list)


@dataclass
class PathEdge:
    """路径边"""
    source: str
    target: str
    distance: float
    energy_cost: float
    latency: float
    bandwidth: float
    reliability: float


class PathScoringNetwork(nn.Module):
    """路径评分网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # 单个分数输出
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        score = torch.sigmoid(self.fc3(x))
        return score


class MultiObjectiveOptimizer:
    """多目标优化器"""
    
    def __init__(self, objectives: List[str] = None):
        self.objectives = objectives or ['energy', 'latency', 'reliability', 'cost', 'throughput']
        self.weights = {obj: 1.0 / len(self.objectives) for obj in self.objectives}
        
    def optimize_weights(self, performance_history: List[Dict]):
        """基于历史性能优化权重"""
        if not performance_history:
            return
        
        # 简化的权重优化：基于性能方差
        objective_performance = {obj: [] for obj in self.objectives}
        
        for entry in performance_history[-100:]:  # 最近100条记录
            for obj in self.objectives:
                if obj in entry:
                    objective_performance[obj].append(entry[obj])
        
        # 计算权重（方差小的目标给予更高权重）
        for obj in self.objectives:
            if objective_performance[obj]:
                variance = np.var(objective_performance[obj])
                self.weights[obj] = 1.0 / (1.0 + variance)
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for obj in self.objectives:
                self.weights[obj] /= total_weight


class IntelligentPathSelector:
    """智能路径选择器"""
    
    def __init__(self,
                 num_nodes: int = 20,
                 num_objectives: int = 5,
                 input_dim: int = 32,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        
        self.num_nodes = num_nodes
        self.num_objectives = num_objectives
        self.device = device
        
        # 网络图
        self.graph = nx.Graph()
        self.nodes = {}
        self.edges = {}
        
        # 初始化网络拓扑
        self._initialize_network_topology()
        
        # 路径评分网络
        self.scoring_network = PathScoringNetwork(input_dim).to(device)
        self.optimizer = optim.Adam(self.scoring_network.parameters(), lr=learning_rate)
        
        # 多目标优化器
        self.multi_objective_optimizer = MultiObjectiveOptimizer()
        
        # 路径历史
        self.path_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=5000)
        
        # 统计信息
        self.total_paths_selected = 0
        self.successful_paths = 0
        self.average_energy_consumption = 0.0
        self.average_latency = 0.0
        
        # 动态参数
        self.load_balancing_factor = 0.1
        self.reliability_threshold = 0.8
        self.energy_efficiency_threshold = 0.5
        
    def _initialize_network_topology(self):
        """初始化网络拓扑"""
        # 创建节点
        for i in range(self.num_nodes):
            node_id = f"node_{i}"
            node = PathNode(
                id=node_id,
                x=np.random.uniform(0, 100),
                y=np.random.uniform(0, 100),
                energy_consumption=np.random.uniform(1.0, 5.0),
                processing_capacity=np.random.uniform(50, 150),
                current_load=np.random.uniform(0.1, 0.8)
            )
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node.__dict__)
        
        # 创建边（保证连通性）
        # 首先创建最小生成树
        positions = {node_id: (node.x, node.y) for node_id, node in self.nodes.items()}
        min_edges = list(nx.minimum_spanning_tree(self.graph, pos=positions).edges())
        
        # 添加额外的边以提高容错性
        additional_edges = []
        for _ in range(self.num_nodes):
            node1, node2 = np.random.choice(list(self.nodes.keys()), 2, replace=False)
            if (node1, node2) not in min_edges and (node2, node1) not in min_edges:
                additional_edges.append((node1, node2))
        
        all_edges = min_edges + additional_edges[:self.num_nodes//2]
        
        # 创建边
        for source, target in all_edges:
            edge = self._create_edge(source, target)
            self.edges[f"{source}_{target}"] = edge
            self.edges[f"{target}_{source}"] = edge  # 无向图
            
            # 更新节点的连接列表
            self.nodes[source].connectivity.append(target)
            self.nodes[target].connectivity.append(source)
            
            # 添加到图
            self.graph.add_edge(source, target, **edge.__dict__)
    
    def _create_edge(self, source: str, target: str) -> PathEdge:
        """创建路径边"""
        source_node = self.nodes[source]
        target_node = self.nodes[target]
        
        # 计算距离
        distance = np.sqrt((source_node.x - target_node.x)**2 + 
                          (source_node.y - target_node.y)**2)
        
        # 边的属性
        edge = PathEdge(
            source=source,
            target=target,
            distance=distance,
            energy_cost=distance * 0.1,  # 基于距离的能耗
            latency=distance / 100.0,  # 基于距离的延迟
            bandwidth=np.random.uniform(50, 200),  # 带宽
            reliability=np.random.uniform(0.85, 0.99)  # 可靠性
        )
        
        return edge
    
    def find_optimal_path(self, 
                         source: str, 
                         target: str, 
                         requirements: Dict[str, float] = None) -> Dict:
        """寻找最优路径"""
        if source not in self.nodes or target not in self.nodes:
            return {'error': 'Invalid source or target node'}
        
        if requirements is None:
            requirements = {}
        
        # 获取所有可能路径
        all_paths = self._find_all_paths(source, target)
        
        if not all_paths:
            return {'error': 'No path found between source and target'}
        
        # 评估每条路径
        path_scores = []
        for i, path in enumerate(all_paths):
            score_info = self._evaluate_path(path, requirements)
            path_scores.append({
                'path': path,
                'score': score_info['total_score'],
                'details': score_info
            })
        
        # 按评分排序
        path_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # 选择最佳路径
        best_path = path_scores[0]
        
        # 更新统计
        self.total_paths_selected += 1
        self.path_history.append({
            'source': source,
            'target': target,
            'selected_path': best_path['path'],
            'score': best_path['score'],
            'timestamp': len(self.path_history)
        })
        
        return {
            'selected_path': best_path['path'],
            'total_score': best_path['score'],
            'path_details': best_path['details'],
            'alternative_paths': [ps['path'] for ps in path_scores[1:3]]  # 提供备选路径
        }
    
    def _find_all_paths(self, source: str, target: str, max_depth: int = 6) -> List[List[str]]:
        """寻找所有可能路径（使用深度限制）"""
        def dfs(current: str, target: str, path: List[str], visited: set, depth: int):
            if depth > max_depth:
                return []
            
            if current == target:
                return [path + [current]]
            
            paths = []
            for neighbor in self.nodes[current].connectivity:
                if neighbor not in visited:
                    new_paths = dfs(neighbor, target, path + [current], visited | {neighbor}, depth + 1)
                    paths.extend(new_paths)
            
            return paths
        
        return dfs(source, target, [], {source}, 0)
    
    def _evaluate_path(self, path: List[str], requirements: Dict[str, float]) -> Dict:
        """评估路径性能"""
        # 计算路径指标
        total_energy = 0.0
        total_latency = 0.0
        total_distance = 0.0
        path_reliability = 1.0
        min_bandwidth = float('inf')
        avg_utilization = 0.0
        
        # 计算各边的属性
        for i in range(len(path) - 1):
            source, target = path[i], path[i + 1]
            edge = self.edges[f"{source}_{target}"]
            
            total_energy += edge.energy_cost
            total_latency += edge.latency
            total_distance += edge.distance
            path_reliability *= edge.reliability
            min_bandwidth = min(min_bandwidth, edge.bandwidth)
        
        # 计算节点负载
        for node_id in path:
            node = self.nodes[node_id]
            avg_utilization += node.current_load
        avg_utilization /= len(path)
        
        # 归一化指标
        energy_score = 1.0 / (1.0 + total_energy)
        latency_score = 1.0 / (1.0 + total_latency)
        reliability_score = path_reliability
        bandwidth_score = min(1.0, min_bandwidth / 200.0)  # 假设最大带宽200
        utilization_score = 1.0 - avg_utilization
        
        # 需求约束检查
        constraint_penalty = 0.0
        if 'max_energy' in requirements and total_energy > requirements['max_energy']:
            constraint_penalty += 0.3
        if 'max_latency' in requirements and total_latency > requirements['max_latency']:
            constraint_penalty += 0.3
        if 'min_reliability' in requirements and path_reliability < requirements['min_reliability']:
            constraint_penalty += 0.2
        
        # 多目标优化评分
        scores = {
            'energy': energy_score,
            'latency': latency_score,
            'reliability': reliability_score,
            'bandwidth': bandwidth_score,
            'utilization': utilization_score
        }
        
        # 使用优化权重计算总分
        weights = self.multi_objective_optimizer.weights
        total_score = sum(scores[obj] * weights.get(obj, 0) for obj in scores.keys())
        
        # 应用负载均衡
        load_balance_bonus = self._calculate_load_balance_bonus(path)
        total_score += load_balance_bonus
        
        # 应用约束惩罚
        total_score -= constraint_penalty
        
        # 确保分数在[0,1]范围内
        total_score = max(0, min(1, total_score))
        
        return {
            'total_score': total_score,
            'energy_score': energy_score,
            'latency_score': latency_score,
            'reliability_score': reliability_score,
            'bandwidth_score': bandwidth_score,
            'utilization_score': utilization_score,
            'constraint_penalty': constraint_penalty,
            'load_balance_bonus': load_balance_bonus,
            'metrics': {
                'total_energy': total_energy,
                'total_latency': total_latency,
                'total_distance': total_distance,
                'reliability': path_reliability,
                'min_bandwidth': min_bandwidth,
                'avg_utilization': avg_utilization
            }
        }
    
    def _calculate_load_balance_bonus(self, path: List[str]) -> float:
        """计算负载均衡奖励"""
        # 计算路径中节点的负载均衡度
        utilizations = [self.nodes[node_id].current_load for node_id in path]
        
        if len(utilizations) < 2:
            return 0.0
        
        # 计算负载方差（越小越好）
        load_variance = np.var(utilizations)
        balance_score = 1.0 / (1.0 + load_variance)
        
        return balance_score * self.load_balancing_factor
    
    def update_node_load(self, node_id: str, load_change: float):
        """更新节点负载"""
        if node_id in self.nodes:
            node = self.nodes[node_id]
            node.current_load = max(0, min(1, node.current_load + load_change))
    
    def update_path_performance(self, 
                               path: List[str], 
                               actual_energy: float,
                               actual_latency: float,
                               success: bool):
        """更新路径性能"""
        # 记录性能
        performance_entry = {
            'path': path.copy(),
            'actual_energy': actual_energy,
            'actual_latency': actual_latency,
            'success': success,
            'timestamp': len(self.performance_history)
        }
        self.performance_history.append(performance_entry)
        
        # 更新统计
        if success:
            self.successful_paths += 1
        
        # 更新平均指标
        total_paths = self.total_paths_selected
        self.average_energy_consumption = (
            self.average_energy_consumption * (total_paths - 1) + actual_energy
        ) / total_paths
        self.average_latency = (
            self.average_latency * (total_paths - 1) + actual_latency
        ) / total_paths
        
        # 优化权重
        if len(self.performance_history) % 50 == 0:
            self.multi_objective_optimizer.optimize_weights(list(self.performance_history))
    
    def train_scoring_network(self, batch_size: int = 32):
        """训练评分网络"""
        if len(self.path_history) < batch_size:
            return 0.0
        
        # 采样历史数据
        recent_history = list(self.path_history)[-batch_size:]
        
        # 准备训练数据
        states = []
        target_scores = []
        
        for entry in recent_history:
            # 构建路径状态向量
            path = entry['selected_path']
            path_state = self._path_to_state_vector(path)
            target_score = entry['score']
            
            states.append(path_state)
            target_scores.append(target_score)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        targets_tensor = torch.FloatTensor(target_scores).to(self.device)
        
        # 前向传播
        predicted_scores = self.scoring_network(states_tensor).squeeze()
        
        # 计算损失
        loss = nn.MSELoss()(predicted_scores, targets_tensor)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _path_to_state_vector(self, path: List[str]) -> np.ndarray:
        """将路径转换为状态向量"""
        # 简化的路径状态表示
        state = np.zeros(32)  # 假设状态维度为32
        
        # 路径长度
        state[0] = len(path) / 10.0  # 归一化
        
        # 节点信息
        for i, node_id in enumerate(path[:8]):  # 最多8个节点
            if i * 4 + 3 < len(state):
                node = self.nodes[node_id]
                state[i * 4 + 1] = node.energy_consumption / 5.0
                state[i * 4 + 2] = node.processing_capacity / 150.0
                state[i * 4 + 3] = node.current_load
        
        return state
    
    def get_path_analysis(self, source: str, target: str) -> Dict:
        """获取路径分析"""
        if source not in self.nodes or target not in self.nodes:
            return {'error': 'Invalid nodes'}
        
        all_paths = self._find_all_paths(source, target)
        
        if not all_paths:
            return {'error': 'No paths found'}
        
        # 分析所有路径
        path_analyses = []
        for i, path in enumerate(all_paths):
            analysis = self._evaluate_path(path, {})
            path_analyses.append({
                'path_id': i,
                'path': path,
                'total_score': analysis['total_score'],
                'energy_consumption': analysis['metrics']['total_energy'],
                'latency': analysis['metrics']['total_latency'],
                'reliability': analysis['metrics']['reliability'],
                'path_length': len(path)
            })
        
        # 按评分排序
        path_analyses.sort(key=lambda x: x['total_score'], reverse=True)
        
        return {
            'source': source,
            'target': target,
            'total_paths_found': len(all_paths),
            'path_analyses': path_analyses[:10],  # 前10条路径
            'best_path': path_analyses[0] if path_analyses else None,
            'network_statistics': self._get_network_statistics()
        }
    
    def _get_network_statistics(self) -> Dict:
        """获取网络统计信息"""
        node_utilizations = [node.current_load for node in self.nodes.values()]
        edge_reliabilities = [edge.reliability for edge in self.edges.values()]
        
        return {
            'total_nodes': len(self.nodes),
            'total_edges': len(self.edges),
            'avg_node_utilization': np.mean(node_utilizations),
            'max_node_utilization': np.max(node_utilizations),
            'min_node_utilization': np.min(node_utilizations),
            'avg_edge_reliability': np.mean(edge_reliabilities),
            'connectivity': nx.is_connected(self.graph)
        }
    
    def simulate_traffic_load(self, 
                             num_requests: int = 1000,
                             callback: Optional[callable] = None):
        """模拟流量负载"""
        node_ids = list(self.nodes.keys())
        
        for i in range(num_requests):
            # 随机选择源和目标
            source, target = np.random.choice(node_ids, 2, replace=False)
            
            # 寻找最优路径
            result = self.find_optimal_path(source, target)
            
            if 'selected_path' in result:
                # 模拟路径使用
                path = result['selected_path']
                for node_id in path:
                    self.update_node_load(node_id, 0.01)  # 增加负载
                
                # 模拟完成
                actual_energy = result['path_details']['metrics']['total_energy']
                actual_latency = result['path_details']['metrics']['total_latency']
                success = np.random.random() < result['path_details']['reliability_score']
                
                self.update_path_performance(path, actual_energy, actual_latency, success)
                
                # 回调函数
                if callback:
                    callback(i, source, target, result)
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        return {
            'total_paths_selected': self.total_paths_selected,
            'successful_paths': self.successful_paths,
            'success_rate': self.successful_paths / max(self.total_paths_selected, 1),
            'average_energy_consumption': self.average_energy_consumption,
            'average_latency': self.average_latency,
            'path_history_size': len(self.path_history),
            'performance_history_size': len(self.performance_history),
            'network_connectivity': nx.is_connected(self.graph),
            'objective_weights': self.multi_objective_optimizer.weights.copy()
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        model_data = {
            'scoring_network_state_dict': self.scoring_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'nodes': self.nodes,
            'edges': self.edges,
            'objective_weights': self.multi_objective_optimizer.weights,
            'performance_metrics': self.get_performance_metrics()
        }
        
        torch.save(model_data, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        model_data = torch.load(filepath, map_location=self.device)
        
        self.scoring_network.load_state_dict(model_data['scoring_network_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        self.nodes = model_data['nodes']
        self.edges = model_data['edges']
        self.multi_objective_optimizer.weights = model_data['objective_weights']