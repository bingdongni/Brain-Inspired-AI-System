"""
自适应负载均衡器
实现多级负载均衡策略和动态资源分配
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import heapq
import time


@dataclass
class LoadMetrics:
    """负载指标"""
    current_load: float
    capacity: float
    utilization: float
    response_time: float
    throughput: float
    error_rate: float
    priority: int = 0
    
    @property
    def load_factor(self) -> float:
        return self.current_load / self.capacity if self.capacity > 0 else 0
    
    @property
    def health_score(self) -> float:
        """健康评分"""
        # 综合评分：低负载 + 快响应 + 高吞吐量 + 低错误率
        utilization_score = 1.0 - min(1.0, self.load_factor)
        response_score = 1.0 / (1.0 + self.response_time)
        throughput_score = min(1.0, self.throughput / 100.0)  # 假设最大吞吐量100
        error_penalty = max(0, 1.0 - self.error_rate)
        
        return (utilization_score + response_score + throughput_score + error_penalty) / 4


@dataclass
class ServerNode:
    """服务器节点"""
    id: str
    capacity: float
    current_load: float = 0.0
    connections: int = 0
    max_connections: int = 100
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    last_heartbeat: float = field(default_factory=time.time)
    
    @property
    def utilization(self) -> float:
        return self.current_load / self.capacity if self.capacity > 0 else 0
    
    @property
    def avg_response_time(self) -> float:
        return np.mean(list(self.response_times)) if self.response_times else 0.0
    
    @property
    def error_rate(self) -> float:
        return self.error_count / max(self.total_requests, 1)
    
    @property
    def success_rate(self) -> float:
        return self.successful_requests / max(self.total_requests, 1)
    
    @property
    def health_score(self) -> float:
        return (1.0 - self.utilization) * (1.0 / (1.0 + self.avg_response_time)) * self.success_rate


class LoadBalancingNetwork(nn.Module):
    """负载均衡决策网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_nodes: int = 8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_nodes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        node_scores = torch.softmax(self.fc3(x), dim=-1)
        return node_scores


class AdaptiveLoadBalancer:
    """自适应负载均衡器"""
    
    def __init__(self,
                 num_nodes: int = 8,
                 state_dim: int = 64,
                 learning_rate: float = 1e-3,
                 balancing_strategy: str = 'adaptive',
                 device: str = 'cpu'):
        
        self.num_nodes = num_nodes
        self.state_dim = state_dim
        self.balancing_strategy = balancing_strategy
        self.device = device
        
        # 服务器节点
        self.nodes = self._initialize_nodes()
        
        # 负载均衡策略
        self.strategy_weights = {
            'round_robin': 0.0,
            'least_connections': 0.3,
            'weighted_response_time': 0.3,
            'adaptive': 0.4
        }
        
        # 负载均衡网络
        self.load_balancer_network = LoadBalancingNetwork(state_dim, num_nodes=num_nodes).to(device)
        self.optimizer = optim.Adam(self.load_balancer_network.parameters(), lr=learning_rate)
        
        # 历史数据
        self.request_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=5000)
        
        # 统计信息
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
        self.total_response_time = 0.0
        
        # 负载均衡参数
        self.health_threshold = 0.5
        self.breakdown_threshold = 0.9
        self.recovery_threshold = 0.3
        
        # 动态调整
        self.adjustment_frequency = 100
        self.last_adjustment = 0
        
    def _initialize_nodes(self) -> List[ServerNode]:
        """初始化服务器节点"""
        nodes = []
        for i in range(self.num_nodes):
            node = ServerNode(
                id=f"server_{i}",
                capacity=np.random.uniform(80, 120),
                max_connections=np.random.randint(80, 120)
            )
            nodes.append(node)
        return nodes
    
    def select_node(self, 
                   request: Dict = None,
                   state: np.ndarray = None) -> int:
        """选择最佳服务器节点"""
        if request is None:
            request = {}
        if state is None:
            state = self._get_current_state()
        
        # 获取节点评分
        node_scores = self._calculate_node_scores(state)
        
        # 应用负载均衡策略
        if self.balancing_strategy == 'adaptive':
            selected_node = self._adaptive_selection(node_scores)
        elif self.balancing_strategy == 'least_connections':
            selected_node = self._least_connections_selection()
        elif self.balancing_strategy == 'weighted_response_time':
            selected_node = self._weighted_response_time_selection()
        elif self.balancing_strategy == 'round_robin':
            selected_node = self._round_robin_selection()
        else:
            # 默认使用评分最高的节点
            selected_node = np.argmax(node_scores)
        
        # 更新节点连接
        self._add_connection(selected_node, request)
        
        # 记录请求
        self._record_request(selected_node, request)
        
        return selected_node
    
    def _calculate_node_scores(self, state: np.ndarray) -> np.ndarray:
        """计算节点评分"""
        node_scores = np.zeros(self.num_nodes)
        
        for i, node in enumerate(self.nodes):
            # 基于当前状态的评分
            load_score = 1.0 - node.utilization
            response_score = 1.0 / (1.0 + node.avg_response_time)
            success_score = node.success_rate
            capacity_score = min(1.0, node.capacity / 100.0)
            
            # 综合评分
            node_scores[i] = (load_score * 0.3 + 
                            response_score * 0.3 + 
                            success_score * 0.2 + 
                            capacity_score * 0.2)
            
            # 防止选择过载节点
            if node.utilization > self.breakdown_threshold:
                node_scores[i] *= 0.1
        
        return node_scores
    
    def _adaptive_selection(self, node_scores: np.ndarray) -> int:
        """自适应节点选择"""
        # 使用神经网络预测最佳节点
        state_tensor = torch.FloatTensor(self._get_current_state()).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predicted_scores = self.load_balancer_network(state_tensor).cpu().numpy().squeeze()
        
        # 结合预测评分和实际评分
        combined_scores = 0.6 * node_scores + 0.4 * predicted_scores
        
        # 添加随机性避免过度集中在某个节点
        noise = np.random.normal(0, 0.05, len(combined_scores))
        combined_scores += noise
        
        return np.argmax(combined_scores)
    
    def _least_connections_selection(self) -> int:
        """最少连接选择"""
        min_connections = float('inf')
        selected_node = 0
        
        for i, node in enumerate(self.nodes):
            if node.connections < min_connections and node.utilization < 1.0:
                min_connections = node.connections
                selected_node = i
        
        return selected_node
    
    def _weighted_response_time_selection(self) -> int:
        """加权响应时间选择"""
        best_score = -1
        selected_node = 0
        
        for i, node in enumerate(self.nodes):
            if node.total_requests > 10:  # 至少处理过10个请求
                # 响应时间加权评分
                score = node.success_rate / (1.0 + node.avg_response_time)
                if score > best_score:
                    best_score = score
                    selected_node = i
        
        return selected_node
    
    def _round_robin_selection(self) -> int:
        """轮询选择"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        selected_node = self._round_robin_index
        self._round_robin_index = (self._round_robin_index + 1) % self.num_nodes
        
        # 确保节点可用
        for _ in range(self.num_nodes):
            if self.nodes[selected_node].utilization < 1.0:
                break
            selected_node = (selected_node + 1) % self.num_nodes
        
        return selected_node
    
    def _add_connection(self, node_idx: int, request: Dict):
        """添加连接"""
        if 0 <= node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            node.connections += 1
            node.total_requests += 1
            node.last_heartbeat = time.time()
    
    def _record_request(self, node_idx: int, request: Dict):
        """记录请求"""
        self.request_history.append({
            'node_idx': node_idx,
            'timestamp': time.time(),
            'request_size': request.get('size', 1),
            'priority': request.get('priority', 0)
        })
    
    def complete_request(self, 
                        node_idx: int, 
                        response_time: float, 
                        success: bool,
                        error_type: str = None):
        """完成请求"""
        if 0 <= node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            
            # 更新响应时间
            node.response_times.append(response_time)
            self.total_response_time += response_time
            
            # 更新成功/失败统计
            if success:
                node.successful_requests += 1
                self.successful_requests += 1
            else:
                node.error_count += 1
                self.failed_requests += 1
                if error_type:
                    # 可以记录错误类型统计
                    pass
            
            # 移除连接
            node.connections = max(0, node.connections - 1)
            
            # 更新平均响应时间
            self.avg_response_time = self.total_response_time / max(self.total_requests, 1)
            
            self.total_requests += 1
    
    def _get_current_state(self) -> np.ndarray:
        """获取当前状态"""
        state = np.zeros(self.state_dim)
        
        # 节点信息
        for i, node in enumerate(self.nodes):
            if i * 6 < self.state_dim:
                state[i * 6] = node.utilization
                state[i * 6 + 1] = node.connections / max(node.max_connections, 1)
                state[i * 6 + 2] = node.avg_response_time
                state[i * 6 + 3] = node.success_rate
                state[i * 6 + 4] = node.error_rate
                state[i * 6 + 5] = node.capacity
        
        # 全局统计
        if self.state_dim >= len(self.nodes) * 6 + 4:
            total_utilization = np.mean([n.utilization for n in self.nodes])
            max_utilization = np.max([n.utilization for n in self.nodes])
            min_utilization = np.min([n.utilization for n in self.nodes])
            global_avg_response = np.mean([n.avg_response_time for n in self.nodes])
            
            start_idx = len(self.nodes) * 6
            state[start_idx] = total_utilization
            state[start_idx + 1] = max_utilization
            state[start_idx + 2] = min_utilization
            state[start_idx + 3] = global_avg_response
        
        return state
    
    def update_node_load(self, node_idx: int, load_change: float):
        """更新节点负载"""
        if 0 <= node_idx < len(self.nodes):
            node = self.nodes[node_idx]
            node.current_load = max(0, node.current_load + load_change)
            node.current_load = min(node.current_load, node.capacity * 1.2)  # 防止超过容量
    
    def train_load_balancer(self, batch_size: int = 32):
        """训练负载均衡器"""
        if len(self.request_history) < batch_size:
            return 0.0
        
        # 采样历史数据
        recent_requests = list(self.request_history)[-batch_size:]
        
        # 准备训练数据
        states = []
        target_nodes = []
        
        for request in recent_requests:
            # 简化的状态构建
            state = self._get_current_state() + np.random.normal(0, 0.1, self.state_dim)
            # 目标节点：实际使用的节点
            target_nodes.append(request['node_idx'])
            
            states.append(state)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        targets_tensor = torch.LongTensor(target_nodes).to(self.device)
        
        # 前向传播
        node_scores = self.load_balancer_network(states_tensor)
        
        # 计算损失（这里简化实现，实际中需要更复杂的损失函数）
        loss = nn.CrossEntropyLoss()(node_scores, targets_tensor)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def get_load_balancing_stats(self) -> Dict:
        """获取负载均衡统计"""
        if self.total_requests == 0:
            return {
                'total_requests': 0,
                'success_rate': 0.0,
                'avg_response_time': 0.0,
                'avg_utilization': 0.0,
                'load_variance': 0.0
            }
        
        # 节点统计
        node_stats = []
        for node in self.nodes:
            node_stats.append({
                'id': node.id,
                'utilization': node.utilization,
                'connections': node.connections,
                'avg_response_time': node.avg_response_time,
                'success_rate': node.success_rate,
                'total_requests': node.total_requests
            })
        
        # 全局统计
        total_utilization = np.mean([n.utilization for n in self.nodes])
        utilization_variance = np.var([n.utilization for n in self.nodes])
        success_rate = self.successful_requests / self.total_requests
        
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': success_rate,
            'avg_response_time': self.avg_response_time,
            'total_utilization': total_utilization,
            'utilization_variance': utilization_variance,
            'strategy': self.balancing_strategy,
            'node_stats': node_stats,
            'strategy_weights': self.strategy_weights
        }
    
    def get_node_recommendations(self) -> Dict:
        """获取节点优化建议"""
        recommendations = {
            'scale_up': [],
            'scale_down': [],
            'maintenance': [],
            'optimization': []
        }
        
        for i, node in enumerate(self.nodes):
            node_id = node.id
            
            # 检查是否需要扩容
            if node.utilization > 0.9:
                recommendations['scale_up'].append({
                    'node_id': node_id,
                    'reason': 'High utilization',
                    'current_utilization': node.utilization,
                    'suggested_capacity': node.capacity * 1.5
                })
            
            # 检查是否需要缩容
            if node.utilization < 0.3 and node.total_requests > 100:
                recommendations['scale_down'].append({
                    'node_id': node_id,
                    'reason': 'Low utilization',
                    'current_utilization': node.utilization,
                    'suggested_capacity': node.capacity * 0.7
                })
            
            # 检查是否需要维护
            if node.error_rate > 0.1 or node.success_rate < 0.8:
                recommendations['maintenance'].append({
                    'node_id': node_id,
                    'reason': 'High error rate or low success rate',
                    'error_rate': node.error_rate,
                    'success_rate': node.success_rate
                })
            
            # 优化建议
            if node.avg_response_time > 2.0:
                recommendations['optimization'].append({
                    'node_id': node_id,
                    'reason': 'High response time',
                    'current_response_time': node.avg_response_time,
                    'suggested_actions': ['optimize_queries', 'increase_cache', 'upgrade_hardware']
                })
        
        return recommendations
    
    def simulate_workload(self, 
                         pattern: str = 'normal',
                         duration: int = 1000,
                         callback=None):
        """模拟工作负载"""
        for step in range(duration):
            # 生成请求
            if pattern == 'burst':
                # 突发负载
                if step % 50 == 0:  # 每50步一个突发
                    request_count = np.random.randint(20, 50)
                else:
                    request_count = np.random.randint(1, 5)
            elif pattern == 'gradual':
                # 渐进负载
                request_count = int(1 + step * 0.1)
            elif pattern == 'diurnal':
                # 昼夜模式
                hour = (step // 60) % 24
                if 9 <= hour <= 17:  # 工作时间
                    request_count = np.random.randint(15, 25)
                else:
                    request_count = np.random.randint(3, 8)
            else:  # normal
                request_count = np.random.randint(5, 15)
            
            # 处理请求
            for _ in range(request_count):
                # 选择节点
                node_idx = self.select_node({'size': np.random.uniform(0.1, 2.0)})
                
                # 模拟处理时间
                node = self.nodes[node_idx]
                processing_time = (node.avg_response_time + 
                                 np.random.exponential(node.avg_response_time))
                
                # 模拟成功率
                success = np.random.random() < node.success_rate
                
                # 完成请求
                self.complete_request(node_idx, processing_time, success)
                
                # 回调函数
                if callback:
                    callback(step, node_idx, processing_time, success)
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'load_balancer_network_state_dict': self.load_balancer_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'balancing_strategy': self.balancing_strategy,
            'strategy_weights': self.strategy_weights,
            'nodes': self.nodes,
            'statistics': self.get_load_balancing_stats()
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_balancer_network.load_state_dict(
            checkpoint['load_balancer_network_state_dict']
        )
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.balancing_strategy = checkpoint['balancing_strategy']
        self.strategy_weights = checkpoint['strategy_weights']
        self.nodes = checkpoint['nodes']