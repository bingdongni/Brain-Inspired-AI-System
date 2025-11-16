"""
神经启发路由算法
基于生物神经网络的路由决策机制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import math


@dataclass
class NeuronState:
    """神经元状态"""
    id: str
    membrane_potential: float = 0.0
    activation_level: float = 0.0
    last_spike_time: float = 0.0
    refractory_period: float = 0.0
    connection_strengths: Dict[str, float] = field(default_factory=dict)


@dataclass
class Synapse:
    """突触连接"""
    pre_neuron: str
    post_neuron: str
    weight: float
    delay: float
    plasticity: float = 1.0


class NeuralInspirationNetwork(nn.Module):
    """神经启发网络"""
    
    def __init__(self, 
                 num_neurons: int = 64,
                 num_inputs: int = 32,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.num_neurons = num_neurons
        
        # 输入层到神经元层
        self.input_weights = nn.Parameter(torch.randn(num_inputs, num_neurons) * 0.1)
        
        # 神经元间连接
        self.recurrent_weights = nn.Parameter(torch.randn(num_neurons, num_neurons) * 0.1)
        
        # 神经元输出层
        self.output_weights = nn.Parameter(torch.randn(num_neurons, 3) * 0.1)  # 3个输出：路由决策、能效评分、置信度
        
        # 层归一化
        self.layer_norm1 = nn.LayerNorm(num_neurons)
        self.layer_norm2 = nn.LayerNorm(num_neurons)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = inputs.size(0)
        
        # 输入到神经元的连接
        neuron_inputs = torch.matmul(inputs, self.input_weights)
        
        # 神经元的递归连接
        recurrent_input = torch.matmul(neuron_inputs, self.recurrent_weights)
        
        # 结合输入和递归输入
        combined_input = neuron_inputs + recurrent_input
        
        # 层归一化
        normalized_input = self.layer_norm1(combined_input)
        
        # 激活函数（使用ReLU和Sigmoid的组合）
        neuron_activation = torch.relu(normalized_input)
        neuron_activation = self.dropout(neuron_activation)
        
        # 再次层归一化
        final_activation = self.layer_norm2(neuron_activation)
        
        # 输出层
        outputs = torch.matmul(final_activation, self.output_weights)
        
        # 分离不同的输出
        routing_decision = torch.sigmoid(outputs[:, 0])  # 路由决策
        energy_score = torch.sigmoid(outputs[:, 1])  # 能效评分
        confidence = torch.sigmoid(outputs[:, 2])  # 置信度
        
        return routing_decision, energy_score, confidence


class NeuralInspiredRouter:
    """神经启发路由器"""
    
    def __init__(self,
                 num_neurons: int = 64,
                 input_dim: int = 32,
                 num_paths: int = 8,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        
        self.num_neurons = num_neurons
        self.input_dim = input_dim
        self.num_paths = num_paths
        self.device = device
        
        # 神经网络
        self.neural_network = NeuralInspirationNetwork(num_neurons, input_dim).to(device)
        self.optimizer = optim.Adam(self.neural_network.parameters(), lr=learning_rate)
        
        # 神经元状态
        self.neurons = self._initialize_neurons()
        self.synapses = self._initialize_synapses()
        
        # 学习参数
        self.spike_threshold = 0.5
        self.refractory_period = 5.0
        self.synaptic_plasticity_rate = 0.01
        self.homeostatic_regulation = 0.001
        
        # 路径信息
        self.paths = self._initialize_paths()
        
        # 训练历史
        self.training_history = deque(maxlen=10000)
        self.performance_history = deque(maxlen=5000)
        
        # 统计信息
        self.total_decisions = 0
        self.successful_routes = 0
        self.average_energy_consumption = 0.0
        self.neuron_activity_levels = deque(maxlen=1000)
        
    def _initialize_neurons(self) -> Dict[str, NeuronState]:
        """初始化神经元"""
        neurons = {}
        
        # 输入神经元（感知环境状态）
        for i in range(self.input_dim):
            neurons[f'input_{i}'] = NeuronState(
                id=f'input_{i}',
                connection_strengths={}
            )
        
        # 隐藏神经元（处理和决策）
        for i in range(self.num_neurons):
            neurons[f'hidden_{i}'] = NeuronState(
                id=f'hidden_{i}',
                connection_strengths={}
            )
        
        # 输出神经元（路由决策）
        for i in range(3):  # 3个输出神经元
            neurons[f'output_{i}'] = NeuronState(
                id=f'output_{i}',
                connection_strengths={}
            )
        
        return neurons
    
    def _initialize_synapses(self) -> List[Synapse]:
        """初始化突触连接"""
        synapses = []
        
        # 输入到隐藏层的连接
        for input_id in [n for n in self.neurons.keys() if n.startswith('input_')]:
            for hidden_id in [n for n in self.neurons.keys() if n.startswith('hidden_')]:
                synapses.append(Synapse(
                    pre_neuron=input_id,
                    post_neuron=hidden_id,
                    weight=np.random.normal(0, 0.1),
                    delay=np.random.uniform(0.1, 1.0)
                ))
        
        # 隐藏层内部的连接（递归）
        hidden_neurons = [n for n in self.neurons.keys() if n.startswith('hidden_')]
        for i, pre_id in enumerate(hidden_neurons):
            for j, post_id in enumerate(hidden_neurons):
                if i != j:  # 无自连接
                    synapses.append(Synapse(
                        pre_neuron=pre_id,
                        post_neuron=post_id,
                        weight=np.random.normal(0, 0.05),
                        delay=np.random.uniform(0.1, 0.5)
                    ))
        
        # 隐藏层到输出层的连接
        for hidden_id in [n for n in self.neurons.keys() if n.startswith('hidden_')]:
            for output_id in [n for n in self.neurons.keys() if n.startswith('output_')]:
                synapses.append(Synapse(
                    pre_neuron=hidden_id,
                    post_neuron=output_id,
                    weight=np.random.normal(0, 0.1),
                    delay=np.random.uniform(0.1, 0.3)
                ))
        
        return synapses
    
    def _initialize_paths(self) -> List[Dict]:
        """初始化路径信息"""
        paths = []
        for i in range(self.num_paths):
            paths.append({
                'id': f'path_{i}',
                'energy_consumption': np.random.uniform(0.5, 2.0),
                'latency': np.random.uniform(0.1, 1.0),
                'reliability': np.random.uniform(0.8, 0.99),
                'throughput': np.random.uniform(50, 200),
                'current_load': np.random.uniform(0.1, 0.8),
                'capacity': np.random.uniform(80, 120)
            })
        return paths
    
    def process_input(self, state: np.ndarray) -> Tuple[int, float, float]:
        """处理输入状态并生成路由决策"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # 前向传播
        routing_decision, energy_score, confidence = self.neural_network(state_tensor)
        
        # 转换为具体决策
        route_idx = int(torch.argmax(routing_decision).item())
        energy_rating = energy_score.item()
        confidence_level = confidence.item()
        
        # 更新神经元活动
        self._update_neuron_activity(state, route_idx, energy_rating, confidence_level)
        
        # 记录决策
        self.total_decisions += 1
        
        return route_idx, energy_rating, confidence_level
    
    def _update_neuron_activity(self, 
                               input_state: np.ndarray, 
                               route_idx: int, 
                               energy_rating: float, 
                               confidence: float):
        """更新神经元活动状态"""
        current_time = len(self.training_history)
        
        # 更新输入神经元
        for i, value in enumerate(input_state):
            input_neuron = f'input_{i}'
            if input_neuron in self.neurons:
                neuron = self.neurons[input_neuron]
                neuron.membrane_potential = value
                neuron.activation_level = max(0, value)
                
                if value > self.spike_threshold:
                    neuron.last_spike_time = current_time
        
        # 更新隐藏神经元（简化模型）
        hidden_neurons = [n for n in self.neurons.keys() if n.startswith('hidden_')]
        for i, hidden_neuron in enumerate(hidden_neurons):
            if hidden_neuron in self.neurons:
                neuron = self.neurons[hidden_neuron]
                
                # 简化的激活计算
                input_sum = np.sum(input_state[i % len(input_state):i + len(input_state)])
                neuron.membrane_potential = 0.7 * neuron.membrane_potential + 0.3 * input_sum
                neuron.activation_level = max(0, neuron.membrane_potential)
        
        # 更新输出神经元
        output_neurons = ['output_0', 'output_1', 'output_2']  # 路由、能效、置信度
        output_values = [route_idx / self.num_paths, energy_rating, confidence]
        
        for i, (output_neuron, value) in enumerate(zip(output_neurons, output_values)):
            if output_neuron in self.neurons:
                neuron = self.neurons[output_neuron]
                neuron.membrane_potential = value
                neuron.activation_level = value
                neuron.last_spike_time = current_time if value > 0.5 else neuron.last_spike_time
    
    def train_step(self, 
                   state: np.ndarray, 
                   target_route: int, 
                   target_energy_score: float,
                   target_confidence: float) -> Dict[str, float]:
        """训练步骤"""
        # 前向传播
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        routing_decision, energy_score, confidence = self.neural_network(state_tensor)
        
        # 计算损失
        target_route_tensor = torch.FloatTensor([target_route / self.num_paths]).to(self.device)
        target_energy_tensor = torch.FloatTensor([target_energy_score]).to(self.device)
        target_confidence_tensor = torch.FloatTensor([target_confidence]).to(self.device)
        
        loss_routing = nn.MSELoss()(routing_decision, target_route_tensor)
        loss_energy = nn.MSELoss()(energy_score, target_energy_tensor)
        loss_confidence = nn.MSELoss()(confidence, target_confidence_tensor)
        
        total_loss = loss_routing + loss_energy + loss_confidence
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # 更新突触可塑性
        self._update_synaptic_plasticity()
        
        # 记录训练历史
        self.training_history.append({
            'loss': total_loss.item(),
            'routing_loss': loss_routing.item(),
            'energy_loss': loss_energy.item(),
            'confidence_loss': loss_confidence.item(),
            'timestamp': len(self.training_history)
        })
        
        return {
            'total_loss': total_loss.item(),
            'routing_loss': loss_routing.item(),
            'energy_loss': loss_energy.item(),
            'confidence_loss': loss_confidence.item()
        }
    
    def _update_synaptic_plasticity(self):
        """更新突触可塑性"""
        # 简化的Hebbian学习规则
        for synapse in self.synapses:
            pre_neuron = self.neurons.get(synapse.pre_neuron)
            post_neuron = self.neurons.get(synapse.post_neuron)
            
            if pre_neuron and post_neuron:
                # Hebbian规则：如果两个神经元同时活跃，增强连接
                if (pre_neuron.activation_level > 0.5 and post_neuron.activation_level > 0.5):
                    synapse.weight += self.synaptic_plasticity_rate * pre_neuron.activation_level * post_neuron.activation_level
                    synapse.weight = np.clip(synapse.weight, -1.0, 1.0)  # 限制权重范围
                
                # 稳态调节
                synapse.weight *= (1 - self.homeostatic_regulation)
                
                # 更新突触连接强度
                post_neuron.connection_strengths[synapse.pre_neuron] = synapse.weight
    
    def update_path_metrics(self, path_idx: int, actual_energy: float, 
                           actual_latency: float, success: bool):
        """更新路径指标"""
        if 0 <= path_idx < len(self.paths):
            path = self.paths[path_idx]
            
            # 使用指数移动平均更新指标
            alpha = 0.1
            path['energy_consumption'] = (alpha * actual_energy + 
                                        (1 - alpha) * path['energy_consumption'])
            path['latency'] = (alpha * actual_latency + 
                             (1 - alpha) * path['latency'])
            
            # 更新可靠性
            if success:
                path['reliability'] = min(0.99, path['reliability'] + 0.001)
            else:
                path['reliability'] = max(0.5, path['reliability'] - 0.01)
            
            # 更新成功率
            if success:
                self.successful_routes += 1
    
    def get_energy_efficiency_analysis(self, path_idx: int) -> Dict:
        """获取能效分析"""
        if path_idx >= len(self.paths):
            return {}
        
        path = self.paths[path_idx]
        
        # 计算能效指标
        energy_per_bit = path['energy_consumption'] / max(path['throughput'], 1)
        power_efficiency = 1.0 / (1.0 + path['energy_consumption'])
        latency_efficiency = 1.0 / (1.0 + path['latency'])
        
        # 综合能效评分
        efficiency_score = (power_efficiency * 0.4 + 
                          latency_efficiency * 0.3 + 
                          path['reliability'] * 0.3)
        
        # 与其他路径比较
        other_scores = []
        for i, other_path in enumerate(self.paths):
            if i != path_idx:
                other_energy = other_path['energy_consumption']
                other_latency = other_path['latency']
                other_reliability = other_path['reliability']
                
                other_score = ((1.0 / (1.0 + other_energy)) * 0.4 + 
                              (1.0 / (1.0 + other_latency)) * 0.3 + 
                              other_reliability * 0.3)
                other_scores.append(other_score)
        
        relative_efficiency = efficiency_score / (np.mean(other_scores) + 0.001)
        
        return {
            'path_id': path_idx,
            'energy_consumption': path['energy_consumption'],
            'latency': path['latency'],
            'reliability': path['reliability'],
            'throughput': path['throughput'],
            'current_load': path['current_load'],
            'energy_per_bit': energy_per_bit,
            'power_efficiency': power_efficiency,
            'latency_efficiency': latency_efficiency,
            'overall_efficiency_score': efficiency_score,
            'relative_efficiency': relative_efficiency,
            'capacity_utilization': path['current_load'] / max(path['capacity'], 1)
        }
    
    def get_neuron_activity_analysis(self) -> Dict:
        """获取神经元活动分析"""
        if not self.training_history:
            return {'error': 'No training history available'}
        
        # 计算平均神经元活动水平
        avg_activity = np.mean([entry.get('neuron_activity', 0) for entry in self.training_history])
        
        # 活动方差
        activities = [entry.get('neuron_activity', 0) for entry in self.training_history[-100:]]
        activity_variance = np.var(activities) if activities else 0
        
        # 神经元类型统计
        neuron_stats = {
            'input_neurons': len([n for n in self.neurons.keys() if n.startswith('input_')]),
            'hidden_neurons': len([n for n in self.neurons.keys() if n.startswith('hidden_')]),
            'output_neurons': len([n for n in self.neurons.keys() if n.startswith('output_')])
        }
        
        # 突触统计
        synapse_stats = {
            'total_synapses': len(self.synapses),
            'avg_weight': np.mean([s.weight for s in self.synapses]),
            'max_weight': np.max([s.weight for s in self.synapses]),
            'min_weight': np.min([s.weight for s in self.synapses])
        }
        
        return {
            'average_activity': avg_activity,
            'activity_variance': activity_variance,
            'neuron_statistics': neuron_stats,
            'synapse_statistics': synapse_stats,
            'training_history_size': len(self.training_history)
        }
    
    def simulate_neural_activity(self, duration: int = 1000) -> Dict:
        """模拟神经元活动"""
        activities = []
        
        for step in range(duration):
            # 模拟输入状态
            input_state = np.random.randn(self.input_dim) * 0.5
            input_state = np.clip(input_state, 0, 1)  # 归一化到[0,1]
            
            # 处理输入
            route_idx, energy_score, confidence = self.process_input(input_state)
            
            # 记录活动
            activities.append({
                'step': step,
                'route_decision': route_idx,
                'energy_score': energy_score,
                'confidence': confidence,
                'input_state': input_state.copy()
            })
            
            # 模拟训练（随机目标）
            if step % 10 == 0:
                target_route = np.random.randint(0, self.num_paths)
                target_energy = np.random.uniform(0, 1)
                target_confidence = np.random.uniform(0.5, 1.0)
                
                self.train_step(input_state, target_route, target_energy, target_confidence)
        
        return {
            'simulation_duration': duration,
            'total_activities': len(activities),
            'avg_route_decisions': np.mean([a['route_decision'] for a in activities]),
            'avg_energy_score': np.mean([a['energy_score'] for a in activities]),
            'avg_confidence': np.mean([a['confidence'] for a in activities])
        }
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        return {
            'total_decisions': self.total_decisions,
            'successful_routes': self.successful_routes,
            'success_rate': self.successful_routes / max(self.total_decisions, 1),
            'average_energy_consumption': self.average_energy_consumption,
            'neuron_count': len(self.neurons),
            'synapse_count': len(self.synapses),
            'training_history_size': len(self.training_history),
            'recent_loss': np.mean([entry['loss'] for entry in list(self.training_history)[-10:]]) if self.training_history else 0
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'neural_network_state_dict': self.neural_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'neurons': self.neurons,
            'synapses': self.synapses,
            'paths': self.paths,
            'performance_metrics': self.get_performance_metrics()
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.neural_network.load_state_dict(checkpoint['neural_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.neurons = checkpoint['neurons']
        self.synapses = checkpoint['synapses']
        self.paths = checkpoint['paths']