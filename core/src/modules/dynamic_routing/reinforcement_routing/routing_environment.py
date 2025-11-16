"""
路由环境模拟器
用于训练和测试强化学习路由策略的环境
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import time
from dataclasses import dataclass


@dataclass
class Module:
    """模块信息"""
    id: str
    capacity: float
    load: float
    latency: float
    energy: float
    accuracy: float
    availability: float
    
    @property
    def utilization(self) -> float:
        return self.load / self.capacity
    
    @property
    def score(self) -> float:
        """综合评分"""
        # 综合考虑利用率、延迟、能耗和准确性
        utilization_penalty = max(0, self.utilization - 0.8) * 2  # 利用率超过80%时惩罚
        latency_score = 1.0 / (1.0 + self.latency)  # 延迟越低越好
        energy_score = 1.0 / (1.0 + self.energy)  # 能耗越低越好
        accuracy_score = self.accuracy  # 准确性越高越好
        availability_score = self.availability  # 可用性越高越好
        
        return (accuracy_score + availability_score + latency_score + energy_score) / 4 - utilization_penalty


class RoutingEnvironment:
    """路由环境"""
    
    def __init__(self, 
                 num_modules: int = 8,
                 state_dim: int = 32,
                 max_steps: int = 1000,
                 seed: int = 42):
        
        self.num_modules = num_modules
        self.state_dim = state_dim
        self.max_steps = max_steps
        self.current_step = 0
        self.seed = seed
        
        # 初始化模块
        self.modules = self._initialize_modules()
        
        # 状态信息
        self.state = np.zeros(state_dim)
        self.action_space = list(range(num_modules))
        self.observation_space = (state_dim,)
        
        # 统计信息
        self.total_reward = 0
        self.rewards_history = []
        self.path_history = []
        
        # 性能指标
        self.throughput_history = []
        self.latency_history = []
        self.energy_history = []
        
        np.random.seed(seed)
    
    def _initialize_modules(self) -> List[Module]:
        """初始化模块"""
        modules = []
        module_types = [
            "computation", "storage", "communication", "ai_inference", 
            "data_processing", "cache", "backup", "monitoring"
        ]
        
        for i in range(self.num_modules):
            module_type = module_types[i % len(module_types)]
            
            # 根据模块类型设置不同的性能特征
            if module_type == "computation":
                capacity = np.random.uniform(80, 120)
                latency = np.random.uniform(0.1, 0.3)
                energy = np.random.uniform(2.0, 4.0)
                accuracy = np.random.uniform(0.9, 0.95)
                availability = np.random.uniform(0.95, 0.99)
            elif module_type == "storage":
                capacity = np.random.uniform(100, 150)
                latency = np.random.uniform(0.2, 0.5)
                energy = np.random.uniform(1.0, 2.0)
                accuracy = np.random.uniform(0.98, 0.99)
                availability = np.random.uniform(0.99, 1.0)
            elif module_type == "communication":
                capacity = np.random.uniform(60, 100)
                latency = np.random.uniform(0.05, 0.2)
                energy = np.random.uniform(1.5, 3.0)
                accuracy = np.random.uniform(0.92, 0.97)
                availability = np.random.uniform(0.93, 0.98)
            elif module_type == "ai_inference":
                capacity = np.random.uniform(50, 80)
                latency = np.random.uniform(0.5, 1.0)
                energy = np.random.uniform(5.0, 8.0)
                accuracy = np.random.uniform(0.85, 0.95)
                availability = np.random.uniform(0.90, 0.95)
            else:
                capacity = np.random.uniform(70, 110)
                latency = np.random.uniform(0.3, 0.7)
                energy = np.random.uniform(2.0, 3.5)
                accuracy = np.random.uniform(0.88, 0.94)
                availability = np.random.uniform(0.92, 0.97)
            
            modules.append(Module(
                id=f"module_{i}_{module_type}",
                capacity=capacity,
                load=0.0,
                latency=latency,
                energy=energy,
                accuracy=accuracy,
                availability=availability
            ))
        
        return modules
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        self.total_reward = 0
        self.rewards_history.clear()
        self.path_history.clear()
        
        # 重置模块负载
        for module in self.modules:
            module.load = 0.0
        
        # 生成初始状态
        self.state = self._get_state()
        return self.state.copy()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作"""
        if action not in self.action_space:
            raise ValueError(f"Invalid action: {action}")
        
        self.current_step += 1
        selected_module = self.modules[action]
        
        # 模拟请求处理
        reward = self._process_request(selected_module)
        
        # 更新模块负载
        self._update_module_loads()
        
        # 更新状态
        self.state = self._get_state()
        
        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        # 记录历史
        self.path_history.append(action)
        self.rewards_history.append(reward)
        self.total_reward += reward
        
        # 记录性能指标
        self.throughput_history.append(1.0)  # 简化为每步一个请求
        self.latency_history.append(selected_module.latency)
        self.energy_history.append(selected_module.energy)
        
        info = {
            'selected_module': selected_module.id,
            'module_score': selected_module.score,
            'module_utilization': selected_module.utilization,
            'step': self.current_step,
            'total_reward': self.total_reward
        }
        
        return self.state.copy(), reward, done, info
    
    def _process_request(self, module: Module) -> float:
        """处理请求并计算奖励"""
        # 基本奖励基于模块评分
        base_reward = module.score
        
        # 利用率惩罚
        if module.utilization > 0.9:
            utilization_penalty = (module.utilization - 0.9) * 5
            base_reward -= utilization_penalty
        
        # 负载均衡奖励
        avg_utilization = np.mean([m.utilization for m in self.modules])
        balance_bonus = max(0, 0.1 - abs(module.utilization - avg_utilization)) * 2
        
        # 随机扰动
        noise = np.random.normal(0, 0.1)
        
        final_reward = base_reward + balance_bonus + noise
        return max(0, final_reward)  # 确保奖励非负
    
    def _update_module_loads(self):
        """更新模块负载"""
        for module in self.modules:
            # 模拟负载变化
            load_change = np.random.normal(0, 5)
            module.load = max(0, module.load + load_change)
            
            # 模拟恢复过程
            module.load = max(0, module.load - np.random.uniform(0, 2))
            
            # 保持负载在合理范围内
            module.load = min(module.load, module.capacity * 1.2)
    
    def _get_state(self) -> np.ndarray:
        """获取环境状态"""
        state = np.zeros(self.state_dim)
        
        # 模块信息（前num_modules个维度）
        for i, module in enumerate(self.modules):
            if i < self.state_dim // 4:
                state[i * 4] = module.utilization
                state[i * 4 + 1] = module.latency
                state[i * 4 + 2] = module.energy
                state[i * 4 + 3] = module.accuracy
        
        # 全局统计信息
        if self.state_dim >= self.num_modules * 4 + 4:
            avg_utilization = np.mean([m.utilization for m in self.modules])
            max_utilization = np.max([m.utilization for m in self.modules])
            min_utilization = np.min([m.utilization for m in self.modules])
            utilization_variance = np.var([m.utilization for m in self.modules])
            
            state[self.num_modules * 4] = avg_utilization
            state[self.num_modules * 4 + 1] = max_utilization
            state[self.num_modules * 4 + 2] = min_utilization
            state[self.num_modules * 4 + 3] = utilization_variance
        
        # 时间特征
        if self.state_dim >= self.num_modules * 4 + 8:
            state[self.num_modules * 4 + 4] = self.current_step / self.max_steps
            state[self.num_modules * 4 + 5] = np.sin(2 * np.pi * self.current_step / 100)
            state[self.num_modules * 4 + 6] = np.cos(2 * np.pi * self.current_step / 100)
            state[self.num_modules * 4 + 7] = len(self.path_history)
        
        return state
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """可视化环境"""
        if mode == 'human':
            self._plot_environment()
        elif mode == 'rgb_array':
            return self._get_rgb_array()
        return None
    
    def _plot_environment(self):
        """绘制环境状态"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 利用率分布
        utilization_values = [m.utilization for m in self.modules]
        axes[0, 0].bar(range(len(self.modules)), utilization_values)
        axes[0, 0].set_title('模块利用率')
        axes[0, 0].set_ylabel('利用率')
        axes[0, 0].set_xlabel('模块ID')
        
        # 奖励历史
        if self.rewards_history:
            axes[0, 1].plot(self.rewards_history)
            axes[0, 1].set_title('奖励历史')
            axes[0, 1].set_ylabel('奖励')
            axes[0, 1].set_xlabel('步骤')
        
        # 路径选择历史
        if self.path_history:
            axes[1, 0].hist(self.path_history, bins=self.num_modules)
            axes[1, 0].set_title('模块选择分布')
            axes[1, 0].set_ylabel('选择次数')
            axes[1, 0].set_xlabel('模块ID')
        
        # 性能指标
        if self.throughput_history:
            x = range(len(self.throughput_history))
            axes[1, 1].plot(x, self.throughput_history, label='吞吐量')
            axes[1, 1].plot(x, self.latency_history, label='延迟')
            axes[1, 1].plot(x, np.array(self.energy_history) / 10, label='能耗/10')
            axes[1, 1].set_title('性能指标')
            axes[1, 1].set_ylabel('值')
            axes[1, 1].set_xlabel('步骤')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def _get_rgb_array(self) -> np.ndarray:
        """获取RGB数组（用于保存）"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        utilization_values = [m.utilization for m in self.modules]
        colors = ['red' if u > 0.9 else 'orange' if u > 0.7 else 'green' 
                 for u in utilization_values]
        
        ax.bar(range(len(self.modules)), utilization_values, color=colors)
        ax.set_title('当前模块利用率')
        ax.set_ylabel('利用率')
        ax.set_xlabel('模块ID')
        
        # 转换为RGB数组
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        return {
            'total_reward': self.total_reward,
            'avg_reward': np.mean(self.rewards_history) if self.rewards_history else 0,
            'avg_utilization': np.mean([m.utilization for m in self.modules]),
            'max_utilization': np.max([m.utilization for m in self.modules]),
            'utilization_variance': np.var([m.utilization for m in self.modules]),
            'avg_latency': np.mean([m.latency for m in self.modules]),
            'avg_energy': np.mean([m.energy for m in self.modules]),
            'avg_accuracy': np.mean([m.accuracy for m in self.modules]),
            'throughput': np.mean(self.throughput_history) if self.throughput_history else 0,
            'path_diversity': len(set(self.path_history)) / len(self.path_history) if self.path_history else 0
        }
    
    def get_module_status(self) -> Dict:
        """获取模块状态信息"""
        status = {}
        for i, module in enumerate(self.modules):
            status[module.id] = {
                'utilization': module.utilization,
                'load': module.load,
                'capacity': module.capacity,
                'latency': module.latency,
                'energy': module.energy,
                'accuracy': module.accuracy,
                'availability': module.availability,
                'score': module.score
            }
        return status
    
    def simulate_workload_pattern(self, pattern: str = 'burst'):
        """模拟不同的工作负载模式"""
        if pattern == 'burst':
            # 突发负载
            burst_size = np.random.randint(5, 20)
            burst_modules = np.random.choice(self.num_modules, burst_size, replace=False)
            for module_idx in burst_modules:
                self.modules[module_idx].load += np.random.uniform(20, 40)
        elif pattern == 'gradual':
            # 渐进负载
            for module in self.modules:
                module.load += np.random.uniform(1, 5)
        elif pattern == 'balanced':
            # 均衡负载
            for module in self.modules:
                module.load += np.random.uniform(-2, 2)
        
        # 确保负载在合理范围内
        for module in self.modules:
            module.load = max(0, min(module.load, module.capacity * 1.2))