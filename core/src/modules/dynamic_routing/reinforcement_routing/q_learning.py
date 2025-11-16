"""
Q-Learning路由策略
实现基于表格和深度Q学习的动态模块选择
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from collections import deque, defaultdict
import random


class QNetwork(nn.Module):
    """深度Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        q_values = self.fc3(x)
        return q_values


class QLearningRouter:
    """基于Q-Learning的智能路由选择器"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 device: str = 'cpu',
                 use_deep_q: bool = True):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = device
        self.use_deep_q = use_deep_q
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=memory_size)
        self.states_memory = deque(maxlen=memory_size)
        
        if use_deep_q:
            # 深度Q网络
            self.q_network = QNetwork(state_dim, action_dim).to(device)
            self.target_network = QNetwork(state_dim, action_dim).to(device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.update_target_network()
        else:
            # 表格Q学习
            self.q_table = defaultdict(lambda: np.zeros(action_dim))
            
        # 训练统计
        self.training_steps = 0
        self.episode_rewards = []
        self.losses = []
        
    def update_target_network(self):
        """更新目标网络（用于深度Q学习）"""
        if self.use_deep_q:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作（模块路径）"""
        if training and random.random() < self.epsilon:
            # 探索：随机选择
            return random.randint(0, self.action_dim - 1)
        
        if self.use_deep_q:
            # 利用：选择Q值最大的动作
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
        else:
            # 表格Q学习的利用
            state_key = tuple(state)
            return np.argmax(self.q_table[state_key])
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        if self.use_deep_q:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.states_memory.append(state)
            self.q_table[tuple(state)][action] += reward
    
    def train_step(self, batch_size: int = 32):
        """执行一步训练"""
        if self.use_deep_q:
            self._train_deep_q(batch_size)
        else:
            self._train_tabular_q()
    
    def _train_deep_q(self, batch_size: int):
        """深度Q网络训练"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        dones = torch.BoolTensor([t[4] for t in batch]).to(self.device)
        
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 下一个Q值（目标网络）
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + self.gamma * next_q_values * (~dones)
        
        # 损失函数
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.training_steps += 1
        
        # 更新探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # 定期更新目标网络
        if self.training_steps % 1000 == 0:
            self.update_target_network()
    
    def _train_tabular_q(self):
        """表格Q学习训练"""
        if len(self.states_memory) < 2:
            return
            
        # 随机选择两个连续的状态
        if len(self.states_memory) >= 2:
            state = self.states_memory[-2]
            next_state = self.states_memory[-1]
            
            # 查找最佳动作
            best_action = np.argmax(self.q_table[tuple(next_state)])
            
            # 更新Q值
            state_key = tuple(state)
            current_q = self.q_table[state_key]
            reward = self._get_last_reward()
            
            # Q学习更新规则
            target = reward + self.gamma * self.q_table[tuple(next_state)][best_action]
            current_q = current_q + 0.1 * (target - current_q)
            
            self.training_steps += 1
    
    def _get_last_reward(self) -> float:
        """获取最后一次的奖励值（简化实现）"""
        return 1.0  # 默认奖励
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """获取Q值分布"""
        if self.use_deep_q:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor).cpu().numpy().squeeze()
            return q_values
        else:
            state_key = tuple(state)
            return self.q_table[state_key]
    
    def save_model(self, filepath: str):
        """保存模型"""
        if self.use_deep_q:
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_steps': self.training_steps,
                'epsilon': self.epsilon
            }, filepath)
        else:
            # 保存Q表
            np.save(filepath, dict(self.q_table))
    
    def load_model(self, filepath: str):
        """加载模型"""
        if self.use_deep_q:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_steps = checkpoint['training_steps']
            self.epsilon = checkpoint['epsilon']
        else:
            # 加载Q表
            q_data = np.load(filepath, allow_pickle=True).item()
            self.q_table = defaultdict(lambda: np.zeros(self.action_dim), q_data)
    
    def get_statistics(self) -> Dict:
        """获取训练统计信息"""
        stats = {
            'training_steps': self.training_steps,
            'epsilon': self.epsilon,
            'memory_size': len(self.memory) if self.use_deep_q else len(self.states_memory)
        }
        
        if self.use_deep_q:
            stats['avg_loss'] = np.mean(self.losses[-100:]) if self.losses else 0
        else:
            stats['q_table_size'] = len(self.q_table)
            
        return stats
    
    def get_action_value(self, state: np.ndarray, action: int) -> float:
        """获取特定状态-动作对的Q值"""
        if self.use_deep_q:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values[0, action].item()
        else:
            state_key = tuple(state)
            return self.q_table[state_key][action]