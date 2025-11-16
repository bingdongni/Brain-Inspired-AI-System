"""
Actor-Critic路由策略
实现智能体Actor-Critic框架用于动态模块选择
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import deque
import random


class ActorNetwork(nn.Module):
    """Actor网络：学习策略π(a|s)"""
    
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
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs


class CriticNetwork(nn.Module):
    """Critic网络：学习价值函数V(s)"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        value = self.fc3(x)
        return value


class ActorCriticRouter:
    """基于Actor-Critic的智能路由选择器"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.device = device
        
        # 初始化网络
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=10000)
        
        # 训练统计
        self.training_steps = 0
        self.losses = {'actor': [], 'critic': []}
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作（模块路径）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if training:
                action_probs = self.actor(state)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
            else:
                action_probs = self.actor(state)
                action = torch.argmax(action_probs)
                
        return action.item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size: int = 32):
        """执行一步训练"""
        if len(self.memory) < batch_size:
            return
        
        # 采样批次数据
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        dones = torch.BoolTensor([t[4] for t in batch]).to(self.device)
        
        # 计算当前价值
        current_values = self.critic(states).squeeze()
        
        # 计算下一个价值（使用GAE）
        next_values = self.critic(next_states).squeeze()
        target_values = rewards + self.gamma * next_values * (~dones)
        
        # Critic损失
        critic_loss = F.mse_loss(current_values, target_values.detach())
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 计算优势
        advantages = target_values - current_values
        
        # Actor损失
        action_probs = self.actor(states)
        action_dist = torch.distributions.Categorical(action_probs)
        log_probs = action_dist.log_prob(actions)
        
        actor_loss = -(log_probs * advantages.detach()).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 记录损失
        self.losses['actor'].append(actor_loss.item())
        self.losses['critic'].append(critic_loss.item())
        self.training_steps += 1
    
    def get_routing_probabilities(self, state: np.ndarray) -> np.ndarray:
        """获取路由概率分布"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.actor(state).cpu().numpy().squeeze()
        return action_probs
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_steps': self.training_steps
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_steps = checkpoint['training_steps']
    
    def get_statistics(self) -> Dict:
        """获取训练统计信息"""
        return {
            'training_steps': self.training_steps,
            'memory_size': len(self.memory),
            'avg_actor_loss': np.mean(self.losses['actor'][-100:]) if self.losses['actor'] else 0,
            'avg_critic_loss': np.mean(self.losses['critic'][-100:]) if self.losses['critic'] else 0
        }