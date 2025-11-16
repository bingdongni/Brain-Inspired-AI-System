"""
多智能体路由协作系统
实现多个智能体协作的动态路由策略
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import random
import asyncio


class Agent(nn.Module):
    """单个智能体网络"""
    
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


class CommunicationProtocol:
    """智能体间通信协议"""
    
    def __init__(self, max_agents: int):
        self.max_agents = max_agents
        self.agent_states = {}
        self.message_queue = deque(maxlen=1000)
        self.collaboration_weights = {}
        
    def register_agent(self, agent_id: str):
        """注册智能体"""
        self.agent_states[agent_id] = {
            'state': None,
            'q_values': None,
            'timestamp': 0,
            'reliability': 1.0
        }
        self.collaboration_weights[agent_id] = 1.0
        
    def update_agent_state(self, agent_id: str, state: np.ndarray, q_values: np.ndarray):
        """更新智能体状态"""
        if agent_id in self.agent_states:
            self.agent_states[agent_id].update({
                'state': state,
                'q_values': q_values,
                'timestamp': self.agent_states[agent_id]['timestamp'] + 1
            })
    
    def send_message(self, sender_id: str, receiver_id: str, message_type: str, data: Any):
        """发送消息"""
        message = {
            'sender': sender_id,
            'receiver': receiver_id,
            'type': message_type,
            'data': data,
            'timestamp': len(self.message_queue)
        }
        self.message_queue.append(message)
        
    def get_global_state(self) -> Dict:
        """获取全局状态信息"""
        return {
            'agent_states': self.agent_states,
            'active_agents': len([a for a in self.agent_states.values() if a['timestamp'] > 0]),
            'message_count': len(self.message_queue)
        }
    
    def calculate_collaboration_weights(self) -> Dict[str, float]:
        """计算协作权重"""
        active_agents = [aid for aid, state in self.agent_states.items() 
                        if state['timestamp'] > 0]
        
        if not active_agents:
            return {}
        
        # 基于可靠性和活跃度的权重计算
        total_weight = 0
        for agent_id in active_agents:
            agent = self.agent_states[agent_id]
            weight = agent['reliability'] * (1 + agent['timestamp'] * 0.1)
            self.collaboration_weights[agent_id] = weight
            total_weight += weight
        
        # 归一化权重
        if total_weight > 0:
            for agent_id in active_agents:
                self.collaboration_weights[agent_id] /= total_weight
        
        return self.collaboration_weights.copy()


class MultiAgentRouter:
    """多智能体协作路由系统"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 num_agents: int = 4,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.device = device
        
        # 初始化智能体
        self.agents = {}
        self.optimizers = {}
        
        for i in range(num_agents):
            agent_id = f"agent_{i}"
            self.agents[agent_id] = Agent(state_dim, action_dim).to(device)
            self.optimizers[agent_id] = optim.Adam(
                self.agents[agent_id].parameters(), lr=learning_rate
            )
        
        # 通信协议
        self.comm_protocol = CommunicationProtocol(num_agents)
        for i in range(num_agents):
            self.comm_protocol.register_agent(f"agent_{i}")
        
        # 全局经验回放
        self.global_memory = deque(maxlen=20000)
        self.agent_memories = {f"agent_{i}": deque(maxlen=5000) for i in range(num_agents)}
        
        # 协作机制
        self.collaboration_rate = 0.1
        self.knowledge_sharing_frequency = 10
        self.global_target_update_freq = 100
        
        # 训练统计
        self.training_steps = 0
        self.collaboration_stats = []
        
    def select_action(self, state: np.ndarray, agent_id: str, training: bool = True) -> int:
        """单个智能体选择动作"""
        if agent_id not in self.agents:
            return random.randint(0, self.action_dim - 1)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.agents[agent_id](state_tensor)
            
        # 协作决策：融合全局信息
        if training:
            collaborative_q = self._apply_collaboration(agent_id, q_values)
            action = torch.argmax(collaborative_q).item()
        else:
            action = torch.argmax(q_values).item()
        
        # 更新通信状态
        self.comm_protocol.update_agent_state(agent_id, state, q_values.cpu().numpy().squeeze())
        
        return action
    
    def _apply_collaboration(self, agent_id: str, local_q_values: torch.Tensor) -> torch.Tensor:
        """应用协作机制"""
        # 获取协作权重
        collab_weights = self.comm_protocol.calculate_collaboration_weights()
        
        if agent_id not in collab_weights:
            return local_q_values
        
        # 融合其他智能体的Q值
        collaborative_q = local_q_values.clone()
        
        for other_agent_id, weight in collab_weights.items():
            if other_agent_id != agent_id and other_agent_id in self.agents:
                other_state = self.comm_protocol.agent_states[other_agent_id]
                if other_state['state'] is not None:
                    # 简化的Q值融合（实际应用中需要更复杂的相似度计算）
                    collaborative_q += weight * 0.1 * torch.randn_like(collaborative_q)
        
        return collaborative_q
    
    def store_experience(self, agent_id: str, state, action, reward, next_state, done):
        """存储经验"""
        experience = (state, action, reward, next_state, done, agent_id)
        self.global_memory.append(experience)
        self.agent_memories[agent_id].append(experience)
    
    async def knowledge_sharing(self):
        """知识共享过程"""
        # 简化的知识共享实现
        global_state = self.comm_protocol.get_global_state()
        
        # 基于性能的知识共享
        agent_performances = {}
        for agent_id, memory in self.agent_memories.items():
            # 简化的性能评估
            agent_performances[agent_id] = len(memory)
        
        # 识别表现最好的智能体
        best_agent = max(agent_performances.keys(), key=lambda x: agent_performances[x])
        
        # 广播最佳策略（简化实现）
        if best_agent != "agent_0":  # 避免与自己比较
            for agent_id, agent in self.agents.items():
                if agent_id != best_agent:
                    # 简化的参数同步
                    with torch.no_grad():
                        for param, best_param in zip(agent.parameters(), 
                                                   self.agents[best_agent].parameters()):
                            param.copy_(param * (1 - self.collaboration_rate) + 
                                      best_param * self.collaboration_rate)
    
    def train_step(self, batch_size: int = 32):
        """执行一步训练"""
        if len(self.global_memory) < batch_size:
            return
        
        # 采样批次经验
        batch = random.sample(self.global_memory, batch_size)
        states = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        actions = torch.LongTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        dones = torch.BoolTensor([t[4] for t in batch]).to(self.device)
        agent_ids = [t[5] for t in batch]
        
        # 按智能体分组
        agent_groups = defaultdict(list)
        for i, agent_id in enumerate(agent_ids):
            agent_groups[agent_id].append(i)
        
        total_loss = 0
        
        # 为每个智能体计算损失
        for agent_id, indices in agent_groups.items():
            if agent_id not in self.agents:
                continue
                
            agent_group_states = states[indices]
            agent_group_actions = actions[indices]
            agent_group_rewards = rewards[indices]
            agent_group_next_states = next_states[indices]
            agent_group_dones = dones[indices]
            
            # 当前Q值
            current_q_values = self.agents[agent_id](agent_group_states).gather(1, agent_group_actions.unsqueeze(1))
            
            # 目标Q值
            with torch.no_grad():
                # 使用所有智能体的最大Q值作为目标
                max_q_values = torch.zeros_like(current_q_values)
                for other_agent_id, other_agent in self.agents.items():
                    other_q = other_agent(agent_group_next_states).max(1)[0].unsqueeze(1)
                    max_q_values = torch.max(max_q_values, other_q)
                
                target_q_values = agent_group_rewards.unsqueeze(1) + self.gamma * max_q_values * (~agent_group_dones).unsqueeze(1)
            
            # 损失函数
            loss = F.mse_loss(current_q_values, target_q_values)
            
            # 更新智能体
            self.optimizers[agent_id].zero_grad()
            loss.backward()
            self.optimizers[agent_id].step()
            
            total_loss += loss.item()
        
        self.training_steps += 1
        
        # 定期进行知识共享
        if self.training_steps % self.knowledge_sharing_frequency == 0:
            asyncio.create_task(self.knowledge_sharing())
        
        # 记录协作统计
        self.collaboration_stats.append(total_loss / len(agent_groups))
    
    def get_collaborative_decision(self, state: np.ndarray) -> int:
        """获取协作决策"""
        # 获取所有智能体的决策
        agent_decisions = {}
        for agent_id in self.agents:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.agents[agent_id](state_tensor)
            agent_decisions[agent_id] = torch.argmax(q_values).item()
        
        # 使用多数投票
        decision_counts = {}
        for decision in agent_decisions.values():
            decision_counts[decision] = decision_counts.get(decision, 0) + 1
        
        best_decision = max(decision_counts.keys(), key=lambda x: decision_counts[x])
        return best_decision
    
    def get_global_q_values(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """获取所有智能体的Q值"""
        global_q_values = {}
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        for agent_id, agent in self.agents.items():
            with torch.no_grad():
                q_values = agent(state_tensor).cpu().numpy().squeeze()
                global_q_values[agent_id] = q_values
        
        return global_q_values
    
    def save_models(self, filepath: str):
        """保存所有智能体模型"""
        model_data = {}
        for agent_id, agent in self.agents.items():
            model_data[agent_id] = {
                'state_dict': agent.state_dict(),
                'optimizer_state_dict': self.optimizers[agent_id].state_dict()
            }
        
        torch.save({
            'agents': model_data,
            'training_steps': self.training_steps,
            'collaboration_stats': self.collaboration_stats
        }, filepath)
    
    def load_models(self, filepath: str):
        """加载所有智能体模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        model_data = checkpoint['agents']
        
        for agent_id, agent_data in model_data.items():
            if agent_id in self.agents:
                self.agents[agent_id].load_state_dict(agent_data['state_dict'])
                self.optimizers[agent_id].load_state_dict(agent_data['optimizer_state_dict'])
        
        self.training_steps = checkpoint['training_steps']
        self.collaboration_stats = checkpoint['collaboration_stats']
    
    def get_statistics(self) -> Dict:
        """获取训练统计信息"""
        return {
            'training_steps': self.training_steps,
            'num_agents': self.num_agents,
            'global_memory_size': len(self.global_memory),
            'avg_collaboration_loss': np.mean(self.collaboration_stats[-100:]) if self.collaboration_stats else 0,
            'active_agents': len([a for a in self.comm_protocol.agent_states.values() 
                                if a['timestamp'] > 0])
        }