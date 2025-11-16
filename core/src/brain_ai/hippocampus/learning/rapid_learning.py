#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
快速学习机制
===========

实现海马体的快速学习能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class RapidLearningMechanism:
    """快速学习机制"""
    
    def __init__(self, learning_rate: float = 0.1, plasticity: float = 0.9):
        """
        初始化快速学习机制
        
        Args:
            learning_rate: 学习率
            plasticity: 可塑性参数
        """
        self.learning_rate = learning_rate
        self.plasticity = plasticity
        self.synaptic_traces = {}  # 突触痕迹
        self.learning_history = []
    
    def learn_episode(self, state: torch.Tensor, action: torch.Tensor, 
                     reward: float, next_state: torch.Tensor) -> Dict[str, Any]:
        """
        学习一个情景
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 奖励信号
            next_state: 下一个状态
            
        Returns:
            学习结果
        """
        # 计算学习信号
        learning_signal = self._compute_learning_signal(reward)
        
        # 更新突触权重
        weight_changes = self._update_synaptic_weights(
            state, action, learning_signal
        )
        
        # 记录学习历史
        self.learning_history.append({
            'state': state,
            'action': action,
            'reward': reward,
            'learning_signal': learning_signal,
            'weight_changes': weight_changes
        })
        
        return {
            'learning_signal': learning_signal,
            'weight_changes': weight_changes,
            'episodic_strength': self._compute_episodic_strength(reward)
        }
    
    def _compute_learning_signal(self, reward: float) -> float:
        """计算学习信号"""
        # 根据奖励信号计算学习强度
        base_signal = max(0, reward)  # 只对正奖励进行学习
        
        # 应用时间折扣
        time_factor = 1.0  # 可以根据时间进行调整
        
        return self.learning_rate * base_signal * time_factor
    
    def _update_synaptic_weights(self, state: torch.Tensor, 
                                action: torch.Tensor, 
                                learning_signal: float) -> Dict[str, float]:
        """更新突触权重"""
        changes = {}
        
        # 简化的权重更新（实际应用中应使用更复杂的突触可塑性模型）
        state_norm = torch.norm(state).item()
        action_norm = torch.norm(action).item()
        
        # 计算权重变化
        weight_change = learning_signal * state_norm * action_norm * self.plasticity
        changes['magnitude'] = weight_change
        
        # 应用到突触痕迹
        connection_id = self._get_connection_id(state, action)
        if connection_id not in self.synaptic_traces:
            self.synaptic_traces[connection_id] = 0.0
        
        self.synaptic_traces[connection_id] += weight_change
        
        # 限制权重变化范围
        self.synaptic_traces[connection_id] = np.clip(
            self.synaptic_traces[connection_id], -10.0, 10.0
        )
        
        return changes
    
    def _get_connection_id(self, state: torch.Tensor, action: torch.Tensor) -> str:
        """生成连接ID"""
        state_hash = hash(tuple(state.detach().cpu().numpy().flatten()))
        action_hash = hash(tuple(action.detach().cpu().numpy().flatten()))
        return f"{state_hash}_{action_hash}"
    
    def _compute_episodic_strength(self, reward: float) -> float:
        """计算情景强度"""
        # 根据奖励计算情景的持久性强度
        base_strength = 1.0 + max(0, reward)
        reward_boost = 0.1 * reward if reward > 0 else 0
        
        return base_strength + reward_boost
    
    def retrieve_relevant_memories(self, query_state: torch.Tensor, 
                                  threshold: float = 0.5) -> List[Tuple[str, float]]:
        """检索相关记忆"""
        relevant_memories = []
        
        for connection_id, trace_strength in self.synaptic_traces.items():
            if trace_strength > threshold:
                # 计算与查询状态的相似度
                similarity = self._compute_state_similarity(query_state, connection_id)
                relevant_memories.append((connection_id, similarity))
        
        # 按相似度排序
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return relevant_memories
    
    def _compute_state_similarity(self, state: torch.Tensor, 
                                 connection_id: str) -> float:
        """计算状态相似度（简化实现）"""
        # 这里应该实现更复杂的状态相似度计算
        # 暂时返回随机值
        return np.random.random()
    
    def consolidate_learning(self, consolidation_threshold: float = 0.8) -> int:
        """巩固学习"""
        consolidated_count = 0
        
        # 强化高强度的突触连接
        for connection_id, trace_strength in self.synaptic_traces.items():
            if trace_strength > consolidation_threshold:
                self.synaptic_traces[connection_id] *= 1.2
                consolidated_count += 1
        
        return consolidated_count
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        if not self.learning_history:
            return {'total_episodes': 0}
        
        rewards = [episode['reward'] for episode in self.learning_history]
        learning_signals = [episode['learning_signal'] for episode in self.learning_history]
        
        return {
            'total_episodes': len(self.learning_history),
            'average_reward': np.mean(rewards),
            'max_reward': np.max(rewards),
            'average_learning_signal': np.mean(learning_signals),
            'max_learning_signal': np.max(learning_signals),
            'active_connections': len(self.synaptic_traces),
            'average_connection_strength': np.mean(list(self.synaptic_traces.values())),
            'max_connection_strength': np.max(list(self.synaptic_traces.values()))
        }
    
    def reset_learning(self):
        """重置学习状态"""
        self.synaptic_traces.clear()
        self.learning_history.clear()

if __name__ == "__main__":
    # 测试快速学习机制
    learning_mechanism = RapidLearningMechanism()
    
    # 模拟学习过程
    for i in range(10):
        state = torch.randn(10)
        action = torch.randn(5)
        reward = np.random.uniform(-1, 1)
        next_state = torch.randn(10)
        
        result = learning_mechanism.learn_episode(state, action, reward, next_state)
        print(f"Episode {i+1}: reward={reward:.3f}, signal={result['learning_signal']:.3f}")
    
    # 获取统计信息
    stats = learning_mechanism.get_learning_statistics()
    print(f"\n学习统计: {stats}")
    
    # 测试记忆检索
    query_state = torch.randn(10)
    relevant_memories = learning_mechanism.retrieve_relevant_memories(query_state)
    print(f"\n相关记忆: {len(relevant_memories)} 个")
    
    # 巩固学习
    consolidated = learning_mechanism.consolidate_learning()
    print(f"巩固了 {consolidated} 个连接")