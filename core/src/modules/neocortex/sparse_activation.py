"""
稀疏激活和权重巩固机制
======================

实现新皮层的稀疏激活和慢速权重巩固机制，包括：
- 稀疏激活：模拟新皮层中稀疏的神经激活模式
- 权重巩固：实现长期记忆的权重巩固
- 印记细胞：形成长期记忆的印记细胞
- 记忆巩固：睡眠和清醒状态下的记忆巩固

基于论文：
- "Complementary learning systems" (McClelland et al., 1995)
- "Synaptic tagging and capture" (Frey & Morris, 1997)
- "Memory consolidation" (Diekelmann & Born, 2010)
- "Engram cells" (Tonegawa et al., 2015)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class CellType(Enum):
    """细胞类型"""
    REGULAR = "regular"           # 常规神经元
    ENGRAM = "engram"            # 印记细胞
    SILENT = "silent"            # 静默细胞
    INTERNEURON = "interneuron"  # 中间神经元


class MemoryState(Enum):
    """记忆状态"""
    ACTIVE = "active"            # 活跃状态
    LABILE = "labile"           # 易变状态
    CONSOLIDATED = "consolidated" # 巩固状态
    FORGOTTEN = "forgotten"     # 遗忘状态


@dataclass
class ConsolidationConfig:
    """巩固配置"""
    feature_dim: int                    # 特征维度
    
    # 稀疏激活参数
    sparsity_level: float = 0.95       # 稀疏性水平（95%神经元保持沉默）
    activation_threshold: float = 0.7  # 激活阈值
    adaptive_sparsity: bool = True     # 自适应稀疏性
    
    # 巩固参数
    consolidation_rate: float = 0.01   # 巩固速率
    forgetting_rate: float = 0.001     # 遗忘速率
    stability_threshold: float = 0.8   # 稳定性阈值
    
    # 印记细胞参数
    engram_formation_threshold: float = 0.9  # 印记形成阈值
    engram_strength_decay: float = 0.95      # 印记强度衰减
    engram_reactivation_threshold: float = 0.6  # 印记重激活阈值
    
    # 时间参数
    consolidation_delay: float = 20.0  # 巩固延迟（时间步）
    reactivation_interval: float = 100.0  # 重激活间隔
    sleep_consolidation_duration: float = 1000.0  # 睡眠巩固持续时间


class SparseActivation(nn.Module):
    """稀疏激活模块
    
    模拟新皮层中稀疏的神经激活模式。
    """
    
    def __init__(self, config: ConsolidationConfig, feature_dim: int):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        
        # 激活门控器
        self.activation_gater = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
        # 稀疏性调节器
        self.sparsity_regulator = SparsityRegulator(config)
        
        # 竞争机制
        self.competition_mechanism = CompetitionMechanism(config, feature_dim)
        
        # 统计信息
        self.register_buffer('activation_history', torch.zeros(100, feature_dim))
        self.register_buffer('history_index', torch.tensor(0))
        self.sparsity_stats = {
            'avg_sparsity': 0.0,
            'activation_count': 0,
            'sparsity_stability': 0.0
        }
    
    def forward(self, features: torch.Tensor, 
                attention: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        稀疏激活前向传播
        
        Args:
            features: 输入特征 [batch, feature_dim]
            attention: 注意力信号
            context: 上下文信息
            
        Returns:
            dict: 稀疏激活结果
        """
        batch_size = features.shape[0]
        
        # 门控激活
        gated_activations = self.activation_gater(features)
        
        # 应用注意力调制
        if attention is not None:
            gated_activations = gated_activations * attention
        
        # 竞争机制
        competitive_activations = self.competition_mechanism(gated_activations)
        
        # 稀疏性处理
        sparse_result = self.sparsity_regulator.apply_sparsity(
            competitive_activations, context
        )
        
        # 更新统计
        self._update_statistics(sparse_result['sparse_activations'])
        
        return {
            'sparse_activations': sparse_result['sparse_activations'],
            'sparsity_mask': sparse_result['sparsity_mask'],
            'activation_strength': sparse_result['activation_strength'],
            'competition_weights': competitive_activations,
            'sparsity_level': sparse_result['sparsity_level'],
            'actual_sparsity': sparse_result['actual_sparsity'],
            'sparse_efficiency': sparse_result['sparse_efficiency']
        }
    
    def _update_statistics(self, activations: torch.Tensor):
        """更新稀疏性统计"""
        # 计算实际稀疏性
        non_zero_count = (activations > self.config.activation_threshold).sum().item()
        total_count = activations.numel()
        actual_sparsity = 1.0 - non_zero_count / total_count
        
        # 更新历史记录
        idx = int(self.history_index % self.activation_history.shape[0])
        if len(activations.shape) == 2:
            self.activation_history[idx] = activations.mean(dim=0)
        else:
            self.activation_history[idx] = activations
        
        self.history_index = (self.history_index + 1) % self.activation_history.shape[0]
        
        # 更新统计信息
        self.sparsity_stats['avg_sparsity'] = (
            (self.sparsity_stats['avg_sparsity'] * self.sparsity_stats['activation_count'] +
             actual_sparsity) / (self.sparsity_stats['activation_count'] + 1)
        )
        self.sparsity_stats['activation_count'] += 1


class SparsityRegulator(nn.Module):
    """稀疏性调节器"""
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__()
        self.config = config
        
        # 自适应阈值调节器
        self.threshold_adaptor = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
    
    def apply_sparsity(self, activations: torch.Tensor, 
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """应用稀疏性"""
        batch_size, feature_dim = activations.shape
        
        # 计算自适应阈值
        if self.config.adaptive_sparsity:
            activation_magnitude = activations.abs().mean()
            threshold_factor = self.threshold_adaptor(activation_magnitude.unsqueeze(-1))
            threshold = self.config.activation_threshold * threshold_factor.squeeze(-1)
        else:
            threshold = self.config.activation_threshold
        
        # 创建稀疏性掩码
        sparsity_mask = (activations > threshold).float()
        
        # 应用稀疏性
        sparse_activations = activations * sparsity_mask
        
        # 计算激活强度
        activation_strength = sparse_activations.sum(dim=1, keepdim=True)
        
        # 计算实际稀疏性
        non_zero_count = sparsity_mask.sum(dim=1)
        total_count = feature_dim
        actual_sparsity = 1.0 - non_zero_count.float() / total_count
        
        # 计算稀疏效率（激活强度/使用的神经元数量）
        sparse_efficiency = activation_strength / (non_zero_count.float().unsqueeze(-1) + 1e-8)
        sparse_efficiency = sparse_efficiency.mean()
        
        # 计算目标稀疏性
        target_sparsity = torch.tensor(self.config.sparsity_level)
        
        return {
            'sparse_activations': sparse_activations,
            'sparsity_mask': sparsity_mask,
            'activation_strength': activation_strength,
            'sparsity_level': target_sparsity,
            'actual_sparsity': actual_sparsity.mean(),
            'sparse_efficiency': sparse_efficiency
        }


class CompetitionMechanism(nn.Module):
    """竞争机制"""
    
    def __init__(self, config: ConsolidationConfig, feature_dim: int):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        
        # 竞争权重生成器
        self.competition_weights = nn.Parameter(torch.ones(feature_dim))
        
        # 侧向抑制
        self.lateral_inhibition = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        
        # 初始化侧向抑制权重（抑制性连接）
        with torch.no_grad():
            self.lateral_inhibition.weight.fill_(-0.1)
    
    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """应用竞争机制"""
        # 应用竞争权重
        weighted_activations = activations * self.competition_weights.unsqueeze(0)
        
        # 侧向抑制
        if len(weighted_activations.shape) == 2:
            # 添加通道维度
            activations_expanded = weighted_activations.unsqueeze(1)
            inhibited = self.lateral_inhibition(activations_expanded).squeeze(1)
        else:
            inhibited = self.lateral_inhibition(weighted_activations)
        
        # 归一化
        normalized = F.softmax(inhibited, dim=-1)
        
        return normalized


class WeightConsolidation(nn.Module):
    """权重巩固模块"""
    
    def __init__(self, config: ConsolidationConfig, weight_shape: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.weight_shape = weight_shape
        
        # 巩固网络
        self.consolidation_network = nn.Sequential(
            nn.Linear(weight_shape[0] * weight_shape[1], weight_shape[0] * weight_shape[1] // 2),
            nn.ReLU(),
            nn.Linear(weight_shape[0] * weight_shape[1] // 2, weight_shape[0] * weight_shape[1]),
            nn.Sigmoid()
        )
        
        # 稳定性评估器
        self.stability_evaluator = nn.Sequential(
            nn.Linear(weight_shape[0] * weight_shape[1], 1),
            nn.Sigmoid()
        )
        
        # 遗忘控制器
        self.forgetting_controller = ForgettingController(config)
        
        # 统计信息
        self.consolidation_history = []
        self.stability_scores = []
    
    def forward(self, current_weights: torch.Tensor, 
                weight_changes: torch.Tensor,
                consolidation_signals: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        权重巩固前向传播
        
        Args:
            current_weights: 当前权重
            weight_changes: 权重变化
            consolidation_signals: 巩固信号
            
        Returns:
            dict: 巩固结果
        """
        batch_size = current_weights.shape[0]
        
        # 展平权重用于处理
        current_weights_flat = current_weights.view(batch_size, -1)
        weight_changes_flat = weight_changes.view(batch_size, -1)
        
        # 计算巩固强度
        consolidation_strength = self._compute_consolidation_strength(
            current_weights_flat, weight_changes_flat, consolidation_signals
        )
        
        # 应用巩固网络
        consolidated_weights_flat = self.consolidation_network(
            current_weights_flat + weight_changes_flat * consolidation_strength
        )
        
        # 重塑为原始形状
        consolidated_weights = consolidated_weights_flat.view(current_weights.shape)
        
        # 评估稳定性
        stability_score = self.stability_evaluator(consolidated_weights_flat).mean()
        
        # 应用遗忘机制
        final_weights, forgetting_result = self.forgetting_controller(
            consolidated_weights, current_weights
        )
        
        # 更新统计
        self.consolidation_history.append({
            'consolidation_strength': consolidation_strength.mean().item(),
            'stability_score': stability_score.item(),
            'timestamp': len(self.consolidation_history)
        })
        self.stability_scores.append(stability_score.item())
        
        return {
            'consolidated_weights': final_weights,
            'consolidation_strength': consolidation_strength,
            'stability_score': stability_score,
            'weight_stability': stability_score,
            'memory_strength': self._compute_memory_strength(current_weights, final_weights),
            'forgetting_result': forgetting_result,
            'consolidation_quality': self._compute_consolidation_quality()
        }
    
    def _compute_consolidation_strength(self, current_weights: torch.Tensor,
                                      weight_changes: torch.Tensor,
                                      consolidation_signals: Optional[torch.Tensor]) -> torch.Tensor:
        """计算巩固强度"""
        # 基于权重变化幅度计算巩固强度
        change_magnitude = weight_changes.abs().mean(dim=1, keepdim=True)
        
        # 基于当前权重稳定性计算
        weight_stability = 1.0 / (1.0 + current_weights.var(dim=1, keepdim=True))
        
        # 结合信号
        if consolidation_signals is not None:
            signal_strength = consolidation_signals.mean(dim=1, keepdim=True)
        else:
            signal_strength = torch.ones_like(change_magnitude)
        
        # 综合计算巩固强度
        consolidation_strength = (
            change_magnitude * self.config.consolidation_rate +
            weight_stability * 0.1 +
            signal_strength * 0.1
        )
        
        # 限制在合理范围内
        consolidation_strength = torch.clamp(consolidation_strength, 0.0, 1.0)
        
        return consolidation_strength
    
    def _compute_memory_strength(self, original_weights: torch.Tensor, 
                               consolidated_weights: torch.Tensor) -> torch.Tensor:
        """计算记忆强度"""
        # 基于权重相似性计算记忆强度
        weight_similarity = F.cosine_similarity(
            original_weights.flatten(1), 
            consolidated_weights.flatten(1), 
            dim=1
        )
        return (weight_similarity + 1) / 2  # 归一化到[0,1]
    
    def _compute_consolidation_quality(self) -> float:
        """计算巩固质量"""
        if len(self.stability_scores) < 2:
            return 0.5
        
        # 基于稳定性分数的趋势和质量
        recent_stability = np.mean(self.stability_scores[-10:])
        stability_trend = self.stability_scores[-1] - self.stability_scores[-2] if len(self.stability_scores) > 1 else 0
        
        quality = 0.7 * recent_stability + 0.3 * max(0, stability_trend)
        return min(1.0, max(0.0, quality))


class EngramCell(nn.Module):
    """印记细胞
    
    负责形成和维持长期记忆的特殊细胞类型。
    """
    
    def __init__(self, config: ConsolidationConfig, cell_id: str):
        super().__init__()
        self.config = config
        self.cell_id = cell_id
        
        # 印记强度
        self.register_buffer('engram_strength', torch.tensor(0.0))
        
        # 形成时间
        self.register_buffer('formation_time', torch.tensor(0.0))
        
        # 最后激活时间
        self.register_buffer('last_activation_time', torch.tensor(-1.0))
        
        # 激活次数
        self.activation_count = 0
        
        # 关联细胞
        self.associated_cells = []
        
        # 印记网络
        self.engram_network = nn.Sequential(
            nn.Linear(128, 64),  # 假设输入维度为128
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 重激活阈值
        self.reactivation_threshold = config.engram_reactivation_threshold
        
        # 状态
        self.memory_state = MemoryState.ACTIVE
    
    def forward(self, input_signal: torch.Tensor, 
                context: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """印记细胞前向传播"""
        batch_size = input_signal.shape[0]
        
        # 计算激活强度
        activation = self.engram_network(input_signal.mean(dim=-1, keepdim=True))
        
        # 更新印记强度
        if activation > self.config.engram_formation_threshold:
            self._strengthen_engram(activation)
        
        # 更新激活状态
        current_activation = activation.mean()
        if current_activation > self.reactivation_threshold:
            self._reactivate_engram()
        
        # 评估记忆状态
        memory_state = self._assess_memory_state()
        
        return {
            'engram_activation': activation,
            'engram_strength': self.engram_strength,
            'memory_state': memory_state.value,
            'is_reactivated': current_activation > self.reactivation_threshold,
            'cell_id': self.cell_id
        }
    
    def _strengthen_engram(self, activation: torch.Tensor):
        """强化印记"""
        # 印记强度更新
        current_strength = self.engram_strength
        new_strength = current_strength + activation.mean() * self.config.consolidation_rate
        
        # 限制最大强度
        self.engram_strength = torch.clamp(new_strength, 0.0, 1.0)
        
        # 更新形成时间和激活次数
        if self.formation_time == 0.0:
            self.formation_time = torch.tensor(float(len(self.associated_cells)))
        
        self.activation_count += 1
    
    def _reactivate_engram(self):
        """重激活印记"""
        self.last_activation_time = torch.tensor(float(self.activation_count))
        
        # 重激活时稍微增强印记
        self.engram_strength = torch.clamp(
            self.engram_strength * self.config.engram_strength_decay + 0.01,
            0.0, 1.0
        )
        
        # 如果长期未激活，可能进入巩固状态
        if (self.activation_count - self.last_activation_time) > 1000:
            self.memory_state = MemoryState.CONSOLIDATED
    
    def _assess_memory_state(self) -> MemoryState:
        """评估记忆状态"""
        # 基于印记强度和激活频率评估状态
        if self.engram_strength < 0.1:
            return MemoryState.FORGOTTEN
        elif self.engram_strength < 0.5:
            return MemoryState.LABILE
        elif self.engram_strength > 0.8:
            return MemoryState.CONSOLIDATED
        else:
            return MemoryState.ACTIVE
    
    def associate_with(self, other_cell: 'EngramCell', strength: float = 0.5):
        """与其他印记细胞关联"""
        self.associated_cells.append((other_cell, strength))
        other_cell.associated_cells.append((self, strength))


class ConsolidationEngine(nn.Module):
    """巩固引擎
    
    整合稀疏激活、权重巩固和印记细胞的综合巩固系统。
    """
    
    def __init__(self, config: ConsolidationConfig, feature_dim: int, 
                 num_engram_cells: int = 100):
        super().__init__()
        self.config = config
        self.feature_dim = feature_dim
        
        # 核心组件
        self.sparse_activation = SparseActivation(config, feature_dim)
        self.engram_cells = nn.ModuleList([
            EngramCell(config, f'engram_{i}') for i in range(num_engram_cells)
        ])
        
        # 权重巩固器（简化版本，用于演示）
        self.weight_consolidators = nn.ModuleList([
            nn.Linear(feature_dim, feature_dim) for _ in range(3)
        ])
        
        # 巩固协调器
        self.consolidation_coordinator = ConsolidationCoordinator(config)
        
        # 睡眠模式模拟
        self.sleep_mode = False
        self.sleep_consolidation_active = False
        
        # 统计信息
        self.consolidation_stats = {
            'total_consolidations': 0,
            'successful_engrams': 0,
            'avg_engram_strength': 0.0,
            'memory_retention_rate': 0.0
        }
    
    def forward(self, features: torch.Tensor, 
                weight_changes: Optional[torch.Tensor] = None,
                context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        巩固引擎前向传播
        
        Args:
            features: 输入特征
            weight_changes: 权重变化（可选）
            context: 上下文信息
            
        Returns:
            dict: 巩固结果
        """
        # 稀疏激活处理
        sparse_result = self.sparse_activation(features, context=context)
        
        # 印记细胞处理
        engram_results = []
        for engram_cell in self.engram_cells:
            engram_result = engram_cell(features)
            engram_results.append(engram_result)
        
        # 权重巩固（如果提供了权重变化）
        consolidation_result = None
        if weight_changes is not None:
            # 简化的权重巩固
            current_weights = torch.randn_like(weight_changes) * 0.1
            for consolidator in self.weight_consolidators:
                result = consolidator(weight_changes)
                current_weights = current_weights + result * self.config.consolidation_rate
            consolidation_result = {'consolidated_weights': current_weights}
        
        # 协调巩固过程
        coordination_result = self.consolidation_coordinator.coordinate(
            sparse_result, engram_results, consolidation_result, context
        )
        
        # 更新统计
        self._update_statistics(sparse_result, engram_results, coordination_result)
        
        return {
            'sparse_activation': sparse_result,
            'engram_results': engram_results,
            'consolidation_result': consolidation_result,
            'coordination_result': coordination_result,
            'memory_state': self._get_global_memory_state(),
            'consolidation_summary': {
                'sparse_efficiency': sparse_result['sparse_efficiency'],
                'active_engram_cells': sum(1 for r in engram_results if r['is_reactivated']),
                'formed_engrams': sum(1 for r in engram_results if r['engram_strength'] > 0.5),
                'avg_engram_strength': torch.stack([r['engram_strength'] for r in engram_results]).mean().item(),
                'consolidation_quality': coordination_result.get('quality', 0.0)
            }
        }
    
    def _update_statistics(self, sparse_result: Dict, 
                          engram_results: List[Dict], coordination_result: Dict):
        """更新统计信息"""
        self.consolidation_stats['total_consolidations'] += 1
        
        # 成功印记数量
        successful_engrams = sum(1 for r in engram_results if r['engram_strength'] > 0.5)
        self.consolidation_stats['successful_engrams'] += successful_engrams
        
        # 平均印记强度
        avg_strength = torch.stack([r['engram_strength'] for r in engram_results]).mean().item()
        self.consolidation_stats['avg_engram_strength'] = (
            (self.consolidation_stats['avg_engram_strength'] * (self.consolidation_stats['total_consolidations'] - 1) +
             avg_strength) / self.consolidation_stats['total_consolidations']
        )
        
        # 记忆保留率
        retention_rate = sparse_result['sparse_efficiency'] * (successful_engrams / len(engram_results))
        self.consolidation_stats['memory_retention_rate'] = (
            (self.consolidation_stats['memory_retention_rate'] * (self.consolidation_stats['total_consolidations'] - 1) +
             retention_rate) / self.consolidation_stats['total_consolidations']
        )
    
    def _get_global_memory_state(self) -> Dict[str, Any]:
        """获取全局记忆状态"""
        return {
            'active_cells': sum(1 for cell in self.engram_cells 
                              if cell.memory_state == MemoryState.ACTIVE),
            'consolidated_cells': sum(1 for cell in self.engram_cells 
                                    if cell.memory_state == MemoryState.CONSOLIDATED),
            'labil_cells': sum(1 for cell in self.engram_cells 
                             if cell.memory_state == MemoryState.LABILE),
            'forgotten_cells': sum(1 for cell in self.engram_cells 
                                 if cell.memory_state == MemoryState.FORGOTTEN),
            'total_cells': len(self.engram_cells),
            'avg_strength': torch.stack([cell.engram_strength for cell in self.engram_cells]).mean().item()
        }
    
    def enter_sleep_mode(self):
        """进入睡眠模式"""
        self.sleep_mode = True
        self.sleep_consolidation_active = True
        print("进入睡眠巩固模式")
    
    def exit_sleep_mode(self):
        """退出睡眠模式"""
        self.sleep_mode = False
        self.sleep_consolidation_active = False
        print("退出睡眠巩固模式")


class ConsolidationCoordinator(nn.Module):
    """巩固协调器"""
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__()
        self.config = config
        
        # 协调网络
        self.coordination_network = nn.Sequential(
            nn.Linear(200, 100),  # 假设输入为200维（稀疏激活+印记结果）
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )
    
    def coordinate(self, sparse_result: Dict, engram_results: List[Dict],
                  consolidation_result: Optional[Dict], 
                  context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """协调巩固过程"""
        # 准备协调输入
        coordination_input = self._prepare_coordination_input(sparse_result, engram_results)
        
        # 计算协调权重
        coordination_weights = self.coordination_network(coordination_input)
        
        # 生成协调信号
        coordination_signals = {
            'sparse_weights': coordination_weights[:50],  # 假设前50维用于稀疏激活
            'engram_weights': coordination_weights[50:150],  # 中间100维用于印记
            'consolidation_weights': coordination_weights[150:]  # 剩余用于巩固
        }
        
        # 计算巩固质量
        quality = self._compute_coordination_quality(
            sparse_result, engram_results, consolidation_result
        )
        
        return {
            'coordination_weights': coordination_weights,
            'coordination_signals': coordination_signals,
            'quality': quality,
            'coordination_active': True
        }
    
    def _prepare_coordination_input(self, sparse_result: Dict, 
                                  engram_results: List[Dict]) -> torch.Tensor:
        """准备协调输入"""
        # 简化的输入准备
        sparse_features = sparse_result['sparse_activations'].flatten()
        engram_features = torch.stack([r['engram_strength'] for r in engram_results])
        
        # 拼接特征
        combined_features = torch.cat([sparse_features, engram_features])
        
        # 调整到期望维度
        if combined_features.shape[0] > 200:
            combined_features = combined_features[:200]
        elif combined_features.shape[0] < 200:
            padding = torch.zeros(200 - combined_features.shape[0])
            combined_features = torch.cat([combined_features, padding])
        
        return combined_features.unsqueeze(0)
    
    def _compute_coordination_quality(self, sparse_result: Dict,
                                    engram_results: List[Dict],
                                    consolidation_result: Optional[Dict]) -> float:
        """计算协调质量"""
        quality_scores = []
        
        # 稀疏激活质量
        quality_scores.append(sparse_result['sparse_efficiency'].item())
        
        # 印记质量
        if engram_results:
            engram_quality = torch.stack([r['engram_strength'] for r in engram_results]).mean().item()
            quality_scores.append(engram_quality)
        
        # 巩固质量
        if consolidation_result and 'consolidation_quality' in consolidation_result:
            quality_scores.append(consolidation_result['consolidation_quality'])
        
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5


class ForgettingController(nn.Module):
    """遗忘控制器"""
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__()
        self.config = config
        
        # 遗忘概率计算器
        self.forgetting_probability = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, current_weights: torch.Tensor, 
                reference_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """控制遗忘过程"""
        # 计算权重变化
        weight_changes = current_weights - reference_weights
        change_magnitude = weight_changes.abs().mean()
        
        # 计算遗忘概率
        forgetting_prob = self.forgetting_probability(change_magnitude.unsqueeze(-1))
        
        # 应用遗忘
        forgotten_weights = current_weights * (1 - forgetting_prob)
        
        # 计算遗忘结果
        forgetting_result = {
            'forgetting_probability': forgetting_prob.mean(),
            'weight_change_magnitude': change_magnitude.item(),
            'forgotten_weights_norm': forgotten_weights.norm().item(),
            'forgetting_effective': forgetting_prob.mean() > 0.1
        }
        
        return forgotten_weights, forgetting_result


# 工厂函数
def create_consolidation_engine(feature_dim: int, 
                               num_engram_cells: int = 100,
                               sparsity_level: float = 0.95,
                               **kwargs) -> ConsolidationEngine:
    """创建巩固引擎"""
    config = ConsolidationConfig(
        feature_dim=feature_dim,
        sparsity_level=sparsity_level,
        **kwargs
    )
    
    return ConsolidationEngine(config, feature_dim, num_engram_cells)