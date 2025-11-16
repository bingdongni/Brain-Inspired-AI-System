"""
稀疏激活与慢速权重巩固机制 (Sparse Activation & Weight Consolidation)
=====================================================================

实现新皮层的稀疏激活特性和慢速权重巩固机制：
- 稀疏显式编码（概念单元的稀疏激活）
- 慢速权重巩固（长期记忆形成）
- 海马印记细胞（Engram Cells）模拟
- 星形胶质细胞调节机制
- 权重巩固与遗忘平衡

基于论文：
- "Learning-associated astrocyte ensembles regulate memory recall"
- "Concept cells: the building blocks of declarative memory functions"
- 神经科学中的稀疏编码和长期巩固理论
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import math
import random


class ConsolidationType(Enum):
    """巩固类型"""
    SYNAPTIC = "synaptic"              # 突触水平巩固
    CELLULAR = "cellular"              # 细胞水平巩固
    NETWORK = "network"               # 网络水平巩固
    SYSTEM = "system"                 # 系统水平巩固


class MemoryState(Enum):
    """记忆状态"""
    LABILE = "labile"                 # 不稳定（易受干扰）
    REACTIVATING = "reactivating"     # 重新激活
    CONSOLIDATING = "consolidating"   # 巩固中
    STABLE = "stable"                 # 稳定长期记忆
    FORGETTING = "forgetting"         # 遗忘中


class CellType(Enum):
    """细胞类型"""
    PYRAMIDAL = "pyramidal"           # 锥体神经元
    INTERNEURON = "interneuron"       # 中间神经元
    ENGRAM_CELL = "engram_cell"       # 印记细胞
    ASTROCYTE = "astrocyte"           # 星形胶质细胞


@dataclass
class ConsolidationConfig:
    """巩固配置参数"""
    # 基础巩固参数
    consolidation_rate: float = 0.001     # 巩固速率
    forgetting_rate: float = 0.0001       # 遗忘速率
    reactivation_threshold: float = 0.7   # 重新激活阈值
    
    # 时间参数
    initial_consolidation_time: int = 100   # 初始巩固时间步数
    stable_memory_threshold: float = 0.9    # 稳定记忆阈值
    
    # 稀疏性参数
    target_sparsity: float = 0.1            # 目标稀疏程度
    sparse_threshold: float = 0.3           # 稀疏激活阈值
    competition_factor: float = 2.0         # 竞争因子
    
    # 印记参数
    engram_threshold: float = 0.8           # 印记形成阈值
    engram_strength_factor: float = 1.5     # 印记强度因子
    astro_regulation: float = 0.3           # 星形胶质调节强度
    
    # 巩固层次权重
    synaptic_weight: float = 0.4            # 突触巩固权重
    cellular_weight: float = 0.3            # 细胞巩固权重
    network_weight: float = 0.3             # 网络巩固权重


@dataclass
class EngramConfig:
    """印记细胞配置"""
    cell_id: str
    cell_type: CellType
    selectivity_threshold: float = 0.7
    consolidation_threshold: float = 0.8
    reactivation_threshold: float = 0.6
    
    # 印记特征
    concept_association: Optional[str] = None
    temporal_pattern: Optional[List[float]] = None
    strength: float = 0.0
    state: MemoryState = MemoryState.LABILE


class SparseActivation(nn.Module):
    """
    稀疏激活机制
    
    实现新皮层的稀疏编码特性：
    - 竞争性稀疏激活
    - 门控稀疏性控制
    - 动态稀疏模式
    - 稀疏性与性能平衡
    """
    
    def __init__(self, config: ConsolidationConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.target_sparsity = config.target_sparsity
        
        # 稀疏门控网络
        self.sparse_gating = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        
        # 竞争性稀疏控制器
        self.competitive_sparse = CompetitiveSparse(
            input_dim, config.sparse_threshold, config.competition_factor
        )
        
        # 动态稀疏调节器
        self.dynamic_sparse_regulator = DynamicSparseRegulator(
            input_dim, config.target_sparsity
        )
        
        # 稀疏性监控器
        self.sparse_monitor = SparseMonitor(input_dim)
        
    def forward(self, input_features: torch.Tensor,
                context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        稀疏激活前向传播
        
        Args:
            input_features: 输入特征 [batch, input_dim]
            context: 上下文信息
            
        Returns:
            稀疏激活结果和相关信息
        """
        batch_size = input_features.shape[0]
        
        # 步骤1：基础稀疏门控
        gating_weights = self.sparse_gating(input_features)
        gated_features = input_features * gating_weights
        
        # 步骤2：竞争性稀疏激活
        competitive_output = self.competitive_sparse(gated_features)
        
        # 步骤3：动态稀疏调节
        dynamic_output = self.dynamic_sparse_regulator(competitive_output)
        
        # 步骤4：稀疏性监控和调整
        sparse_info = self.sparse_monitor(dynamic_output)
        
        # 应用最终的稀疏掩码
        sparse_mask = self._create_sparse_mask(dynamic_output, sparse_info)
        final_sparse_output = dynamic_output * sparse_mask
        
        # 计算稀疏性指标
        actual_sparsity = self._calculate_sparsity(final_sparse_output)
        
        return {
            'sparse_output': final_sparse_output,
            'sparse_mask': sparse_mask,
            'gating_weights': gating_weights,
            'sparsity_info': sparse_info,
            'actual_sparsity': actual_sparsity,
            'target_sparsity': torch.tensor(self.target_sparsity),
            'sparse_efficiency': self._calculate_sparsity_efficiency(
                final_sparse_output, actual_sparsity
            )
        }
    
    def _create_sparse_mask(self, features: torch.Tensor, 
                          sparse_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """创建稀疏掩码"""
        # 基于稀疏信息生成掩码
        threshold = sparse_info['adaptive_threshold']
        
        # 稀疏掩码：大于阈值的保留，其他置零
        mask = (features.abs() > threshold).float()
        
        return mask
    
    def _calculate_sparsity(self, features: torch.Tensor) -> torch.Tensor:
        """计算实际稀疏性"""
        # 计算零元素比例作为稀疏性指标
        zero_elements = (features.abs() < 1e-6).float().sum(dim=-1)
        total_elements = features.shape[-1]
        sparsity = zero_elements / total_elements
        
        return sparsity
    
    def _calculate_sparsity_efficiency(self, features: torch.Tensor,
                                     sparsity: torch.Tensor) -> torch.Tensor:
        """计算稀疏性效率（信息含量/稀疏性比率）"""
        # 计算特征的信息含量（基于方差）
        information_content = torch.var(features, dim=-1)
        
        # 效率 = 信息含量 / 稀疏性
        efficiency = information_content / (sparsity + 1e-8)
        
        return efficiency


class CompetitiveSparse(nn.Module):
    """竞争性稀疏激活器"""
    
    def __init__(self, input_dim: int, threshold: float, competition_factor: float):
        super().__init__()
        self.input_dim = input_dim
        self.threshold = threshold
        self.competition_factor = competition_factor
        
        # 竞争强度计算器
        self.competition_calculator = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """竞争性稀疏处理"""
        # 计算每个特征的竞争强度
        competition_strength = self.competition_calculator(features)
        
        # 应用竞争机制：强激活抑制弱激活
        # 实现WTA (Winner-Take-All)机制
        batch_size = features.shape[0]
        
        sparse_output = torch.zeros_like(features)
        
        for b in range(batch_size):
            # 找到激活最强的特征
            feature_activations = features[b]
            competition_values = competition_strength[b]
            
            # 应用竞争抑制
            for i in range(self.input_dim):
                if feature_activations[i] > self.threshold:
                    # 当前神经元激活，抑制其他神经元
                    inhibition = self.competition_factor * competition_values[i]
                    
                    # 应用抑制到其他神经元
                    mask = torch.ones(self.input_dim)
                    mask[i] = 0  # 不抑制自己
                    
                    other_activations = feature_activations * (1 - inhibition * mask)
                    
                    # 保留最强的激活
                    if other_activations.max() < feature_activations[i]:
                        sparse_output[b, i] = feature_activations[i]
                    else:
                        # 有更强的激活，不保留当前激活
                        pass
                else:
                    sparse_output[b, i] = feature_activations[i]
        
        return sparse_output


class DynamicSparseRegulator(nn.Module):
    """动态稀疏性调节器"""
    
    def __init__(self, input_dim: int, target_sparsity: float):
        super().__init__()
        self.input_dim = input_dim
        self.target_sparsity = target_sparsity
        
        # 稀疏性反馈调节器
        self.sparsity_feedback = nn.Sequential(
            nn.Linear(2, 16),  # 当前稀疏性和目标稀疏性
            nn.ReLU(),
            nn.Linear(16, input_dim),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """动态稀疏调节"""
        # 计算当前稀疏性
        current_sparsity = (features.abs() < 0.1).float().mean(dim=-1)
        
        # 稀疏性反馈调节
        sparsity_input = torch.stack([current_sparsity, 
                                    torch.ones_like(current_sparsity) * self.target_sparsity], 
                                   dim=-1)
        
        regulation_weights = self.sparsity_feedback(sparsity_input)
        
        # 应用调节权重
        regulated_features = features * regulation_weights
        
        return regulated_features


class SparseMonitor(nn.Module):
    """稀疏性监控器"""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        
        # 自适应阈值计算器
        self.adaptive_threshold = nn.Sequential(
            nn.Linear(3, 16),  # 特征统计量
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """稀疏性监控"""
        # 计算特征统计量
        feature_mean = features.mean(dim=-1, keepdim=True)
        feature_std = features.std(dim=-1, keepdim=True)
        feature_max = features.max(dim=-1, keepdim=True)[0]
        
        # 生成自适应阈值
        threshold_input = torch.cat([feature_mean, feature_std, feature_max], dim=-1)
        adaptive_threshold = self.adaptive_threshold(threshold_input)
        
        return {
            'feature_stats': {
                'mean': feature_mean,
                'std': feature_std,
                'max': feature_max
            },
            'adaptive_threshold': adaptive_threshold
        }


class WeightConsolidation(nn.Module):
    """
    慢速权重巩固机制
    
    实现长期记忆的慢速巩固过程：
    - 突触权重巩固
    - 网络结构重塑
    - 巩固与遗忘平衡
    - 时间依赖性巩固
    """
    
    def __init__(self, config: ConsolidationConfig, weight_dim: Tuple[int, ...]):
        super().__init__()
        self.config = config
        self.weight_dim = weight_dim
        self.num_weights = np.prod(weight_dim)
        
        # 权重历史记录（用于跟踪权重变化）
        self.weight_history = deque(maxlen=config.initial_consolidation_time)
        
        # 巩固强度计算器
        self.consolidation_calculator = nn.Sequential(
            nn.Linear(4, 16),  # 权重变化统计
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 遗忘门控器
        self.forgetting_gater = nn.Sequential(
            nn.Linear(2, 8),  # 时间衰减和巩固强度
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 巩固动力学模型
        self.consolidation_dynamics = ConsolidationDynamics(config)
        
    def forward(self, current_weights: torch.Tensor,
                weight_changes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        权重巩固前向传播
        
        Args:
            current_weights: 当前权重 [weight_dim]
            weight_changes: 权重变化量（如果提供）
            
        Returns:
            巩固后的权重和相关信息
        """
        # 记录权重历史
        self.weight_history.append(current_weights.detach().clone())
        
        # 计算巩固强度
        consolidation_info = self._calculate_consolidation_strength(
            current_weights, weight_changes
        )
        
        # 应用遗忘机制
        forgetting_info = self._calculate_forgetting(current_weights)
        
        # 巩固动力学更新
        consolidated_weights = self.consolidation_dynamics(
            current_weights, consolidation_info, forgetting_info
        )
        
        # 计算巩固质量
        consolidation_quality = self._assess_consolidation_quality(consolidated_weights)
        
        return {
            'consolidated_weights': consolidated_weights,
            'consolidation_info': consolidation_info,
            'forgetting_info': forgetting_info,
            'consolidation_quality': consolidation_quality,
            'weight_stability': self._calculate_weight_stability(),
            'memory_strength': self._calculate_memory_strength()
        }
    
    def _calculate_consolidation_strength(self, weights: torch.Tensor,
                                        changes: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算巩固强度"""
        # 基础巩固强度（基于权重幅值）
        base_strength = torch.sigmoid(torch.norm(weights))
        
        # 基于权重变化的增强
        if changes is not None and len(self.weight_history) > 1:
            # 计算权重变化的统计量
            change_magnitude = torch.norm(changes)
            change_consistency = self._calculate_change_consistency(changes)
            
            # 综合巩固强度
            strength = base_strength * (1 + 0.5 * change_magnitude + 0.3 * change_consistency)
        else:
            strength = base_strength
            
        # 规范化到 [0, 1]
        strength = torch.clamp(strength, 0, 1)
        
        return {
            'base_strength': base_strength,
            'change_magnitude': changes.norm() if changes is not None else torch.tensor(0.0),
            'change_consistency': self._calculate_change_consistency(changes) if changes is not None else torch.tensor(0.0),
            'final_strength': strength
        }
    
    def _calculate_forgetting(self, weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """计算遗忘影响"""
        # 遗忘强度基于时间（历史长度）
        time_factor = min(len(self.weight_history) / self.config.initial_consolidation_time, 1.0)
        
        # 遗忘门控信号
        forgetting_input = torch.cat([
            torch.tensor(time_factor).unsqueeze(0),
            torch.tensor(self.config.forgetting_rate).unsqueeze(0)
        ])
        
        forgetting_strength = self.forgetting_gater(forgetting_input.unsqueeze(0))
        
        return {
            'time_factor': torch.tensor(time_factor),
            'forgetting_strength': forgetting_strength.squeeze(-1)
        }
    
    def _calculate_change_consistency(self, changes: torch.Tensor) -> torch.Tensor:
        """计算权重变化的一致性"""
        if len(self.weight_history) < 2:
            return torch.tensor(0.0)
            
        # 计算当前变化与历史变化的相似性
        if len(self.weight_history) >= 5:
            # 使用最近5个权重变化
            recent_changes = list(self.weight_history)[-5:]
            historical_gradients = []
            
            for i in range(1, len(recent_changes)):
                gradient = recent_changes[i] - recent_changes[i-1]
                historical_gradients.append(gradient)
                
            if len(historical_gradients) > 0:
                historical_consistency = sum(
                    F.cosine_similarity(changes.flatten(), g.flatten(), dim=0)
                    for g in historical_gradients
                ) / len(historical_gradients)
                
                return torch.clamp(historical_consistency, 0, 1)
        
        return torch.tensor(0.0)
    
    def _assess_consolidation_quality(self, weights: torch.Tensor) -> torch.Tensor:
        """评估巩固质量"""
        # 权重稳定性
        weight_stability = self._calculate_weight_stability()
        
        # 权重分布合理性
        weight_distribution = self._assess_weight_distribution(weights)
        
        # 综合质量分数
        quality = (weight_stability + weight_distribution) / 2
        
        return quality
    
    def _calculate_weight_stability(self) -> torch.Tensor:
        """计算权重稳定性"""
        if len(self.weight_history) < 10:
            return torch.tensor(0.5)  # 初始低稳定性
            
        # 计算最近权重的方差
        recent_weights = torch.stack(list(self.weight_history)[-10:])
        weight_variance = torch.var(recent_weights)
        
        # 稳定性 = 1 / (1 + 方差)
        stability = 1.0 / (1.0 + weight_variance)
        
        return stability
    
    def _assess_weight_distribution(self, weights: torch.Tensor) -> torch.Tensor:
        """评估权重分布"""
        # 理想分布应该是适度的，不是全零也不是全饱和
        mean_abs_weight = torch.mean(torch.abs(weights))
        
        # 分布质量：适中的平均权重幅值
        if 0.1 < mean_abs_weight < 0.9:
            distribution_quality = 1.0
        else:
            distribution_quality = mean_abs_weight
            
        return distribution_quality
    
    def _calculate_memory_strength(self) -> torch.Tensor:
        """计算记忆强度"""
        # 基于历史权重变化和巩固强度
        if len(self.weight_history) < 5:
            return torch.tensor(0.0)
            
        # 计算权重轨迹的平滑性
        recent_weights = torch.stack(list(self.weight_history)[-5:])
        
        # 计算相邻权重间的差异
        weight_diffs = []
        for i in range(1, len(recent_weights)):
            diff = torch.norm(recent_weights[i] - recent_weights[i-1])
            weight_diffs.append(diff)
            
        # 记忆强度与权重变化的平滑性成反比
        if weight_diffs:
            avg_diff = sum(weight_diffs) / len(weight_diffs)
            memory_strength = 1.0 / (1.0 + avg_diff)
        else:
            memory_strength = 0.0
            
        return memory_strength


class ConsolidationDynamics(nn.Module):
    """巩固动力学模型"""
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__()
        self.config = config
        
        # 巩固更新规则
        self.update_rule = nn.Sequential(
            nn.Linear(3, 16),  # 当前权重、巩固强度、遗忘强度
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh()
        )
        
    def forward(self, current_weights: torch.Tensor,
                consolidation_info: Dict[str, torch.Tensor],
                forgetting_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """巩固动力学更新"""
        batch_size = current_weights.numel()
        
        # 展平权重进行更新
        flat_weights = current_weights.flatten()
        
        # 构建更新输入
        consolidation_strength = consolidation_info['final_strength'].item()
        forgetting_strength = forgetting_info['forgetting_strength'].item()
        
        updates = torch.zeros_like(flat_weights)
        
        for i in range(len(flat_weights)):
            # 计算当前权重的更新
            update_input = torch.tensor([
                flat_weights[i].item(),
                consolidation_strength,
                forgetting_strength
            ])
            
            update = self.update_rule(update_input.unsqueeze(0))
            updates[i] = update * self.config.consolidation_rate
            
        # 应用更新
        new_weights = flat_weights + updates
        
        # 重塑回原形状
        new_weights = new_weights.view(current_weights.shape)
        
        # 权重裁剪（防止数值爆炸）
        new_weights = torch.clamp(new_weights, -10, 10)
        
        return new_weights


class EngramCell(nn.Module):
    """
    印记细胞 (Engram Cell)
    
    模拟海马和新皮层中的印记细胞：
    - 概念选择性激活
    - 记忆印记形成
    - 重新激活机制
    - 与星形胶质细胞的相互作用
    """
    
    def __init__(self, config: EngramConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.cell_id = config.cell_id
        self.cell_type = config.cell_type
        
        # 印记编码器
        self.engram_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 选择性权重（决定细胞对什么概念响应）
        if config.cell_type == CellType.ENGRAM_CELL:
            self.selectivity_weights = nn.Parameter(torch.randn(input_dim))
        
        # 印记强度跟踪器
        self.strength_tracker = EngramStrengthTracker(config)
        
        # 星形胶质调节器
        self.astrocyte_regulator = AstrocyteRegulator(config)
        
    def forward(self, input_features: torch.Tensor,
                context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        印记细胞前向传播
        
        Args:
            input_features: 输入特征 [batch, input_dim]
            context: 上下文信息
            
        Returns:
            印记细胞响应和记忆信息
        """
        batch_size = input_features.shape[0]
        
        # 基础印记响应
        engram_response = self.engram_encoder(input_features)
        
        # 选择性响应（如果是印记细胞）
        if self.config.cell_type == CellType.ENGRAM_CELL:
            selective_response = F.linear(input_features, self.selectivity_weights)
            engram_response = engram_response + 0.5 * torch.sigmoid(selective_response)
        
        # 星形胶质调节
        astro_regulated_response = self.astrocyte_regulator(
            engram_response, context
        )
        
        # 更新印记强度
        strength_update = self.strength_tracker(astro_regulated_response)
        
        # 检查印记形成条件
        engram_formed = self._check_engram_formation(astro_regulated_response, strength_update)
        
        # 计算重新激活可能性
        reactivation_prob = self._calculate_reactivation_probability(
            astro_regulated_response, strength_update
        )
        
        return {
            'engram_response': astro_regulated_response,
            'selective_response': selective_response if self.config.cell_type == CellType.ENGRAM_CELL else engram_response,
            'engram_strength': strength_update['current_strength'],
            'engram_formed': engram_formed,
            'reactivation_probability': reactivation_prob,
            'cell_state': strength_update['memory_state'],
            'astrocyte_influence': strength_update['astrocyte_influence']
        }
    
    def _check_engram_formation(self, response: torch.Tensor, 
                              strength_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """检查印记是否形成"""
        # 印记形成条件：响应强度足够且持续时间足够
        response_threshold = response > self.config.consolidation_threshold
        strength_sufficient = strength_info['current_strength'] > self.config.engram_threshold
        
        # 结合时间和强度条件
        formation = (response_threshold & strength_sufficient).float()
        
        return formation
    
    def _calculate_reactivation_probability(self, response: torch.Tensor,
                                          strength_info: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算重新激活概率"""
        # 基础概率基于印记强度
        base_prob = strength_info['current_strength']
        
        # 响应强度调制
        response_factor = torch.sigmoid(response * 2)
        
        # 状态调制（稳定记忆更容易重新激活）
        state_factor = torch.where(
            strength_info['memory_state'] == MemoryState.STABLE,
            1.0, 0.5
        )
        
        # 综合重新激活概率
        reactivation_prob = base_prob * response_factor * state_factor
        
        return torch.clamp(reactivation_prob, 0, 1)


class EngramStrengthTracker(nn.Module):
    """印记强度跟踪器"""
    
    def __init__(self, config: EngramConfig):
        super().__init__()
        self.config = config
        
        # 强度更新网络
        self.strength_updater = nn.Sequential(
            nn.Linear(2, 8),  # 当前响应、当前强度
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 状态转换器
        self.state_transitioner = nn.Sequential(
            nn.Linear(2, 4),  # 强度、时间因子
            nn.ReLU(),
            nn.Linear(4, len(MemoryState)),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, response: torch.Tensor) -> Dict[str, torch.Tensor]:
        """更新印记强度"""
        batch_size = response.shape[0]
        
        # 模拟当前强度（简化为响应强度）
        current_strength = response
        
        # 强度更新
        strength_input = torch.cat([response, current_strength], dim=-1)
        new_strength = self.strength_updater(strength_input)
        
        # 状态转换
        time_factor = torch.ones(batch_size, 1)  # 简化的时间因子
        state_input = torch.cat([new_strength, time_factor], dim=-1)
        state_probs = self.state_transitioner(state_input)
        
        # 选择最可能的状态
        memory_state = MemoryState(list(MemoryState)[torch.argmax(state_probs, dim=-1)])
        
        return {
            'current_strength': new_strength,
            'memory_state': memory_state,
            'state_confidence': torch.max(state_probs, dim=-1)[0]
        }


class AstrocyteRegulator(nn.Module):
    """星形胶质细胞调节器"""
    
    def __init__(self, config: EngramConfig):
        super().__init__()
        self.config = config
        self.astro_regulation = config.astro_regulation
        
        # 星形胶质活动计算器
        self.astro_activity_calc = nn.Sequential(
            nn.Linear(1, 8),  # 输入响应
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 调节信号生成器
        self.regulation_signal_gen = nn.Sequential(
            nn.Linear(2, 8),  # 星形胶质活动、上下文
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()
        )
        
    def forward(self, response: torch.Tensor,
                context: Optional[Dict] = None) -> torch.Tensor:
        """星形胶质调节"""
        batch_size = response.shape[0]
        
        # 计算星形胶质活动
        astro_activity = self.astro_activity_calc(response)
        
        # 生成调节信号
        regulation_input = torch.cat([
            astro_activity,
            torch.ones(batch_size, 1) * 0.5  # 默认上下文
        ], dim=-1)
        
        regulation_signal = self.regulation_signal_gen(regulation_input)
        
        # 应用调节
        regulated_response = response + self.astro_regulation * regulation_signal
        
        return {
            'regulated_response': regulated_response,
            'astrocyte_activity': astro_activity,
            'regulation_signal': regulation_signal
        }


class ConsolidationEngine(nn.Module):
    """
    巩固引擎
    
    整合稀疏激活和权重巩固机制的主引擎：
    - 协调稀疏激活和巩固过程
    - 管理印记细胞网络
    - 实现长期记忆形成
    - 平衡巩固与遗忘
    """
    
    def __init__(self, config: ConsolidationConfig, input_dim: int, 
                 num_engram_cells: int = 50):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # 稀疏激活模块
        self.sparse_activation = SparseActivation(config, input_dim)
        
        # 权重巩固模块
        weight_shape = (input_dim, input_dim)  # 简化的权重形状
        self.weight_consolidation = WeightConsolidation(config, weight_shape)
        
        # 印记细胞网络
        self.engram_network = self._create_engram_network(num_engram_cells)
        
        # 巩固协调器
        self.consolidation_coordinator = ConsolidationCoordinator(config)
        
        # 记忆状态跟踪器
        self.memory_tracker = MemoryStateTracker()
        
    def _create_engram_network(self, num_cells: int) -> nn.ModuleList:
        """创建印记细胞网络"""
        cells = nn.ModuleList()
        
        # 创建不同类型的细胞
        cell_types = [CellType.ENGRAM_CELL, CellType.PYRAMIDAL, CellType.INTERNEURON]
        
        for i in range(num_cells):
            cell_type = cell_types[i % len(cell_types)]
            
            config = EngramConfig(
                cell_id=f"engram_cell_{i}",
                cell_type=cell_type,
                concept_association=f"concept_{i % 10}" if i % 5 == 0 else None
            )
            
            cells.append(EngramCell(config, self.input_dim))
            
        return cells
    
    def forward(self, input_features: torch.Tensor,
                context: Optional[Dict] = None) -> Dict[str, Union[torch.Tensor, List, Dict]]:
        """
        巩固引擎前向传播
        
        Args:
            input_features: 输入特征 [batch, input_dim]
            context: 上下文信息
            
        Returns:
            完整的巩固处理结果
        """
        batch_size = input_features.shape[0]
        
        # 步骤1：稀疏激活
        sparse_results = self.sparse_activation(input_features, context)
        
        # 步骤2：权重巩固
        # 模拟权重矩阵（简化）
        current_weights = torch.eye(self.input_dim) * 0.5
        weight_changes = None  # 在实际应用中需要跟踪权重变化
        
        consolidation_results = self.weight_consolidation(current_weights, weight_changes)
        
        # 步骤3：印记细胞网络处理
        engram_responses = []
        for engram_cell in self.engram_network:
            response = engram_cell(input_features, context)
            engram_responses.append(response)
        
        # 步骤4：巩固协调
        coordination_results = self.consolidation_coordinator(
            sparse_results, consolidation_results, engram_responses
        )
        
        # 步骤5：记忆状态跟踪
        memory_state = self.memory_tracker(engram_responses)
        
        # 整合最终结果
        final_output = self._integrate_results(
            sparse_results, consolidation_results, engram_responses,
            coordination_results, memory_state
        )
        
        return {
            'sparse_activation': sparse_results,
            'weight_consolidation': consolidation_results,
            'engram_responses': engram_responses,
            'coordination': coordination_results,
            'memory_state': memory_state,
            'final_output': final_output,
            'consolidation_summary': self._generate_consolidation_summary(
                sparse_results, consolidation_results, engram_responses
            )
        }
    
    def _integrate_results(self, sparse_results: Dict,
                         consolidation_results: Dict,
                         engram_responses: List[Dict],
                         coordination_results: Dict,
                         memory_state: Dict) -> Dict[str, torch.Tensor]:
        """整合所有结果"""
        
        # 提取关键信息
        sparse_output = sparse_results['sparse_output']
        consolidated_weights = consolidation_results['consolidated_weights']
        engram_activations = torch.stack([
            resp['engram_response'] for resp in engram_responses
        ])  # [num_cells, batch_size]
        
        # 计算激活细胞的平均响应
        active_cell_response = engram_activations.mean(dim=0)
        
        # 整合输出
        integrated_output = (
            0.4 * sparse_output +  # 稀疏特征
            0.3 * active_cell_response +  # 印记细胞响应
            0.3 * consolidated_weights.mean(dim=0)  # 巩固权重
        )
        
        # 计算整体记忆强度
        memory_strength = (sparse_results['actual_sparsity'].mean() + 
                          consolidation_results['memory_strength'] +
                          torch.sigmoid(engram_activations.mean())) / 3
        
        return {
            'integrated_output': integrated_output,
            'memory_strength': memory_strength,
            'sparse_efficiency': sparse_results['sparse_efficiency'].mean(),
            'consolidation_quality': consolidation_results['consolidation_quality']
        }
    
    def _generate_consolidation_summary(self, sparse_results: Dict,
                                      consolidation_results: Dict,
                                      engram_responses: List[Dict]) -> Dict[str, any]:
        """生成巩固摘要"""
        
        # 激活的印记细胞数量
        active_engram_cells = sum(
            1 for resp in engram_responses
            if resp['engram_response'].mean() > 0.5
        )
        
        # 形成的印记数量
        formed_engrams = sum(
            1 for resp in engram_responses
            if resp['engram_formed'].mean() > 0.5
        )
        
        # 平均印记强度
        avg_engram_strength = torch.stack([
            resp['engram_strength'] for resp in engram_responses
        ]).mean().item()
        
        # 巩固质量评估
        consolidation_quality = consolidation_results['consolidation_quality'].item()
        
        return {
            'active_engram_cells': active_engram_cells,
            'formed_engrams': formed_engrams,
            'avg_engram_strength': avg_engram_strength,
            'consolidation_quality': consolidation_quality,
            'sparse_efficiency': sparse_results['sparse_efficiency'].mean().item(),
            'memory_strength': consolidation_results['memory_strength'].item()
        }


class ConsolidationCoordinator(nn.Module):
    """巩固协调器"""
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__()
        self.config = config
        
        # 协调权重学习器
        self.coordination_weights = nn.Parameter(torch.ones(3) / 3)  # 稀疏、巩固、印记
        
    def forward(self, sparse_results: Dict,
                consolidation_results: Dict,
                engram_responses: List[Dict]) -> Dict[str, torch.Tensor]:
        """协调各种机制"""
        
        # 归一化协调权重
        weights = F.softmax(self.coordination_weights, dim=0)
        
        # 计算各机制的贡献度
        sparse_contribution = sparse_results['sparse_efficiency'].mean()
        consolidation_contribution = consolidation_results['consolidation_quality']
        
        engram_activations = torch.stack([resp['engram_response'] for resp in engram_responses])
        engram_contribution = torch.sigmoid(engram_activations.mean())
        
        # 协调输出
        coordinated_signal = (
            weights[0] * sparse_contribution +
            weights[1] * consolidation_contribution +
            weights[2] * engram_contribution
        )
        
        return {
            'coordination_weights': weights,
            'sparse_contribution': sparse_contribution,
            'consolidation_contribution': consolidation_contribution,
            'engram_contribution': engram_contribution,
            'coordinated_signal': coordinated_signal
        }


class MemoryStateTracker(nn.Module):
    """记忆状态跟踪器"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, engram_responses: List[Dict]) -> Dict[str, any]:
        """跟踪记忆状态"""
        
        # 统计各种状态的细胞数量
        state_counts = defaultdict(int)
        total_strength = 0.0
        reactivation_probs = []
        
        for response in engram_responses:
            state = response['cell_state']
            state_counts[state.value] += 1
            total_strength += response['engram_strength'].item()
            reactivation_probs.append(response['reactivation_probability'].item())
        
        # 计算整体记忆健康度
        avg_reactivation_prob = np.mean(reactivation_probs)
        memory_health = total_strength / len(engram_responses) * avg_reactivation_prob
        
        return {
            'state_counts': dict(state_counts),
            'total_strength': total_strength,
            'avg_reactivation_probability': avg_reactivation_prob,
            'memory_health': memory_health
        }


# 工厂函数

def create_consolidation_engine(input_dim: int, 
                              num_engram_cells: int = 50) -> ConsolidationEngine:
    """创建巩固引擎"""
    config = ConsolidationConfig()
    return ConsolidationEngine(config, input_dim, num_engram_cells)


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建巩固引擎
    consolidation_engine = create_consolidation_engine(256, 50).to(device)
    
    # 测试输入
    test_features = torch.randn(4, 256).to(device)
    
    # 前向传播
    results = consolidation_engine(test_features)
    
    print(f"巩固引擎测试:")
    print(f"输入特征形状: {test_features.shape}")
    print(f"稀疏效率: {results['consolidation_summary']['sparse_efficiency']:.3f}")
    print(f"激活印记细胞: {results['consolidation_summary']['active_engram_cells']}")
    print(f"形成印记数量: {results['consolidation_summary']['formed_engrams']}")
    print(f"平均印记强度: {results['consolidation_summary']['avg_engram_strength']:.3f}")
    print(f"巩固质量: {results['consolidation_summary']['consolidation_quality']:.3f}")
    print(f"记忆健康度: {results['memory_state']['memory_health']:.3f}")
    print(f"整合输出形状: {results['final_output']['integrated_output'].shape}")
