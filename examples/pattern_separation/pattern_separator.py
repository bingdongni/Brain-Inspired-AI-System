"""
模式分离机制
基于海马体CA3区的模式分离功能
实现输入的模式分离和差异增强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class GranuleCellLayer(nn.Module):
    """
    颗粒细胞层
    模拟齿状回颗粒细胞的稀疏激活模式
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 sparsity: float = 0.02,
                 activation_threshold: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.activation_threshold = activation_threshold
        
        # 编码器权重
        self.encoder_weights = nn.Parameter(
            torch.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        )
        
        # 偏置项
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # 稀疏约束层
        self.sparsity_gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )
        
        # 归一化层
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # 侧抑制机制
        self.lateral_inhibition = nn.Parameter(
            torch.ones(output_dim, output_dim) - torch.eye(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        颗粒细胞层前向传播
        
        Args:
            x: 输入 [batch_size, input_dim]
            
        Returns:
            稀疏输出、激活掩码
        """
        # 线性变换
        pre_activation = F.linear(x, self.encoder_weights, self.bias)
        
        # 应用稀疏约束
        sparsity_gate = self.sparsity_gate(pre_activation)
        
        # 阈值激活（实现稀疏性）
        activation_mask = sparsity_gate > self.activation_threshold
        
        # 侧抑制机制
        if activation_mask.any():
            # 计算激活神经元的平均激活
            active_activations = pre_activation * activation_mask.float()
            mean_activation = active_activations.sum(dim=-1, keepdim=True) / (activation_mask.sum(dim=-1, keepdim=True) + 1e-8)
            
            # 侧抑制
            inhibited_activations = pre_activation - mean_activation
            
            # 应用侧抑制（避免原位操作）
            inhibition_strength = 0.1
            lateral_inhibition_effect = inhibition_strength * torch.matmul(
                activation_mask.float(), self.lateral_inhibition
            )
            pre_activation = pre_activation - lateral_inhibition_effect
        
        # 稀疏激活
        sparse_output = pre_activation * activation_mask.float()
        
        # 归一化
        sparse_output = self.layer_norm(sparse_output)
        
        return sparse_output, activation_mask
    
    def compute_sparsity(self, activation_mask: torch.Tensor) -> float:
        """计算实际稀疏性"""
        return 1.0 - activation_mask.float().mean().item()
    
    def update_sparsity_target(self, current_sparsity: float, target_sparsity: float = 0.02):
        """动态调整稀疏性目标"""
        if current_sparsity > target_sparsity:
            self.activation_threshold *= 0.99
        else:
            self.activation_threshold *= 1.01


class MossyFiberProjection(nn.Module):
    """
    苔藓纤维投射
    连接CA3锥体细胞到颗粒细胞的投射机制
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 projection_sparsity: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.projection_sparsity = projection_sparsity
        
        # 投射权重
        self.projection_weights = nn.Parameter(
            torch.randn(output_dim, input_dim) * 0.1
        )
        
        # 投射偏置
        self.projection_bias = nn.Parameter(torch.zeros(output_dim))
        
        # 门控机制
        self.gating_mechanism = nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.ReLU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        # 动态连接强度
        self.connection_strength = nn.Parameter(torch.ones(output_dim))
        
    def forward(self, x: torch.Tensor, ca3_activities: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        苔藓纤维投射前向传播
        
        Args:
            x: 输入激活
            ca3_activities: CA3活动模式
            
        Returns:
            投射输出
        """
        batch_size = x.shape[0]
        
        # 基础投射
        projected = F.linear(x, self.projection_weights, self.projection_bias)
        
        # 应用门控
        if ca3_activities is not None:
            gating = self.gating_mechanism(ca3_activities)
            projected = projected * gating
        
        # 应用连接强度
        projected = projected * self.connection_strength.unsqueeze(0)
        
        # 投影稀疏性控制
        projection_mask = torch.rand_like(projected) > self.projection_sparsity
        projected = projected * projection_mask.float()
        
        return projected


class CA3RecurrentNetwork(nn.Module):
    """
    CA3递归网络
    实现模式分离和模式完成的递归连接
    """
    
    def __init__(self,
                 hidden_dim: int,
                 recurrent_connections: int = 100,
                 separation_strength: float = 0.8,
                 completion_rate: float = 0.6):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.recurrent_connections = recurrent_connections
        self.separation_strength = separation_strength
        self.completion_rate = completion_rate
        
        # 递归权重矩阵
        self.recurrent_weights = nn.Parameter(
            torch.randn(recurrent_connections, hidden_dim, hidden_dim) * 0.01
        )
        
        # 递归偏置
        self.recurrent_bias = nn.Parameter(torch.zeros(recurrent_connections, hidden_dim))
        
        # 模式分离权重
        self.separation_weights = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)  # 3个分离组件
        ])
        
        # 模式完成权重
        self.completion_weights = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(3)  # 3个完成组件
        ])
        
        # 迭代控制
        self.iteration_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, 
                x: torch.Tensor,
                iterations: int = 3) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        CA3递归网络前向传播
        
        Args:
            x: 输入
            iterations: 递归迭代次数
            
        Returns:
            递归输出、分离统计
        """
        batch_size = x.shape[0]
        current_state = x.clone()
        
        separation_stats = {
            'separation_magnitudes': [],
            'completion_magnitudes': [],
            'stability_scores': []
        }
        
        for iteration in range(iterations):
            # 模式分离
            separated_state = self._apply_separation(current_state)
            
            # 模式完成
            completed_state = self._apply_completion(separated_state)
            
            # 递归连接
            recurrent_input = self._apply_recurrent_connections(completed_state)
            
            # 迭代控制
            iteration_control = self.iteration_controller(recurrent_input)
            
            # 更新状态
            current_state = (1 - iteration_control) * current_state + \
                           iteration_control * recurrent_input
            
            # 记录统计
            separation_magnitude = torch.norm(separated_state - current_state).item()
            completion_magnitude = torch.norm(completed_state - separated_state).item()
            stability = iteration_control.mean().item()
            
            separation_stats['separation_magnitudes'].append(separation_magnitude)
            separation_stats['completion_magnitudes'].append(completion_magnitude)
            separation_stats['stability_scores'].append(stability)
        
        return current_state, separation_stats
    
    def _apply_separation(self, x: torch.Tensor) -> torch.Tensor:
        """应用模式分离"""
        separated = x.clone()
        
        for i, sep_weight in enumerate(self.separation_weights):
            # 增强差异性
            sep_output = sep_weight(x)
            sep_output = torch.tanh(sep_output) * self.separation_strength
            
            if i == 0:
                separated = sep_output
            else:
                # 交替分离策略
                separated = (separated + sep_output) / 2
        
        return separated
    
    def _apply_completion(self, x: torch.Tensor) -> torch.Tensor:
        """应用模式完成"""
        completed = x.clone()
        
        for i, comp_weight in enumerate(self.completion_weights):
            # 增强相似性
            comp_output = comp_weight(x)
            comp_output = torch.sigmoid(comp_output) * self.completion_rate
            
            if i == 0:
                completed = comp_output
            else:
                # 权重平均
                completed = (completed + comp_output) / 2
        
        return completed
    
    def _apply_recurrent_connections(self, x: torch.Tensor) -> torch.Tensor:
        """应用递归连接"""
        batch_size = x.shape[0]
        
        # 计算递归输入
        recurrent_input = torch.zeros_like(x)
        
        for i in range(min(self.recurrent_connections, len(self.recurrent_weights))):
            # 选择性的递归连接
            connection_strength = 1.0 / (1.0 + i * 0.1)  # 递减权重
            
            # 应用递归权重
            recurrent_contribution = torch.einsum(
                'bhd,hdh->bh', x, self.recurrent_weights[i]
            )
            recurrent_contribution = recurrent_contribution * connection_strength
            
            recurrent_input = recurrent_input + recurrent_contribution
        
        # 添加偏置
        recurrent_input = recurrent_input + self.recurrent_bias[:recurrent_input.shape[0]].unsqueeze(0)
        
        # 激活函数
        recurrent_output = F.relu(recurrent_input)
        
        return recurrent_output


class PatternSeparationNetwork(nn.Module):
    """
    完整模式分离网络
    集成颗粒细胞层、苔藓纤维投射和CA3递归网络
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 512,
                 num_granule_cells: int = 1000,
                 num_ca3_cells: int = 200,
                 sparsity: float = 0.02,
                 separation_strength: float = 0.8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_granule_cells = num_granule_cells
        self.num_ca3_cells = num_ca3_cells
        self.separation_strength = separation_strength
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 颗粒细胞层（模式分离的主要执行者）
        self.granule_layer = GranuleCellLayer(
            input_dim=hidden_dim,
            output_dim=num_granule_cells,
            sparsity=sparsity
        )
        
        # 苔藓纤维投射
        self.mossy_fiber = MossyFiberProjection(
            input_dim=num_granule_cells,
            output_dim=num_ca3_cells
        )
        
        # CA3递归网络
        self.ca3_network = CA3RecurrentNetwork(
            hidden_dim=num_ca3_cells,
            recurrent_connections=100,
            separation_strength=separation_strength
        )
        
        # 输出投影
        self.output_projection = nn.Sequential(
            nn.Linear(num_ca3_cells, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # 模式分离度量器
        self.separation_evaluator = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 自适应控制
        self.adaptive_controller = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # sparsity, separation, completion
            nn.Softmax(dim=-1)
        )
        
    def forward(self, 
                x1: torch.Tensor,
                x2: Optional[torch.Tensor] = None,
                adaptive_control: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        模式分离前向传播
        
        Args:
            x1: 输入1 [batch_size, input_dim]
            x2: 输入2（用于比较分离效果）
            adaptive_control: 是否使用自适应控制
            
        Returns:
            分离后的表示、分离统计信息
        """
        batch_size = x1.shape[0]
        
        # 输入投影
        projected1 = self.input_projection(x1)
        
        stats = {
            'input1_norm': torch.norm(x1).item(),
            'projected1_norm': torch.norm(projected1).item()
        }
        
        if x2 is not None:
            projected2 = self.input_projection(x2)
            stats['input2_norm'] = torch.norm(x2).item()
            stats['projected2_norm'] = torch.norm(projected2).item()
            
            # 原始输入分离程度
            input_similarity = F.cosine_similarity(x1, x2).mean().item()
            stats['input_similarity'] = input_similarity
        else:
            projected2 = None
        
        # 自适应控制
        if adaptive_control:
            control_weights = self.adaptive_controller(projected1.mean(dim=0, keepdim=True))
            stats['control_weights'] = control_weights.detach().cpu().numpy()
        else:
            control_weights = torch.ones(1, 3, device=x1.device) / 3
        
        # 颗粒细胞层处理
        granule_output1, granule_mask1 = self.granule_layer(projected1)
        
        stats['granule_sparsity'] = self.granule_layer.compute_sparsity(granule_mask1)
        
        if projected2 is not None:
            granule_output2, granule_mask2 = self.granule_layer(projected2)
            stats['granule_sparsity_2'] = self.granule_layer.compute_sparsity(granule_mask2)
        
        # 苔藓纤维投射
        mossy_output1 = self.mossy_fiber(granule_output1)
        stats['mossy_output_norm_1'] = torch.norm(mossy_output1).item()
        
        if projected2 is not None:
            mossy_output2 = self.mossy_fiber(granule_output2)
            stats['mossy_output_norm_2'] = torch.norm(mossy_output2).item()
        
        # CA3递归网络处理（确保维度正确）
        if mossy_output1.shape[-1] != self.ca3_network.hidden_dim:
            # 如果维度不匹配，需要调整
            mossy_output1 = mossy_output1.mean(dim=-1, keepdim=True).expand(-1, self.ca3_network.hidden_dim)
        
        ca3_output1, ca3_stats = self.ca3_network(mossy_output1)
        
        stats.update({
            f'ca3_{k}': v for k, v in ca3_stats.items()
        })
        
        if projected2 is not None:
            # 确保维度正确
            if mossy_output2.shape[-1] != self.ca3_network.hidden_dim:
                mossy_output2 = mossy_output2.mean(dim=-1, keepdim=True).expand(-1, self.ca3_network.hidden_dim)
            ca3_output2, _ = self.ca3_network(mossy_output2)
        
        # 输出投影
        final_output1 = self.output_projection(ca3_output1)
        stats['final_output_norm'] = torch.norm(final_output1).item()
        
        if projected2 is not None:
            final_output2 = self.output_projection(ca3_output2)
            
            # 计算分离效果
            output_similarity = F.cosine_similarity(final_output1, final_output2).mean().item()
            stats['output_similarity'] = output_similarity
            
            # 分离程度
            separation_degree = 1.0 - output_similarity
            stats['separation_degree'] = separation_degree
            
            # 模式分离评估
            separation_input = torch.cat([x1, x2], dim=-1)
            separation_score = self.separation_evaluator(separation_input)
            stats['separation_score'] = separation_score.item()
            
            return final_output1, final_output2, stats
        else:
            return final_output1, None, stats
    
    def compute_separation_metrics(self,
                                 x1: torch.Tensor,
                                 x2: torch.Tensor) -> Dict[str, float]:
        """
        计算详细的模式分离指标
        
        Args:
            x1: 第一组输入
            x2: 第二组输入
            
        Returns:
            分离指标字典
        """
        with torch.no_grad():
            # 前向传播
            sep1, sep2, stats = self.forward(x1, x2)
            
            # 计算多种相似性度量
            metrics = {
                'euclidean_distance': torch.norm(x1 - x2, dim=-1).mean().item(),
                'cosine_similarity': F.cosine_similarity(x1, x2).mean().item(),
                'separated_euclidean': torch.norm(sep1 - sep2, dim=-1).mean().item(),
                'separated_cosine': F.cosine_similarity(sep1, sep2).mean().item(),
                'separation_improvement': stats['input_similarity'] - stats['output_similarity']
            }
            
            # 添加其他统计指标
            metrics.update({
                'input_sparsity': stats['granule_sparsity'],
                'ca3_stability': np.mean(stats['stability_scores']),
                'separation_strength': stats['separation_degree']
            })
            
        return metrics
    
    def adaptive_separate(self,
                         x1: torch.Tensor,
                         x2: torch.Tensor,
                         target_separation: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自适应模式分离
        根据目标分离程度动态调整参数
        
        Args:
            x1: 输入1
            x2: 输入2
            target_separation: 目标分离程度
            
        Returns:
            分离后的表示
        """
        current_separation = 0.0
        iterations = 0
        max_iterations = 10
        
        while current_separation < target_separation and iterations < max_iterations:
            sep1, sep2, stats = self.forward(x1, x2, adaptive_control=True)
            
            current_separation = stats['separation_degree']
            
            # 动态调整分离强度
            if current_separation < target_separation:
                # 增加分离强度
                self.ca3_network.separation_strength = min(
                    self.ca3_network.separation_strength * 1.1, 2.0
                )
                
                # 减少稀疏性以获得更多激活
                self.granule_layer.activation_threshold = max(
                    self.granule_layer.activation_threshold * 0.9, 0.01
                )
            
            iterations += 1
        
        return sep1, sep2
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        granule_stats = {
            'num_granule_cells': self.num_granule_cells,
            'current_sparsity': self.granule_layer.sparsity,
            'activation_threshold': self.granule_layer.activation_threshold
        }
        
        ca3_stats = {
            'num_ca3_cells': self.num_ca3_cells,
            'recurrent_connections': self.ca3_network.recurrent_connections,
            'separation_strength': self.ca3_network.separation_strength,
            'completion_rate': self.ca3_network.completion_rate
        }
        
        return {
            'granule_layer': granule_stats,
            'ca3_network': ca3_stats,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }