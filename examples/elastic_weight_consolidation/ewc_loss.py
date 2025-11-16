"""
EWC损失函数模块

实现弹性权重巩固的核心损失函数，结合Fisher信息矩阵的重要性加权。
基于Kirkpatrick et al. (2017)的理论框架。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class EWCLossFunction(nn.Module):
    """
    弹性权重巩固损失函数
    
    结合新任务损失和Fisher信息矩阵加权的二次惩罚项，
    用于在学习新任务时保护重要参数不被覆盖。
    """
    
    def __init__(self,
                 base_criterion: nn.Module,
                 fisher_dict: Optional[Dict[str, torch.Tensor]] = None,
                 old_params: Optional[Dict[str, torch.Tensor]] = None,
                 lambda_ewc: float = 1000.0,
                 adaptive_lambda: bool = True,
                 lambda_schedule: str = 'cosine',
                 importance_threshold: float = 0.001,
                 use_relative_importance: bool = True):
        """
        初始化EWC损失函数
        
        Args:
            base_criterion: 基础损失函数（如交叉熵）
            fisher_dict: Fisher信息矩阵字典
            old_params: 旧任务参数值字典
            lambda_ewc: EWC权重系数
            adaptive_lambda: 是否启用自适应lambda
            lambda_schedule: lambda调度策略 ('constant', 'cosine', 'linear')
            importance_threshold: 重要性阈值，低于此值的参数不保护
            use_relative_importance: 是否使用相对重要性
        """
        super(EWCLossFunction, self).__init__()
        
        self.base_criterion = base_criterion
        self.fisher_dict = fisher_dict or {}
        self.old_params = old_params or {}
        self.lambda_ewc = lambda_ewc
        self.adaptive_lambda = adaptive_lambda
        self.lambda_schedule = lambda_schedule
        self.importance_threshold = importance_threshold
        self.use_relative_importance = use_relative_importance
        
        # 学习进度跟踪
        self.epoch = 0
        self.num_epochs = 1
        
        # 任务相似度估计
        self.task_similarity = 1.0
        
        logger.info(f"初始化EWC损失函数，lambda_ewc={lambda_ewc}")
    
    def forward(self, 
                outputs: torch.Tensor, 
                targets: torch.Tensor,
                current_params: Dict[str, torch.Tensor],
                epoch: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        前向传播计算EWC损失
        
        Args:
            outputs: 模型输出
            targets: 目标标签
            current_params: 当前参数值字典
            epoch: 当前训练轮次
            
        Returns:
            总损失和各部分损失的字典
        """
        # 计算基础损失
        base_loss = self.base_criterion(outputs, targets)
        
        # 计算EWC损失
        ewc_loss, ewc_components = self.compute_ewc_loss(current_params)
        
        # 计算自适应lambda
        if self.adaptive_lambda:
            adaptive_lambda = self.compute_adaptive_lambda()
        else:
            adaptive_lambda = self.lambda_ewc
        
        # 总损失
        total_loss = base_loss + adaptive_lambda * ewc_loss
        
        # 损失组成信息
        loss_info = {
            'base_loss': base_loss.item(),
            'ewc_loss': ewc_loss.item(),
            'total_loss': total_loss.item(),
            'adaptive_lambda': adaptive_lambda,
            'task_similarity': self.task_similarity
        }
        loss_info.update(ewc_components)
        
        return total_loss, loss_info
    
    def compute_ewc_loss(self, current_params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算EWC损失项
        
        Args:
            current_params: 当前参数值字典
            
        Returns:
            EWC损失值和各参数贡献的字典
        """
        if not self.fisher_dict or not self.old_params:
            return torch.tensor(0.0, device=next(iter(current_params.values())).device), {}
        
        ewc_loss = 0.0
        component_losses = {}
        total_importance = 0.0
        
        device = next(iter(current_params.values())).device
        
        for name in self.fisher_dict.keys():
            if name in current_params and name in self.old_params:
                current_param = current_params[name]
                old_param = self.old_params[name]
                fisher = self.fisher_dict[name]
                
                # 计算参数差异
                param_diff = current_param - old_param
                
                # 应用重要性阈值和相对重要性
                if self.use_relative_importance:
                    # 相对重要性归一化
                    fisher_normalized = self._normalize_fisher(fisher, name)
                else:
                    fisher_normalized = fisher
                
                # 重要性阈值过滤
                important_mask = fisher_normalized > self.importance_threshold
                if important_mask.sum() == 0:
                    continue
                
                # 计算加权二次损失
                masked_diff = param_diff * important_mask
                masked_fisher = fisher_normalized * important_mask
                
                param_loss = (masked_fisher * masked_diff ** 2).sum()
                ewc_loss += param_loss
                
                # 记录组件信息
                component_losses[f'ewc_{name}'] = param_loss.item()
                total_importance += masked_fisher.sum().item()
        
        if len(component_losses) == 0:
            return torch.tensor(0.0, device=device), {}
        
        # 记录统计信息
        component_losses['total_params_protected'] = len(self.fisher_dict)
        component_losses['total_importance'] = total_importance
        component_losses['avg_param_importance'] = total_importance / max(len(self.fisher_dict), 1)
        
        return ewc_loss, component_losses
    
    def _normalize_fisher(self, fisher: torch.Tensor, param_name: str) -> torch.Tensor:
        """
        标准化Fisher信息矩阵
        
        Args:
            fisher: Fisher信息矩阵
            param_name: 参数名
            
        Returns:
            标准化后的Fisher矩阵
        """
        # 使用min-max标准化
        fisher_min = fisher.min()
        fisher_max = fisher.max()
        
        if fisher_max > fisher_min:
            normalized = (fisher - fisher_min) / (fisher_max - fisher_min + 1e-8)
        else:
            normalized = torch.ones_like(fisher)
        
        return normalized
    
    def compute_adaptive_lambda(self) -> float:
        """
        计算自适应lambda权重
        
        Returns:
            自适应lambda值
        """
        if not self.adaptive_lambda:
            return self.lambda_ewc
        
        # 基于训练进度的调度
        progress = self.epoch / max(self.num_epochs, 1)
        
        if self.lambda_schedule == 'cosine':
            # 余弦调度：开始时大（保护旧知识），结束时小（专注新任务）
            adaptive_lambda = self.lambda_ewc * 0.5 * (1 + np.cos(np.pi * progress))
            
        elif self.lambda_schedule == 'linear':
            # 线性调度：随时间线性递减
            adaptive_lambda = self.lambda_ewc * (1 - progress)
            
        elif self.lambda_schedule == 'inverse_sigmoid':
            # 反sigmoid调度：早期强保护，后期放松
            k = 3.0
            adaptive_lambda = self.lambda_ewc / (1 + np.exp(k * (progress - 0.5)))
            
        else:  # constant
            adaptive_lambda = self.lambda_ewc
        
        # 基于任务相似度调整
        adaptive_lambda *= self.task_similarity
        
        return adaptive_lambda
    
    def update_task_similarity(self, 
                             old_dataset_stats: Dict[str, float],
                             new_dataset_stats: Dict[str, float]) -> None:
        """
        更新任务相似度估计
        
        Args:
            old_dataset_stats: 旧任务数据集统计
            new_dataset_stats: 新任务数据集统计
        """
        # 计算分布相似度（简单的特征均值比较）
        if 'feature_mean' in old_dataset_stats and 'feature_mean' in new_dataset_stats:
            old_mean = torch.tensor(old_dataset_stats['feature_mean'])
            new_mean = torch.tensor(new_dataset_stats['feature_mean'])
            
            # 余弦相似度
            similarity = F.cosine_similarity(old_mean.unsqueeze(0), 
                                           new_mean.unsqueeze(0))
            self.task_similarity = max(0.0, min(1.0, similarity.item()))
        else:
            # 默认中等相似度
            self.task_similarity = 0.5
            
        logger.info(f"更新任务相似度: {self.task_similarity:.3f}")
    
    def set_fisher_matrix(self, fisher_dict: Dict[str, torch.Tensor]) -> None:
        """
        设置Fisher信息矩阵
        
        Args:
            fisher_dict: Fisher信息矩阵字典
        """
        self.fisher_dict = fisher_dict
        logger.info(f"设置Fisher信息矩阵，包含 {len(fisher_dict)} 个参数")
    
    def set_old_parameters(self, old_params: Dict[str, torch.Tensor]) -> None:
        """
        设置旧任务参数
        
        Args:
            old_params: 旧任务参数值字典
        """
        self.old_params = old_params
        logger.info(f"设置旧任务参数，包含 {len(old_params)} 个参数")
    
    def set_training_progress(self, epoch: int, num_epochs: int) -> None:
        """
        设置训练进度
        
        Args:
            epoch: 当前轮次
            num_epochs: 总轮次
        """
        self.epoch = epoch
        self.num_epochs = num_epochs
    
    def get_important_parameters(self, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        获取最重要的参数列表
        
        Args:
            top_k: 返回前k个重要参数
            
        Returns:
            (参数名, 重要性分数)元组列表
        """
        if not self.fisher_dict:
            return []
        
        importance_scores = []
        for name, fisher in self.fisher_dict.items():
            # 计算参数的平均重要性
            avg_importance = fisher.mean().item()
            importance_scores.append((name, avg_importance))
        
        # 按重要性排序
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        return importance_scores[:top_k]


class AdaptiveEWC(nn.Module):
    """
    自适应EWC损失函数
    
    动态调整Fisher信息矩阵和权重系数，适应不同任务特性。
    """
    
    def __init__(self,
                 base_criterion: nn.Module,
                 base_lambda: float = 1000.0,
                 fisher_update_frequency: int = 10,
                 importance_decay: float = 0.95,
                 similarity_threshold: float = 0.3):
        """
        初始化自适应EWC
        
        Args:
            base_criterion: 基础损失函数
            base_lambda: 基础lambda权重
            fisher_update_frequency: Fisher矩阵更新频率
            importance_decay: 重要性衰减系数
            similarity_threshold: 相似度阈值
        """
        super(AdaptiveEWC, self).__init__()
        
        self.base_criterion = base_criterion
        self.base_lambda = base_lambda
        self.fisher_update_frequency = fisher_update_frequency
        self.importance_decay = importance_decay
        self.similarity_threshold = similarity_threshold
        
        # 存储历史Fisher矩阵
        self.fisher_history = []
        self.param_history = []
        self.task_similarities = []
        
        self.step = 0
        
    def forward(self, 
                outputs: torch.Tensor,
                targets: torch.Tensor,
                current_params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        自适应EWC前向传播
        """
        base_loss = self.base_criterion(outputs, targets)
        
        # 计算自适应EWC损失
        ewc_loss = self._compute_adaptive_ewc_loss(current_params)
        
        # 自适应lambda计算
        adaptive_lambda = self._compute_adaptive_lambda()
        
        total_loss = base_loss + adaptive_lambda * ewc_loss
        
        loss_info = {
            'base_loss': base_loss.item(),
            'ewc_loss': ewc_loss.item(),
            'total_loss': total_loss.item(),
            'adaptive_lambda': adaptive_lambda
        }
        
        return total_loss, loss_info
    
    def _compute_adaptive_ewc_loss(self, current_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算自适应EWC损失
        """
        if not self.fisher_history:
            return torch.tensor(0.0, device=next(iter(current_params.values())).device)
        
        total_ewc_loss = 0.0
        
        # 组合历史Fisher信息
        combined_fisher = self._combine_historical_fisher()
        
        if combined_fisher:
            for name in combined_fisher.keys():
                if name in current_params and name in self.param_history[-1]:
                    current_param = current_params[name]
                    old_param = self.param_history[-1][name]
                    fisher = combined_fisher[name]
                    
                    param_diff = current_param - old_param
                    ewc_loss = (fisher * param_diff ** 2).sum()
                    total_ewc_loss += ewc_loss
        
        return total_ewc_loss
    
    def _combine_historical_fisher(self) -> Dict[str, torch.Tensor]:
        """
        组合历史的Fisher信息矩阵
        """
        if not self.fisher_history:
            return {}
        
        device = next(iter(self.fisher_history[0].values())).device
        combined_fisher = {}
        
        # 找到所有共同的参数名
        common_params = set(self.fisher_history[0].keys())
        for fisher_dict in self.fisher_history[1:]:
            common_params &= set(fisher_dict.keys())
        
        for name in common_params:
            # 指数移动平均组合
            combined = None
            alpha = 1.0
            
            for fisher_dict in reversed(self.fisher_history):
                if name in fisher_dict:
                    if combined is None:
                        combined = fisher_dict[name].clone()
                    else:
                        combined = alpha * combined + (1 - alpha) * fisher_dict[name]
                    alpha *= self.importance_decay
            
            if combined is not None:
                combined_fisher[name] = combined
        
        return combined_fisher
    
    def _compute_adaptive_lambda(self) -> float:
        """
        计算自适应lambda权重
        """
        if not self.task_similarities:
            return self.base_lambda
        
        # 基于任务相似度的动态调整
        avg_similarity = np.mean(self.task_similarities[-10:])  # 最近10个任务的平均相似度
        
        if avg_similarity < self.similarity_threshold:
            # 低相似度：减少保护强度
            lambda_factor = avg_similarity / self.similarity_threshold
        else:
            # 高相似度：增加保护强度
            lambda_factor = 1.0 + (avg_similarity - self.similarity_threshold)
        
        return self.base_lambda * lambda_factor
    
    def update_fisher_matrix(self, fisher_dict: Dict[str, torch.Tensor]) -> None:
        """
        更新Fisher信息矩阵历史
        """
        self.fisher_history.append(fisher_dict)
        self.step += 1
        
        # 限制历史长度
        if len(self.fisher_history) > 5:
            self.fisher_history.pop(0)
    
    def update_old_parameters(self, old_params: Dict[str, torch.Tensor]) -> None:
        """
        更新旧参数历史
        """
        self.param_history.append(old_params)
        
        # 限制历史长度
        if len(self.param_history) > 5:
            self.param_history.pop(0)
    
    def update_task_similarity(self, similarity: float) -> None:
        """
        更新任务相似度历史
        """
        self.task_similarities.append(similarity)
        
        # 限制历史长度
        if len(self.task_similarities) > 20:
            self.task_similarities.pop(0)


class ProportionalEWC(nn.Module):
    """
    比例EWC损失函数
    
    根据参数分布动态调整EWC损失的比例权重。
    """
    
    def __init__(self, 
                 base_criterion: nn.Module,
                 base_lambda: float = 1000.0,
                 distribution_aware: bool = True,
                 entropy_threshold: float = 0.1):
        super(ProportionalEWC, self).__init__()
        
        self.base_criterion = base_criterion
        self.base_lambda = base_lambda
        self.distribution_aware = distribution_aware
        self.entropy_threshold = entropy_threshold
    
    def forward(self,
                outputs: torch.Tensor,
                targets: torch.Tensor,
                current_params: Dict[str, torch.Tensor],
                old_params: Dict[str, torch.Tensor],
                fisher_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        比例EWC前向传播
        """
        base_loss = self.base_criterion(outputs, targets)
        
        # 计算比例因子
        proportional_factor = self._compute_proportional_factor(
            current_params, old_params, fisher_dict)
        
        # 计算EWC损失
        ewc_loss = self._compute_weighted_ewc_loss(
            current_params, old_params, fisher_dict, proportional_factor)
        
        total_loss = base_loss + self.base_lambda * ewc_loss
        
        loss_info = {
            'base_loss': base_loss.item(),
            'ewc_loss': ewc_loss.item(),
            'total_loss': total_loss.item(),
            'proportional_factor': proportional_factor.mean().item()
        }
        
        return total_loss, loss_info
    
    def _compute_proportional_factor(self,
                                   current_params: Dict[str, torch.Tensor],
                                   old_params: Dict[str, torch.Tensor],
                                   fisher_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算比例因子
        """
        device = next(iter(current_params.values())).device
        
        if not self.distribution_aware:
            return torch.ones(1, device=device)
        
        # 计算参数变化分布
        all_factors = []
        
        for name in fisher_dict.keys():
            if name in current_params and name in old_params:
                current = current_params[name]
                old = old_params[name]
                fisher = fisher_dict[name]
                
                # 参数变化幅度
                param_change = (current - old).abs()
                relative_change = param_change / (old.abs() + 1e-8)
                
                # 归一化变化幅度
                normalized_change = torch.clamp(relative_change, 0, 1)
                
                # Fisher重要性权重
                fisher_weight = fisher / (fisher.max() + 1e-8)
                
                # 比例因子：变化小且重要性高的参数需要更强保护
                factor = fisher_weight / (normalized_change + 0.01)
                
                all_factors.extend(factor.flatten())
        
        if not all_factors:
            return torch.ones(1, device=device)
        
        factors = torch.stack(all_factors)
        # 归一化比例因子
        factors = torch.clamp(factors / (factors.max() + 1e-8), 0, 1)
        
        return factors.mean()
    
    def _compute_weighted_ewc_loss(self,
                                 current_params: Dict[str, torch.Tensor],
                                 old_params: Dict[str, torch.Tensor],
                                 fisher_dict: Dict[str, torch.Tensor],
                                 weight_factor: float) -> torch.Tensor:
        """
        计算加权EWC损失
        """
        ewc_loss = 0.0
        
        for name in fisher_dict.keys():
            if name in current_params and name in old_params:
                current = current_params[name]
                old = old_params[name]
                fisher = fisher_dict[name]
                
                param_diff = current - old
                weighted_loss = weight_factor * fisher * param_diff ** 2
                ewc_loss += weighted_loss.sum()
        
        return ewc_loss