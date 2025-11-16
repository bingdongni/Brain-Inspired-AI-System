"""
Fisher信息矩阵计算模块

计算Fisher信息矩阵对角近似，用于估计神经网络参数的重要性。
这是弹性权重巩固(EWC)算法的核心组件。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class FisherInformationMatrix:
    """
    Fisher信息矩阵计算器
    
    实现Fisher信息矩阵的对角近似，用于估计参数重要性。
    支持对角、块对角和全矩阵三种计算模式。
    """
    
    def __init__(self, 
                 model: nn.Module,
                 mode: str = 'diagonal',
                 stable_numeric: bool = True,
                 epsilon: float = 1e-8):
        """
        初始化Fisher信息矩阵计算器
        
        Args:
            model: 神经网络模型
            mode: 计算模式 ('diagonal', 'block_diagonal', 'full')
            stable_numeric: 是否启用数值稳定化
            epsilon: 数值稳定化参数
        """
        self.model = model
        self.mode = mode
        self.stable_numeric = stable_numeric
        self.epsilon = epsilon
        self.fisher_matrices = {}
        self.num_samples = 0
        
        # 验证模式
        if mode not in ['diagonal', 'block_diagonal', 'full']:
            raise ValueError(f"不支持的Fisher模式: {mode}")
            
        logger.info(f"初始化Fisher信息矩阵计算器，模式: {mode}")
    
    def compute_fisher_diagonal(self, 
                              data_loader: torch.utils.data.DataLoader,
                              num_samples: Optional[int] = None,
                              device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        计算Fisher信息矩阵对角近似
        
        Args:
            data_loader: 数据加载器
            num_samples: 使用的样本数量，None表示使用全部
            device: 计算设备
            
        Returns:
            Fisher对角元素的字典，键为参数名，值为对角元素
        """
        if device is None:
            device = next(self.model.parameters()).device
            
        fisher_dict = {}
        
        # 初始化Fisher矩阵
        for name, param in self.model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param, device=device)
        
        sample_count = 0
        max_samples = num_samples if num_samples is not None else len(data_loader)
        
        logger.info(f"开始计算Fisher对角近似，目标样本数: {max_samples}")
        
        # 确保模型处于训练模式以计算梯度
        self.model.train()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
                if sample_count >= max_samples:
                    break
                    
                inputs, targets = inputs.to(device), targets.to(device)
                batch_size = inputs.size(0)
                
                # 前向传播
                outputs = self.model(inputs)
                log_probs = F.log_softmax(outputs, dim=-1)
                
                # 对每个样本计算梯度
                for i in range(batch_size):
                    if sample_count >= max_samples:
                        break
                        
                    # 计算单样本的梯度
                    sample_log_probs = log_probs[i:i+1]
                    sample_targets = targets[i:i+1]
                    
                    # 计算对数概率的梯度
                    self.model.zero_grad()
                    log_prob = torch.gather(sample_log_probs, 1, sample_targets.unsqueeze(1))
                    loss = -log_prob.sum()
                    
                    loss.backward(retain_graph=True)
                    
                    # 累积Fisher信息
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad = param.grad.detach()
                            fisher_dict[name] += grad ** 2
                    
                    sample_count += 1
                    
                    if batch_idx % 100 == 0:
                        logger.info(f"已处理样本: {sample_count}/{max_samples}")
        
        # 平均化
        for name in fisher_dict:
            fisher_dict[name] /= sample_count
            
        self.fisher_matrices = fisher_dict
        self.num_samples = sample_count
        
        logger.info(f"Fisher对角计算完成，处理样本数: {sample_count}")
        return fisher_dict
    
    def compute_fisher_empirical(self,
                               data_loader: torch.utils.data.DataLoader,
                               num_samples: Optional[int] = None,
                               device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        计算经验Fisher信息矩阵（使用真实标签）
        
        Args:
            data_loader: 数据加载器
            num_samples: 使用的样本数量
            device: 计算设备
            
        Returns:
            Fisher信息矩阵字典
        """
        if device is None:
            device = next(self.model.parameters()).device
            
        fisher_dict = {}
        
        # 初始化Fisher矩阵
        for name, param in self.model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param, device=device)
        
        sample_count = 0
        max_samples = num_samples if num_samples is not None else len(data_loader)
        
        # 在训练模式下计算梯度
        self.model.train()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            if sample_count >= max_samples:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算交叉熵损失的梯度
            self.model.zero_grad()
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            
            # 累积Fisher信息
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_dict[name] += param.grad ** 2
            
            sample_count += inputs.size(0)
                
        # 平均化
        for name in fisher_dict:
            fisher_dict[name] /= sample_count
            
        self.fisher_matrices = fisher_dict
        self.num_samples = sample_count
        
        return fisher_dict
    
    def compute_fisher_natural_gradient(self,
                                      data_loader: torch.utils.data.DataLoader,
                                      num_samples: Optional[int] = None,
                                      device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """
        计算自然梯度的Fisher信息矩阵
        
        Args:
            data_loader: 数据加载器
            num_samples: 使用的样本数量
            device: 计算设备
            
        Returns:
            Fisher信息矩阵字典
        """
        if device is None:
            device = next(self.model.parameters()).device
            
        # 自然梯度Fisher需要二阶信息
        # 这里实现一个简化版本，使用对角近似
        return self.compute_fisher_diagonal(data_loader, num_samples, device)
    
    def update_fisher(self,
                     new_fisher: Dict[str, torch.Tensor],
                     alpha: float = 0.5) -> None:
        """
        更新Fisher信息矩阵（指数移动平均）
        
        Args:
            new_fisher: 新的Fisher信息矩阵
            alpha: 平滑系数 (0-1)
        """
        if not self.fisher_matrices:
            self.fisher_matrices = new_fisher
            return
            
        for name in new_fisher:
            if name in self.fisher_matrices:
                self.fisher_matrices[name] = (
                    alpha * self.fisher_matrices[name] + 
                    (1 - alpha) * new_fisher[name]
                )
            else:
                self.fisher_matrices[name] = new_fisher[name]
    
    def get_parameter_importance(self, 
                               layer_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        获取参数重要性分数
        
        Args:
            layer_names: 指定的层名列表，None表示所有层
            
        Returns:
            参数重要性字典
        """
        if not self.fisher_matrices:
            raise ValueError("Fisher信息矩阵尚未计算")
            
        importance = {}
        for name, fisher in self.fisher_matrices.items():
            if layer_names is None or any(layer_name in name for layer_name in layer_names):
                # 计算相对重要性（归一化）
                relative_importance = fisher / (fisher.max() + self.epsilon)
                importance[name] = relative_importance
                
        return importance
    
    def compute_fisher_statistics(self) -> Dict[str, float]:
        """
        计算Fisher信息矩阵的统计信息
        
        Returns:
            统计信息字典
        """
        if not self.fisher_matrices:
            return {}
            
        stats = {}
        total_importance = 0.0
        num_params = 0
        
        for name, fisher in self.fisher_matrices.values():
            importance_sum = fisher.sum().item()
            total_importance += importance_sum
            num_params += fisher.numel()
            
        stats['total_importance'] = total_importance
        stats['num_parameters'] = num_params
        stats['avg_importance'] = total_importance / max(num_params, 1)
        
        # 计算top重要参数
        all_importances = []
        for fisher in self.fisher_matrices.values():
            all_importances.extend(fisher.flatten().tolist())
            
        all_importances.sort(reverse=True)
        if all_importances:
            stats['top_1_percent_importance'] = np.percentile(all_importances, 99)
            stats['top_10_percent_importance'] = np.percentile(all_importances, 90)
            
        return stats
    
    def save_fisher(self, filepath: str) -> None:
        """
        保存Fisher信息矩阵到文件
        
        Args:
            filepath: 保存路径
        """
        if not self.fisher_matrices:
            raise ValueError("Fisher信息矩阵尚未计算")
            
        fisher_data = {}
        for name, fisher in self.fisher_matrices.items():
            fisher_data[name] = fisher.cpu().numpy()
            
        np.save(filepath, fisher_data)
        logger.info(f"Fisher信息矩阵已保存到: {filepath}")
    
    def load_fisher(self, filepath: str, device: Optional[torch.device] = None) -> None:
        """
        从文件加载Fisher信息矩阵
        
        Args:
            filepath: 文件路径
            device: 目标设备
        """
        if device is None:
            device = next(self.model.parameters()).device
            
        fisher_data = np.load(filepath, allow_pickle=True).item()
        
        self.fisher_matrices = {}
        for name, fisher_np in fisher_data.items():
            self.fisher_matrices[name] = torch.from_numpy(fisher_np).to(device)
            
        logger.info(f"Fisher信息矩阵已从 {filepath} 加载")
    
    def normalize_fisher(self, method: str = 'min_max') -> Dict[str, torch.Tensor]:
        """
        标准化Fisher信息矩阵
        
        Args:
            method: 标准化方法 ('min_max', 'z_score', 'robust')
            
        Returns:
            标准化后的Fisher字典
        """
        if not self.fisher_matrices:
            raise ValueError("Fisher信息矩阵尚未计算")
            
        normalized_fisher = {}
        
        if method == 'min_max':
            for name, fisher in self.fisher_matrices.items():
                fisher_min = fisher.min()
                fisher_max = fisher.max()
                if fisher_max > fisher_min:
                    normalized_fisher[name] = (fisher - fisher_min) / (fisher_max - fisher_min + self.epsilon)
                else:
                    normalized_fisher[name] = torch.ones_like(fisher)
                    
        elif method == 'z_score':
            for name, fisher in self.fisher_matrices.items():
                mean_val = fisher.mean()
                std_val = fisher.std()
                if std_val > self.epsilon:
                    normalized_fisher[name] = (fisher - mean_val) / (std_val + self.epsilon)
                else:
                    normalized_fisher[name] = torch.zeros_like(fisher)
                    
        elif method == 'robust':
            for name, fisher in self.fisher_matrices.items():
                median_val = fisher.median()
                mad = (fisher - median_val).abs().median()
                if mad > self.epsilon:
                    normalized_fisher[name] = (fisher - median_val) / (mad + self.epsilon)
                else:
                    normalized_fisher[name] = torch.zeros_like(fisher)
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
            
        return normalized_fisher


class EfficientFisherComputation:
    """
    高效Fisher信息矩阵计算器
    使用蒙特卡洛采样和近似方法加速计算
    """
    
    def __init__(self, 
                 model: nn.Module,
                 sample_size: int = 1000,
                 importance_threshold: float = 0.01):
        """
        初始化高效Fisher计算器
        
        Args:
            model: 神经网络模型
            sample_size: 采样大小
            importance_threshold: 重要性阈值
        """
        self.model = model
        self.sample_size = sample_size
        self.importance_threshold = importance_threshold
    
    def monte_carlo_fisher(self,
                          data_loader: torch.utils.data.DataLoader,
                          num_mc_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        蒙特卡洛Fisher信息矩阵计算
        
        Args:
            data_loader: 数据加载器
            num_mc_samples: 蒙特卡洛采样次数
            
        Returns:
            Fisher信息矩阵字典
        """
        device = next(self.model.parameters()).device
        fisher_accumulator = defaultdict(lambda: torch.zeros(1, device=device))
        
        # 获取采样数据
        samples = self._sample_data(data_loader, self.sample_size)
        
        for mc_step in range(num_mc_samples):
            # 添加噪声模拟分布
            noisy_samples = self._add_noise(samples)
            
            # 计算单步Fisher
            step_fisher = self._compute_step_fisher(noisy_samples)
            
            # 累积
            for name, fisher in step_fisher.items():
                fisher_accumulator[name] += fisher / num_mc_samples
                
        return dict(fisher_accumulator)
    
    def _sample_data(self, data_loader: torch.utils.data.DataLoader, sample_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样数据"""
        device = next(self.model.parameters()).device
        samples = []
        targets = []
        
        for inputs, target in data_loader:
            samples.append(inputs)
            targets.append(target)
            if len(samples) * inputs.size(0) >= sample_size:
                break
                
        samples = torch.cat(samples, dim=0)[:sample_size].to(device)
        targets = torch.cat(targets, dim=0)[:sample_size].to(device)
        
        return samples, targets
    
    def _add_noise(self, samples: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """添加高斯噪声模拟数据分布"""
        inputs, targets = samples
        noise_std = 0.01  # 噪声标准差
        noisy_inputs = inputs + torch.randn_like(inputs) * noise_std
        return noisy_inputs, targets
    
    def _compute_step_fisher(self, samples: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算单步Fisher信息"""
        inputs, targets = samples
        self.model.eval()
        
        fisher_dict = {}
        for name, param in self.model.named_parameters():
            fisher_dict[name] = torch.zeros_like(param)
        
        with torch.no_grad():
            outputs = self.model(inputs)
            log_probs = F.log_softmax(outputs, dim=-1)
            
            for i in range(inputs.size(0)):
                sample_log_probs = log_probs[i:i+1]
                sample_targets = targets[i:i+1]
                
                self.model.zero_grad()
                log_prob = torch.gather(sample_log_probs, 1, sample_targets.unsqueeze(1))
                loss = -log_prob.sum()
                
                loss.backward(retain_graph=True)
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_dict[name] += param.grad ** 2
        
        # 平均化
        for name in fisher_dict:
            fisher_dict[name] /= inputs.size(0)
            
        return fisher_dict