"""
知识蒸馏机制实现 (Knowledge Distillation)

基于终身学习研究报告的理论基础，实现高效的知识蒸馏和迁移机制：
1. 基础知识蒸馏 - 教师-学生模型知识传递
2. 温度软化蒸馏 - 使用温度参数的软标签蒸馏
3. 特征蒸馏 - 中间层特征的知识传递
4. 自适应蒸馏 - 根据任务相似度调整蒸馏强度
5. 在线蒸馏和离线蒸馏
6. 渐进式蒸馏 - 多阶段知识传递

知识蒸馏的核心思想是通过教师模型的输出分布来指导学生模型的学习，
实现知识的高效迁移，特别适用于解决终身学习中的灾难性遗忘问题。

Author: Lifelong Learning Team  
Date: 2025-11-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import copy
import math
from collections import defaultdict
import warnings


class DistillationLoss(nn.Module):
    """蒸馏损失基类"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7, beta: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # 蒸馏损失权重
        self.beta = beta    # 原始损失权重
    
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                target_outputs: Optional[torch.Tensor] = None,
                target_labels: Optional[torch.Tensor] = None,
                criterion: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """计算蒸馏损失
        
        Args:
            student_outputs: 学生模型输出
            teacher_outputs: 教师模型输出
            target_outputs: 目标输出（可选）
            target_labels: 真实标签（可选）
            criterion: 原始任务损失函数（可选）
        """
        raise NotImplementedError


class TemperatureSoftmax(nn.Module):
    """温度软化模块"""
    
    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """应用温度软化的Softmax"""
        if self.temperature <= 0:
            warnings.warn(f"温度参数 {self.temperature} 必须大于0，使用1.0替代")
            temperature = 1.0
        else:
            temperature = self.temperature
            
        return F.softmax(logits / temperature, dim=dim)
    
    def log_softmax(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """应用温度软化的LogSoftmax"""
        if self.temperature <= 0:
            temperature = 1.0
        else:
            temperature = self.temperature
            
        return F.log_softmax(logits / temperature, dim=dim)


class KLDivLoss(nn.Module):
    """KL散度损失"""
    
    def __init__(self, temperature: float = 4.0, reduction: str = 'batchmean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """计算KL散度蒸馏损失"""
        # 应用温度软化
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # 计算KL散度
        kl_loss = F.kl_div(
            student_log_probs, 
            teacher_probs, 
            reduction=self.reduction
        )
        
        # 乘以温度平方
        return kl_loss * (self.temperature ** 2)


class LogitsDistillation(DistillationLoss):
    """基于Logits的知识蒸馏"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7, beta: float = 1.0,
                 kl_div_loss: bool = True):
        super().__init__(temperature, alpha, beta)
        self.kl_div_loss = kl_div_loss
        
        if self.kl_div_loss:
            self.kl_criterion = KLDivLoss(temperature)
    
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                target_outputs: Optional[torch.Tensor] = None,
                target_labels: Optional[torch.Tensor] = None,
                criterion: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """计算Logits蒸馏损失"""
        losses = {}
        
        # 1. 蒸馏损失
        if self.kl_div_loss:
            distillation_loss = self.kl_criterion(student_outputs, teacher_outputs)
        else:
            # MSE损失
            distillation_loss = F.mse_loss(
                F.softmax(student_outputs / self.temperature, dim=-1),
                F.softmax(teacher_outputs / self.temperature, dim=-1)
            )
        
        losses['distillation_loss'] = distillation_loss
        
        # 2. 原始任务损失
        if target_labels is not None and criterion is not None:
            task_loss = criterion(student_outputs, target_labels)
            losses['task_loss'] = task_loss
        elif target_outputs is not None:
            task_loss = F.mse_loss(student_outputs, target_outputs)
            losses['task_loss'] = task_loss
        else:
            task_loss = torch.tensor(0.0, device=student_outputs.device)
            losses['task_loss'] = task_loss
        
        # 3. 总损失
        total_loss = self.alpha * distillation_loss + self.beta * task_loss
        losses['total_loss'] = total_loss
        
        return losses


class FeatureDistillation(DistillationLoss):
    """基于特征的蒸馏损失"""
    
    def __init__(self, temperature: float = 1.0, alpha: float = 0.5, beta: float = 1.0,
                 feature_matching: str = 'l2', normalize: bool = True):
        super().__init__(temperature, alpha, beta)
        self.feature_matching = feature_matching
        self.normalize = normalize
    
    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor,
                target_outputs: Optional[torch.Tensor] = None,
                target_labels: Optional[torch.Tensor] = None,
                criterion: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """计算特征蒸馏损失"""
        losses = {}
        
        # 特征归一化
        if self.normalize:
            student_features = F.normalize(student_features, p=2, dim=-1)
            teacher_features = F.normalize(teacher_features, p=2, dim=-1)
        
        # 特征蒸馏损失
        if self.feature_matching == 'l2':
            feature_loss = F.mse_loss(student_features, teacher_features)
        elif self.feature_matching == 'cosine':
            feature_loss = 1 - F.cosine_similarity(
                student_features, teacher_features, dim=-1
            ).mean()
        else:
            raise ValueError(f"不支持的特征匹配方式: {self.feature_matching}")
        
        losses['feature_distillation_loss'] = feature_loss
        
        # 原始任务损失
        if target_labels is not None and criterion is not None:
            task_loss = criterion(target_outputs, target_labels)
            losses['task_loss'] = task_loss
        elif target_outputs is not None:
            task_loss = F.mse_loss(target_outputs, target_outputs)
            losses['task_loss'] = task_loss
        else:
            task_loss = torch.tensor(0.0, device=student_features.device)
            losses['task_loss'] = task_loss
        
        # 总损失
        total_loss = self.alpha * feature_loss + self.beta * task_loss
        losses['total_loss'] = total_loss
        
        return losses


class AttentionDistillation(DistillationLoss):
    """基于注意力机制的蒸馏损失"""
    
    def __init__(self, temperature: float = 1.0, alpha: float = 0.6, beta: float = 1.0,
                 attention_type: str = 'spatial', normalize: bool = True):
        super().__init__(temperature, alpha, beta)
        self.attention_type = attention_type
        self.normalize = normalize
    
    def forward(self, student_attention: torch.Tensor, teacher_attention: torch.Tensor,
                target_outputs: Optional[torch.Tensor] = None,
                target_labels: Optional[torch.Tensor] = None,
                criterion: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """计算注意力蒸馏损失"""
        losses = {}
        
        # 计算注意力权重
        if self.attention_type == 'spatial':
            # 空间注意力
            student_att = torch.softmax(student_attention.view(*student_attention.shape[:2], -1), dim=-1)
            teacher_att = torch.softmax(teacher_attention.view(*teacher_attention.shape[:1], -1), dim=-1)
        elif self.attention_type == 'channel':
            # 通道注意力
            student_att = torch.softmax(student_attention.mean(dim=[-1, -2]), dim=-1)
            teacher_att = torch.softmax(teacher_attention.mean(dim=[-1, -2]), dim=-1)
        else:
            raise ValueError(f"不支持的注意力类型: {self.attention_type}")
        
        # 注意力蒸馏损失
        attention_loss = F.mse_loss(student_att, teacher_att)
        losses['attention_distillation_loss'] = attention_loss
        
        # 原始任务损失
        if target_labels is not None and criterion is not None:
            task_loss = criterion(target_outputs, target_labels)
            losses['task_loss'] = task_loss
        else:
            task_loss = torch.tensor(0.0, device=student_attention.device)
            losses['task_loss'] = task_loss
        
        # 总损失
        total_loss = self.alpha * attention_loss + self.beta * task_loss
        losses['total_loss'] = total_loss
        
        return losses


class AdaptiveDistillation(DistillationLoss):
    """自适应蒸馏损失"""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7, beta: float = 1.0,
                 similarity_threshold: float = 0.5, adaptation_strategy: str = 'similarity'):
        super().__init__(temperature, alpha, beta)
        self.similarity_threshold = similarity_threshold
        self.adaptation_strategy = adaptation_strategy
        self.task_similarities = []
        self.adaptation_history = []
    
    def compute_task_similarity(self, task1_features: torch.Tensor, 
                              task2_features: torch.Tensor) -> float:
        """计算任务间相似度"""
        # 使用余弦相似度
        similarity = F.cosine_similarity(
            task1_features.mean(dim=0), 
            task2_features.mean(dim=0), 
            dim=0
        )
        return similarity.item()
    
    def adaptive_alpha(self, similarity: float) -> float:
        """根据任务相似度自适应调整蒸馏权重"""
        if self.adaptation_strategy == 'similarity':
            # 相似度越高，蒸馏权重越大
            if similarity > self.similarity_threshold:
                return min(self.alpha * (1 + similarity), 1.0)
            else:
                return max(self.alpha * 0.5, 0.1)
        elif self.adaptation_strategy == 'inverse_similarity':
            # 相似度越低，蒸馏权重越大
            return self.alpha * (2 - similarity)
        else:
            return self.alpha
    
    def forward(self, student_outputs: torch.Tensor, teacher_outputs: torch.Tensor,
                student_features: Optional[torch.Tensor] = None,
                teacher_features: Optional[torch.Tensor] = None,
                target_outputs: Optional[torch.Tensor] = None,
                target_labels: Optional[torch.Tensor] = None,
                criterion: Optional[nn.Module] = None) -> Dict[str, torch.Tensor]:
        """自适应蒸馏损失"""
        losses = {}
        
        # 计算任务相似度
        if student_features is not None and teacher_features is not None:
            similarity = self.compute_task_similarity(student_features, teacher_features)
            self.task_similarities.append(similarity)
            
            # 自适应蒸馏权重
            adaptive_alpha = self.adaptive_alpha(similarity)
            self.adaptation_history.append(adaptive_alpha)
        else:
            similarity = 0.5  # 默认相似度
            adaptive_alpha = self.alpha
        
        # KL散度蒸馏损失
        kl_loss = KLDivLoss(self.temperature)(student_outputs, teacher_outputs)
        losses['distillation_loss'] = kl_loss
        
        # 原始任务损失
        if target_labels is not None and criterion is not None:
            task_loss = criterion(student_outputs, target_labels)
            losses['task_loss'] = task_loss
        elif target_outputs is not None:
            task_loss = F.mse_loss(student_outputs, target_outputs)
            losses['task_loss'] = task_loss
        else:
            task_loss = torch.tensor(0.0, device=student_outputs.device)
            losses['task_loss'] = task_loss
        
        # 自适应总损失
        total_loss = adaptive_alpha * kl_loss + self.beta * task_loss
        losses['total_loss'] = total_loss
        losses['task_similarity'] = similarity
        losses['adaptive_alpha'] = adaptive_alpha
        
        return losses


class ProgressiveDistillation:
    """渐进式蒸馏"""
    
    def __init__(self, num_stages: int = 3, stage_weights: Optional[List[float]] = None):
        self.num_stages = num_stages
        self.stage_weights = stage_weights or [1.0 / num_stages] * num_stages
        self.current_stage = 0
        self.stage_progress = 0.0
    
    def get_current_weight(self) -> float:
        """获取当前阶段的权重"""
        if self.current_stage < len(self.stage_weights):
            return self.stage_weights[self.current_stage]
        return 1.0
    
    def update_progress(self, progress: float):
        """更新进度"""
        self.stage_progress = progress
        
        # 检查是否需要进入下一阶段
        if self.current_stage < self.num_stages - 1:
            next_stage_threshold = (self.current_stage + 1) / self.num_stages
            if progress >= next_stage_threshold:
                self.current_stage += 1
                self.stage_progress = 0.0
    
    def get_stage_distillation_loss(self, student_outputs: torch.Tensor, 
                                  teacher_outputs: torch.Tensor,
                                  temperature: float = 4.0) -> torch.Tensor:
        """获取当前阶段的蒸馏损失"""
        stage_weight = self.get_current_weight()
        return stage_weight * KLDivLoss(temperature)(student_outputs, teacher_outputs)


class KnowledgeDistillationTrainer:
    """知识蒸馏训练器"""
    
    def __init__(self, student_model: nn.Module, teacher_model: nn.Module,
                 distillation_type: str = 'logits', temperature: float = 4.0,
                 alpha: float = 0.7, beta: float = 1.0, device: str = 'cpu',
                 criterion: Optional[nn.Module] = None):
        
        self.device = device
        self.student_model = student_model.to(device)
        self.teacher_model = teacher_model.to(device)
        
        # 冻结教师模型
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        self.teacher_model.eval()
        
        # 蒸馏损失函数
        if distillation_type == 'logits':
            self.distillation_loss = LogitsDistillation(temperature, alpha, beta)
        elif distillation_type == 'feature':
            self.distillation_loss = FeatureDistillation(temperature, alpha, beta)
        elif distillation_type == 'attention':
            self.distillation_loss = AttentionDistillation(temperature, alpha, beta)
        elif distillation_type == 'adaptive':
            self.distillation_loss = AdaptiveDistillation(temperature, alpha, beta)
        else:
            raise ValueError(f"不支持的蒸馏类型: {distillation_type}")
        
        # 原始任务损失
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.progressive_distillation = None
        
        # 训练历史
        self.training_history = defaultdict(list)
    
    def enable_progressive_distillation(self, num_stages: int = 3,
                                      stage_weights: Optional[List[float]] = None):
        """启用渐进式蒸馏"""
        self.progressive_distillation = ProgressiveDistillation(num_stages, stage_weights)
    
    def train_step(self, batch_data: torch.Tensor, batch_labels: torch.Tensor,
                   optimizer: torch.optim.Optimizer,
                   feature_layers: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """一步训练"""
        self.student_model.train()
        
        # 移动到设备
        batch_data = batch_data.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # 前向传播 - 学生模型
        student_outputs = self.student_model(batch_data)
        
        # 前向传播 - 教师模型
        with torch.no_grad():
            teacher_outputs = self.teacher_model(batch_data)
        
        # 特征蒸馏
        feature_losses = {}
        if feature_layers and hasattr(self.student_model, 'get_features'):
            student_features = self.student_model.get_features()
            teacher_features = self.teacher_model.get_features()
            
            for layer in feature_layers:
                if layer in student_features and layer in teacher_features:
                    feature_loss = FeatureDistillation()
                    feature_losses[f'{layer}_loss'] = feature_loss(
                        student_features[layer], teacher_features[layer]
                    )['feature_distillation_loss']
        
        # 计算蒸馏损失
        distillation_results = self.distillation_loss(
            student_outputs, teacher_outputs,
            target_labels=batch_labels,
            criterion=self.criterion
        )
        
        # 如果启用渐进式蒸馏
        if self.progressive_distillation:
            progressive_loss = self.progressive_distillation.get_stage_distillation_loss(
                student_outputs, teacher_outputs
            )
            distillation_results['total_loss'] += progressive_loss
            distillation_results['progressive_loss'] = progressive_loss
        
        # 总损失（包含特征损失）
        total_loss = distillation_results['total_loss']
        if feature_losses:
            feature_loss_sum = sum(feature_losses.values())
            total_loss += 0.1 * feature_loss_sum  # 特征损失权重
            distillation_results['feature_loss'] = feature_loss_sum
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 记录训练历史
        for key, value in distillation_results.items():
            if isinstance(value, torch.Tensor):
                self.training_history[key].append(value.item())
            else:
                self.training_history[key].append(value)
        
        return distillation_results
    
    def online_distillation_train(self, new_task_data: DataLoader,
                                optimizer: torch.optim.Optimizer,
                                num_epochs: int = 10) -> Dict[str, List[float]]:
        """在线蒸馏训练"""
        self.student_model.train()
        self.teacher_model.eval()
        
        training_logs = defaultdict(list)
        
        for epoch in range(num_epochs):
            epoch_losses = defaultdict(list)
            
            for batch_idx, (data, labels) in enumerate(new_task_data):
                results = self.train_step(data, labels, optimizer)
                
                for key, value in results.items():
                    epoch_losses[key].append(value)
            
            # 记录epoch统计
            for key, values in epoch_losses.items():
                training_logs[f'{key}_epoch_{epoch}'] = np.mean(values)
        
        return training_logs
    
    def offline_distillation_train(self, teacher_data: DataLoader,
                                 student_data: DataLoader,
                                 optimizer: torch.optim.Optimizer,
                                 num_epochs: int = 20,
                                 temperature_schedule: Optional[List[float]] = None) -> Dict[str, List[float]]:
        """离线蒸馏训练"""
        training_logs = defaultdict(list)
        
        if temperature_schedule is None:
            temperature_schedule = [4.0] * num_epochs
        
        for epoch in range(num_epochs):
            # 调整温度
            if hasattr(self.distillation_loss, 'temperature'):
                self.distillation_loss.temperature = temperature_schedule[epoch]
            
            epoch_losses = defaultdict(list)
            
            # 交替使用教师数据和学生数据
            teacher_iter = iter(teacher_data)
            student_iter = iter(student_data)
            max_batches = max(len(teacher_data), len(student_data))
            
            for batch_idx in range(max_batches):
                # 教师数据训练
                try:
                    teacher_batch = next(teacher_iter)
                    results = self.train_step(teacher_batch[0], teacher_batch[1], optimizer)
                except StopIteration:
                    teacher_iter = iter(teacher_data)
                    teacher_batch = next(teacher_iter)
                    results = self.train_step(teacher_batch[0], teacher_batch[1], optimizer)
                
                for key, value in results.items():
                    epoch_losses[key].append(value)
            
            # 记录epoch统计
            for key, values in epoch_losses.items():
                training_logs[f'{key}_epoch_{epoch}'] = np.mean(values)
        
        return training_logs
    
    def evaluate_distillation(self, test_data: DataLoader) -> Dict[str, float]:
        """评估蒸馏效果"""
        self.student_model.eval()
        self.teacher_model.eval()
        
        student_correct = 0
        teacher_correct = 0
        total_samples = 0
        total_kl_divergence = 0.0
        
        with torch.no_grad():
            for data, labels in test_data:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                student_outputs = self.student_model(data)
                teacher_outputs = self.teacher_model(data)
                
                # 计算准确率
                student_pred = torch.argmax(student_outputs, dim=1)
                teacher_pred = torch.argmax(teacher_outputs, dim=1)
                
                student_correct += (student_pred == labels).sum().item()
                teacher_correct += (teacher_pred == labels).sum().item()
                total_samples += labels.size(0)
                
                # 计算KL散度
                student_log_probs = F.log_softmax(student_outputs, dim=-1)
                teacher_probs = F.softmax(teacher_outputs, dim=-1)
                kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
                total_kl_divergence += kl_div.item()
        
        # 计算指标
        student_accuracy = student_correct / total_samples
        teacher_accuracy = teacher_correct / total_samples
        average_kl_divergence = total_kl_divergence / len(test_data)
        
        return {
            'student_accuracy': student_accuracy,
            'teacher_accuracy': teacher_accuracy,
            'accuracy_gap': teacher_accuracy - student_accuracy,
            'average_kl_divergence': average_kl_divergence,
            'distillation_quality': 1.0 / (1.0 + average_kl_divergence)  # 越高越好
        }


class ProgressiveKnowledgeDistillation:
    """渐进式知识蒸馏管理器"""
    
    def __init__(self, num_tasks: int, base_temperature: float = 4.0,
                 adaptation_strategy: str = 'linear', min_alpha: float = 0.1, 
                 max_alpha: float = 0.9):
        self.num_tasks = num_tasks
        self.base_temperature = base_temperature
        self.adaptation_strategy = adaptation_strategy
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        
        self.task_models = []  # 保存所有任务的学生模型
        self.teacher_models = []  # 保存教师模型
        self.task_distillation_losses = []  # 保存蒸馏损失函数
        self.current_task = 0
    
    def add_task_model(self, student_model: nn.Module, teacher_model: nn.Module,
                      task_similarity: float = 0.5):
        """添加新任务的模型"""
        # 创建自适应蒸馏损失
        distillation_loss = AdaptiveDistillation(
            temperature=self.base_temperature,
            alpha=self.get_task_alpha(task_similarity),
            beta=1.0,
            similarity_threshold=0.3
        )
        
        self.task_models.append(student_model)
        self.teacher_models.append(teacher_model)
        self.task_distillation_losses.append(distillation_loss)
        
        self.current_task = len(self.task_models) - 1
    
    def get_task_alpha(self, similarity: float) -> float:
        """根据任务相似度获取蒸馏权重"""
        if self.adaptation_strategy == 'linear':
            # 线性插值
            progress = len(self.task_models) / self.num_tasks
            base_alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * progress
            return base_alpha * (0.5 + 0.5 * similarity)
        elif self.adaptation_strategy == 'exponential':
            # 指数衰减
            return self.max_alpha * math.exp(-0.1 * len(self.task_models)) * similarity
        elif self.adaptation_strategy == 'similarity_based':
            # 基于相似度
            return self.min_alpha + (self.max_alpha - self.min_alpha) * similarity
        else:
            return (self.min_alpha + self.max_alpha) / 2
    
    def distill_between_tasks(self, source_task_idx: int, target_task_idx: int,
                            source_data: torch.Tensor, target_data: torch.Tensor,
                            source_labels: torch.Tensor, target_labels: torch.Tensor,
                            optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """在任务间进行知识蒸馏"""
        if source_task_idx >= len(self.task_models) or target_task_idx >= len(self.task_models):
            raise ValueError("任务索引超出范围")
        
        source_model = self.task_models[source_task_idx]
        target_model = self.task_models[target_task_idx]
        source_teacher = self.teacher_models[source_task_idx]
        
        target_model.train()
        
        # 源任务到目标任务的知识蒸馏
        with torch.no_grad():
            source_outputs = source_teacher(source_data)
        
        target_outputs = target_model(target_data)
        
        # 计算蒸馏损失
        distillation_loss_func = self.task_distillation_losses[source_task_idx]
        results = distillation_loss_func(
            target_outputs, source_outputs,
            target_labels=target_labels,
            criterion=nn.CrossEntropyLoss()
        )
        
        # 反向传播
        optimizer.zero_grad()
        results['total_loss'].backward()
        optimizer.step()
        
        return {key: value.item() if isinstance(value, torch.Tensor) else value 
                for key, value in results.items()}
    
    def get_distillation_statistics(self) -> Dict[str, Any]:
        """获取蒸馏统计信息"""
        return {
            'num_tasks': len(self.task_models),
            'current_task': self.current_task,
            'task_alphas': [loss.alpha for loss in self.task_distillation_losses],
            'task_temperatures': [loss.temperature for loss in self.task_distillation_losses],
            'total_models': len(self.task_models)
        }


def create_simple_mlp(input_dim: int, hidden_dim: int = 128, num_classes: int = 10) -> nn.Module:
    """创建简单的MLP模型"""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes)
    )


def demo_knowledge_distillation():
    """演示知识蒸馏的使用"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 参数设置
    input_dim = 784
    hidden_dim = 128
    num_classes = 10
    batch_size = 64
    
    # 创建教师和学生模型
    teacher_model = create_simple_mlp(input_dim, hidden_dim, num_classes)
    student_model = create_simple_mlp(input_dim, hidden_dim, num_classes)
    
    # 创建训练器
    trainer = KnowledgeDistillationTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        distillation_type='logits',
        temperature=4.0,
        alpha=0.7,
        beta=1.0
    )
    
    print("知识蒸馏训练演示")
    print(f"教师模型参数数量: {sum(p.numel() for p in teacher_model.parameters())}")
    print(f"学生模型参数数量: {sum(p.numel() for p in student_model.parameters())}")
    
    # 创建模拟数据
    num_samples = 1000
    data = torch.randn(num_samples, input_dim)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # 创建数据加载器
    dataset = torch.utils.data.TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 优化器
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    
    # 训练
    print("\n开始训练...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        epoch_results = defaultdict(list)
        
        for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
            results = trainer.train_step(batch_data, batch_labels, optimizer)
            
            for key, value in results.items():
                if isinstance(value, torch.Tensor):
                    epoch_results[key].append(value.item())
                else:
                    epoch_results[key].append(value)
        
        # 打印epoch结果
        print(f"Epoch {epoch + 1}:")
        for key, values in epoch_results.items():
            avg_value = np.mean(values)
            print(f"  {key}: {avg_value:.4f}")
        print()
    
    print("知识蒸馏训练完成!")


if __name__ == "__main__":
    demo_knowledge_distillation()