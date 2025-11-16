"""
元学习机制实现 (Meta-Learning)

基于终身学习研究报告的理论基础，实现快速适应新任务的元学习机制：
1. MAML (Model-Agnostic Meta-Learning) - 模型无关的元学习
2. Reptile - 简单的元学习算法
3. FOMAML - 一阶元学习
4. Meta-SGD - 元学习的随机梯度下降

元学习的核心思想是"学习如何学习"，通过在多个相关任务上的训练，
学习到能够快速适应新任务的初始化参数或更新规则。

Author: Lifelong Learning Team
Date: 2025-11-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import copy
import math
from collections import defaultdict


class MetaLearner(nn.Module):
    """元学习基类"""
    
    def __init__(self, model: nn.Module, meta_lr: float = 0.01, 
                 inner_lr: float = 0.1, num_inner_steps: int = 5):
        super().__init__()
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        
        # 元参数（用于学习如何更新参数）
        self.meta_parameters = nn.ParameterDict()
        self._initialize_meta_parameters()
    
    def _initialize_meta_parameters(self):
        """初始化元参数"""
        # 为每个参数组创建学习率元参数
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # 初始化为该参数的梯度方向的元学习率
                self.meta_parameters[f'{name}_lr'] = nn.Parameter(
                    torch.ones_like(param) * self.inner_lr
                )
    
    def inner_loop(self, support_data: torch.Tensor, support_targets: torch.Tensor,
                   task_id: str = None) -> nn.Module:
        """内循环：在支持集上更新参数"""
        # 复制模型进行内循环更新
        inner_model = copy.deepcopy(self.model)
        inner_model.train()
        
        # 使用元参数进行多次梯度更新
        for step in range(self.num_inner_steps):
            # 前向传播
            outputs = inner_model(support_data)
            loss = F.cross_entropy(outputs, support_targets)
            
            # 反向传播
            inner_model.zero_grad()
            loss.backward()
            
            # 使用元参数进行参数更新
            with torch.no_grad():
                for name, param in inner_model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        meta_lr = self.meta_parameters.get(f'{name}_lr', 
                                                          torch.tensor(self.inner_lr))
                        param.data -= meta_lr * param.grad.data
        
        return inner_model
    
    def meta_loss(self, inner_model: nn.Module, query_data: torch.Tensor,
                  query_targets: torch.Tensor) -> torch.Tensor:
        """计算元损失（在查询集上的损失）"""
        with torch.no_grad():
            outputs = inner_model(query_data)
            return F.cross_entropy(outputs, query_targets)
    
    def forward(self, x):
        return self.model(x)


class MAML(MetaLearner):
    """模型无关的元学习 (MAML)"""
    
    def __init__(self, model: nn.Module, meta_lr: float = 0.01, 
                 inner_lr: float = 0.1, num_inner_steps: int = 5,
                 second_order: bool = True):
        super().__init__(model, meta_lr, inner_lr, num_inner_steps)
        self.second_order = second_order
        
    def adapt(self, support_data: torch.Tensor, support_targets: torch.Tensor) -> nn.Module:
        """适应新任务"""
        return self.inner_loop(support_data, support_targets)
    
    def meta_train(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, 
                                         torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """元训练：在一个批次任务上训练"""
        total_meta_loss = 0.0
        total_inner_loss = 0.0
        num_tasks = len(tasks)
        
        # 计算所有任务的梯度
        meta_grads = defaultdict(list)
        
        for task_idx, (support_data, support_targets, query_data, query_targets) in enumerate(tasks):
            # 内循环：更新参数
            inner_model = self.adapt(support_data, support_targets)
            
            # 元损失：在查询集上计算损失
            query_outputs = inner_model(query_data)
            meta_loss = F.cross_entropy(query_outputs, query_targets)
            
            total_meta_loss += meta_loss.item()
            
            # 如果使用二阶梯度，计算完整的meta梯度
            if self.second_order:
                meta_loss.backward()
                
                # 收集参数梯度
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            meta_grads[name].append(param.grad.clone())
                            param.grad.zero_()
        
        # 计算平均meta梯度并更新
        with torch.no_grad():
            for name, grads in meta_grads.items():
                if grads:
                    avg_grad = torch.stack(grads).mean(dim=0)
                    if hasattr(self.model, name.split('.')[0]):
                        # 使用元学习率更新参数
                        self.model.parameters().__next__().data -= self.meta_lr * avg_grad
        
        # 简单的一阶MAML
        if not self.second_order:
            # 使用一阶近似
            self.first_order_meta_update(tasks)
        
        return {
            'meta_loss': total_meta_loss / num_tasks,
            'inner_loss': total_inner_loss / num_tasks
        }
    
    def first_order_meta_update(self, tasks: List[Tuple]):
        """一阶MAML更新"""
        self.model.zero_grad()
        
        # 计算每个任务的元损失
        total_loss = 0
        for support_data, support_targets, query_data, query_targets in tasks:
            # 内循环更新
            inner_model = copy.deepcopy(self.model)
            
            # 简单的内循环更新
            for step in range(self.num_inner_steps):
                outputs = inner_model(support_data)
                loss = F.cross_entropy(outputs, support_targets)
                
                # 手动计算梯度并更新
                grads = torch.autograd.grad(
                    loss, inner_model.parameters(), 
                    create_graph=False
                )
                
                with torch.no_grad():
                    for param, grad in zip(inner_model.parameters(), grads):
                        param.data -= self.inner_lr * grad
            
            # 计算查询损失
            query_outputs = inner_model(query_data)
            query_loss = F.cross_entropy(query_outputs, query_targets)
            total_loss += query_loss
        
        # 元更新
        total_loss.backward()
        
        # 更新原始模型
        with torch.no_grad():
            for param in self.model.parameters():
                param.data -= self.meta_lr * param.grad
                param.grad.zero_()


class Reptile(MetaLearner):
    """Reptile元学习算法"""
    
    def __init__(self, model: nn.Module, meta_lr: float = 0.01, 
                 inner_lr: float = 0.1, num_inner_steps: int = 5):
        super().__init__(model, meta_lr, inner_lr, num_inner_steps)
    
    def meta_train(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, 
                                         torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Reptile元训练"""
        total_loss = 0.0
        num_tasks = len(tasks)
        
        for support_data, support_targets, query_data, query_targets in tasks:
            # 保存初始参数
            initial_params = [p.data.clone() for p in self.model.parameters()]
            
            # 内循环更新
            inner_model = copy.deepcopy(self.model)
            
            for step in range(self.num_inner_steps):
                outputs = inner_model(support_data)
                loss = F.cross_entropy(outputs, support_targets)
                
                # 更新内部模型
                inner_model.zero_grad()
                loss.backward()
                
                with torch.no_grad():
                    for param in inner_model.parameters():
                        param.data -= self.inner_lr * param.grad
            
            # 计算Reptile损失：原始参数与内循环后参数的差异
            reptile_loss = 0
            for initial_param, final_param in zip(initial_params, inner_model.parameters()):
                reptile_loss += F.mse_loss(final_param, initial_param)
            
            total_loss += reptile_loss.item()
            
            # 用内循环后的参数更新元模型
            with torch.no_grad():
                for param, final_param in zip(self.model.parameters(), inner_model.parameters()):
                    param.data += self.meta_lr * (final_param.data - param.data)
        
        return {
            'meta_loss': total_loss / num_tasks,
            'inner_loss': 0.0
        }
    
    def adapt(self, support_data: torch.Tensor, support_targets: torch.Tensor) -> nn.Module:
        """适应新任务（Reptile版本）"""
        inner_model = copy.deepcopy(self.model)
        
        for step in range(self.num_inner_steps):
            outputs = inner_model(support_data)
            loss = F.cross_entropy(outputs, support_targets)
            
            inner_model.zero_grad()
            loss.backward()
            
            with torch.no_grad():
                for param in inner_model.parameters():
                    param.data -= self.inner_lr * param.grad
        
        return inner_model


class MetaSGD(MetaLearner):
    """Meta-SGD：学习如何更新参数的元学习算法"""
    
    def __init__(self, model: nn.Module, meta_lr: float = 0.01, 
                 inner_lr: float = 0.1, num_inner_steps: int = 5,
                 learnable_lr: bool = True, learnable_update_dir: bool = True):
        super().__init__(model, meta_lr, inner_lr, num_inner_steps)
        self.learnable_lr = learnable_lr
        self.learnable_update_dir = learnable_update_dir
        
        # 学习方向参数
        self.update_directions = nn.ParameterDict()
        self._initialize_update_directions()
    
    def _initialize_update_directions(self):
        """初始化更新方向参数"""
        if self.learnable_update_dir:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.update_directions[f'{name}_dir'] = nn.Parameter(
                        torch.randn_like(param) * 0.01
                    )
    
    def meta_train(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, 
                                         torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Meta-SGD训练"""
        total_meta_loss = 0.0
        num_tasks = len(tasks)
        
        for task_idx, (support_data, support_targets, query_data, query_targets) in enumerate(tasks):
            # 内循环
            inner_model = copy.deepcopy(self.model)
            
            for step in range(self.num_inner_steps):
                outputs = inner_model(support_data)
                loss = F.cross_entropy(outputs, support_targets)
                
                inner_model.zero_grad()
                loss.backward()
                
                # 使用可学习的更新方向和幅度
                with torch.no_grad():
                    for name, param in inner_model.named_parameters():
                        if param.grad is not None:
                            if self.learnable_lr:
                                lr = self.meta_parameters.get(f'{name}_lr', 
                                                            torch.tensor(self.inner_lr))
                            else:
                                lr = torch.tensor(self.inner_lr)
                            
                            if self.learnable_update_dir:
                                update_dir = self.update_directions.get(f'{name}_dir', 
                                                                      torch.ones_like(param))
                                update_dir = F.normalize(update_dir, dim=0)
                                param.data -= lr * update_dir * param.grad
                            else:
                                param.data -= lr * param.grad
            
            # 计算元损失
            query_outputs = inner_model(query_data)
            meta_loss = F.cross_entropy(query_outputs, query_targets)
            total_meta_loss += meta_loss.item()
        
        # 计算平均meta梯度
        total_meta_loss.backward()
        
        # 元更新
        with torch.no_grad():
            for param in self.model.parameters():
                param.data -= self.meta_lr * param.grad
                param.grad.zero_()
        
        return {
            'meta_loss': total_meta_loss / num_tasks,
            'inner_loss': 0.0
        }


class MAMLClassifier(nn.Module):
    """基于MAML的分类器示例"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # 简单的MLP作为基础模型
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)


class OmniglotNShotTask:
    """Omniglot风格的N-way K-shot任务生成器"""
    
    def __init__(self, num_classes: int = 5, num_support: int = 1, 
                 num_query: int = 15, input_dim: int = 784):
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.input_dim = input_dim
    
    def generate_task(self, class_prototypes: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """生成一个N-way K-shot任务"""
        # 随机选择类别
        selected_classes = torch.randperm(len(class_prototypes))[:self.num_classes]
        
        support_data = []
        support_targets = []
        query_data = []
        query_targets = []
        
        for class_idx, class_id in enumerate(selected_classes):
            prototype = class_prototypes[class_id]
            
            # 生成支持集样本（带噪声）
            for _ in range(self.num_support):
                noise = torch.randn_like(prototype) * 0.1
                support_data.append(prototype + noise)
                support_targets.append(class_idx)
            
            # 生成查询集样本（带噪声）
            for _ in range(self.num_query):
                noise = torch.randn_like(prototype) * 0.1
                query_data.append(prototype + noise)
                query_targets.append(class_idx)
        
        # 转换为张量
        support_data = torch.stack(support_data)
        support_targets = torch.tensor(support_targets, dtype=torch.long)
        query_data = torch.stack(query_data)
        query_targets = torch.tensor(query_targets, dtype=torch.long)
        
        return support_data, support_targets, query_data, query_targets
    
    def generate_task_batch(self, class_prototypes: torch.Tensor, 
                          batch_size: int) -> List[Tuple]:
        """生成一个批次的任务"""
        tasks = []
        for _ in range(batch_size):
            task = self.generate_task(class_prototypes)
            tasks.append(task)
        return tasks


class MetaLearningTrainer:
    """元学习训练器"""
    
    def __init__(self, model: nn.Module, meta_algorithm: str = 'maml',
                 meta_lr: float = 0.01, inner_lr: float = 0.1,
                 num_inner_steps: int = 5, device: str = 'cpu'):
        self.device = device
        self.meta_algorithm = meta_algorithm.lower()
        
        # 创建元学习器
        if self.meta_algorithm == 'maml':
            self.meta_learner = MAML(model, meta_lr, inner_lr, num_inner_steps)
        elif self.meta_algorithm == 'reptile':
            self.meta_learner = Reptile(model, meta_lr, inner_lr, num_inner_steps)
        elif self.meta_algorithm == 'meta_sgd':
            self.meta_learner = MetaSGD(model, meta_lr, inner_lr, num_inner_steps)
        else:
            raise ValueError(f"不支持的元学习算法: {meta_algorithm}")
        
        self.model = self.meta_learner.model.to(device)
        self.meta_learner.to(device)
    
    def train_epoch(self, class_prototypes: torch.Tensor, 
                   task_batch_size: int = 4) -> Dict[str, float]:
        """训练一个epoch"""
        self.meta_learner.train()
        
        # 生成任务批次
        task_generator = OmniglotNShotTask(
            num_classes=5, num_support=1, num_query=15
        )
        tasks = task_generator.generate_task_batch(class_prototypes, task_batch_size)
        
        # 移动到设备
        processed_tasks = []
        for support_data, support_targets, query_data, query_targets in tasks:
            support_data = support_data.to(self.device)
            support_targets = support_targets.to(self.device)
            query_data = query_data.to(self.device)
            query_targets = query_targets.to(self.device)
            processed_tasks.append((support_data, support_targets, query_data, query_targets))
        
        # 元训练
        results = self.meta_learner.meta_train(processed_tasks)
        
        return results
    
    def evaluate_few_shot(self, test_class_prototypes: torch.Tensor,
                         num_test_tasks: int = 100) -> Dict[str, float]:
        """在few-shot任务上评估"""
        self.meta_learner.eval()
        
        task_generator = OmniglotNShotTask(
            num_classes=5, num_support=1, num_query=15
        )
        
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for _ in range(num_test_tasks):
                support_data, support_targets, query_data, query_targets = \
                    task_generator.generate_task(test_class_prototypes)
                
                # 移动到设备
                support_data = support_data.to(self.device)
                support_targets = support_targets.to(self.device)
                query_data = query_data.to(self.device)
                query_targets = query_targets.to(self.device)
                
                # 适应任务
                adapted_model = self.meta_learner.adapt(support_data, support_targets)
                
                # 在查询集上评估
                query_outputs = adapted_model(query_data)
                predictions = torch.argmax(query_outputs, dim=1)
                
                correct_predictions += (predictions == query_targets).sum().item()
                total_predictions += len(query_targets)
        
        accuracy = correct_predictions / total_predictions
        
        return {
            'few_shot_accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'total_predictions': total_predictions
        }


class MetaSGDOptimizer:
    """Meta-SGD优化器"""
    
    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8):
        self.defaults = {'lr': lr, 'betas': betas, 'eps': eps}
        self.state = defaultdict(dict)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                
                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # 更新平均值
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
                # 偏差校正
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # 计算步长
                step_size = group['lr'] / bias_correction1
                
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                # 更新参数
                p.data.addcdiv_(-step_size, exp_avg, denom)
        
        return loss
    
    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad = None


def create_meta_learning_model(input_dim: int, hidden_dim: int = 64, 
                             num_classes: int = 5) -> nn.Module:
    """创建元学习模型"""
    return MAMLClassifier(input_dim, hidden_dim, num_classes)


def main():
    """主函数：演示元学习的使用"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 参数设置
    input_dim = 784  # 28x28图像
    hidden_dim = 64
    num_classes = 20  # 总共20个类别
    meta_lr = 0.01
    inner_lr = 0.1
    num_inner_steps = 5
    
    # 创建模型
    model = create_meta_learning_model(input_dim, hidden_dim, num_classes)
    
    # 创建元学习器
    meta_learner = MAML(model, meta_lr, inner_lr, num_inner_steps, second_order=True)
    
    # 生成一些类别原型（模拟Omniglot字符原型）
    class_prototypes = torch.randn(num_classes, input_dim)
    
    # 创建训练器
    trainer = MetaLearningTrainer(
        model, meta_algorithm='maml', 
        meta_lr=meta_lr, inner_lr=inner_lr,
        num_inner_steps=num_inner_steps
    )
    
    # 训练
    print("开始元学习训练...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        results = trainer.train_epoch(class_prototypes, task_batch_size=4)
        
        if epoch % 2 == 0:
            # 评估
            eval_results = trainer.evaluate_few_shot(
                class_prototypes, num_test_tasks=20
            )
            
            print(f"Epoch {epoch}:")
            print(f"  Meta Loss: {results['meta_loss']:.4f}")
            print(f"  Few-shot Accuracy: {eval_results['few_shot_accuracy']:.4f}")
            print()
    
    print("元学习训练完成!")


if __name__ == "__main__":
    main()