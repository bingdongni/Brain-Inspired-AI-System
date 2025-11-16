#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
训练脚本
=======

提供统一的模型训练功能，包括:
- 训练循环
- 验证和测试
- 学习率调度
- 早停机制
- 模型保存
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain_ai.utils import ConfigManager, Logger, MetricsCollector, ModelCheckpoint, Timer
from brain_ai.utils.visualization import TrainingVisualizer

class TrainingConfig:
    """训练配置"""
    
    def __init__(self, **kwargs):
        self.epochs = kwargs.get('epochs', 100)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.optimizer = kwargs.get('optimizer', 'adam')
        self.scheduler = kwargs.get('scheduler', 'cosine')
        self.early_stopping = kwargs.get('early_stopping', True)
        self.patience = kwargs.get('patience', 10)
        self.save_frequency = kwargs.get('save_frequency', 10)
        self.monitor_metric = kwargs.get('monitor_metric', 'val_loss')
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_workers = kwargs.get('num_workers', 4)
        self.seed = kwargs.get('seed', 42)

class BaseTrainer:
    """基础训练器"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        """
        初始化训练器
        
        Args:
            model: 要训练的模型
            config: 训练配置
        """
        self.model = model.to(config.device)
        self.config = config
        
        # 设置随机种子
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
        
        # 初始化组件
        self.logger = Logger("trainer")
        self.metrics = MetricsCollector()
        self.timer = Timer("training")
        
        # 优化器和调度器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # 早停机制
        self.best_metric = float('inf') if 'loss' in config.monitor_metric else float('-inf')
        self.patience_counter = 0
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.logger.info(f"训练器初始化完成，设备: {config.device}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """创建优化器"""
        if self.config.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=0.9)
        elif self.config.optimizer.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        else:
            raise ValueError(f"不支持的优化器: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[Any]:
        """创建学习率调度器"""
        if self.config.scheduler.lower() == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler.lower() == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.config.scheduler.lower() == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', patience=10, factor=0.5
            )
        elif self.config.scheduler.lower() == 'none':
            return None
        else:
            raise ValueError(f"不支持的调度器: {self.config.scheduler}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.config.device), target.to(self.config.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self._compute_loss(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item() * data.size(0)
            if hasattr(output, 'argmax'):
                pred = output.argmax(dim=1)
                correct = (pred == target).sum().item()
                total_correct += correct
            
            total_samples += data.size(0)
            
            # 更新进度条
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples if total_correct > 0 else 0
            
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{avg_acc:.4f}'
            })
        
        epoch_metrics = {
            'train_loss': total_loss / total_samples,
            'train_acc': total_correct / total_samples
        }
        
        return epoch_metrics
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                output = self.model(data)
                loss = self._compute_loss(output, target)
                
                total_loss += loss.item() * data.size(0)
                if hasattr(output, 'argmax'):
                    pred = output.argmax(dim=1)
                    correct = (pred == target).sum().item()
                    total_correct += correct
                
                total_samples += data.size(0)
        
        epoch_metrics = {
            'val_loss': total_loss / total_samples,
            'val_acc': total_correct / total_samples
        }
        
        return epoch_metrics
    
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算损失函数"""
        if hasattr(self.model, 'criterion'):
            return self.model.criterion(output, target)
        else:
            # 默认使用交叉熵损失
            return nn.functional.cross_entropy(output, target)
    
    def _check_early_stopping(self, metrics: Dict[str, float]) -> bool:
        """检查是否应该早停"""
        current_metric = metrics.get(self.config.monitor_metric)
        if current_metric is None:
            return False
        
        if ('loss' in self.config.monitor_metric and 
            current_metric < self.best_metric):
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        elif ('acc' in self.config.monitor_metric and 
              current_metric > self.best_metric):
            self.best_metric = current_metric
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"早停触发，已patience {self.config.patience} 个epoch")
                return True
            return False
    
    def train(self, train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None,
              save_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            save_dir: 模型保存目录
            
        Returns:
            训练历史字典
        """
        save_dir = Path(save_dir) if save_dir else Path("output")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建检查点管理器
        checkpoint_manager = ModelCheckpoint(
            save_dir / "checkpoints",
            save_best_only=True,
            monitor_metric=self.config.monitor_metric
        )
        
        self.logger.info("开始训练...")
        
        with self.timer:
            for epoch in range(1, self.config.epochs + 1):
                # 训练
                train_metrics = self.train_epoch(train_loader)
                
                # 验证
                val_metrics = {}
                if val_loader:
                    val_metrics = self.validate_epoch(val_loader)
                
                # 更新历史
                self.train_history['train_loss'].append(train_metrics['train_loss'])
                self.train_history['train_acc'].append(train_metrics['train_acc'])
                if val_metrics:
                    self.train_history['val_loss'].append(val_metrics['val_loss'])
                    self.train_history['val_acc'].append(val_metrics['val_acc'])
                
                # 学习率调度
                if self.scheduler:
                    if self.config.scheduler.lower() == 'plateau':
                        self.scheduler.step(val_metrics.get('val_loss', 0))
                    else:
                        self.scheduler.step()
                
                # 保存检查点
                all_metrics = {**train_metrics, **val_metrics}
                checkpoint_manager.save_checkpoint(
                    self.model, epoch, all_metrics
                )
                
                # 早停检查
                if self.config.early_stopping and val_metrics:
                    if self._check_early_stopping(val_metrics):
                        break
                
                # 日志记录
                if epoch % 10 == 0 or epoch == 1:
                    log_msg = f"Epoch {epoch}/{self.config.epochs}: "
                    log_msg += f"Train Loss: {train_metrics['train_loss']:.4f}, "
                    log_msg += f"Train Acc: {train_metrics['train_acc']:.4f}"
                    if val_metrics:
                        log_msg += f", Val Loss: {val_metrics['val_loss']:.4f}, "
                        log_msg += f"Val Acc: {val_metrics['val_acc']:.4f}"
                    
                    self.logger.info(log_msg)
        
        # 保存训练历史
        history_path = save_dir / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)
        
        # 生成训练曲线
        visualizer = TrainingVisualizer(save_dir / "plots")
        visualizer.plot_training_curves(
            {k: v for k, v in self.train_history.items() if 'train' in k},
            {k.replace('train', 'val'): v for k, v in self.train_history.items() if 'train' in k}
        )
        
        self.logger.info("训练完成！")
        return self.train_history

class HippocampusTrainer(BaseTrainer):
    """海马体专门训练器"""
    
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算海马体特定损失"""
        if hasattr(self.model, 'memory_loss'):
            return self.model.memory_loss(output, target)
        return super()._compute_loss(output, target)

class NeocortexTrainer(BaseTrainer):
    """新皮层专门训练器"""
    
    def _compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算新皮层特定损失"""
        if hasattr(self.model, 'hierarchy_loss'):
            return self.model.hierarchy_loss(output, target)
        return super()._compute_loss(output, target)

def train_model(model: nn.Module, 
                train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                config: Optional[TrainingConfig] = None,
                save_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    通用模型训练函数
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 训练配置
        save_dir: 保存目录
        
    Returns:
        训练历史
    """
    if config is None:
        config = TrainingConfig()
    
    # 根据模型类型选择训练器
    if model.__class__.__name__.lower().find('hippocampus') >= 0:
        trainer = HippocampusTrainer(model, config)
    elif model.__class__.__name__.lower().find('neocortex') >= 0:
        trainer = NeocortexTrainer(model, config)
    else:
        trainer = BaseTrainer(model, config)
    
    return trainer.train(train_loader, val_loader, save_dir)

def create_data_loaders(data_path: str, 
                       config: TrainingConfig,
                       transform: Optional[Any] = None) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    创建数据加载器
    
    Args:
        data_path: 数据路径
        config: 训练配置
        transform: 数据变换
        
    Returns:
        训练和验证数据加载器
    """
    # 这里应该实现具体的数据加载逻辑
    # 取决于具体的数据格式和模型需求
    raise NotImplementedError("请实现具体的数据加载逻辑")

if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description='训练Brain-Inspired AI模型')
    parser.add_argument('--model_path', type=str, help='模型文件路径')
    parser.add_argument('--data_path', type=str, required=True, help='数据路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )
    
    # 加载配置（如果提供）
    if args.config:
        config_manager = ConfigManager(args.config)
        config_dict = config_manager.get_config()
        # 更新配置参数
    
    # 创建数据加载器
    # train_loader, val_loader = create_data_loaders(args.data_path, config)
    
    print("训练脚本需要根据具体模型和数据格式进行实现")
    print("请在集成时实现具体的模型和数据加载逻辑")