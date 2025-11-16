"""
生成式回放训练器模块

整合生成对抗网络、经验重放和双侧记忆巩固的完整训练系统。
实现端到端的生成式回放训练流程。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from collections import defaultdict
import time
import os

from .generative_adversarial_network import GenerativeAdversarialNetwork
from .experience_replay import ExperienceReplayBuffer, Experience
from .bilateral_consolidation import BilateralMemoryConsolidation

logger = logging.getLogger(__name__)


class GenerativeReplayTrainer:
    """
    生成式回放训练器
    
    完整的生成式回放训练系统，集成了GAN生成、经验重放和记忆巩固功能。
    支持多种重放策略和生成模式的组合使用。
    """
    
    def __init__(self,
                 model: nn.Module,
                 generator_config: Optional[Dict] = None,
                 replay_config: Optional[Dict] = None,
                 consolidation_config: Optional[Dict] = None,
                 device: Optional[torch.device] = None):
        """
        初始化生成式回放训练器
        
        Args:
            model: 神经网络模型
            generator_config: 生成器配置
            replay_config: 重放配置
            consolidation_config: 巩固配置
            device: 计算设备
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 默认配置
        default_generator_config = {
            'generator_type': 'conditional',
            'latent_dim': 100,
            'output_shape': None,
            'num_classes': 10
        }
        default_generator_config.update(generator_config or {})
        
        default_replay_config = {
            'buffer_size': 10000,
            'strategies': ['uniform', 'generative'],
            'strategy_configs': {
                'generative': {
                    'real_ratio': 0.7
                }
            }
        }
        default_replay_config.update(replay_config or {})
        
        default_consolidation_config = {
            'hippocampal_capacity': 5000,
            'cortical_capacity': 50000,
            'consolidation_interval': 1800  # 30分钟
        }
        default_consolidation_config.update(consolidation_config or {})
        
        # 初始化组件
        self.generator = None
        self.replay_buffer = None
        self.memory_system = None
        
        self._initialize_components(default_generator_config, default_replay_config, default_consolidation_config)
        
        # 训练状态
        self.current_task = 0
        self.task_history = []
        self.training_stats = defaultdict(list)
        
        # 性能监控
        self.best_generation_quality = 0.0
        self.replay_effectiveness_history = []
        
        logger.info("生成式回放训练器初始化完成")
    
    def _initialize_components(self, 
                             generator_config: Dict,
                             replay_config: Dict,
                             consolidation_config: Dict) -> None:
        """
        初始化各个组件
        """
        # 创建生成器
        if generator_config['output_shape'] is None:
            # 尝试从模型推断输出形状
            try:
                with torch.no_grad():
                    dummy_input = torch.randn(1, *get_model_input_shape(self.model), device=self.device)
                    dummy_output = self.model(dummy_input)
                    generator_config['output_shape'] = dummy_output.shape[1:]
            except:
                logger.warning("无法推断输出形状，使用默认值 (784,)")
                generator_config['output_shape'] = (784,)
        
        self.generator = GenerativeAdversarialNetwork(
            **generator_config, device=self.device)
        
        # 配置生成器关联
        if 'generative' in replay_config.get('strategies', []):
            generative_config = replay_config['strategy_configs'].get('generative', {})
            generative_config['generator'] = self.generator
            replay_config['strategy_configs']['generative'] = generative_config
        
        # 创建经验重放缓冲区
        self.replay_buffer = ExperienceReplayBuffer(**replay_config)
        
        # 创建记忆巩固系统
        self.memory_system = BilateralMemoryConsolidation(**consolidation_config)
    
    def train_generation_model(self,
                             train_loader: DataLoader,
                             val_loader: Optional[DataLoader] = None,
                             num_epochs: int = 100,
                             save_path: Optional[str] = None) -> Dict[str, List[float]]:
        """
        训练生成模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮次
            save_path: 模型保存路径
            
        Returns:
            训练历史字典
        """
        logger.info("开始训练生成模型")
        
        training_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'generation_quality': []
        }
        
        best_quality = 0.0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # 训练GAN
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            num_batches = 0
            
            self.model.eval()
            for batch_idx, (real_data, real_labels) in enumerate(train_loader):
                real_data = real_data.to(self.device)
                
                if real_labels is not None:
                    real_labels = real_labels.to(self.device)
                
                # 训练GAN
                losses = self.generator.train_step(real_data, real_labels)
                
                epoch_g_loss += losses['g_loss']
                epoch_d_loss += losses['d_loss']
                num_batches += 1
                
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch}/{num_epochs}, Batch {batch_idx}, "
                              f"G Loss: {losses['g_loss']:.4f}, D Loss: {losses['d_loss']:.4f}")
            
            # 计算平均损失
            avg_g_loss = epoch_g_loss / max(num_batches, 1)
            avg_d_loss = epoch_d_loss / max(num_batches, 1)
            
            # 评估生成质量
            quality_score = self._evaluate_generation_quality(val_loader)
            
            # 记录历史
            training_history['generator_loss'].append(avg_g_loss)
            training_history['discriminator_loss'].append(avg_d_loss)
            training_history['generation_quality'].append(quality_score)
            
            # 更新最佳质量
            if quality_score > best_quality:
                best_quality = quality_score
                if save_path:
                    self.generator.save_model(os.path.join(save_path, 'best_generator.pt'))
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch}/{num_epochs} 完成，耗时: {epoch_time:.2f}s, "
                       f"质量分数: {quality_score:.4f}")
        
        self.best_generation_quality = best_quality
        logger.info(f"生成模型训练完成，最佳质量: {best_quality:.4f}")
        
        return training_history
    
    def _evaluate_generation_quality(self, val_loader: Optional[DataLoader]) -> float:
        """
        评估生成质量
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            质量分数
        """
        if val_loader is None:
            return 0.0
        
        self.generator.generator.eval()
        
        # 生成样本
        num_samples = min(100, len(val_loader.dataset))
        generated_samples = self.generator.generate_samples(num_samples)
        
        # 计算统计指标（简化版本）
        with torch.no_grad():
            # 样本多样性（方差）
            sample_var = torch.var(generated_samples).item()
            
            # 样本范围
            sample_range = (generated_samples.max() - generated_samples.min()).item()
        
        # 简单的质量分数
        quality_score = min(1.0, (sample_var * 10 + sample_range) / 2)
        
        return quality_score
    
    def add_task_experience(self,
                          train_loader: DataLoader,
                          task_id: int,
                          experience_ratio: float = 0.1) -> None:
        """
        添加任务经验到重放缓冲区
        
        Args:
            train_loader: 任务数据加载器
            task_id: 任务ID
            experience_ratio: 经验比例
        """
        logger.info(f"为任务 {task_id} 添加经验到重放缓冲区")
        
        # 计算要添加的经验数量
        total_samples = len(train_loader.dataset)
        num_experiences = int(total_samples * experience_ratio)
        
        # 采样经验
        sample_indices = torch.randperm(total_samples)[:num_experiences]
        
        added_count = 0
        for idx in sample_indices:
            try:
                # 获取样本
                sample_data, sample_label = train_loader.dataset[idx]
                
                # 创建经验
                experience = Experience(
                    state=sample_data,
                    label=sample_label if hasattr(sample_label, 'item') else torch.tensor(sample_label),
                    task_id=task_id,
                    metadata={
                        'task_id': task_id,
                        'sample_idx': idx.item(),
                        'source': 'real_data'
                    }
                )
                
                # 添加到缓冲区
                self.replay_buffer.add_experience(experience)
                added_count += 1
                
            except Exception as e:
                logger.warning(f"添加经验失败 {idx}: {str(e)}")
        
        logger.info(f"成功添加 {added_count} 个经验到重放缓冲区")
    
    def generate_replay_data(self,
                           task_id: int,
                           num_samples: int,
                           strategy: str = 'generative') -> List[Experience]:
        """
        生成重放数据
        
        Args:
            task_id: 任务ID
            num_samples: 生成样本数量
            strategy: 重放策略
            
        Returns:
            生成的体验列表
        """
        if strategy != 'generative' or not hasattr(self.generator, 'generate_samples'):
            logger.warning("生成器未就绪或策略不支持生成")
            return []
        
        # 生成合成样本
        try:
            generated_data = self.generator.generate_samples(num_samples)
            
            experiences = []
            for i, data in enumerate(generated_data):
                experience = Experience(
                    state=data,
                    label=None,  # 合成数据可能没有标签
                    task_id=task_id,
                    metadata={
                        'task_id': task_id,
                        'synthetic': True,
                        'generation_step': i
                    }
                )
                experiences.append(experience)
            
            logger.info(f"生成 {len(experiences)} 个重放样本")
            return experiences
            
        except Exception as e:
            logger.error(f"生成重放数据失败: {str(e)}")
            return []
    
    def get_replay_batch(self,
                        batch_size: int,
                        strategy: Optional[str] = None) -> List[Experience]:
        """
        获取重放批次
        
        Args:
            batch_size: 批次大小
            strategy: 重放策略
            
        Returns:
            经验批次
        """
        return self.replay_buffer.sample_batch(batch_size, strategy)
    
    def train_task_with_replay(self,
                             train_loader: DataLoader,
                             task_id: int,
                             val_loader: Optional[DataLoader] = None,
                             num_epochs: int = 50,
                             replay_ratio: float = 0.3,
                             use_generative_replay: bool = True) -> Dict[str, Any]:
        """
        使用重放训练任务
        
        Args:
            train_loader: 任务训练数据
            task_id: 任务ID
            val_loader: 验证数据
            num_epochs: 训练轮次
            replay_ratio: 重放数据比例
            use_generative_replay: 是否使用生成式重放
            
        Returns:
            训练结果字典
        """
        logger.info(f"开始使用重放训练任务 {task_id}")
        
        # 添加当前任务的经验
        self.add_task_experience(train_loader, task_id)
        
        # 如果使用生成式重放，生成一些合成数据
        synthetic_experiences = []
        if use_generative_replay:
            synthetic_count = int(len(train_loader.dataset) * replay_ratio * 0.5)
            synthetic_experiences = self.generate_replay_data(task_id, synthetic_count)
            for exp in synthetic_experiences:
                self.replay_buffer.add_experience(exp)
        
        # 训练循环
        training_history = {
            'task_id': task_id,
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'replay_effectiveness': []
        }
        
        best_val_acc = 0.0
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # 计算重放数据数量
            batch_size = 64
            replay_batch_size = int(batch_size * replay_ratio)
            new_data_batch_size = batch_size - replay_batch_size
            
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            replay_effectiveness_scores = []
            
            for batch_idx, (new_data, new_labels) in enumerate(train_loader):
                # 获取重放数据
                replay_experiences = self.get_replay_batch(replay_batch_size) if replay_batch_size > 0 else []
                
                # 处理重放数据
                replay_loss = 0.0
                if replay_experiences:
                    replay_loss, replay_effectiveness = self._process_replay_batch(
                        replay_experiences, epoch)
                    replay_effectiveness_scores.append(replay_effectiveness)
                
                # 处理新数据
                new_loss, new_correct, new_total = self._process_new_data_batch(
                    new_data, new_labels, new_data_batch_size)
                
                # 总损失
                total_loss = new_loss + replay_loss
                
                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 优化器步进
                # 这里需要根据具体的优化器设置
                
                # 统计
                epoch_loss += total_loss.item()
                epoch_correct += new_correct
                epoch_total += new_total
                
                if batch_idx % 100 == 0:
                    logger.info(f"Task {task_id}, Epoch {epoch}, Batch {batch_idx}, "
                              f"Loss: {total_loss.item():.4f}, Acc: {new_correct/new_total:.4f}")
            
            # 计算epoch统计
            avg_loss = epoch_loss / len(train_loader)
            avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
            
            # 验证
            if val_loader is not None:
                val_metrics = self._validate_model(val_loader)
                val_acc = val_metrics['accuracy']
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                
                training_history['val_loss'].append(val_metrics['loss'])
                training_history['val_acc'].append(val_acc)
            
            # 记录训练指标
            training_history['train_loss'].append(avg_loss)
            training_history['train_acc'].append(avg_acc)
            
            if replay_effectiveness_scores:
                avg_replay_effectiveness = np.mean(replay_effectiveness_scores)
                training_history['replay_effectiveness'].append(avg_replay_effectiveness)
                self.replay_effectiveness_history.append(avg_replay_effectiveness)
            
            epoch_time = time.time() - epoch_start
            logger.info(f"Task {task_id}, Epoch {epoch} 完成，耗时: {epoch_time:.2f}s")
        
        # 运行记忆巩固
        consolidation_results = self.memory_system.run_consolidation_cycle()
        
        # 记录任务历史
        task_result = {
            'task_id': task_id,
            'training_history': training_history,
            'best_val_acc': best_val_acc,
            'consolidation_results': consolidation_results,
            'synthetic_experiences_added': len(synthetic_experiences),
            'final_replay_effectiveness': training_history['replay_effectiveness'][-1] if training_history['replay_effectiveness'] else 0.0
        }
        
        self.task_history.append(task_result)
        self.current_task += 1
        
        logger.info(f"任务 {task_id} 训练完成，最佳验证准确率: {best_val_acc:.4f}")
        
        return task_result
    
    def _process_replay_batch(self, 
                            replay_experiences: List[Experience],
                            epoch: int) -> Tuple[torch.Tensor, float]:
        """
        处理重放批次
        
        Args:
            replay_experiences: 重放经验列表
            epoch: 当前轮次
            
        Returns:
            (重放损失, 重放效果分数)
        """
        if not replay_experiences:
            return torch.tensor(0.0, device=self.device), 0.0
        
        # 准备重放数据 - 确保tensor大小一致
        try:
            # 先获取所有state的形状，确保一致性
            state_shape = replay_experiences[0].state.shape
            replay_states = []
            for exp in replay_experiences:
                # 调整tensor形状
                if exp.state.shape != state_shape:
                    # 如果形状不匹配，则使用零填充
                    padded_state = torch.zeros(state_shape)
                    padded_state.flat[:min(exp.state.numel(), padded_state.numel())] = exp.state.flat[:min(exp.state.numel(), padded_state.numel())]
                    replay_states.append(padded_state)
                else:
                    replay_states.append(exp.state)
            
            replay_states = torch.stack(replay_states).to(self.device)
            replay_labels = torch.stack([exp.label for exp in replay_experiences if exp.label is not None]).to(self.device)
        except Exception as e:
            logger.warning(f"处理重放批次时出错: {e}")
            return torch.tensor(0.0, device=self.device), 0.0
        
        # 前向传播
        self.optimizer.zero_grad()
        replay_outputs = self.model(replay_states)
        
        # 计算损失
        if len(replay_labels) > 0:
            replay_loss = nn.functional.cross_entropy(replay_outputs[:len(replay_labels)], replay_labels)
        else:
            # 无标签情况，使用特征重构或其他方法
            replay_loss = torch.tensor(0.0, device=self.device)
        
        # 计算重放效果分数
        replay_effectiveness = self._compute_replay_effectiveness(replay_experiences, replay_outputs)
        
        return replay_loss, replay_effectiveness
    
    def _compute_replay_effectiveness(self,
                                    replay_experiences: List[Experience],
                                    outputs: torch.Tensor) -> float:
        """
        计算重放效果分数
        
        Args:
            replay_experiences: 重放经验
            outputs: 模型输出
            
        Returns:
            效果分数
        """
        # 简化的效果评估：基于输出分布的熵
        with torch.no_grad():
            probs = torch.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            avg_entropy = entropy.mean().item()
        
        # 归一化到[0,1]
        effectiveness_score = 1.0 / (1.0 + avg_entropy)
        
        return effectiveness_score
    
    def _process_new_data_batch(self,
                              new_data: torch.Tensor,
                              new_labels: torch.Tensor,
                              target_batch_size: int) -> Tuple[torch.Tensor, int, int]:
        """
        处理新数据批次
        
        Args:
            new_data: 新数据
            new_labels: 新标签
            target_batch_size: 目标批次大小
            
        Returns:
            (损失, 正确数, 总数)
        """
        # 限制批次大小
        if len(new_data) > target_batch_size:
            indices = torch.randperm(len(new_data))[:target_batch_size]
            new_data = new_data[indices]
            new_labels = new_labels[indices]
        
        new_data, new_labels = new_data.to(self.device), new_labels.to(self.device)
        
        # 前向传播
        outputs = self.model(new_data)
        loss = nn.functional.cross_entropy(outputs, new_labels)
        
        # 计算准确率
        predicted = outputs.argmax(dim=1)
        correct = predicted.eq(new_labels).sum().item()
        total = new_labels.size(0)
        
        return loss, correct, total
    
    def _validate_model(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证指标字典
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = nn.functional.cross_entropy(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
        
        self.model.train()
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': total_correct / total_samples
        }
    
    def evaluate_generative_replay_performance(self,
                                             task_data_loaders: Dict[int, DataLoader]) -> Dict[str, float]:
        """
        评估生成式重放性能
        
        Args:
            task_data_loaders: 任务数据加载器字典
            
        Returns:
            性能评估结果
        """
        results = {}
        
        for task_id, data_loader in task_data_loaders.items():
            # 获取重放效果
            replay_experiences = self.replay_buffer.sample_batch(min(100, len(data_loader.dataset)))
            
            # 计算重放多样性
            replay_diversity = self._compute_replay_diversity(replay_experiences)
            
            # 计算重放质量
            replay_quality = self._compute_replay_quality(replay_experiences)
            
            # 计算任务间干扰
            interference = self._compute_task_interference(task_id)
            
            results[f'task_{task_id}'] = {
                'replay_diversity': replay_diversity,
                'replay_quality': replay_quality,
                'interference_score': interference
            }
        
        # 计算总体指标
        all_diversities = [metrics['replay_diversity'] for metrics in results.values()]
        all_qualities = [metrics['replay_quality'] for metrics in results.values()]
        
        results['overall'] = {
            'average_replay_diversity': np.mean(all_diversities),
            'average_replay_quality': np.mean(all_qualities),
            'generation_quality': self.best_generation_quality,
            'replay_effectiveness_trend': self.replay_effectiveness_history[-5:] if len(self.replay_effectiveness_history) >= 5 else self.replay_effectiveness_history
        }
        
        return results
    
    def _compute_replay_diversity(self, experiences: List[Experience]) -> float:
        """
        计算重放多样性
        
        Args:
            experiences: 经验列表
            
        Returns:
            多样性分数
        """
        if not experiences:
            return 0.0
        
        # 计算状态的多样性
        states = torch.stack([exp.state for exp in experiences])
        
        # 计算状态间的平均距离
        with torch.no_grad():
            pairwise_distances = []
            for i in range(len(states)):
                for j in range(i + 1, len(states)):
                    distance = torch.norm(states[i] - states[j]).item()
                    pairwise_distances.append(distance)
            
            avg_distance = np.mean(pairwise_distances) if pairwise_distances else 0.0
        
        # 归一化
        max_distance = torch.norm(states.max() - states.min()).item()
        diversity_score = avg_distance / (max_distance + 1e-8)
        
        return diversity_score
    
    def _compute_replay_quality(self, experiences: List[Experience]) -> float:
        """
        计算重放质量
        
        Args:
            experiences: 经验列表
            
        Returns:
            质量分数
        """
        synthetic_count = sum(1 for exp in experiences if exp.metadata.get('synthetic', False))
        total_count = len(experiences)
        
        # 合成数据比例作为质量指标的一部分
        synthetic_ratio = synthetic_count / max(total_count, 1)
        
        # 合成数据的平均重要性
        synthetic_importances = [exp.importance for exp in experiences 
                              if exp.metadata.get('synthetic', False)]
        avg_importance = np.mean(synthetic_importances) if synthetic_importances else 0.0
        
        # 综合质量分数
        quality_score = 0.7 * synthetic_ratio + 0.3 * avg_importance
        
        return quality_score
    
    def _compute_task_interference(self, task_id: int) -> float:
        """
        计算任务间干扰
        
        Args:
            task_id: 当前任务ID
            
        Returns:
            干扰分数
        """
        # 获取记忆系统统计
        system_stats = self.memory_system.get_system_statistics()
        interference_rate = system_stats['cortex']['interference_rate']
        
        # 考虑任务间相似度（简化）
        task_interference = interference_rate * 0.1  # 简单的加权
        
        return task_interference
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """
        获取训练统计信息
        
        Returns:
            完整统计信息字典
        """
        # 生成器统计
        generator_stats = self.generator.get_training_statistics() if self.generator else {}
        
        # 重放缓冲区统计
        replay_stats = self.replay_buffer.get_statistics()
        
        # 记忆系统统计
        memory_stats = self.memory_system.get_system_statistics()
        
        # 任务历史统计
        task_stats = {
            'total_tasks': len(self.task_history),
            'completed_tasks': self.current_task,
            'average_replay_effectiveness': np.mean(self.replay_effectiveness_history) if self.replay_effectiveness_history else 0.0,
            'best_generation_quality': self.best_generation_quality
        }
        
        # 组合所有统计
        comprehensive_stats = {
            'generator': generator_stats,
            'replay_buffer': replay_stats,
            'memory_system': memory_stats,
            'training_history': task_stats,
            'component_integration': {
                'total_operations': memory_stats['system']['total_operations'],
                'consolidation_cycles': memory_stats['system']['consolidation_cycles'],
                'memory_utilization': {
                    'hippocampus': memory_stats['hippocampus']['capacity_utilization'],
                    'cortex': memory_stats['cortex']['capacity_utilization']
                }
            }
        }
        
        return comprehensive_stats
    
    def save_complete_system(self, save_dir: str) -> None:
        """
        保存完整系统
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存生成器
        if self.generator:
            self.generator.save_model(os.path.join(save_dir, 'generator.pt'))
        
        # 保存训练器状态
        trainer_state = {
            'model_state_dict': self.model.state_dict(),
            'current_task': self.current_task,
            'task_history': self.task_history,
            'best_generation_quality': self.best_generation_quality,
            'replay_effectiveness_history': self.replay_effectiveness_history,
            'training_stats': dict(self.training_stats)
        }
        
        torch.save(trainer_state, os.path.join(save_dir, 'trainer_state.pt'))
        
        # 保存记忆系统（需要序列化处理）
        memory_data = {
            'hippocampus_memories': {mid: {
                'content': mem.content.cpu().numpy(),
                'metadata': mem.metadata,
                'timestamp': mem.timestamp,
                'consolidation_level': mem.consolidation_level,
                'access_count': mem.access_count,
                'strength': mem.strength,
                'memory_type': mem.memory_type
            } for mid, mem in self.memory_system.hippocampus.memories.items()},
            'cortex_memories': {mid: {
                'content': mem.content.cpu().numpy(),
                'metadata': mem.metadata,
                'timestamp': mem.timestamp,
                'consolidation_level': mem.consolidation_level,
                'access_count': mem.access_count,
                'strength': mem.strength,
                'memory_type': mem.memory_type
            } for mid, mem in self.memory_system.cortex.memories.items()}
        }
        
        np.save(os.path.join(save_dir, 'memory_system.npy'), memory_data)
        
        logger.info(f"完整系统已保存到: {save_dir}")


# 辅助函数
def get_model_input_shape(model: nn.Module) -> Tuple[int, ...]:
    """
    获取模型的输入形状
    
    Args:
        model: 神经网络模型
        
    Returns:
        输入形状元组
    """
    try:
        # 尝试从模型结构推断
        if hasattr(model, 'input_shape'):
            return model.input_shape
        elif hasattr(model, 'conv1'):
            return (3, 224, 224)  # 默认图像输入
        else:
            # 通用方法
            return (784,)  # 默认向量输入
    except:
        return (784,)