"""
学习无遗忘(LwF)模块

实现学习无 forgetting(LwF)算法，通过知识蒸馏保持旧任务知识的同时学习新任务。
参考 Li & Hoiem (2016) "Learning without Forgetting"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)


class OldKnowledgeExtractor:
    """
    旧知识提取器
    
    从旧模型中提取和存储知识，用于后续的知识蒸馏。
    """
    
    def __init__(self, 
                 old_model: nn.Module,
                 temperature: float = 4.0,
                 storage_method: str = 'output_distribution'):
        """
        初始化旧知识提取器
        
        Args:
            old_model: 旧模型
            temperature: 蒸馏温度
            storage_method: 存储方法 ('output_distribution', 'feature_maps', 'attention_weights')
        """
        self.old_model = old_model
        self.temperature = temperature
        self.storage_method = storage_method
        
        # 知识存储
        self.knowledge_store = {}
        self.knowledge_statistics = {}
        
        # 冻结旧模型
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad = False
        
        logger.info(f"旧知识提取器初始化完成，存储方法: {storage_method}")
    
    def extract_knowledge_from_data(self,
                                  data_loader: torch.utils.data.DataLoader,
                                  task_id: int,
                                  num_samples: Optional[int] = None,
                                  sample_strategy: str = 'random') -> Dict[str, Any]:
        """
        从数据中提取知识
        
        Args:
            data_loader: 数据加载器
            task_id: 任务ID
            num_samples: 样本数量限制
            sample_strategy: 采样策略
            
        Returns:
            提取的知识字典
        """
        logger.info(f"开始提取任务 {task_id} 的知识")
        
        knowledge_data = {
            'task_id': task_id,
            'extraction_time': torch.tensor(0.0),
            'sample_count': 0,
            'knowledge_type': self.storage_method
        }
        
        if self.storage_method == 'output_distribution':
            knowledge_data.update(self._extract_output_distribution(data_loader, task_id, num_samples))
        elif self.storage_method == 'feature_maps':
            knowledge_data.update(self._extract_feature_maps(data_loader, task_id, num_samples))
        elif self.storage_method == 'attention_weights':
            knowledge_data.update(self._extract_attention_weights(data_loader, task_id, num_samples))
        else:
            raise ValueError(f"不支持的存储方法: {self.storage_method}")
        
        # 存储知识
        self.knowledge_store[task_id] = knowledge_data
        
        # 更新统计信息
        self._update_knowledge_statistics(task_id, knowledge_data)
        
        logger.info(f"任务 {task_id} 知识提取完成，样本数: {knowledge_data['sample_count']}")
        
        return knowledge_data
    
    def _extract_output_distribution(self,
                                   data_loader: torch.utils.data.DataLoader,
                                   task_id: int,
                                   num_samples: Optional[int]) -> Dict[str, Any]:
        """提取输出分布知识"""
        outputs = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(data_loader):
                if num_samples and sample_count >= num_samples:
                    break
                
                # 计算输出
                batch_outputs = self.old_model(inputs)
                batch_outputs = F.softmax(batch_outputs / self.temperature, dim=1)
                
                outputs.append(batch_outputs.cpu())
                sample_count += inputs.size(0)
        
        # 拼接所有输出
        all_outputs = torch.cat(outputs, dim=0)
        if num_samples:
            all_outputs = all_outputs[:num_samples]
        
        # 计算统计信息
        output_mean = all_outputs.mean(dim=0)
        output_std = all_outputs.std(dim=0)
        output_entropy = -torch.sum(all_outputs * torch.log(all_outputs + 1e-8), dim=1).mean()
        
        return {
            'output_distribution': all_outputs,
            'output_mean': output_mean,
            'output_std': output_std,
            'output_entropy': output_entropy,
            'sample_count': len(all_outputs)
        }
    
    def _extract_feature_maps(self,
                            data_loader: torch.utils.data.DataLoader,
                            task_id: int,
                            num_samples: Optional[int]) -> Dict[str, Any]:
        """提取特征图知识"""
        feature_maps = []
        sample_count = 0
        
        # 注册钩子来获取特征图
        feature_extractor = {}
        
        def extract_features(name):
            def hook(module, input, output):
                feature_extractor[name] = output.detach()
            return hook
        
        # 假设我们从中间层提取特征
        hooks = []
        for name, module in self.old_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and 'relu' not in name.lower():
                hook = module.register_forward_hook(extract_features(name))
                hooks.append(hook)
        
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(data_loader):
                if num_samples and sample_count >= num_samples:
                    break
                
                # 前向传播提取特征
                self.old_model(inputs)
                
                for name, features in feature_extractor.items():
                    if name not in feature_maps:
                        feature_maps[name] = []
                    feature_maps[name].append(features.cpu())
                
                sample_count += inputs.size(0)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        # 处理特征图
        processed_features = {}
        for name, feature_list in feature_maps.items():
            all_features = torch.cat(feature_list, dim=0)
            if num_samples:
                all_features = all_features[:num_samples]
            
            processed_features[name] = {
                'features': all_features,
                'mean': all_features.mean(dim=0),
                'std': all_features.std(dim=0)
            }
        
        return {
            'feature_maps': processed_features,
            'sample_count': sample_count
        }
    
    def _extract_attention_weights(self,
                                 data_loader: torch.utils.data.DataLoader,
                                 task_id: int,
                                 num_samples: Optional[int]) -> Dict[str, Any]:
        """提取注意力权重知识"""
        attention_maps = []
        sample_count = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(data_loader):
                if num_samples and sample_count >= num_samples:
                    break
                
                # 计算注意力权重（简化版本）
                outputs = self.old_model(inputs)
                
                # 简单的注意力计算
                attention_weights = F.softmax(outputs, dim=1)
                attention_maps.append(attention_weights.cpu())
                
                sample_count += inputs.size(0)
        
        all_attention = torch.cat(attention_maps, dim=0)
        if num_samples:
            all_attention = all_attention[:num_samples]
        
        return {
            'attention_weights': all_attention,
            'attention_mean': all_attention.mean(dim=0),
            'sample_count': len(all_attention)
        }
    
    def _update_knowledge_statistics(self, task_id: int, knowledge_data: Dict[str, Any]):
        """更新知识统计信息"""
        stats = {
            'task_id': task_id,
            'sample_count': knowledge_data.get('sample_count', 0),
            'extraction_timestamp': time.time() if 'extraction_time' not in knowledge_data else knowledge_data['extraction_time'],
            'knowledge_type': self.storage_method,
            'knowledge_quality': self._assess_knowledge_quality(knowledge_data)
        }
        
        self.knowledge_statistics[task_id] = stats
    
    def _assess_knowledge_quality(self, knowledge_data: Dict[str, Any]) -> float:
        """评估知识质量"""
        if 'output_entropy' in knowledge_data:
            # 基于熵的质量评估
            entropy = knowledge_data['output_entropy'].item()
            # 适中的熵值表示知识分布较为均匀
            quality = 1.0 / (1.0 + abs(entropy - 2.0))  # 假设最佳熵值为2.0
            return quality
        elif 'feature_maps' in knowledge_data:
            # 基于特征多样性的质量评估
            feature_diversity = 0.0
            for name, features in knowledge_data['feature_maps'].items():
                feature_std = features['std'].mean().item()
                feature_diversity += feature_std
            return feature_diversity / len(knowledge_data['feature_maps'])
        else:
            return 1.0  # 默认质量
    
    def get_distillation_targets(self, 
                               task_id: int,
                               new_inputs: torch.Tensor,
                               temperature: Optional[float] = None) -> torch.Tensor:
        """
        获取蒸馏目标
        
        Args:
            task_id: 任务ID
            new_inputs: 新输入
            temperature: 温度参数
            
        Returns:
            蒸馏目标
        """
        if task_id not in self.knowledge_store:
            raise ValueError(f"任务 {task_id} 的知识未存储")
        
        if self.storage_method == 'output_distribution':
            # 直接返回存储的输出分布
            knowledge = self.knowledge_store[task_id]
            
            # 如果有对应的输入，使用相似性匹配
            if 'output_distribution' in knowledge:
                stored_outputs = knowledge['output_distribution']
                
                # 简单匹配：使用存储分布的均值作为目标
                return stored_outputs.mean(dim=0, keepdim=True).repeat(new_inputs.size(0), 1)
        
        # 默认：使用旧模型计算
        with torch.no_grad():
            temp = temperature or self.temperature
            old_outputs = self.old_model(new_inputs)
            distillation_targets = F.softmax(old_outputs / temp, dim=1)
        
        return distillation_targets
    
    def get_knowledge_summary(self, task_id: int) -> Dict[str, Any]:
        """获取知识摘要"""
        if task_id not in self.knowledge_store:
            return {}
        
        knowledge = self.knowledge_store[task_id]
        stats = self.knowledge_statistics.get(task_id, {})
        
        summary = {
            'task_id': task_id,
            'knowledge_type': self.storage_method,
            'sample_count': stats.get('sample_count', 0),
            'knowledge_quality': stats.get('knowledge_quality', 0.0),
            'extraction_timestamp': stats.get('extraction_timestamp', 0.0),
            'stored_data_size': self._calculate_data_size(knowledge)
        }
        
        return summary
    
    def _calculate_data_size(self, knowledge_data: Dict[str, Any]) -> int:
        """计算存储数据大小"""
        total_size = 0
        
        for key, value in knowledge_data.items():
            if torch.is_tensor(value):
                total_size += value.numel() * value.element_size()
            elif isinstance(value, dict):
                total_size += self._calculate_data_size(value)
        
        return total_size
    
    def cleanup_old_knowledge(self, keep_recent_k: int = 5) -> int:
        """清理旧知识"""
        task_ids = sorted(self.knowledge_store.keys())
        
        # 保留最近的k个任务
        tasks_to_remove = task_ids[:-keep_recent_k] if len(task_ids) > keep_recent_k else []
        
        removed_count = 0
        for task_id in tasks_to_remove:
            if task_id in self.knowledge_store:
                del self.knowledge_store[task_id]
            if task_id in self.knowledge_statistics:
                del self.knowledge_statistics[task_id]
            removed_count += 1
        
        logger.info(f"清理了 {removed_count} 个任务的旧知识")
        return removed_count


class LearningWithoutForgetting:
    """
    学习无遗忘(LwF)算法
    
    通过知识蒸馏在学习新任务时保持旧任务的知识。
    """
    
    def __init__(self,
                 model: nn.Module,
                 base_criterion: Optional[nn.Module] = None,
                 distillation_config: Optional[Dict] = None):
        """
        初始化LwF算法
        
        Args:
            model: 当前模型
            base_criterion: 基础损失函数
            distillation_config: 蒸馏配置
        """
        self.model = model
        self.base_criterion = base_criterion or nn.CrossEntropyLoss()
        
        # 默认蒸馏配置
        default_config = {
            'temperature': 4.0,
            'distillation_weight': 1.0,
            'task_specific_weight': 1.0,
            'knowledge_storage': 'output_distribution',
            'adaptive_weights': True,
            'gradient_scaling': True
        }
        default_config.update(distillation_config or {})
        self.config = default_config
        
        # 旧知识提取器
        self.old_knowledge_extractor = None
        
        # 任务历史
        self.task_history = []
        self.distillation_weights_history = []
        
        # 性能监控
        self.performance_history = defaultdict(list)
        
        logger.info("学习无遗忘算法初始化完成")
    
    def setup_old_knowledge_extractor(self, old_model: nn.Module) -> OldKnowledgeExtractor:
        """设置旧知识提取器"""
        self.old_knowledge_extractor = OldKnowledgeExtractor(
            old_model=old_model,
            temperature=self.config['temperature'],
            storage_method=self.config['knowledge_storage']
        )
        
        logger.info("旧知识提取器设置完成")
        return self.old_knowledge_extractor
    
    def train_with_lwf(self,
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: Optional[torch.utils.data.DataLoader],
                      new_task_id: int,
                      old_task_ids: List[int],
                      num_epochs: int = 50,
                      learning_rate: float = 0.001,
                      distillation_lr_factor: float = 0.1) -> Dict[str, Any]:
        """
        使用LwF训练新任务
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            new_task_id: 新任务ID
            old_task_ids: 旧任务ID列表
            num_epochs: 训练轮次
            learning_rate: 学习率
            distillation_lr_factor: 蒸馏损失学习率因子
            
        Returns:
            训练结果字典
        """
        logger.info(f"开始LwF训练，新任务: {new_task_id}, 旧任务: {old_task_ids}")
        
        # 准备优化器
        if self.config['adaptive_weights']:
            # 自适应权重：主任务使用正常学习率，蒸馏任务使用较小学习率
            main_params = []
            distillation_params = []
            
            for name, param in self.model.named_parameters():
                if 'head' in name or 'classifier' in name or f'task_{new_task_id}' in name:
                    main_params.append(param)
                else:
                    distillation_params.append(param)
            
            optimizer = optim.Adam([
                {'params': main_params, 'lr': learning_rate},
                {'params': distillation_params, 'lr': learning_rate * distillation_lr_factor}
            ])
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 训练历史
        training_history = {
            'new_task_loss': [],
            'distillation_loss': [],
            'total_loss': [],
            'new_task_accuracy': [],
            'old_task_accuracies': {},
            'val_accuracy': []
        }
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # 训练阶段
            epoch_metrics = self._train_epoch_lwf(
                train_loader, new_task_id, old_task_ids, optimizer
            )
            
            # 验证阶段
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, new_task_id)
                training_history['val_accuracy'].append(val_metrics['accuracy'])
                best_val_acc = max(best_val_acc, val_metrics['accuracy'])
            
            # 更新历史
            training_history['new_task_loss'].append(epoch_metrics['new_task_loss'])
            training_history['distillation_loss'].append(epoch_metrics['distillation_loss'])
            training_history['total_loss'].append(epoch_metrics['total_loss'])
            training_history['new_task_accuracy'].append(epoch_metrics['new_task_accuracy'])
            
            # 记录旧任务准确率
            for task_id, acc in epoch_metrics['old_task_accuracies'].items():
                if task_id not in training_history['old_task_accuracies']:
                    training_history['old_task_accuracies'][task_id] = []
                training_history['old_task_accuracies'][task_id].append(acc)
            
            epoch_time = time.time() - epoch_start
            
            if epoch % 10 == 0:
                log_msg = f"LwF Epoch {epoch}/{num_epochs} - "
                log_msg += f"New Task Loss: {epoch_metrics['new_task_loss']:.4f}, "
                log_msg += f"Distill Loss: {epoch_metrics['distillation_loss']:.4f}, "
                log_msg += f"Total Loss: {epoch_metrics['total_loss']:.4f}, "
                log_msg += f"New Acc: {epoch_metrics['new_task_accuracy']:.4f}"
                
                old_task_str = ", ".join([f"T{tid}:{acc:.3f}" for tid, acc in epoch_metrics['old_task_accuracies'].items()])
                if old_task_str:
                    log_msg += f", Old Tasks: {old_task_str}"
                
                log_msg += f", Time: {epoch_time:.2f}s"
                
                logger.info(log_msg)
        
        # 计算最终指标
        final_results = {
            'new_task_performance': training_history['new_task_accuracy'][-1],
            'old_task_performance': {tid: accs[-1] for tid, accs in training_history['old_task_accuracies'].items()},
            'forgetting_measure': self._calculate_forgetting_measure(old_task_ids),
            'knowledge_retention': self._calculate_knowledge_retention(old_task_ids),
            'training_history': training_history,
            'best_val_accuracy': best_val_acc,
            'training_time': sum(epoch.get('time', 0) for epoch in self.training_stats) if hasattr(self, 'training_stats') else 0
        }
        
        # 记录任务历史
        self.task_history.append({
            'task_id': new_task_id,
            'old_tasks': old_task_ids,
            'results': final_results,
            'training_config': self.config.copy()
        })
        
        logger.info(f"LwF训练完成，新任务性能: {final_results['new_task_performance']:.4f}")
        
        return final_results
    
    def _train_epoch_lwf(self,
                        train_loader: torch.utils.data.DataLoader,
                        new_task_id: int,
                        old_task_ids: List[int],
                        optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """LwF训练轮次"""
        self.model.train()
        
        total_new_task_loss = 0.0
        total_distillation_loss = 0.0
        total_loss = 0.0
        
        new_task_correct = 0
        new_task_total = 0
        
        old_task_corrects = {task_id: 0 for task_id in old_task_ids}
        old_task_totals = {task_id: 0 for task_id in old_task_ids}
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to('cpu'), targets.to('cpu')  # 简化处理
            
            optimizer.zero_grad()
            
            # 前向传播
            if hasattr(self.model, 'forward_task'):
                # 渐进式网络
                outputs = self.model.forward_task(inputs, new_task_id)
                old_task_outputs = {}
                for old_task_id in old_task_ids:
                    old_task_outputs[old_task_id] = self.model.forward_task(inputs, old_task_id)
            else:
                # 标准网络
                outputs = self.model(inputs)
                old_task_outputs = {task_id: outputs for task_id in old_task_ids}  # 简化：使用相同输出
            
            # 计算新任务损失
            new_task_loss = self.base_criterion(outputs, targets)
            
            # 计算蒸馏损失
            distillation_loss = 0.0
            if old_task_ids and self.old_knowledge_extractor:
                for old_task_id in old_task_ids:
                    # 获取蒸馏目标
                    if old_task_id in self.old_knowledge_extractor.knowledge_store:
                        # 使用存储的知识
                        distillation_targets = self.old_knowledge_extractor.get_distillation_targets(
                            old_task_id, inputs, self.config['temperature']
                        )
                    else:
                        # 使用旧模型计算（如果没有存储）
                        with torch.no_grad():
                            old_outputs = self.old_knowledge_extractor.old_model(inputs)
                            distillation_targets = F.softmax(
                                old_outputs / self.config['temperature'], dim=1
                            )
                    
                    # 计算KL散度损失
                    current_outputs = F.log_softmax(
                        old_task_outputs[old_task_id] / self.config['temperature'], dim=1
                    )
                    
                    task_distillation_loss = F.kl_div(
                        current_outputs, distillation_targets, reduction='batchmean'
                    ) * (self.config['temperature'] ** 2)
                    
                    distillation_loss += task_distillation_loss
                
                distillation_loss /= len(old_task_ids)  # 平均
            
            # 总损失
            total_batch_loss = (self.config['task_specific_weight'] * new_task_loss + 
                              self.config['distillation_weight'] * distillation_loss)
            
            # 反向传播
            total_batch_loss.backward()
            
            if self.config['gradient_scaling']:
                # 梯度缩放防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            total_new_task_loss += new_task_loss.item()
            total_distillation_loss += distillation_loss.item()
            total_loss += total_batch_loss.item()
            
            # 准确率统计
            predicted = outputs.argmax(dim=1)
            new_task_correct += predicted.eq(targets).sum().item()
            new_task_total += targets.size(0)
            
            # 旧任务准确率（简化）
            for old_task_id in old_task_ids:
                old_predicted = old_task_outputs[old_task_id].argmax(dim=1)
                # 这里需要对应的旧任务标签，简化处理
                old_task_corrects[old_task_id] += old_predicted.eq(targets).sum().item()
                old_task_totals[old_task_id] += targets.size(0)
        
        # 计算平均指标
        num_batches = len(train_loader)
        return {
            'new_task_loss': total_new_task_loss / num_batches,
            'distillation_loss': total_distillation_loss / num_batches,
            'total_loss': total_loss / num_batches,
            'new_task_accuracy': new_task_correct / max(new_task_total, 1),
            'old_task_accuracies': {
                task_id: old_task_corrects[task_id] / max(old_task_totals[task_id], 1)
                for task_id in old_task_ids
            },
            'time': time.time() - self.epoch_start_time if hasattr(self, 'epoch_start_time') else 0
        }
    
    def _validate_epoch(self, val_loader: torch.utils.data.DataLoader, task_id: int) -> Dict[str, float]:
        """验证轮次"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to('cpu'), targets.to('cpu')
                
                if hasattr(self.model, 'forward_task'):
                    outputs = self.model.forward_task(inputs, task_id)
                else:
                    outputs = self.model(inputs)
                
                predicted = outputs.argmax(dim=1)
                total_correct += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
        
        self.model.train()
        
        return {
            'accuracy': total_correct / max(total_samples, 1)
        }
    
    def _calculate_forgetting_measure(self, old_task_ids: List[int]) -> float:
        """计算遗忘度量"""
        if not self.task_history:
            return 0.0
        
        forgetting_scores = []
        
        for old_task_id in old_task_ids:
            # 找到该任务最初的最佳性能
            initial_best = None
            current_best = 0.0
            
            for task_record in self.task_history:
                if old_task_id in task_record.get('old_tasks', []):
                    if old_task_id not in task_record['results']['old_task_performance']:
                        continue
                    
                    performance = task_record['results']['old_task_performance'][old_task_id]
                    
                    if initial_best is None:
                        initial_best = performance
                    current_best = max(current_best, performance)
            
            if initial_best is not None:
                forgetting = initial_best - current_best
                forgetting_scores.append(max(0.0, forgetting))
        
        return np.mean(forgetting_scores) if forgetting_scores else 0.0
    
    def _calculate_knowledge_retention(self, old_task_ids: List[int]) -> float:
        """计算知识保持率"""
        if not self.task_history:
            return 1.0
        
        retention_scores = []
        
        for old_task_id in old_task_ids:
            # 找到该任务的所有性能记录
            performances = []
            
            for task_record in self.task_history:
                if old_task_id in task_record.get('old_tasks', []):
                    if old_task_id in task_record['results']['old_task_performance']:
                        performances.append(task_record['results']['old_task_performance'][old_task_id])
            
            if performances:
                # 计算保持率：最终性能 / 初始性能
                retention = performances[-1] / max(performances[0], 1e-8)
                retention_scores.append(retention)
        
        return np.mean(retention_scores) if retention_scores else 1.0
    
    def adaptive_distillation_weight(self, 
                                   epoch: int,
                                   num_epochs: int,
                                   task_similarity: float = 1.0) -> float:
        """自适应蒸馏权重"""
        if not self.config['adaptive_weights']:
            return self.config['distillation_weight']
        
        # 训练进度因子
        progress = epoch / num_epochs
        
        # 蒸馏权重随时间递减
        base_weight = self.config['distillation_weight']
        time_factor = 1.0 - 0.5 * progress
        
        # 任务相似度因子
        similarity_factor = 0.5 + 0.5 * task_similarity
        
        # 最终权重
        adaptive_weight = base_weight * time_factor * similarity_factor
        
        return max(adaptive_weight, 0.1)  # 最小权重
    
    def extract_knowledge_for_tasks(self,
                                  data_loaders: Dict[int, torch.utils.data.DataLoader],
                                  max_samples_per_task: int = 1000) -> None:
        """
        提取多个任务的知识
        
        Args:
            data_loaders: 任务数据加载器字典
            max_samples_per_task: 每个任务的最大样本数
        """
        logger.info("开始提取任务知识")
        
        for task_id, data_loader in data_loaders.items():
            self.old_knowledge_extractor.extract_knowledge_from_data(
                data_loader, task_id, max_samples_per_task
            )
        
        logger.info(f"完成 {len(data_loaders)} 个任务的知识提取")
    
    def get_lwf_statistics(self) -> Dict[str, Any]:
        """获取LwF统计信息"""
        stats = {
            'completed_tasks': len(self.task_history),
            'config': self.config.copy(),
            'task_history': self.task_history,
            'knowledge_statistics': {}
        }
        
        # 旧知识提取器统计
        if self.old_knowledge_extractor:
            stats['knowledge_extraction'] = {
                'stored_tasks': list(self.old_knowledge_extractor.knowledge_store.keys()),
                'storage_method': self.old_knowledge_extractor.storage_method,
                'total_knowledge_size': sum(
                    self.old_knowledge_extractor._calculate_data_size(knowledge)
                    for knowledge in self.old_knowledge_extractor.knowledge_store.values()
                )
            }
        
        # 计算总体性能指标
        if self.task_history:
            all_forgetting = [self._calculate_forgetting_measure(record.get('old_tasks', [])) 
                            for record in self.task_history]
            all_retention = [self._calculate_knowledge_retention(record.get('old_tasks', [])) 
                           for record in self.task_history]
            
            stats['overall_metrics'] = {
                'average_forgetting': np.mean(all_forgetting),
                'average_retention': np.mean(all_retention),
                'stability_score': 1.0 - np.std(all_forgetting)
            }
        
        return stats


# 需要导入优化器
import torch.optim as optim
import time