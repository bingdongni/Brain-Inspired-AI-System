"""
智能权重保护模块

实现基于机器学习的动态权重保护策略，包括自适应阈值调整、
参数分组保护、以及基于任务相似度的保护强度调节。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score
import warnings

logger = logging.getLogger(__name__)


class IntelligentWeightProtection:
    """
    智能权重保护系统
    
    基于参数重要性分析和任务相似度，动态调整保护策略。
    实现多种高级保护功能：自适应阈值、分组保护、层次化保护等。
    """
    
    def __init__(self,
                 model: nn.Module,
                 protection_strategies: List[str] = ['adaptive_threshold', 'parameter_grouping'],
                 min_importance_threshold: float = 0.001,
                 max_importance_threshold: float = 0.1,
                 adaptation_rate: float = 0.1,
                 similarity_window: int = 10):
        """
        初始化智能权重保护系统
        
        Args:
            model: 神经网络模型
            protection_strategies: 保护策略列表
            min_importance_threshold: 最小重要性阈值
            max_importance_threshold: 最大重要性阈值
            adaptation_rate: 自适应速率
            similarity_window: 相似度评估窗口
        """
        self.model = model
        self.protection_strategies = protection_strategies
        self.min_importance_threshold = min_importance_threshold
        self.max_importance_threshold = max_importance_threshold
        self.adaptation_rate = adaptation_rate
        self.similarity_window = similarity_window
        
        # 保护状态跟踪
        self.parameter_groups = {}
        self.protection_history = []
        self.task_similarities = []
        self.adaptive_thresholds = {}
        
        # 性能监控
        self.performance_history = []
        self.forgetting_events = []
        
        logger.info(f"初始化智能权重保护系统，策略: {protection_strategies}")
    
    def analyze_parameter_importance(self,
                                   fisher_dict: Dict[str, torch.Tensor],
                                   gradient_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Dict[str, float]]:
        """
        分析参数重要性
        
        Args:
            fisher_dict: Fisher信息矩阵字典
            gradient_dict: 梯度信息字典（可选）
            
        Returns:
            参数重要性分析结果字典
        """
        importance_analysis = {}
        
        for name, fisher in fisher_dict.items():
            # Fisher重要性
            fisher_importance = fisher.mean().item()
            fisher_std = fisher.std().item()
            fisher_max = fisher.max().item()
            
            # 相对重要性
            fisher_normalized = fisher / (fisher.max() + 1e-8)
            normalized_importance = fisher_normalized.mean().item()
            
            # 稀疏性分析
            sparse_threshold = fisher_mean = fisher.mean().item()
            sparsity = (fisher < sparse_threshold).float().mean().item()
            
            # 梯度信息（如果提供）
            gradient_info = {}
            if gradient_dict and name in gradient_dict:
                grad = gradient_dict[name]
                gradient_info = {
                    'grad_norm': grad.norm().item(),
                    'grad_mean': grad.mean().item(),
                    'grad_std': grad.std().item()
                }
            
            importance_analysis[name] = {
                'fisher_importance': fisher_importance,
                'fisher_std': fisher_std,
                'fisher_max': fisher_max,
                'normalized_importance': normalized_importance,
                'sparsity': sparsity,
                'parameter_shape': fisher.shape,
                'parameter_size': fisher.numel(),
                **gradient_info
            }
        
        return importance_analysis
    
    def create_parameter_groups(self,
                              fisher_dict: Dict[str, torch.Tensor],
                              num_groups: Optional[int] = None,
                              grouping_method: str = 'kmeans') -> Dict[str, List[str]]:
        """
        创建参数分组
        
        Args:
            fisher_dict: Fisher信息矩阵字典
            num_groups: 分组数量，None则自动确定
            grouping_method: 分组方法 ('kmeans', 'threshold', 'layer_based')
            
        Returns:
            参数分组字典
        """
        if grouping_method == 'kmeans':
            return self._group_by_kmeans(fisher_dict, num_groups)
        elif grouping_method == 'threshold':
            return self._group_by_threshold(fisher_dict)
        elif grouping_method == 'layer_based':
            return self._group_by_layer(fisher_dict)
        else:
            raise ValueError(f"不支持的分组方法: {grouping_method}")
    
    def _group_by_kmeans(self, fisher_dict: Dict[str, torch.Tensor], num_groups: Optional[int] = None) -> Dict[str, List[str]]:
        """
        使用K-means聚类创建参数分组
        """
        # 提取特征向量
        features = []
        param_names = []
        
        for name, fisher in fisher_dict.items():
            # 使用统计特征
            stats = [
                fisher.mean().item(),
                fisher.std().item(),
                fisher.max().item(),
                fisher.min().item(),
                fisher.numel()  # 参数数量
            ]
            features.append(stats)
            param_names.append(name)
        
        features = np.array(features)
        
        # 确定分组数量
        if num_groups is None:
            num_groups = min(max(3, len(param_names) // 10), 10)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=num_groups, random_state=42, n_init=10)
        group_labels = kmeans.fit_predict(features)
        
        # 组织分组
        groups = defaultdict(list)
        for param_name, group_label in zip(param_names, group_labels):
            groups[f'group_{group_label}'].append(param_name)
        
        # 保存聚类中心信息
        self.parameter_groups['cluster_centers'] = kmeans.cluster_centers_
        self.parameter_groups['grouping_method'] = 'kmeans'
        
        logger.info(f"K-means分组完成，创建 {len(groups)} 个组")
        return dict(groups)
    
    def _group_by_threshold(self, fisher_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """
        基于重要性阈值创建参数分组
        """
        # 计算重要性分布
        all_importances = []
        for fisher in fisher_dict.values():
            all_importances.append(fisher.mean().item())
        
        all_importances = np.array(all_importances)
        
        # 使用百分位数创建阈值
        low_threshold = np.percentile(all_importances, 33)
        high_threshold = np.percentile(all_importances, 66)
        
        groups = {
            'low_importance': [],
            'medium_importance': [],
            'high_importance': []
        }
        
        for name, fisher in fisher_dict.items():
            importance = fisher.mean().item()
            if importance < low_threshold:
                groups['low_importance'].append(name)
            elif importance < high_threshold:
                groups['medium_importance'].append(name)
            else:
                groups['high_importance'].append(name)
        
        self.parameter_groups['thresholds'] = {
            'low': low_threshold,
            'high': high_threshold
        }
        self.parameter_groups['grouping_method'] = 'threshold'
        
        return groups
    
    def _group_by_layer(self, fisher_dict: Dict[str, torch.Tensor]) -> Dict[str, List[str]]:
        """
        基于网络层创建参数分组
        """
        groups = defaultdict(list)
        
        for name in fisher_dict.keys():
            # 解析层名
            if '.' in name:
                layer_path = name.split('.')
                if len(layer_path) >= 2:
                    module_name = '.'.join(layer_path[:-1])  # 模块路径
                    groups[module_name].append(name)
                else:
                    groups['global'].append(name)
            else:
                groups['global'].append(name)
        
        self.parameter_groups['grouping_method'] = 'layer_based'
        return dict(groups)
    
    def compute_adaptive_thresholds(self,
                                  fisher_dict: Dict[str, torch.Tensor],
                                  task_similarity: float,
                                  forgetting_risk: float) -> Dict[str, float]:
        """
        计算自适应阈值
        
        Args:
            fisher_dict: Fisher信息矩阵字典
            task_similarity: 任务相似度
            forgetting_risk: 遗忘风险评估
            
        Returns:
            自适应阈值字典
        """
        # 基础阈值
        base_threshold = self.min_importance_threshold
        
        # 基于任务相似度调整
        similarity_factor = 1.0 + (1.0 - task_similarity) * 0.5
        
        # 基于遗忘风险调整
        risk_factor = 1.0 + forgetting_risk * 2.0
        
        # 历史性能影响
        performance_factor = 1.0
        if len(self.performance_history) > 0:
            recent_performance = np.mean(self.performance_history[-5:])
            performance_factor = 0.5 + recent_performance * 0.5
        
        # 计算最终阈值
        adaptive_threshold = base_threshold * similarity_factor * risk_factor * performance_factor
        adaptive_threshold = min(adaptive_threshold, self.max_importance_threshold)
        
        thresholds = {
            'base': base_threshold,
            'similarity_factor': similarity_factor,
            'risk_factor': risk_factor,
            'performance_factor': performance_factor,
            'final_threshold': adaptive_threshold
        }
        
        self.adaptive_thresholds[f'task_{len(self.adaptive_thresholds)}'] = thresholds
        
        return thresholds
    
    def protect_parameters(self,
                          fisher_dict: Dict[str, torch.Tensor],
                          current_params: Dict[str, torch.Tensor],
                          old_params: Dict[str, torch.Tensor],
                          lambda_base: float,
                          task_similarity: float = 1.0) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        智能参数保护
        
        Args:
            fisher_dict: Fisher信息矩阵字典
            current_params: 当前参数字典
            old_params: 旧参数字典
            lambda_base: 基础保护强度
            task_similarity: 任务相似度
            
        Returns:
            保护权重字典和重要性权重字典
        """
        protection_weights = {}
        importance_weights = {}
        
        # 分析参数重要性
        importance_analysis = self.analyze_parameter_importance(fisher_dict)
        
        # 计算遗忘风险
        forgetting_risk = self._estimate_forgetting_risk(current_params, old_params, fisher_dict)
        
        # 计算自适应阈值
        thresholds = self.compute_adaptive_thresholds(fisher_dict, task_similarity, forgetting_risk)
        adaptive_threshold = thresholds['final_threshold']
        
        for name in fisher_dict.keys():
            if name in current_params and name in old_params:
                # 基础重要性
                base_importance = fisher_dict[name].mean().item()
                importance_weights[name] = base_importance
                
                # 应用保护策略
                protection_weight = lambda_base
                
                # 自适应阈值过滤
                if base_importance < adaptive_threshold:
                    protection_weight *= 0.1  # 大幅降低保护强度
                elif base_importance < adaptive_threshold * 2:
                    protection_weight *= 0.5  # 适度降低保护强度
                
                # 基于参数组调整
                group_factor = self._get_group_protection_factor(name)
                protection_weight *= group_factor
                
                # 基于任务相似度调整
                similarity_factor = 0.5 + task_similarity * 0.5
                protection_weight *= similarity_factor
                
                # 基于遗忘风险调整
                risk_factor = 1.0 + forgetting_risk
                protection_weight *= risk_factor
                
                protection_weights[name] = protection_weight
        
        return protection_weights, importance_weights
    
    def _estimate_forgetting_risk(self,
                                current_params: Dict[str, torch.Tensor],
                                old_params: Dict[str, torch.Tensor],
                                fisher_dict: Dict[str, torch.Tensor]) -> float:
        """
        估计遗忘风险
        """
        if not old_params:
            return 0.0
        
        total_risk = 0.0
        num_params = 0
        
        for name in fisher_dict.keys():
            if name in current_params and name in old_params:
                # 计算参数变化
                param_diff = (current_params[name] - old_params[name]).norm()
                param_norm = old_params[name].norm() + 1e-8
                relative_change = param_diff / param_norm
                
                # 重要性权重
                importance = fisher_dict[name].mean().item()
                
                # 风险计算：变化大且重要性高的参数风险更高
                risk = importance * relative_change
                total_risk += risk
                num_params += 1
        
        return total_risk / max(num_params, 1)
    
    def _get_group_protection_factor(self, param_name: str) -> float:
        """
        获取基于参数组的保护因子
        """
        if not self.parameter_groups:
            return 1.0
        
        # 查找参数所属组
        for group_name, group_params in self.parameter_groups.items():
            if isinstance(group_params, list) and param_name in group_params:
                # 根据组类型调整保护强度
                if 'high' in group_name.lower():
                    return 1.5  # 高重要性组增强保护
                elif 'low' in group_name.lower():
                    return 0.5  # 低重要性组减弱保护
                else:
                    return 1.0  # 中等重要性组保持正常
        
        return 1.0
    
    def adaptive_lambda_scheduling(self,
                                 base_lambda: float,
                                 task_similarity: float,
                                 gradient_magnitude: float,
                                 epoch: int,
                                 total_epochs: int) -> float:
        """
        自适应lambda调度
        
        Args:
            base_lambda: 基础lambda值
            task_similarity: 任务相似度
            gradient_magnitude: 梯度幅度
            epoch: 当前轮次
            total_epochs: 总轮次
            
        Returns:
            调整后的lambda值
        """
        # 基础进度调度
        progress = epoch / max(total_epochs, 1)
        progress_factor = 1.0 - 0.5 * progress  # 线性衰减
        
        # 相似度因子：相似任务需要更强保护
        similarity_factor = 0.5 + task_similarity * 0.5
        
        # 梯度因子：梯度大时减少保护
        gradient_factor = 1.0 / (1.0 + gradient_magnitude * 0.1)
        
        # 自适应因子：基于历史性能
        adaptive_factor = self._compute_adaptive_factor()
        
        adaptive_lambda = base_lambda * progress_factor * similarity_factor * gradient_factor * adaptive_factor
        
        return max(adaptive_lambda, base_lambda * 0.1)  # 最小保护强度
    
    def _compute_adaptive_factor(self) -> float:
        """
        计算自适应因子
        """
        if len(self.performance_history) < 2:
            return 1.0
        
        # 基于性能趋势调整
        recent_performance = np.mean(self.performance_history[-3:])
        baseline_performance = np.mean(self.performance_history[:-3]) if len(self.performance_history) > 3 else recent_performance
        
        if recent_performance < baseline_performance * 0.9:
            # 性能下降，增加保护
            return 1.2
        elif recent_performance > baseline_performance * 1.1:
            # 性能提升，减少保护
            return 0.8
        else:
            return 1.0
    
    def monitor_forgetting_events(self,
                                current_accuracy: float,
                                historical_best: float) -> bool:
        """
        监控遗忘事件
        
        Args:
            current_accuracy: 当前准确率
            historical_best: 历史最佳准确率
            
        Returns:
            是否检测到遗忘事件
        """
        forgetting_threshold = 0.05  # 5%的性能下降认为是遗忘
        
        if current_accuracy < historical_best * (1 - forgetting_threshold):
            forgetting_event = {
                'accuracy_drop': historical_best - current_accuracy,
                'relative_drop': (historical_best - current_accuracy) / historical_best,
                'severity': min(1.0, (historical_best - current_accuracy) / forgetting_threshold)
            }
            
            self.forgetting_events.append(forgetting_event)
            
            # 如果遗忘事件频繁，调整保护策略
            if len(self.forgetting_events) > 5:
                recent_events = self.forgetting_events[-5:]
                if np.mean([event['severity'] for event in recent_events]) > 0.5:
                    self._adjust_protection_strategies()
            
            return True
        
        return False
    
    def _adjust_protection_strategies(self) -> None:
        """
        调整保护策略
        """
        # 增加自适应速率
        self.adaptation_rate *= 1.1
        self.adaptation_rate = min(self.adaptation_rate, 0.5)
        
        # 提高最小阈值
        self.min_importance_threshold *= 1.1
        
        logger.info("检测到频繁遗忘事件，已调整保护策略")
    
    def update_task_similarity(self, 
                             old_features: torch.Tensor,
                             new_features: torch.Tensor) -> float:
        """
        更新任务相似度
        
        Args:
            old_features: 旧任务的特征表示
            new_features: 新任务的特征表示
            
        Returns:
            任务相似度分数
        """
        # 使用余弦相似度
        similarity = F.cosine_similarity(
            old_features.mean(dim=0).unsqueeze(0),
            new_features.mean(dim=0).unsqueeze(0)
        ).item()
        
        self.task_similarities.append(similarity)
        
        # 保持窗口大小
        if len(self.task_similarities) > self.similarity_window:
            self.task_similarities.pop(0)
        
        return similarity
    
    def get_protection_statistics(self) -> Dict[str, Union[float, int, List]]:
        """
        获取保护统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'num_parameter_groups': len(self.parameter_groups),
            'num_forgetting_events': len(self.forgetting_events),
            'avg_task_similarity': np.mean(self.task_similarities) if self.task_similarities else 0.0,
            'recent_performance': self.performance_history[-5:] if len(self.performance_history) >= 5 else self.performance_history,
            'adaptive_thresholds_count': len(self.adaptive_thresholds),
            'protection_strategies': self.protection_strategies,
            'adaptation_rate': self.adaptation_rate,
            'min_importance_threshold': self.min_importance_threshold,
            'max_importance_threshold': self.max_importance_threshold
        }
        
        return stats


class HierarchicalProtection(nn.Module):
    """
    层次化保护系统
    
    实现多层级的参数保护策略，包括全局、模块级和参数级的保护。
    """
    
    def __init__(self,
                 base_protection: IntelligentWeightProtection,
                 hierarchy_levels: List[str] = ['global', 'module', 'parameter']):
        """
        初始化层次化保护系统
        
        Args:
            base_protection: 基础保护系统
            hierarchy_levels: 保护层级列表
        """
        super(HierarchicalProtection, self).__init__()
        
        self.base_protection = base_protection
        self.hierarchy_levels = hierarchy_levels
        
        # 层级保护配置
        self.protection_configs = {
            'global': {'weight': 0.3, 'threshold': 0.01},
            'module': {'weight': 0.5, 'threshold': 0.005},
            'parameter': {'weight': 0.2, 'threshold': 0.001}
        }
    
    def hierarchical_protection(self,
                              fisher_dict: Dict[str, torch.Tensor],
                              current_params: Dict[str, torch.Tensor],
                              old_params: Dict[str, torch.Tensor],
                              lambda_base: float,
                              task_similarity: float = 1.0) -> Dict[str, float]:
        """
        执行层次化保护
        
        Args:
            fisher_dict: Fisher信息矩阵字典
            current_params: 当前参数字典
            old_params: 旧参数字典
            lambda_base: 基础保护强度
            task_similarity: 任务相似度
            
        Returns:
            层次化保护权重字典
        """
        all_protection_weights = {}
        
        # 全局层级保护
        if 'global' in self.hierarchy_levels:
            global_weights = self._global_protection(fisher_dict, lambda_base, task_similarity)
            all_protection_weights.update(global_weights)
        
        # 模块层级保护
        if 'module' in self.hierarchy_levels:
            module_weights = self._module_protection(fisher_dict, lambda_base, task_similarity)
            all_protection_weights.update(module_weights)
        
        # 参数层级保护
        if 'parameter' in self.hierarchy_levels:
            param_weights = self._parameter_protection(fisher_dict, current_params, old_params, lambda_base)
            all_protection_weights.update(param_weights)
        
        return all_protection_weights
    
    def _global_protection(self,
                         fisher_dict: Dict[str, torch.Tensor],
                         lambda_base: float,
                         task_similarity: float) -> Dict[str, float]:
        """
        全局层级保护
        """
        global_weight = self.protection_configs['global']['weight']
        global_threshold = self.protection_configs['global']['threshold']
        
        # 计算全局Fisher重要性分布
        all_importances = [fisher.mean().item() for fisher in fisher_dict.values()]
        global_importance = np.mean(all_importances)
        
        protection_weights = {}
        for name in fisher_dict.keys():
            if global_importance > global_threshold:
                protection_weights[name] = lambda_base * global_weight * (1 + task_similarity)
            else:
                protection_weights[name] = lambda_base * global_weight * 0.1
        
        return protection_weights
    
    def _module_protection(self,
                         fisher_dict: Dict[str, torch.Tensor],
                         lambda_base: float,
                         task_similarity: float) -> Dict[str, float]:
        """
        模块层级保护
        """
        module_weight = self.protection_configs['module']['weight']
        module_threshold = self.protection_configs['module']['threshold']
        
        # 按模块分组
        module_importances = defaultdict(float)
        module_counts = defaultdict(int)
        
        for name, fisher in fisher_dict.items():
            # 提取模块名
            if '.' in name:
                module_name = '.'.join(name.split('.')[:-1])
            else:
                module_name = 'global'
            
            module_importances[module_name] += fisher.mean().item()
            module_counts[module_name] += 1
        
        # 计算平均模块重要性
        for module_name in module_importances:
            module_importances[module_name] /= module_counts[module_name]
        
        protection_weights = {}
        for name in fisher_dict.items():
            if name[0] in fisher_dict:
                # 提取模块名
                if '.' in name[0]:
                    module_name = '.'.join(name[0].split('.')[:-1])
                else:
                    module_name = 'global'
                
                module_importance = module_importances[module_name]
                
                if module_importance > module_threshold:
                    protection_weights[name[0]] = lambda_base * module_weight * (1 + task_similarity * 0.5)
                else:
                    protection_weights[name[0]] = lambda_base * module_weight * 0.1
        
        return protection_weights
    
    def _parameter_protection(self,
                            fisher_dict: Dict[str, torch.Tensor],
                            current_params: Dict[str, torch.Tensor],
                            old_params: Dict[str, torch.Tensor],
                            lambda_base: float) -> Dict[str, torch.Tensor]:
        """
        参数层级保护
        """
        param_weight = self.protection_configs['parameter']['weight']
        param_threshold = self.protection_configs['parameter']['threshold']
        
        protection_weights = {}
        for name in fisher_dict.keys():
            if name in current_params and name in old_params:
                importance = fisher_dict[name].mean().item()
                
                if importance > param_threshold:
                    # 基于参数变化调整
                    param_change = (current_params[name] - old_params[name]).norm()
                    old_norm = old_params[name].norm() + 1e-8
                    change_ratio = param_change / old_norm
                    
                    # 变化越小，保护越强
                    change_factor = 1.0 / (1.0 + change_ratio * 10)
                    
                    protection_weights[name] = lambda_base * param_weight * change_factor
                else:
                    protection_weights[name] = lambda_base * param_weight * 0.05
        
        return protection_weights