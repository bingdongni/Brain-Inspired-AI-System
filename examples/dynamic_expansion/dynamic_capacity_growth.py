"""
动态容量增长模块

实现动态容量增长机制，包括渐进式学习、剪枝扩容和自适应容量调整。
根据任务复杂度和性能需求动态调整网络容量。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from collections import defaultdict, deque
import time
from copy import deepcopy

logger = logging.getLogger(__name__)


class CapacityMonitor:
    """
    容量监控器
    
    监控网络容量使用情况和性能指标。
    """
    
    def __init__(self, 
                 monitor_window: int = 100,
                 performance_threshold: float = 0.95):
        """
        初始化容量监控器
        
        Args:
            monitor_window: 监控窗口大小
            performance_threshold: 性能阈值
        """
        self.monitor_window = monitor_window
        self.performance_threshold = performance_threshold
        
        # 监控历史
        self.performance_history = deque(maxlen=monitor_window)
        self.capacity_history = deque(maxlen=monitor_window)
        self.efficiency_history = deque(maxlen=monitor_window)
        
        # 容量统计
        self.total_capacity_used = 0
        self.peak_capacity = 0
        self.avg_efficiency = 0.0
        
    def update_metrics(self, 
                      performance: float, 
                      capacity_usage: float,
                      task_complexity: float = 1.0) -> Dict[str, float]:
        """
        更新监控指标
        
        Args:
            performance: 当前性能
            capacity_usage: 容量使用率
            task_complexity: 任务复杂度
            
        Returns:
            监控结果字典
        """
        # 更新历史
        self.performance_history.append(performance)
        self.capacity_history.append(capacity_usage)
        
        # 计算效率（性能/容量）
        efficiency = performance / max(capacity_usage, 0.001)
        self.efficiency_history.append(efficiency)
        
        # 更新统计
        self.total_capacity_used += capacity_usage
        self.peak_capacity = max(self.peak_capacity, capacity_usage)
        self.avg_efficiency = np.mean(list(self.efficiency_history)) if self.efficiency_history else 0.0
        
        # 生成监控报告
        monitor_result = self._generate_monitor_report()
        
        return monitor_result
    
    def _generate_monitor_report(self) -> Dict[str, float]:
        """生成监控报告"""
        report = {}
        
        if self.performance_history:
            report['avg_performance'] = np.mean(list(self.performance_history))
            report['performance_trend'] = self._calculate_trend(list(self.performance_history))
            report['performance_stability'] = np.std(list(self.performance_history))
        
        if self.capacity_history:
            report['avg_capacity'] = np.mean(list(self.capacity_history))
            report['capacity_variance'] = np.var(list(self.capacity_history))
        
        if self.efficiency_history:
            report['current_efficiency'] = list(self.efficiency_history)[-1] if self.efficiency_history else 0.0
            report['efficiency_trend'] = self._calculate_trend(list(self.efficiency_history))
        
        report['total_operations'] = len(self.performance_history)
        report['peak_capacity'] = self.peak_capacity
        
        return report
    
    def _calculate_trend(self, values: List[float]) -> float:
        """计算趋势（简单线性回归斜率）"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # 线性回归
        slope, _ = np.polyfit(x, y, 1)
        return slope
    
    def needs_capacity_expansion(self) -> Tuple[bool, str]:
        """
        判断是否需要容量扩展
        
        Returns:
            (是否需要扩展, 原因)
        """
        if len(self.performance_history) < 10:
            return False, "数据不足"
        
        # 检查性能趋势
        recent_performance = np.mean(list(self.performance_history)[-10:])
        performance_trend = self._calculate_trend(list(self.performance_history)[-20:])
        
        # 检查效率
        avg_efficiency = np.mean(list(self.efficiency_history)[-10:]) if self.efficiency_history else 0
        
        # 扩展条件
        if performance_trend < -0.01:  # 性能下降趋势
            return True, f"性能下降趋势: {performance_trend:.4f}"
        elif recent_performance < self.performance_threshold and avg_efficiency < 0.8:
            return True, f"性能不足且效率低: {recent_performance:.4f}, {avg_efficiency:.4f}"
        
        return False, "当前容量充足"
    
    def get_optimization_suggestions(self) -> List[str]:
        """
        获取优化建议
        
        Returns:
            建议列表
        """
        suggestions = []
        
        if len(self.performance_history) > 0:
            recent_performance = np.mean(list(self.performance_history)[-5:])
            if recent_performance < self.performance_threshold:
                suggestions.append("考虑增加网络容量或优化架构")
        
        if len(self.efficiency_history) > 0:
            current_efficiency = list(self.efficiency_history)[-1]
            if current_efficiency < 0.6:
                suggestions.append("考虑剪枝减少冗余参数")
        
        if len(self.capacity_history) > 0:
            recent_capacity = np.mean(list(self.capacity_history)[-5:])
            if recent_capacity > 0.9:
                suggestions.append("容量使用率过高，建议扩展")
        
        if not suggestions:
            suggestions.append("当前状态良好，无需优化")
        
        return suggestions


class GradualCapacityExpansion:
    """
    渐进式容量扩展
    
    实现渐进式的网络容量增长，支持多种扩展策略。
    """
    
    def __init__(self, 
                 base_network: nn.Module,
                 expansion_strategies: List[str] = ['layer_expansion', 'width_expansion'],
                 growth_rate: float = 1.2,
                 max_expansion_factor: float = 3.0):
        """
        初始化渐进式容量扩展
        
        Args:
            base_network: 基础网络
            expansion_strategies: 扩展策略列表
            growth_rate: 增长率
            max_expansion_factor: 最大扩展因子
        """
        self.base_network = base_network
        self.expansion_strategies = expansion_strategies
        self.growth_rate = growth_rate
        self.max_expansion_factor = max_expansion_factor
        
        # 扩展历史
        self.expansion_history = []
        self.expansion_count = 0
        
        # 容量配置
        self.capacity_configs = {}
        self._initialize_capacity_configs()
        
        logger.info(f"初始化渐进式容量扩展，策略: {expansion_strategies}")
    
    def _initialize_capacity_configs(self):
        """初始化容量配置"""
        for name, module in self.base_network.named_modules():
            if isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                self.capacity_configs[name] = {
                    'input_dim': in_features,
                    'output_dim': out_features,
                    'expansion_factor': 1.0
                }
            elif isinstance(module, nn.Conv2d):
                in_channels = module.in_channels
                out_channels = module.out_channels
                self.capacity_configs[name] = {
                    'input_channels': in_channels,
                    'output_channels': out_channels,
                    'expansion_factor': 1.0
                }
    
    def expand_capacity(self, 
                       target_improvement: float = 0.1,
                       constraint_based: bool = True) -> nn.Module:
        """
        扩展网络容量
        
        Args:
            target_improvement: 目标性能提升
            constraint_based: 是否基于约束扩展
            
        Returns:
            扩展后的网络
        """
        logger.info(f"开始容量扩展，目标提升: {target_improvement:.2%}")
        
        # 创建扩展后的网络副本
        expanded_network = deepcopy(self.base_network)
        
        # 计算扩展因子
        expansion_factor = self._calculate_expansion_factor(target_improvement, constraint_based)
        expansion_factor = min(expansion_factor, self.max_expansion_factor)
        
        # 应用扩展策略
        for strategy in self.expansion_strategies:
            expanded_network = self._apply_expansion_strategy(expanded_network, strategy, expansion_factor)
        
        # 更新配置
        self._update_capacity_configs(expansion_factor)
        
        # 记录扩展历史
        self.expansion_history.append({
            'timestamp': time.time(),
            'expansion_factor': expansion_factor,
            'target_improvement': target_improvement,
            'strategies_used': self.expansion_strategies.copy()
        })
        self.expansion_count += 1
        
        logger.info(f"容量扩展完成，扩展因子: {expansion_factor:.2f}")
        
        return expanded_network
    
    def _calculate_expansion_factor(self, 
                                   target_improvement: float,
                                   constraint_based: bool) -> float:
        """计算扩展因子"""
        if constraint_based:
            # 基于约束的扩展
            base_factor = 1.0 + target_improvement
            # 考虑历史扩展效果
            if self.expansion_history:
                avg_effectiveness = np.mean([h.get('effectiveness', 1.0) for h in self.expansion_history[-3:]])
                base_factor *= avg_effectiveness
        else:
            # 固定增长率
            base_factor = self.growth_rate
        
        return base_factor
    
    def _apply_expansion_strategy(self, 
                                 network: nn.Module, 
                                 strategy: str, 
                                 expansion_factor: float) -> nn.Module:
        """
        应用扩展策略
        
        Args:
            network: 网络
            strategy: 扩展策略
            expansion_factor: 扩展因子
            
        Returns:
            扩展后的网络
        """
        if strategy == 'layer_expansion':
            return self._expand_layers(network, expansion_factor)
        elif strategy == 'width_expansion':
            return self._expand_width(network, expansion_factor)
        elif strategy == 'depth_expansion':
            return self._expand_depth(network, expansion_factor)
        elif strategy == 'residual_expansion':
            return self._expand_residual(network, expansion_factor)
        else:
            logger.warning(f"未知扩展策略: {strategy}")
            return network
    
    def _expand_layers(self, network: nn.Module, expansion_factor: float) -> nn.Module:
        """扩展层数"""
        for name, module in network.named_children():
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                # 为序列模块添加新层
                expanded_module = self._add_layers_to_module(module, expansion_factor)
                setattr(network, name, expanded_module)
            else:
                # 递归扩展子模块
                setattr(network, name, self._expand_layers(module, expansion_factor))
        
        return network
    
    def _add_layers_to_module(self, module: nn.Module, expansion_factor: float) -> nn.Module:
        """为模块添加层"""
        if isinstance(module, nn.Sequential):
            # 计算要添加的层数
            num_layers = len(module)
            additional_layers = int(num_layers * (expansion_factor - 1))
            
            if additional_layers > 0:
                # 获取最后一层的配置
                last_layer = module[-1]
                if isinstance(last_layer, nn.Linear):
                    new_layers = []
                    for _ in range(additional_layers):
                        new_layer = nn.Linear(last_layer.in_features, last_layer.out_features)
                        new_layers.append(new_layer)
                    
                    return nn.Sequential(*module.children(), *new_layers)
        
        return module
    
    def _expand_width(self, network: nn.Module, expansion_factor: float) -> nn.Module:
        """扩展宽度"""
        for name, module in network.named_children():
            if isinstance(module, nn.Linear):
                # 扩展线性层宽度
                expanded_module = self._expand_linear_layer(module, expansion_factor)
                setattr(network, name, expanded_module)
            elif isinstance(module, nn.Conv2d):
                # 扩展卷积层通道数
                expanded_module = self._expand_conv_layer(module, expansion_factor)
                setattr(network, name, expanded_module)
            else:
                # 递归扩展
                setattr(network, name, self._expand_width(module, expansion_factor))
        
        return network
    
    def _expand_linear_layer(self, layer: nn.Linear, expansion_factor: float) -> nn.Linear:
        """扩展线性层"""
        in_features = layer.in_features
        out_features = layer.out_features
        
        new_out_features = int(out_features * expansion_factor)
        
        # 创建新的线性层
        new_layer = nn.Linear(in_features, new_out_features)
        
        # 复制权重
        min_out = min(out_features, new_out_features)
        new_layer.weight.data[:min_out] = layer.weight.data
        new_layer.bias.data[:min_out] = layer.bias.data
        
        return new_layer
    
    def _expand_conv_layer(self, layer: nn.Conv2d, expansion_factor: float) -> nn.Conv2d:
        """扩展卷积层"""
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        
        new_out_channels = int(out_channels * expansion_factor)
        
        # 创建新的卷积层
        new_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=new_out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias is not None
        )
        
        # 复制权重
        min_out = min(out_channels, new_out_channels)
        new_layer.weight.data[:min_out] = layer.weight.data
        if layer.bias is not None and new_layer.bias is not None:
            new_layer.bias.data[:min_out] = layer.bias.data
        
        return new_layer
    
    def _expand_depth(self, network: nn.Module, expansion_factor: float) -> nn.Module:
        """扩展深度"""
        # 为每个主要块添加残差连接和额外层
        for name, module in network.named_children():
            if hasattr(module, 'blocks') or 'block' in name.lower():
                # 这是一个块结构，添加新的块
                expanded_blocks = self._add_blocks_to_module(module, expansion_factor)
                setattr(network, name, expanded_blocks)
            else:
                setattr(network, name, self._expand_depth(module, expansion_factor))
        
        return network
    
    def _add_blocks_to_module(self, module: nn.Module, expansion_factor: float) -> nn.Module:
        """为模块添加块"""
        if hasattr(module, 'blocks') and isinstance(module.blocks, nn.ModuleList):
            num_blocks = len(module.blocks)
            additional_blocks = int(num_blocks * (expansion_factor - 1))
            
            if additional_blocks > 0:
                # 复制最后一个块的配置
                last_block = module.blocks[-1]
                new_blocks = []
                
                for _ in range(additional_blocks):
                    # 复制块结构
                    new_block = deepcopy(last_block)
                    new_blocks.append(new_block)
                
                module.blocks.extend(new_blocks)
        
        return module
    
    def _expand_residual(self, network: nn.Module, expansion_factor: float) -> nn.Module:
        """扩展残差连接"""
        # 为残差块添加额外的路径
        for name, module in network.named_children():
            if hasattr(module, 'downsample') or 'residual' in name.lower():
                # 这是残差块，添加额外的残差路径
                expanded_module = self._add_residual_path(module, expansion_factor)
                setattr(network, name, expanded_module)
            else:
                setattr(network, name, self._expand_residual(module, expansion_factor))
        
        return network
    
    def _add_residual_path(self, module: nn.Module, expansion_factor: float) -> nn.Module:
        """添加残差路径"""
        if hasattr(module, 'main_branch') and hasattr(module, 'shortcut'):
            # 创建额外的残差路径
            additional_path = deepcopy(module.main_branch)
            module.additional_path = additional_path
        
        return module
    
    def _update_capacity_configs(self, expansion_factor: float):
        """更新容量配置"""
        for config in self.capacity_configs.values():
            config['expansion_factor'] *= expansion_factor
    
    def get_expansion_statistics(self) -> Dict[str, Any]:
        """获取扩展统计信息"""
        stats = {
            'expansion_count': self.expansion_count,
            'expansion_history_length': len(self.expansion_history),
            'current_growth_rate': self.growth_rate,
            'max_expansion_factor': self.max_expansion_factor,
            'strategies_used': self.expansion_strategies,
            'expansion_history': self.expansion_history,
            'capacity_configs': self.capacity_configs
        }
        
        if self.expansion_history:
            total_expansion = np.prod([h['expansion_factor'] for h in self.expansion_history])
            stats['total_expansion_factor'] = total_expansion
            stats['average_expansion_factor'] = np.mean([h['expansion_factor'] for h in self.expansion_history])
        
        return stats


class PruningBasedExpansion:
    """
    基于剪枝的扩展
    
    通过剪枝去除冗余参数，然后扩展重要参数来优化容量。
    """
    
    def __init__(self,
                 network: nn.Module,
                 pruning_threshold: float = 0.1,
                 expansion_multiplier: float = 1.5):
        """
        初始化基于剪枝的扩展
        
        Args:
            network: 基础网络
            pruning_threshold: 剪枝阈值
            expansion_multiplier: 扩展乘数
        """
        self.network = network
        self.pruning_threshold = pruning_threshold
        self.expansion_multiplier = expansion_multiplier
        
        # 剪枝历史
        self.pruning_history = []
        self.expansion_history = []
        
        logger.info("初始化基于剪枝的扩展机制")
    
    def prune_and_expand(self, 
                        expansion_targets: Optional[List[str]] = None) -> nn.Module:
        """
        剪枝并扩展
        
        Args:
            expansion_targets: 扩展目标层名列表
            
        Returns:
            优化后的网络
        """
        logger.info("开始剪枝和扩展")
        
        # 第一步：剪枝
        pruned_network = self._prune_network()
        
        # 第二步：扩展重要参数
        if expansion_targets is None:
            expansion_targets = self._identify_expansion_targets(pruned_network)
        
        expanded_network = self._expand_important_parameters(pruned_network, expansion_targets)
        
        return expanded_network
    
    def _prune_network(self) -> nn.Module:
        """剪枝网络"""
        pruned_network = deepcopy(self.network)
        
        total_params = 0
        pruned_params = 0
        
        for name, module in pruned_network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 计算参数重要性
                if isinstance(module, nn.Linear):
                    importance_scores = torch.abs(module.weight.data).mean(dim=1)
                else:  # Conv2d
                    importance_scores = torch.abs(module.weight.data).mean(dim=(1, 2, 3))
                
                # 确定剪枝掩码
                threshold = torch.quantile(importance_scores, self.pruning_threshold)
                prune_mask = importance_scores >= threshold
                
                # 应用剪枝
                if isinstance(module, nn.Linear):
                    kept_features = prune_mask.sum().item()
                    total_features = module.out_features
                    
                    # 更新层
                    new_layer = nn.Linear(module.in_features, kept_features)
                    new_layer.weight.data = module.weight.data[prune_mask]
                    if module.bias is not None:
                        new_layer.bias.data = module.bias.data[prune_mask]
                    
                    setattr(pruned_network, name.split('.')[-1], new_layer)
                    
                elif isinstance(module, nn.Conv2d):
                    kept_channels = prune_mask.sum().item()
                    total_channels = module.out_channels
                    
                    # 更新层
                    new_layer = nn.Conv2d(
                        module.in_channels,
                        kept_channels,
                        module.kernel_size,
                        module.stride,
                        module.padding,
                        module.bias is not None
                    )
                    new_layer.weight.data = module.weight.data[prune_mask]
                    if module.bias is not None:
                        new_layer.bias.data = module.bias.data[prune_mask]
                    
                    setattr(pruned_network, name.split('.')[-1], new_layer)
                
                pruned_params += (total_params - kept_features if isinstance(module, nn.Linear) 
                                else total_channels - kept_channels)
                total_params += total_features if isinstance(module, nn.Linear) else total_channels
        
        # 记录剪枝历史
        pruning_info = {
            'timestamp': time.time(),
            'pruning_threshold': self.pruning_threshold,
            'pruned_parameters': pruned_params,
            'total_parameters': total_params,
            'pruning_ratio': pruned_params / max(total_params, 1)
        }
        
        self.pruning_history.append(pruning_info)
        
        logger.info(f"剪枝完成，剪枝比例: {pruning_info['pruning_ratio']:.2%}")
        
        return pruned_network
    
    def _identify_expansion_targets(self, network: nn.Module) -> List[str]:
        """识别扩展目标"""
        expansion_targets = []
        
        for name, module in network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # 计算层的重要性（基于参数数量和梯度）
                param_count = sum(p.numel() for p in module.parameters())
                
                # 重要层特征：参数多、中等复杂度
                if 1000 < param_count < 100000:
                    expansion_targets.append(name)
        
        logger.info(f"识别扩展目标: {expansion_targets}")
        return expansion_targets
    
    def _expand_important_parameters(self, 
                                   network: nn.Module, 
                                   expansion_targets: List[str]) -> nn.Module:
        """扩展重要参数"""
        expanded_network = network
        
        for target_name in expansion_targets:
            module = self._get_module_by_name(network, target_name)
            if module is not None:
                if isinstance(module, nn.Linear):
                    expanded_module = self._expand_linear_layer(module, self.expansion_multiplier)
                elif isinstance(module, nn.Conv2d):
                    expanded_module = self._expand_conv_layer(module, self.expansion_multiplier)
                else:
                    continue
                
                # 替换模块
                self._set_module_by_name(expanded_network, target_name, expanded_module)
        
        # 记录扩展历史
        expansion_info = {
            'timestamp': time.time(),
            'expansion_multiplier': self.expansion_multiplier,
            'expansion_targets': expansion_targets,
            'num_targets': len(expansion_targets)
        }
        
        self.expansion_history.append(expansion_info)
        
        logger.info(f"扩展完成，扩展目标数: {len(expansion_targets)}")
        
        return expanded_network
    
    def _get_module_by_name(self, network: nn.Module, name: str) -> Optional[nn.Module]:
        """根据名称获取模块"""
        parts = name.split('.')
        module = network
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        
        return module
    
    def _set_module_by_name(self, network: nn.Module, name: str, new_module: nn.Module):
        """根据名称设置模块"""
        parts = name.split('.')
        module = network
        
        for part in parts[:-1]:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return
        
        setattr(module, parts[-1], new_module)
    
    def _expand_linear_layer(self, layer: nn.Linear, multiplier: float) -> nn.Linear:
        """扩展线性层"""
        in_features = layer.in_features
        out_features = layer.out_features
        new_out_features = int(out_features * multiplier)
        
        new_layer = nn.Linear(in_features, new_out_features)
        
        # 复制原有权重
        min_out = min(out_features, new_out_features)
        new_layer.weight.data[:min_out] = layer.weight.data
        new_layer.bias.data[:min_out] = layer.bias.data
        
        # 初始化新增权重
        if new_out_features > out_features:
            nn.init.xavier_uniform_(new_layer.weight.data[out_features:])
            if new_layer.bias is not None:
                nn.init.zeros_(new_layer.bias.data[out_features:])
        
        return new_layer
    
    def _expand_conv_layer(self, layer: nn.Conv2d, multiplier: float) -> nn.Conv2d:
        """扩展卷积层"""
        in_channels = layer.in_channels
        out_channels = layer.out_channels
        new_out_channels = int(out_channels * multiplier)
        
        new_layer = nn.Conv2d(
            in_channels,
            new_out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.bias is not None
        )
        
        # 复制原有权重
        min_out = min(out_channels, new_out_channels)
        new_layer.weight.data[:min_out] = layer.weight.data
        if layer.bias is not None and new_layer.bias is not None:
            new_layer.bias.data[:min_out] = layer.bias.data
        
        # 初始化新增权重
        if new_out_channels > out_channels:
            nn.init.xavier_normal_(new_layer.weight.data[out_channels:])
            if new_layer.bias is not None:
                nn.init.zeros_(new_layer.bias.data[out_channels:])
        
        return new_layer
    
    def get_pruning_expansion_statistics(self) -> Dict[str, Any]:
        """获取剪枝扩展统计信息"""
        return {
            'pruning_history': self.pruning_history,
            'expansion_history': self.expansion_history,
            'total_pruning_operations': len(self.pruning_history),
            'total_expansion_operations': len(self.expansion_history),
            'pruning_threshold': self.pruning_threshold,
            'expansion_multiplier': self.expansion_multiplier,
            'net_effectiveness': self._calculate_net_effectiveness()
        }
    
    def _calculate_net_effectiveness(self) -> float:
        """计算净效果"""
        if not self.pruning_history or not self.expansion_history:
            return 0.0
        
        total_pruning_ratio = np.mean([h['pruning_ratio'] for h in self.pruning_history])
        avg_expansion = np.mean([h['expansion_multiplier'] for h in self.expansion_history])
        
        # 净效果：扩展效果减去剪枝效果
        net_effectiveness = avg_expansion - total_pruning_ratio
        
        return net_effectiveness


class DynamicCapacityGrowth:
    """
    动态容量增长系统
    
    综合容量监控、渐进式扩展和基于剪枝的扩展的统一框架。
    """
    
    def __init__(self,
                 base_network: nn.Module,
                 config: Optional[Dict] = None):
        """
        初始化动态容量增长系统
        
        Args:
            base_network: 基础网络
            config: 配置字典
        """
        self.base_network = base_network
        self.config = config or {}
        
        # 默认配置
        default_config = {
            'monitor_window': 100,
            'performance_threshold': 0.95,
            'expansion_strategies': ['layer_expansion', 'width_expansion'],
            'growth_rate': 1.2,
            'pruning_threshold': 0.1,
            'expansion_multiplier': 1.5
        }
        default_config.update(self.config)
        self.config = default_config
        
        # 初始化组件
        self.capacity_monitor = CapacityMonitor(
            monitor_window=default_config['monitor_window'],
            performance_threshold=default_config['performance_threshold']
        )
        
        self.gradual_expansion = GradualCapacityExpansion(
            base_network=base_network,
            expansion_strategies=default_config['expansion_strategies'],
            growth_rate=default_config['growth_rate']
        )
        
        self.pruning_expansion = PruningBasedExpansion(
            network=base_network,
            pruning_threshold=default_config['pruning_threshold'],
            expansion_multiplier=default_config['expansion_multiplier']
        )
        
        # 当前网络
        self.current_network = base_network
        
        # 增长历史
        self.growth_history = []
        self.optimization_decisions = []
        
        logger.info("动态容量增长系统初始化完成")
    
    def adapt_capacity(self, 
                      current_performance: float,
                      task_complexity: float = 1.0,
                      resource_constraints: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        自适应容量调整
        
        Args:
            current_performance: 当前性能
            task_complexity: 任务复杂度
            resource_constraints: 资源约束
            
        Returns:
            适应结果字典
        """
        # 更新容量监控
        capacity_usage = self._estimate_current_capacity_usage()
        
        monitor_result = self.capacity_monitor.update_metrics(
            performance=current_performance,
            capacity_usage=capacity_usage,
            task_complexity=task_complexity
        )
        
        # 判断是否需要扩展
        needs_expansion, expansion_reason = self.capacity_monitor.needs_capacity_expansion()
        
        # 记录决策
        decision = {
            'timestamp': time.time(),
            'current_performance': current_performance,
            'capacity_usage': capacity_usage,
            'task_complexity': task_complexity,
            'needs_expansion': needs_expansion,
            'expansion_reason': expansion_reason,
            'monitor_result': monitor_result
        }
        
        adaptation_result = {'decision': decision}
        
        if needs_expansion:
            # 选择扩展策略
            expansion_strategy = self._select_expansion_strategy(
                current_performance, task_complexity, resource_constraints)
            
            # 执行扩展
            expanded_network = self._execute_expansion(expansion_strategy, task_complexity)
            
            adaptation_result.update({
                'expansion_strategy': expansion_strategy,
                'expanded_network': expanded_network,
                'original_network_size': self._count_parameters(self.current_network),
                'expanded_network_size': self._count_parameters(expanded_network)
            })
            
            # 更新当前网络
            self.current_network = expanded_network
            
            logger.info(f"执行容量扩展，策略: {expansion_strategy}")
        
        # 记录适应历史
        self.growth_history.append(adaptation_result)
        self.optimization_decisions.append(decision)
        
        return adaptation_result
    
    def _estimate_current_capacity_usage(self) -> float:
        """估计当前容量使用率"""
        if not self.current_network:
            return 0.0
        
        total_params = self._count_parameters(self.current_network)
        max_params = self._count_parameters(self.base_network) * 10  # 假设最大容量是基础网络的10倍
        
        return min(1.0, total_params / max_params)
    
    def _select_expansion_strategy(self,
                                 performance: float,
                                 complexity: float,
                                 constraints: Optional[Dict[str, float]]) -> str:
        """选择扩展策略"""
        # 基于性能和复杂度选择策略
        if performance < 0.8:
            # 性能较差，优先渐进式扩展
            return 'gradual_expansion'
        elif complexity > 1.5:
            # 任务复杂，选择基于剪枝的扩展
            return 'pruning_expansion'
        elif constraints and constraints.get('memory_limit', 1.0) < 0.8:
            # 内存受限，谨慎扩展
            return 'selective_expansion'
        else:
            # 默认选择混合策略
            return 'hybrid_expansion'
    
    def _execute_expansion(self, strategy: str, complexity: float) -> nn.Module:
        """执行扩展策略"""
        if strategy == 'gradual_expansion':
            target_improvement = max(0.05, complexity * 0.1)
            return self.gradual_expansion.expand_capacity(target_improvement)
        
        elif strategy == 'pruning_expansion':
            return self.pruning_expansion.prune_and_expand()
        
        elif strategy == 'selective_expansion':
            # 选择性扩展：只扩展关键层
            key_layers = self._identify_key_layers()
            return self.gradual_expansion.expand_capacity(
                target_improvement=complexity * 0.05,
                constraint_based=True
            )
        
        elif strategy == 'hybrid_expansion':
            # 混合策略：先剪枝再渐进扩展
            pruned_network = self.pruning_expansion.prune_and_expand()
            expanded_network = self.gradual_expansion.expand_capacity(
                target_improvement=complexity * 0.1
            )
            return expanded_network
        
        else:
            logger.warning(f"未知扩展策略: {strategy}")
            return self.current_network
    
    def _identify_key_layers(self) -> List[str]:
        """识别关键层"""
        key_layers = []
        
        for name, module in self.current_network.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                param_count = sum(p.numel() for p in module.parameters())
                # 识别参数较多的层
                if param_count > 10000:
                    key_layers.append(name)
        
        return key_layers
    
    def _count_parameters(self, network: nn.Module) -> int:
        """计算网络参数数量"""
        return sum(p.numel() for p in network.parameters())
    
    def get_growth_statistics(self) -> Dict[str, Any]:
        """获取增长统计信息"""
        return {
            'capacity_monitor_stats': self.capacity_monitor._generate_monitor_report(),
            'gradual_expansion_stats': self.gradual_expansion.get_expansion_statistics(),
            'pruning_expansion_stats': self.pruning_expansion.get_pruning_expansion_statistics(),
            'growth_history_length': len(self.growth_history),
            'total_optimization_decisions': len(self.optimization_decisions),
            'current_network_size': self._count_parameters(self.current_network),
            'base_network_size': self._count_parameters(self.base_network),
            'growth_ratio': self._count_parameters(self.current_network) / max(self._count_parameters(self.base_network), 1),
            'optimization_effectiveness': self._calculate_optimization_effectiveness()
        }
    
    def _calculate_optimization_effectiveness(self) -> float:
        """计算优化效果"""
        if len(self.growth_history) < 2:
            return 0.0
        
        # 计算最近几次优化的平均效果
        recent_decisions = self.optimization_decisions[-5:]
        
        performance_improvements = []
        for i in range(1, len(recent_decisions)):
            prev_perf = recent_decisions[i-1]['current_performance']
            curr_perf = recent_decisions[i]['current_performance']
            improvement = curr_perf - prev_perf
            performance_improvements.append(improvement)
        
        return np.mean(performance_improvements) if performance_improvements else 0.0
    
    def reset_network(self) -> None:
        """重置网络到基础状态"""
        self.current_network = deepcopy(self.base_network)
        self.growth_history.clear()
        self.optimization_decisions.clear()
        
        logger.info("网络已重置到基础状态")