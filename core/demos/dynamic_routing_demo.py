#!/usr/bin/env python3
"""
动态路由演示 - 神经网络动态路由机制
Dynamic Routing Demo - Neural Network Dynamic Routing

演示动态路由的核心功能：
- 动态连接权重调整
- 路径优化
- 负载均衡
- 效率分析
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
import argparse
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将使用简化版本演示")

class DynamicRouter:
    """动态路由器"""
    
    def __init__(self, 
                 input_size: int = 64,
                 hidden_size: int = 128,
                 num_layers: int = 4,
                 num_routes: int = 8):
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_routes = num_routes
        
        # 路由表
        self.routing_table = []
        self.route_capacities = []
        self.route_loads = []
        
        # 动态权重
        self.dynamic_weights = {}
        self.route_efficiencies = {}
        
        # 统计信息
        self.stats = {
            'total_routes': 0,
            'successful_optimizations': 0,
            'load_balances': 0,
            'path_compressions': 0,
            'efficiency_improvements': 0
        }
        
        # 初始化路由表
        self._initialize_routes()
        
    def _initialize_routes(self):
        """初始化路由表"""
        print("🛣️ 初始化动态路由表...")
        
        # 创建多层路由
        for layer in range(self.num_layers):
            layer_routes = []
            layer_capacities = []
            layer_loads = []
            
            # 每层创建多条路由
            for route_id in range(self.num_routes):
                route = {
                    'layer': layer,
                    'route_id': route_id,
                    'capacity': np.random.uniform(0.8, 1.2),  # 路由容量
                    'efficiency': np.random.uniform(0.7, 0.9),  # 路由效率
                    'connections': [],  # 连接信息
                    'current_load': 0.0,  # 当前负载
                    'queue_length': 0,  # 队列长度
                }
                
                layer_routes.append(route)
                layer_capacities.append(route['capacity'])
                layer_loads.append(0.0)
                
            self.routing_table.append(layer_routes)
            self.route_capacities.append(layer_capacities)
            self.route_loads.append(layer_loads)
            
        print(f"   创建了 {self.num_layers} 层路由，每层 {self.num_routes} 条路径")
        
    def route_input(self, input_data: np.ndarray, 
                   input_size: int = None,
                   optimization_level: str = 'medium') -> Dict[str, Any]:
        """路由输入数据"""
        if input_size is None:
            input_size = self.input_size
            
        # 确保输入是正确维度
        if input_data.ndim == 1:
            input_data = input_data.reshape(1, -1)
            
        batch_size = input_data.shape[0]
        
        # 初始化路由结果
        routing_result = {
            'routed_paths': [],
            'total_cost': 0.0,
            'load_distribution': [],
            'processing_time': 0.0,
            'optimization_applied': False
        }
        
        start_time = time.time()
        
        # 选择路由策略
        if optimization_level == 'high':
            paths = self._optimized_routing(input_data)
        elif optimization_level == 'medium':
            paths = self._balanced_routing(input_data)
        else:
            paths = self._simple_routing(input_data)
            
        # 应用动态调整
        for layer in range(self.num_layers):
            self._adjust_routing_weights(layer, paths[layer])
            
        routing_result['routed_paths'] = paths
        routing_result['processing_time'] = time.time() - start_time
        
        return routing_result
        
    def _simple_routing(self, input_data: np.ndarray) -> List[List[int]]:
        """简单路由策略"""
        paths = []
        
        for layer in range(self.num_layers):
            layer_paths = []
            
            for route_id in range(self.num_routes):
                # 随机选择路由
                route_selection = np.random.randint(0, self.num_routes)
                layer_paths.append(route_selection)
                
            paths.append(layer_paths)
            
        return paths
        
    def _balanced_routing(self, input_data: np.ndarray) -> List[List[int]]:
        """负载均衡路由策略"""
        paths = []
        
        for layer in range(self.num_layers):
            layer_paths = []
            
            # 获取当前层负载
            layer_loads = self.route_loads[layer]
            
            # 选择负载最轻的路由
            for batch_item in range(input_data.shape[0]):
                # 计算负载得分（考虑容量和当前负载）
                load_scores = []
                
                for route_id in range(self.num_routes):
                    capacity = self.route_capacities[layer][route_id]
                    current_load = layer_loads[route_id]
                    
                    # 负载得分 = 容量 / (1 + 当前负载)
                    load_score = capacity / (1.0 + current_load)
                    load_scores.append(load_score)
                    
                # 选择得分最高的路由
                best_route = np.argmax(load_scores)
                layer_paths.append(best_route)
                
                # 更新负载
                layer_loads[best_route] += 1.0
                
            paths.append(layer_paths)
            
        return paths
        
    def _optimized_routing(self, input_data: np.ndarray) -> List[List[int]]:
        """优化路由策略"""
        paths = []
        
        for layer in range(self.num_layers):
            layer_paths = []
            
            # 考虑效率和负载的组合优化
            layer_loads = self.route_loads[layer]
            
            for batch_item in range(input_data.shape[0]):
                # 计算综合得分
                composite_scores = []
                
                for route_id in range(self.num_routes):
                    capacity = self.route_capacities[layer][route_id]
                    efficiency = self.route_efficiencies.get((layer, route_id), 0.8)
                    current_load = layer_loads[route_id]
                    
                    # 综合得分 = 效率 * 容量 / (1 + 负载 + 延迟惩罚)
                    delay_penalty = np.random.uniform(0.05, 0.15)  # 模拟延迟
                    composite_score = efficiency * capacity / (1.0 + current_load + delay_penalty)
                    composite_scores.append(composite_score)
                    
                # 选择综合得分最高的路由
                best_route = np.argmax(composite_scores)
                layer_paths.append(best_route)
                
                # 更新负载和效率
                layer_loads[best_route] += 1.0
                
                # 模拟效率随负载变化
                if layer_loads[best_route] > capacity:
                    self.route_efficiencies[(layer, best_route)] *= 0.95
                else:
                    self.route_efficiencies[(layer, best_route)] = min(0.95, 
                        self.route_efficiencies.get((layer, best_route), 0.8) * 1.01)
                    
            paths.append(layer_paths)
            
        # 记录优化结果
        self.stats['successful_optimizations'] += 1
        
        return paths
        
    def _adjust_routing_weights(self, layer: int, paths: List[int]):
        """调整路由权重"""
        # 统计路径使用情况
        path_usage = {}
        for path in paths:
            path_usage[path] = path_usage.get(path, 0) + 1
            
        # 调整权重
        for path_id, usage_count in path_usage.items():
            # 根据使用频率调整权重
            current_capacity = self.route_capacities[layer][path_id]
            adjustment_factor = 1.0 + (usage_count / len(paths)) * 0.1
            
            # 限制调整幅度
            self.route_capacities[layer][path_id] = np.clip(
                current_capacity * adjustment_factor, 0.5, 2.0
            )
            
        # 记录负载均衡
        if len(set(paths)) > len(paths) * 0.7:  # 如果使用了70%以上的不同路径
            self.stats['load_balances'] += 1
            
    def optimize_routing_table(self, target_efficiency: float = 0.9) -> Dict[str, Any]:
        """优化路由表"""
        print(f"🎯 优化路由表，目标效率: {target_efficiency:.1%}")
        
        optimization_results = {
            'initial_avg_efficiency': 0.0,
            'final_avg_efficiency': 0.0,
            'improvement': 0.0,
            'optimizations_applied': 0
        }
        
        # 计算初始平均效率
        initial_efficiencies = []
        for layer in range(self.num_layers):
            for route_id in range(self.num_routes):
                eff = self.route_efficiencies.get((layer, route_id), 0.8)
                initial_efficiencies.append(eff)
                
        initial_avg_eff = np.mean(initial_efficiencies)
        optimization_results['initial_avg_efficiency'] = initial_avg_eff
        
        # 执行优化
        for layer in range(self.num_layers):
            layer_loads = self.route_loads[layer]
            
            # 负载均衡优化
            total_load = sum(layer_loads)
            if total_load > 0:
                load_variance = np.var(layer_loads)
                
                # 如果负载方差过大，进行均衡调整
                if load_variance > 0.1:
                    avg_load = total_load / self.num_routes
                    
                    for route_id in range(self.num_routes):
                        capacity = self.route_capacities[layer][route_id]
                        
                        # 调整容量以匹配负载
                        target_capacity = avg_load
                        adjustment = (target_capacity - capacity) * 0.1
                        self.route_capacities[layer][route_id] += adjustment
                        
                    optimization_results['optimizations_applied'] += 1
                    
            # 效率优化
            for route_id in range(self.num_routes):
                route_key = (layer, route_id)
                current_eff = self.route_efficiencies.get(route_key, 0.8)
                
                # 如果效率低于目标，进行提升
                if current_eff < target_efficiency:
                    improvement = (target_efficiency - current_eff) * 0.2
                    new_eff = min(target_efficiency, current_eff + improvement)
                    self.route_efficiencies[route_key] = new_eff
                    
                    optimization_results['optimizations_applied'] += 1
                    
        # 计算最终平均效率
        final_efficiencies = []
        for layer in range(self.num_layers):
            for route_id in range(self.num_routes):
                eff = self.route_efficiencies.get((layer, route_id), 0.8)
                final_efficiencies.append(eff)
                
        final_avg_eff = np.mean(final_efficiencies)
        optimization_results['final_avg_efficiency'] = final_avg_eff
        optimization_results['improvement'] = final_avg_eff - initial_avg_eff
        
        # 记录路径压缩
        if optimization_results['improvement'] > 0.05:
            self.stats['path_compressions'] += 1
            self.stats['efficiency_improvements'] += 1
            
        print(f"   初始平均效率: {initial_avg_eff:.3f}")
        print(f"   最终平均效率: {final_avg_eff:.3f}")
        print(f"   效率提升: {optimization_results['improvement']:.3f}")
        print(f"   优化次数: {optimization_results['optimizations_applied']}")
        
        return optimization_results
        
    def analyze_routing_performance(self) -> Dict[str, Any]:
        """分析路由性能"""
        analysis = {
            'total_routes': self.num_layers * self.num_routes,
            'avg_efficiency': 0.0,
            'load_distribution': {},
            'bottleneck_routes': [],
            'efficiency_metrics': {},
            'throughput_estimate': 0.0
        }
        
        # 计算平均效率
        efficiencies = []
        for layer in range(self.num_layers):
            for route_id in range(self.num_routes):
                eff = self.route_efficiencies.get((layer, route_id), 0.8)
                efficiencies.append(eff)
                
        analysis['avg_efficiency'] = np.mean(efficiencies)
        analysis['efficiency_metrics'] = {
            'min_efficiency': np.min(efficiencies),
            'max_efficiency': np.max(efficiencies),
            'std_efficiency': np.std(efficiencies)
        }
        
        # 分析负载分布
        for layer in range(self.num_layers):
            layer_loads = self.route_loads[layer]
            analysis['load_distribution'][f'layer_{layer}'] = {
                'avg_load': np.mean(layer_loads),
                'load_variance': np.var(layer_loads),
                'max_load': np.max(layer_loads),
                'min_load': np.min(layer_loads)
            }
            
            # 识别瓶颈路由
            threshold = np.mean(layer_loads) + np.std(layer_loads)
            for route_id, load in enumerate(layer_loads):
                if load > threshold:
                    analysis['bottleneck_routes'].append({
                        'layer': layer,
                        'route_id': route_id,
                        'load': load
                    })
                    
        # 估算吞吐量
        avg_efficiency = analysis['avg_efficiency']
        total_capacity = sum([sum(capacities) for capacities in self.route_capacities])
        analysis['throughput_estimate'] = avg_efficiency * total_capacity
        
        return analysis
        
    def simulate_traffic(self, num_requests: int = 1000, 
                        traffic_pattern: str = 'uniform') -> Dict[str, Any]:
        """模拟网络流量"""
        print(f"📊 模拟网络流量: {num_requests} 个请求 ({traffic_pattern})")
        
        traffic_results = {
            'total_requests': num_requests,
            'successful_routes': 0,
            'failed_routes': 0,
            'avg_processing_time': 0.0,
            'final_load_distribution': {},
            'routing_efficiency': 0.0
        }
        
        processing_times = []
        successful_routes = 0
        
        # 重置负载
        self.route_loads = [[0.0] * self.num_routes for _ in range(self.num_layers)]
        
        for i in range(num_requests):
            # 生成输入数据
            if traffic_pattern == 'burst':
                # 突发流量模式
                if i % 50 == 0:  # 每50个请求后有一次突发
                    input_size = 64 + np.random.randint(0, 64)
                else:
                    input_size = 32
            elif traffic_pattern == 'skewed':
                # 偏斜流量模式
                if np.random.random() < 0.7:
                    input_size = 16  # 大部分请求很小
                else:
                    input_size = 128  # 少数请求很大
            else:  # uniform
                input_size = 32 + np.random.randint(0, 64)
                
            # 生成随机输入数据
            input_data = np.random.randn(input_size)
            
            start_time = time.time()
            
            try:
                # 路由请求
                result = self.route_input(input_data, input_size, optimization_level='medium')
                processing_time = time.time() - start_time
                
                processing_times.append(processing_time)
                
                if result['processing_time'] > 0:
                    successful_routes += 1
                    
            except Exception as e:
                # 记录失败
                traffic_results['failed_routes'] += 1
                continue
                
            # 定期优化路由表
            if i > 0 and i % 100 == 0:
                self.optimize_routing_table()
                
        # 汇总结果
        traffic_results['successful_routes'] = successful_routes
        traffic_results['failed_routes'] = num_requests - successful_routes
        traffic_results['avg_processing_time'] = np.mean(processing_times) if processing_times else 0.0
        
        # 最终负载分布
        for layer in range(self.num_layers):
            traffic_results['final_load_distribution'][f'layer_{layer}'] = self.route_loads[layer].copy()
            
        # 路由效率
        analysis = self.analyze_routing_performance()
        traffic_results['routing_efficiency'] = analysis['avg_efficiency']
        
        print(f"   成功路由: {successful_routes}/{num_requests}")
        print(f"   平均处理时间: {traffic_results['avg_processing_time']:.4f}秒")
        print(f"   路由效率: {traffic_results['routing_efficiency']:.1%}")
        
        return traffic_results


def run_dynamic_routing_demo():
    """运行动态路由演示"""
    print("🛣️ 动态路由演示")
    print("=" * 50)
    
    # 创建动态路由器
    router = DynamicRouter(
        input_size=64,
        hidden_size=128,
        num_layers=4,
        num_routes=8
    )
    
    print("\n1️⃣ 基础路由功能演示")
    print("-" * 30)
    
    # 测试简单路由
    test_inputs = [
        np.random.randn(32),   # 小输入
        np.random.randn(64),   # 中输入
        np.random.randn(128),  # 大输入
    ]
    
    for i, input_data in enumerate(test_inputs):
        print(f"测试输入 {i+1}: 维度 {input_data.shape[0]}")
        
        result = router.route_input(input_data, optimization_level='simple')
        
        print(f"   路径数量: {len(result['routed_paths'])}")
        print(f"   处理时间: {result['processing_time']:.4f}秒")
        
        # 分析该层的路由分布
        if result['routed_paths']:
            layer_paths = result['routed_paths'][0]  # 第一层
            unique_paths = len(set(layer_paths))
            print(f"   使用路径数: {unique_paths}/{router.num_routes}")
            
    print("\n2️⃣ 负载均衡路由演示")
    print("-" * 30)
    
    # 测试负载均衡
    print("测试负载均衡路由...")
    
    for level in ['simple', 'balanced', 'optimized']:
        print(f"\n   路由策略: {level}")
        
        total_load_variance = 0
        
        for test_run in range(3):
            # 重置负载
            router.route_loads = [[0.0] * router.num_routes for _ in range(router.num_layers)]
            
            # 批量路由测试
            batch_size = 50
            test_data = np.random.randn(batch_size, 32)
            
            for i in range(batch_size):
                input_data = test_data[i]
                result = router.route_input(input_data, optimization_level=level)
                
            # 计算负载方差
            layer_loads = router.route_loads[0]  # 第一层
            load_variance = np.var(layer_loads)
            total_load_variance += load_variance
            
            print(f"     运行 {test_run+1}: 负载方差 = {load_variance:.3f}")
            
        avg_variance = total_load_variance / 3
        print(f"   平均负载方差: {avg_variance:.3f}")
        
    print("\n3️⃣ 路由表优化演示")
    print("-" * 30)
    
    # 初始化一些路由效率
    for layer in range(router.num_layers):
        for route_id in range(router.num_routes):
            router.route_efficiencies[(layer, route_id)] = np.random.uniform(0.6, 0.9)
            
    print("执行路由表优化...")
    
    # 执行多轮优化
    optimization_history = []
    
    for round_num in range(3):
        print(f"\n优化轮次 {round_num + 1}:")
        
        # 生成一些流量来建立负载
        for i in range(20):
            input_data = np.random.randn(32)
            router.route_input(input_data, optimization_level='optimized')
            
        # 优化路由表
        result = router.optimize_routing_table(target_efficiency=0.9)
        optimization_history.append(result)
        
    print("\n4️⃣ 路由性能分析")
    print("-" * 30)
    
    # 分析路由性能
    performance_analysis = router.analyze_routing_performance()
    
    print("路由性能分析:")
    print(f"   总路径数: {performance_analysis['total_routes']}")
    print(f"   平均效率: {performance_analysis['avg_efficiency']:.3f}")
    print(f"   效率范围: {performance_analysis['efficiency_metrics']['min_efficiency']:.3f} - {performance_analysis['efficiency_metrics']['max_efficiency']:.3f}")
    print(f"   估算吞吐量: {performance_analysis['throughput_estimate']:.2f}")
    
    # 显示负载分布
    print("\n各层负载分布:")
    for layer_name, load_info in performance_analysis['load_distribution'].items():
        print(f"   {layer_name}: 平均负载 {load_info['avg_load']:.2f}, 方差 {load_info['load_variance']:.3f}")
        
    # 显示瓶颈路径
    if performance_analysis['bottleneck_routes']:
        print(f"\n瓶颈路径 ({len(performance_analysis['bottleneck_routes'])} 个):")
        for bottleneck in performance_analysis['bottleneck_routes']:
            print(f"   层 {bottleneck['layer']}, 路径 {bottleneck['route_id']}: 负载 {bottleneck['load']:.2f}")
    else:
        print("\n✅ 未检测到瓶颈路径")
        
    print("\n5️⃣ 网络流量模拟")
    print("-" * 30)
    
    # 模拟不同流量模式
    traffic_patterns = ['uniform', 'burst', 'skewed']
    traffic_results = {}
    
    for pattern in traffic_patterns:
        print(f"\n模拟 {pattern} 流量模式:")
        
        # 重置路由器状态
        router.route_efficiencies = {}
        router.route_loads = [[0.0] * router.num_routes for _ in range(router.num_layers)]
        
        result = router.simulate_traffic(num_requests=200, traffic_pattern=pattern)
        traffic_results[pattern] = result
        
        print(f"   成功率: {result['successful_routes']}/{result['total_requests']}")
        print(f"   平均处理时间: {result['avg_processing_time']:.4f}秒")
        print(f"   路由效率: {result['routing_efficiency']:.1%}")
        
    print("\n6️⃣ 效率对比分析")
    print("-" * 30)
    
    # 对比不同路由策略
    print("路由策略效率对比:")
    
    strategies = ['simple', 'balanced', 'optimized']
    strategy_comparison = {}
    
    for strategy in strategies:
        # 重置状态
        router.route_efficiencies = {}
        router.route_loads = [[0.0] * router.num_routes for _ in range(router.num_layers)]
        
        # 测试策略
        total_time = 0
        success_count = 0
        
        for i in range(50):
            input_data = np.random.randn(32)
            
            start_time = time.time()
            result = router.route_input(input_data, optimization_level=strategy)
            end_time = time.time()
            
            if result['processing_time'] > 0:
                success_count += 1
                total_time += end_time - start_time
                
        avg_time = total_time / success_count if success_count > 0 else 0
        success_rate = success_count / 50
        
        strategy_comparison[strategy] = {
            'success_rate': success_rate,
            'avg_processing_time': avg_time
        }
        
        print(f"   {strategy}:")
        print(f"     成功率: {success_rate:.1%}")
        print(f"     平均处理时间: {avg_time:.4f}秒")
        
    print("\n7️⃣ 可视化结果")
    print("-" * 30)
    
    try:
        # 创建可视化图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('动态路由演示结果', fontsize=16)
        
        # 负载分布图
        layer_names = [f'Layer {i+1}' for i in range(router.num_layers)]
        layer_loads = [np.mean(router.route_loads[i]) for i in range(router.num_layers)]
        
        axes[0, 0].bar(layer_names, layer_loads, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('各层平均负载')
        axes[0, 0].set_ylabel('平均负载')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 效率分布
        efficiencies = list(router.route_efficiencies.values()) if router.route_efficiencies else [0.8] * (router.num_layers * router.num_routes)
        axes[0, 1].hist(efficiencies, bins=15, alpha=0.7, color='green')
        axes[0, 1].set_title('路由效率分布')
        axes[0, 1].set_xlabel('效率')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].axvline(x=np.mean(efficiencies), color='red', linestyle='--', label=f'平均值: {np.mean(efficiencies):.3f}')
        axes[0, 1].legend()
        
        # 流量模式对比
        patterns = list(traffic_results.keys())
        success_rates = [traffic_results[p]['successful_routes'] / traffic_results[p]['total_requests'] for p in patterns]
        
        axes[0, 2].bar(patterns, success_rates, color=['blue', 'orange', 'red'])
        axes[0, 2].set_title('不同流量模式成功率')
        axes[0, 2].set_ylabel('成功率')
        axes[0, 2].set_ylim(0, 1)
        
        # 策略性能对比
        strategies = list(strategy_comparison.keys())
        perf_times = [strategy_comparison[s]['avg_processing_time'] for s in strategies]
        
        axes[1, 0].bar(strategies, perf_times, color=['lightcoral', 'lightblue', 'lightgreen'])
        axes[1, 0].set_title('路由策略处理时间对比')
        axes[1, 0].set_ylabel('平均处理时间(秒)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 优化历史
        if optimization_history:
            rounds = list(range(1, len(optimization_history) + 1))
            initial_effs = [h['initial_avg_efficiency'] for h in optimization_history]
            final_effs = [h['final_avg_efficiency'] for h in optimization_history]
            
            axes[1, 1].plot(rounds, initial_effs, 'o-', label='初始效率', color='red')
            axes[1, 1].plot(rounds, final_effs, 's-', label='最终效率', color='blue')
            axes[1, 1].set_title('路由表优化历史')
            axes[1, 1].set_xlabel('优化轮次')
            axes[1, 1].set_ylabel('平均效率')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # 吞吐量分析
        throughputs = [performance_analysis['throughput_estimate']] * 3
        categories = ['估算', '基准', '目标']
        
        axes[1, 2].bar(categories, throughputs, color=['gold', 'silver', 'green'])
        axes[1, 2].set_title('吞吐量分析')
        axes[1, 2].set_ylabel('吞吐量')
        
        plt.tight_layout()
        
        # 保存图表
        import os
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/dynamic_routing_demo.png', dpi=300, bbox_inches='tight')
        print("📊 可视化图表已保存到: visualizations/dynamic_routing_demo.png")
        
        plt.show()
        
    except ImportError:
        print("⚠️ matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
    
    print("\n8️⃣ 保存演示结果")
    print("-" * 30)
    
    # 准备保存的结果
    results = {
        'demo_type': 'dynamic_routing',
        'timestamp': time.time(),
        'router_config': {
            'input_size': router.input_size,
            'hidden_size': router.hidden_size,
            'num_layers': router.num_layers,
            'num_routes': router.num_routes
        },
        'performance_analysis': performance_analysis,
        'traffic_results': traffic_results,
        'strategy_comparison': strategy_comparison,
        'optimization_history': optimization_history,
        'router_stats': router.stats
    }
    
    import os
    os.makedirs('data/results', exist_ok=True)
    
    with open('data/results/dynamic_routing_demo_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("💾 演示结果已保存到: data/results/dynamic_routing_demo_results.json")
    
    print("\n🎉 动态路由演示完成!")
    print("=" * 50)
    
    # 总结
    print("\n📋 演示总结:")
    print(f"✅ 创建了 {router.num_layers} 层路由网络")
    print(f"✅ 总路径数: {performance_analysis['total_routes']}")
    print(f"✅ 平均路由效率: {performance_analysis['avg_efficiency']:.1%}")
    print(f"✅ 优化成功次数: {router.stats['successful_optimizations']}")
    print(f"✅ 负载均衡次数: {router.stats['load_balances']}")
    
    if performance_analysis['avg_efficiency'] > 0.85:
        print("🎯 动态路由系统性能优秀!")
    elif performance_analysis['avg_efficiency'] > 0.75:
        print("👍 动态路由系统运行良好")
    else:
        print("⚠️ 动态路由系统需要优化")
        
    return results


def run_routing_algorithm_comparison():
    """运行路由算法对比测试"""
    print("\n🔬 路由算法对比测试")
    print("-" * 30)
    
    algorithms = ['floyd_warshall', 'dijkstra', 'dynamic_programming', 'genetic_algorithm']
    comparison_results = {}
    
    for algorithm in algorithms:
        print(f"\n测试算法: {algorithm}")
        
        # 模拟算法性能
        np.random.seed(hash(algorithm) % 1000)
        
        # 创建测试网络
        router = DynamicRouter(
            input_size=32,
            hidden_size=64,
            num_layers=3,
            num_routes=6
        )
        
        # 模拟算法特性
        if algorithm == 'floyd_warshall':
            accuracy = np.random.uniform(0.9, 0.95)
            speed = np.random.uniform(0.1, 0.3)  # 较慢
            memory_usage = np.random.uniform(0.8, 0.9)  # 高内存
        elif algorithm == 'dijkstra':
            accuracy = np.random.uniform(0.85, 0.92)
            speed = np.random.uniform(0.4, 0.6)  # 中等速度
            memory_usage = np.random.uniform(0.3, 0.5)  # 中等内存
        elif algorithm == 'dynamic_programming':
            accuracy = np.random.uniform(0.88, 0.94)
            speed = np.random.uniform(0.5, 0.7)  # 较快
            memory_usage = np.random.uniform(0.4, 0.6)  # 中等内存
        else:  # genetic_algorithm
            accuracy = np.random.uniform(0.82, 0.90)
            speed = np.random.uniform(0.6, 0.8)  # 最快
            memory_usage = np.random.uniform(0.2, 0.4)  # 低内存
            
        comparison_results[algorithm] = {
            'accuracy': accuracy,
            'speed': speed,
            'memory_usage': memory_usage,
            'overall_score': accuracy * 0.4 + speed * 0.3 + (1 - memory_usage) * 0.3
        }
        
        print(f"   准确率: {accuracy:.3f}")
        print(f"   速度: {speed:.3f}")
        print(f"   内存使用: {memory_usage:.3f}")
        print(f"   综合得分: {comparison_results[algorithm]['overall_score']:.3f}")
        
    # 找出最佳算法
    best_algorithm = max(comparison_results.keys(), 
                        key=lambda x: comparison_results[x]['overall_score'])
    
    print(f"\n🏆 最佳算法: {best_algorithm}")
    print(f"   综合得分: {comparison_results[best_algorithm]['overall_score']:.3f}")
    
    return comparison_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='动态路由演示')
    parser.add_argument('--demo', choices=['all', 'routing', 'comparison'], default='all',
                       help='演示类型: all(全部), routing(路由演示), comparison(算法对比)')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
    parser.add_argument('--save-results', action='store_true', help='保存结果')
    
    args = parser.parse_args()
    
    if args.demo in ['all', 'routing']:
        results = run_dynamic_routing_demo()
        
    if args.demo in ['all', 'comparison']:
        comparison_results = run_routing_algorithm_comparison()
        
    if args.save_results:
        print("\n💾 所有结果已保存")
        
    if args.visualize:
        print("\n📊 启动交互式可视化...")
        import matplotlib.pyplot as plt
        plt.ion()  # 开启交互模式
        plt.show()


if __name__ == "__main__":
    main()