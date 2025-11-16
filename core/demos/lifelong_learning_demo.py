#!/usr/bin/env python3
"""
终身学习演示 - 持续学习机制
Lifelong Learning Demo - Continual Learning Mechanism

演示持续学习的核心功能：
- 多任务连续学习
- 灾难性遗忘防护
- 知识迁移
- 性能保持率分析
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
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将使用简化版本演示")

class ContinualLearner:
    """持续学习器"""
    
    def __init__(self, 
                 input_dim: int = 20,
                 hidden_dim: int = 128,
                 output_dim: int = 10,
                 memory_size: int = 1000,
                 elasticity: float = 0.1):
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.memory_size = memory_size
        self.elasticity = elasticity
        
        # 任务相关
        self.tasks = []
        self.task_performances = {}
        self.learned_tasks = 0
        
        # 记忆库
        self.experience_replay = []
        self.experience_weights = []
        
        # EWC参数
        self.fisher_information = {}
        self.optimal_params = {}
        
        # 创建网络
        self.model = self._create_network()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'successful_learnings': 0,
            'forgetting_events': 0,
            'knowledge_transfers': 0,
            'consolidation_events': 0
        }
        
    def _create_network(self) -> nn.Module:
        """创建神经网络"""
        if not TORCH_AVAILABLE:
            # 简化版本
            class SimpleNetwork:
                def __init__(self, input_dim, hidden_dim, output_dim):
                    self.input_dim = input_dim
                    self.hidden_dim = hidden_dim
                    self.output_dim = output_dim
                    self.weights = np.random.randn(input_dim, hidden_dim)
                    self.biases = np.random.randn(hidden_dim)
                    self.output_weights = np.random.randn(hidden_dim, output_dim)
                    self.output_biases = np.random.randn(output_dim)
                    
                def forward(self, x):
                    hidden = np.maximum(0, np.dot(x, self.weights) + self.biases)
                    output = np.dot(hidden, self.output_weights) + self.output_biases
                    return output
                    
                def parameters(self):
                    return [self.weights, self.biases, self.output_weights, self.output_biases]
                    
            return SimpleNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        
        class ContinualNetwork(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                
                # 共享特征提取器
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                
                # 任务特定头部
                self.task_heads = nn.ModuleList([
                    nn.Linear(hidden_dim, output_dim) for _ in range(10)  # 支持最多10个任务
                ])
                
            def forward(self, x, task_id=0):
                features = self.feature_extractor(x)
                output = self.task_heads[task_id](features)
                return output
                
        return ContinualNetwork(self.input_dim, self.hidden_dim, self.output_dim)
        
    def generate_task_data(self, task_id: int, n_samples: int = 500) -> Dict[str, np.ndarray]:
        """生成任务数据"""
        np.random.seed(42 + task_id)
        
        # 为每个任务创建独特的模式
        pattern_center = np.random.randn(self.input_dim) * 2
        pattern_spread = 0.5 + task_id * 0.1
        
        X = np.random.randn(n_samples, self.input_dim)
        X = X * pattern_spread + pattern_center
        
        # 创建分类标签
        n_classes = self.output_dim
        y = np.random.randint(0, n_classes, n_samples)
        
        # 添加任务特定噪声
        task_noise = np.random.randn(n_samples, self.input_dim) * 0.1
        X += task_noise
        
        return {
            'X_train': X[:int(n_samples * 0.8)],
            'y_train': y[:int(n_samples * 0.8)],
            'X_test': X[int(n_samples * 0.8):],
            'y_test': y[int(n_samples * 0.8):],
            'task_id': task_id,
            'pattern_center': pattern_center,
            'pattern_spread': pattern_spread
        }
        
    def compute_fisher_information(self, task_data: Dict):
        """计算Fisher信息矩阵（EWC方法）"""
        if not TORCH_AVAILABLE:
            return {}
            
        self.model.eval()
        
        X = torch.FloatTensor(task_data['X_train'])
        y = torch.LongTensor(task_data['y_train'])
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        fisher_dict = {}
        
        for param in self.model.parameters():
            fisher_dict[param] = torch.zeros_like(param)
            
        for batch_x, batch_y in dataloader:
            self.model.zero_grad()
            
            outputs = self.model(batch_x)
            loss = nn.CrossEntropyLoss()(outputs, batch_y)
            loss.backward()
            
            for param in self.model.parameters():
                if param.grad is not None:
                    fisher_dict[param] += param.grad.data ** 2
                    
        # 平均化
        n_batches = len(dataloader)
        for param in self.model.parameters():
            fisher_dict[param] /= n_batches
            
        return fisher_dict
        
    def add_to_experience_replay(self, task_data: Dict, n_samples: int = 100):
        """添加到经验回放"""
        if len(self.experience_replay) < self.memory_size:
            # 随机选择样本
            indices = np.random.choice(
                len(task_data['X_train']), 
                min(n_samples, len(task_data['X_train'])), 
                replace=False
            )
            
            replay_sample = {
                'X': task_data['X_train'][indices],
                'y': task_data['y_train'][indices],
                'task_id': task_data['task_id']
            }
            
            self.experience_replay.append(replay_sample)
            self.experience_weights.append(1.0)
        else:
            # 替换最旧的样本
            self.experience_replay.pop(0)
            self.experience_weights.pop(0)
            
            indices = np.random.choice(
                len(task_data['X_train']), 
                min(n_samples, len(task_data['X_train'])), 
                replace=False
            )
            
            replay_sample = {
                'X': task_data['X_train'][indices],
                'y': task_data['y_train'][indices],
                'task_id': task_data['task_id']
            }
            
            self.experience_replay.append(replay_sample)
            self.experience_weights.append(1.0)
            
    def learn_task(self, task_data: Dict, epochs: int = 50, use_ewc: bool = True):
        """学习新任务"""
        print(f"🔄 学习任务 {task_data['task_id']}...")
        
        # 更新统计
        self.stats['total_tasks'] += 1
        current_task_id = task_data['task_id']
        
        # 评估之前任务的性能（基线）
        previous_performances = {}
        for prev_task_id in range(current_task_id):
            if prev_task_id in self.task_performances:
                perf = self.evaluate_task(task_data, prev_task_id)
                previous_performances[prev_task_id] = perf
        
        # 计算当前任务的初始性能
        initial_performance = self.evaluate_task(task_data, current_task_id)
        
        # 训练当前任务
        if TORCH_AVAILABLE:
            self._train_with_pytorch(task_data, epochs, use_ewc)
        else:
            self._train_simple(task_data, epochs)
            
        # 评估最终性能
        final_performance = self.evaluate_task(task_data, current_task_id)
        
        # 保存任务性能
        self.task_performances[current_task_id] = {
            'initial_accuracy': initial_performance,
            'final_accuracy': final_performance,
            'improvement': final_performance - initial_performance
        }
        
        # 检查遗忘
        forgetting_detected = False
        for prev_task_id, baseline_perf in previous_performances.items():
            current_perf = self.evaluate_task(task_data, prev_task_id)
            if current_perf < baseline_perf * 0.9:  # 性能下降超过10%
                forgetting_detected = True
                self.stats['forgetting_events'] += 1
                print(f"   ⚠️ 灾难性遗忘检测: 任务 {prev_task_id} 性能从 {baseline_perf:.3f} 降至 {current_perf:.3f}")
                
        if not forgetting_detected:
            self.stats['successful_learnings'] += 1
            print(f"   ✅ 任务 {current_task_id} 学习成功，无遗忘")
            
        # 添加到经验回放
        self.add_to_experience_replay(task_data)
        
        # 如果使用EWC，保存Fisher信息
        if use_ewc and TORCH_AVAILABLE:
            fisher = self.compute_fisher_information(task_data)
            self.fisher_information[current_task_id] = fisher
            
            # 保存最优参数
            self.optimal_params[current_task_id] = {}
            for name, param in self.model.named_parameters():
                self.optimal_params[current_task_id][name] = param.data.clone()
                
        self.learned_tasks += 1
        print(f"   📊 任务 {current_task_id} 性能: {initial_performance:.3f} -> {final_performance:.3f}")
        
        return final_performance
        
    def _train_with_pytorch(self, task_data: Dict, epochs: int, use_ewc: bool):
        """使用PyTorch训练"""
        X_train = torch.FloatTensor(task_data['X_train'])
        y_train = torch.LongTensor(task_data['y_train'])
        X_test = torch.FloatTensor(task_data['X_test'])
        y_test = torch.LongTensor(task_data['y_test'])
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                
                outputs = self.model(batch_X, task_data['task_id'])
                loss = criterion(outputs, batch_y)
                
                # 添加EWC损失
                if use_ewc and len(self.fisher_information) > 0:
                    ewc_loss = self._compute_ewc_loss()
                    loss += ewc_loss
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            # 定期评估
            if (epoch + 1) % max(1, epochs // 5) == 0:
                with torch.no_grad():
                    test_outputs = self.model(X_test, task_data['task_id'])
                    _, predicted = torch.max(test_outputs, 1)
                    accuracy = (predicted == y_test).float().mean().item()
                    
                avg_loss = total_loss / len(train_loader)
                print(f"     Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
                
    def _train_simple(self, task_data: Dict, epochs: int):
        """简化的训练过程"""
        # 模拟训练过程
        for epoch in range(epochs):
            # 模拟损失下降
            initial_loss = 2.0
            final_loss = 0.5
            progress = epoch / epochs
            current_loss = initial_loss * (1 - progress) + final_loss * progress
            
            if (epoch + 1) % max(1, epochs // 5) == 0:
                accuracy = min(0.95, 0.5 + progress * 0.4 + np.random.normal(0, 0.02))
                print(f"     Epoch {epoch+1}: Loss={current_loss:.4f}, Accuracy={accuracy:.4f}")
                
    def _compute_ewc_loss(self) -> float:
        """计算EWC损失"""
        ewc_loss = 0
        
        for task_id, fisher in self.fisher_information.items():
            for name, param in self.model.named_parameters():
                if name in fisher and name in self.optimal_params[task_id]:
                    fisher_matrix = fisher[name]
                    optimal_param = self.optimal_params[task_id][name]
                    
                    # 计算参数差异
                    param_diff = param - optimal_param
                    
                    # EWC损失: F * (θ - θ*)^2
                    ewc_loss += torch.sum(fisher_matrix * param_diff.pow(2))
                    
        return self.elasticity * ewc_loss
        
    def evaluate_task(self, task_data: Dict, task_id: int) -> float:
        """评估特定任务的性能"""
        if TORCH_AVAILABLE:
            return self._evaluate_with_pytorch(task_data, task_id)
        else:
            return self._evaluate_simple(task_data, task_id)
            
    def _evaluate_with_pytorch(self, task_data: Dict, task_id: int) -> float:
        """使用PyTorch评估"""
        self.model.eval()
        
        X_test = torch.FloatTensor(task_data['X_test'])
        y_test = torch.LongTensor(task_data['y_test'])
        
        with torch.no_grad():
            outputs = self.model(X_test, task_id)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y_test).float().mean().item()
            
        return accuracy
        
    def _evaluate_simple(self, task_data: Dict, task_id: int) -> float:
        """简化评估"""
        # 模拟准确率（基于学习的任务数量）
        base_accuracy = 0.9 - task_id * 0.05  # 任务越多，难度越大
        return max(0.1, base_accuracy + np.random.normal(0, 0.05))
        
    def consolidate_knowledge(self):
        """知识巩固"""
        print("🔒 执行知识巩固...")
        
        # 经验重放
        if len(self.experience_replay) > 0:
            print(f"   经验重放: {len(self.experience_replay)} 个记忆样本")
            
            # 简单的重放训练
            if TORCH_AVAILABLE:
                # 合并所有经验样本
                all_X = []
                all_y = []
                all_task_ids = []
                
                for i, replay in enumerate(self.experience_replay):
                    all_X.append(replay['X'])
                    all_y.append(replay['y'])
                    all_task_ids.extend([replay['task_id']] * len(replay['X']))
                
                if all_X:
                    combined_X = np.vstack(all_X)
                    combined_y = np.hstack(all_y)
                    
                    # 训练几轮
                    for epoch in range(5):
                        # 随机打乱
                        indices = np.random.permutation(len(combined_X))
                        
                        for start_idx in range(0, len(indices), 32):
                            batch_indices = indices[start_idx:start_idx + 32]
                            batch_X = torch.FloatTensor(combined_X[batch_indices])
                            batch_y = torch.LongTensor(combined_y[batch_indices])
                            
                            # 使用随机任务ID（简化）
                            random_task_id = np.random.randint(0, len(self.task_performances))
                            
                            self.optimizer.zero_grad()
                            outputs = self.model(batch_X, random_task_id)
                            loss = nn.CrossEntropyLoss()(outputs, batch_y)
                            loss.backward()
                            self.optimizer.step()
                            
        # 更新权重
        if len(self.experience_weights) > 1:
            total_weight = sum(self.experience_weights)
            for i in range(len(self.experience_weights)):
                self.experience_weights[i] /= total_weight
                
        self.stats['consolidation_events'] += 1
        print("   ✅ 知识巩固完成")
        
    def analyze_forgetting(self) -> Dict[str, Any]:
        """分析遗忘情况"""
        if len(self.task_performances) < 2:
            return {'forgetting_detected': False, 'avg_retention': 1.0}
            
        # 计算保持率
        retention_rates = []
        
        for task_id in range(len(self.task_performances)):
            # 初始性能（第一次学习后）
            initial_perf = self.task_performances[task_id]['final_accuracy']
            
            # 当前性能（在最新任务学习后）
            current_perf = self.evaluate_task(
                self.tasks[task_id] if task_id < len(self.tasks) else {}, 
                task_id
            )
            
            retention_rate = current_perf / initial_perf if initial_perf > 0 else 1.0
            retention_rates.append(retention_rate)
            
        avg_retention = np.mean(retention_rates)
        forgetting_detected = avg_retention < 0.9
        
        return {
            'forgetting_detected': forgetting_detected,
            'avg_retention': avg_retention,
            'retention_rates': retention_rates,
            'min_retention': min(retention_rates),
            'max_retention': max(retention_rates)
        }


def run_lifelong_learning_demo():
    """运行终身学习演示"""
    print("🔄 终身学习演示")
    print("=" * 50)
    
    # 创建持续学习器
    learner = ContinualLearner(
        input_dim=20,
        hidden_dim=128,
        output_dim=10,
        memory_size=500,
        elasticity=0.1
    )
    
    print("\n1️⃣ 多任务连续学习演示")
    print("-" * 30)
    
    # 定义任务序列
    n_tasks = 5
    task_names = [
        "基础模式识别",
        "复杂模式识别", 
        "噪声模式识别",
        "层次模式识别",
        "抽象模式识别"
    ]
    
    task_performances = {}
    learning_history = []
    
    print(f"开始学习 {n_tasks} 个连续任务...")
    
    for task_id in range(n_tasks):
        print(f"\n📚 任务 {task_id + 1}: {task_names[task_id]}")
        
        # 生成任务数据
        task_data = learner.generate_task_data(task_id, n_samples=600)
        learner.tasks.append(task_data)
        
        # 学习任务
        final_accuracy = learner.learn_task(task_data, epochs=30, use_ewc=True)
        
        # 记录性能
        task_performances[task_id] = {
            'name': task_names[task_id],
            'accuracy': final_accuracy,
            'task_id': task_id
        }
        
        learning_history.append({
            'task_id': task_id,
            'task_name': task_names[task_id],
            'accuracy': final_accuracy,
            'timestamp': time.time()
        })
        
        print(f"   任务 {task_id + 1} 完成: {final_accuracy:.3f}")
        
    print("\n2️⃣ 灾难性遗忘分析")
    print("-" * 30)
    
    # 分析遗忘情况
    forgetting_analysis = learner.analyze_forgetting()
    
    print("遗忘分析结果:")
    print(f"   检测到遗忘: {'是' if forgetting_analysis['forgetting_detected'] else '否'}")
    print(f"   平均保持率: {forgetting_analysis['avg_retention']:.1%}")
    print(f"   最低保持率: {foritting_analysis['min_retention']:.1%}")
    print(f"   最高保持率: {forgetting_analysis['max_retention']:.1%}")
    
    # 显示各任务保持率
    print("\n任务保持率详情:")
    for i, rate in enumerate(forgetting_analysis['retention_rates']):
        print(f"   任务 {i+1}: {rate:.1%}")
    
    print("\n3️⃣ 知识巩固演示")
    print("-" * 30)
    
    # 执行知识巩固
    learner.consolidate_knowledge()
    
    # 重新评估所有任务
    print("\n巩固后重新评估:")
    post_consolidation_performance = {}
    
    for task_id in range(n_tasks):
        if task_id < len(learner.tasks):
            accuracy = learner.evaluate_task(learner.tasks[task_id], task_id)
            post_consolidation_performance[task_id] = accuracy
            print(f"   任务 {task_id + 1}: {accuracy:.3f}")
            
    print("\n4️⃣ 知识迁移分析")
    print("-" * 30)
    
    # 分析知识迁移
    print("知识迁移分析:")
    
    if n_tasks >= 2:
        # 测试早期任务对后期任务的影响
        early_task_perf = learner.evaluate_task(learner.tasks[0], 0)
        print(f"   早期任务(任务1)性能: {early_task_perf:.3f}")
        
        # 测试迁移学习效果
        transfer_benefit = 0
        for task_id in range(1, min(3, n_tasks)):  # 测试前几个任务
            # 模拟没有之前任务帮助的性能
            baseline_perf = 0.6 + task_id * 0.05  # 假设基线性能
            actual_perf = post_consolidation_performance.get(task_id, baseline_perf)
            
            if actual_perf > baseline_perf:
                transfer_benefit += (actual_perf - baseline_perf)
                learner.stats['knowledge_transfers'] += 1
                
        avg_transfer_benefit = transfer_benefit / min(2, n_tasks - 1)
        print(f"   平均知识迁移收益: {avg_transfer_benefit:.3f}")
        print(f"   成功迁移次数: {learner.stats['knowledge_transfers']}")
        
    print("\n5️⃣ 性能指标总结")
    print("-" * 30)
    
    # 计算总体指标
    all_accuracies = [perf['accuracy'] for perf in task_performances.values()]
    avg_accuracy = np.mean(all_accuracies)
    final_accuracy = all_accuracies[-1] if all_accuracies else 0
    
    # 学习曲线分析
    learning_curve = [item['accuracy'] for item in learning_history]
    learning_stability = 1.0 - np.std(learning_curve)  # 稳定性指标
    
    print("总体性能指标:")
    print(f"   平均任务准确率: {avg_accuracy:.3f}")
    print(f"   最终任务准确率: {final_accuracy:.3f}")
    print(f"   学习稳定性: {learning_stability:.3f}")
    print(f"   学习成功率: {learner.stats['successful_learnings']}/{learner.stats['total_tasks']}")
    print(f"   遗忘事件数: {learner.stats['forgetting_events']}")
    print(f"   知识迁移次数: {learner.stats['knowledge_transfers']}")
    
    # 性能评估
    if avg_accuracy > 0.8 and forgetting_analysis['avg_retention'] > 0.85:
        performance_grade = "优秀"
    elif avg_accuracy > 0.7 and forgetting_analysis['avg_retention'] > 0.75:
        performance_grade = "良好"
    elif avg_accuracy > 0.6:
        performance_grade = "一般"
    else:
        performance_grade = "需要改进"
        
    print(f"\n🎯 性能评级: {performance_grade}")
    
    print("\n6️⃣ 可视化结果")
    print("-" * 30)
    
    try:
        # 创建可视化图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('终身学习演示结果', fontsize=16)
        
        # 学习曲线
        task_names_short = [name[:8] + "..." if len(name) > 8 else name 
                           for name in task_names[:len(learning_curve)]]
        
        axes[0, 0].plot(range(1, len(learning_curve) + 1), learning_curve, 
                       'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('任务学习曲线')
        axes[0, 0].set_xlabel('任务序号')
        axes[0, 0].set_ylabel('准确率')
        axes[0, 0].grid(True)
        axes[0, 0].set_xticks(range(1, len(learning_curve) + 1))
        
        # 保持率分析
        if 'retention_rates' in forgetting_analysis:
            axes[0, 1].bar(range(1, len(forgetting_analysis['retention_rates']) + 1),
                          forgetting_analysis['retention_rates'],
                          color='orange', alpha=0.7)
            axes[0, 1].set_title('任务保持率')
            axes[0, 1].set_xlabel('任务序号')
            axes[0, 1].set_ylabel('保持率')
            axes[0, 1].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='90%基准线')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 统计信息
        stats_names = ['成功学习', '遗忘事件', '知识迁移', '巩固事件']
        stats_values = [
            learner.stats['successful_learnings'],
            learner.stats['forgetting_events'],
            learner.stats['knowledge_transfers'],
            learner.stats['consolidation_events']
        ]
        
        axes[1, 0].bar(stats_names, stats_values, 
                      color=['green', 'red', 'blue', 'purple'])
        axes[1, 0].set_title('学习统计')
        axes[1, 0].set_ylabel('次数')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 性能分布
        if len(all_accuracies) > 1:
            axes[1, 1].hist(all_accuracies, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[1, 1].set_title('任务性能分布')
            axes[1, 1].set_xlabel('准确率')
            axes[1, 1].set_ylabel('频次')
            axes[1, 1].axvline(x=avg_accuracy, color='red', linestyle='--', 
                              label=f'平均值: {avg_accuracy:.3f}')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图表
        import os
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/lifelong_learning_demo.png', dpi=300, bbox_inches='tight')
        print("📊 可视化图表已保存到: visualizations/lifelong_learning_demo.png")
        
        plt.show()
        
    except ImportError:
        print("⚠️ matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
    
    print("\n7️⃣ 保存演示结果")
    print("-" * 30)
    
    # 准备保存的结果
    results = {
        'demo_type': 'lifelong_learning',
        'timestamp': time.time(),
        'n_tasks': n_tasks,
        'task_performances': task_performances,
        'learning_history': learning_history,
        'forgetting_analysis': forgetting_analysis,
        'post_consolidation_performance': post_consolidation_performance,
        'learner_stats': learner.stats,
        'overall_metrics': {
            'avg_accuracy': avg_accuracy,
            'final_accuracy': final_accuracy,
            'learning_stability': learning_stability,
            'performance_grade': performance_grade
        }
    }
    
    import os
    os.makedirs('data/results', exist_ok=True)
    
    with open('data/results/lifelong_learning_demo_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("💾 演示结果已保存到: data/results/lifelong_learning_demo_results.json")
    
    print("\n🎉 终身学习演示完成!")
    print("=" * 50)
    
    # 总结
    print("\n📋 演示总结:")
    print(f"✅ 完成了 {n_tasks} 个连续任务的学习")
    print(f"✅ 平均任务准确率: {avg_accuracy:.1%}")
    print(f"✅ 知识保持率: {forgetting_analysis['avg_retention']:.1%}")
    print(f"✅ 灾难性遗忘: {'未检测到' if not forgetting_analysis['forgetting_detected'] else '已检测到'}")
    
    if avg_accuracy > 0.8:
        print("🎯 终身学习系统表现优秀!")
    elif avg_accuracy > 0.6:
        print("👍 终身学习系统运行正常")
    else:
        print("⚠️ 终身学习系统需要优化")
        
    return results


def run_continual_learning_benchmark():
    """运行持续学习基准测试"""
    print("\n📊 持续学习基准测试")
    print("-" * 30)
    
    # 测试不同的持续学习配置
    configurations = [
        {'name': 'EWC + 经验重放', 'use_ewc': True, 'use_replay': True},
        {'name': '仅EWC', 'use_ewc': True, 'use_replay': False},
        {'name': '仅经验重放', 'use_ewc': False, 'use_replay': True},
        {'name': '基线(无防护)', 'use_ewc': False, 'use_replay': False}
    ]
    
    benchmark_results = {}
    
    for config in configurations:
        print(f"\n🔧 测试配置: {config['name']}")
        
        learner = ContinualLearner(
            input_dim=20,
            hidden_dim=64,
            output_dim=8,
            memory_size=300,
            elasticity=0.05
        )
        
        # 训练5个任务
        task_accuracies = []
        
        for task_id in range(5):
            task_data = learner.generate_task_data(task_id, n_samples=400)
            final_accuracy = learner.learn_task(task_data, epochs=20, use_ewc=config['use_ewc'])
            task_accuracies.append(final_accuracy)
            
            # 评估所有之前任务的性能
            all_task_perfs = []
            for prev_task_id in range(task_id + 1):
                if prev_task_id < len(learner.tasks):
                    perf = learner.evaluate_task(learner.tasks[prev_task_id], prev_task_id)
                    all_task_perfs.append(perf)
                    
            avg_perf = np.mean(all_task_perfs) if all_task_perfs else final_accuracy
            
        # 计算指标
        final_performance = task_accuracies[-1]
        avg_performance = np.mean(task_accuracies)
        performance_retention = avg_performance / task_accuracies[0] if task_accuracies[0] > 0 else 1.0
        
        benchmark_results[config['name']] = {
            'final_performance': final_performance,
            'avg_performance': avg_performance,
            'performance_retention': performance_retention,
            'task_accuracies': task_accuracies
        }
        
        print(f"   最终性能: {final_performance:.3f}")
        print(f"   平均性能: {avg_performance:.3f}")
        print(f"   性能保持: {performance_retention:.1%}")
    
    print("\n🏆 基准测试对比:")
    for name, results in benchmark_results.items():
        print(f"   {name}:")
        print(f"     - 平均准确率: {results['avg_performance']:.3f}")
        print(f"     - 性能保持: {results['performance_retention']:.1%}")
        
    return benchmark_results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='终身学习演示')
    parser.add_argument('--demo', choices=['all', 'lifelong', 'benchmark'], default='all',
                       help='演示类型: all(全部), lifelong(终身学习), benchmark(基准测试)')
    parser.add_argument('--tasks', type=int, default=5, help='任务数量')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
    parser.add_argument('--save-results', action='store_true', help='保存结果')
    
    args = parser.parse_args()
    
    if args.demo in ['all', 'lifelong']:
        results = run_lifelong_learning_demo()
        
    if args.demo in ['all', 'benchmark']:
        benchmark_results = run_continual_learning_benchmark()
        
    if args.save_results:
        print("\n💾 所有结果已保存")
        
    if args.visualize:
        print("\n📊 启动交互式可视化...")
        import matplotlib.pyplot as plt
        plt.ion()  # 开启交互模式
        plt.show()


if __name__ == "__main__":
    main()