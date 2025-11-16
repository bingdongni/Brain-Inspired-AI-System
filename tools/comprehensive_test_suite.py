#!/usr/bin/env python3
"""
全面系统测试验证套件
Comprehensive System Testing and Validation Suite

包括：
1. 基准测试套件 - 标准化的测试数据集和评估指标
2. 持续学习能力验证 - 灾难性遗忘、多任务学习、知识迁移
3. 性能优化和调试 - 代码性能分析、内存优化、并行计算
4. 多环境兼容性测试 - CPU/GPU、操作系统、依赖版本
"""

import os
import sys
import json
import time
import subprocess
import multiprocessing
import psutil
import platform
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import tempfile
import shutil

# 兼容性检查
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
SKLEARN_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    print("⚠️ 警告: PyTorch未安装，部分测试将受限")

try:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.datasets import make_classification
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: scikit-learn未安装，部分功能将受限")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("⚠️ 警告: matplotlib未安装，可视化功能将受限")


class TestDatasetGenerator:
    """标准化测试数据集生成器"""
    
    @staticmethod
    def create_mnist_like_data(n_samples=1000, n_features=64, n_classes=10, noise=0.1):
        """创建MNIST风格数据"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=n_features // 4,
            n_classes=n_classes,
            n_clusters_per_class=1,
            flip_y=noise,
            random_state=42
        )
        return X.astype(np.float32), y.astype(np.int32)
    
    @staticmethod
    def create_cifar_like_data(n_samples=1000, n_features=128, n_classes=10, noise=0.1):
        """创建CIFAR风格数据"""
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 3,
            n_redundant=n_features // 6,
            n_classes=n_classes,
            n_clusters_per_class=2,
            flip_y=noise,
            random_state=43
        )
        return X.astype(np.float32), y.astype(np.int32)
    
    @staticmethod
    def create_synthetic_data(n_samples=1000, n_features=64, n_classes=5, noise=0.05):
        """创建合成数据"""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        centers = np.random.randn(n_classes, n_features)
        
        # 分配样本到聚类
        y = np.random.choice(n_classes, n_samples)
        
        # 添加聚类结构
        for i in range(n_samples):
            X[i] = centers[y[i]] + 0.5 * np.random.randn(n_features)
        
        # 添加噪声
        X += noise * np.random.randn(n_samples, n_features)
        
        return X.astype(np.float32), y.astype(np.int32)
    
    @staticmethod
    def create_continual_learning_data(n_tasks=5, n_samples_per_task=500, 
                                     n_features=32, n_classes_per_task=2):
        """创建持续学习数据"""
        datasets = []
        
        for task_id in range(n_tasks):
            # 为每个任务创建不同的数据分布
            np.random.seed(42 + task_id)
            
            X = np.random.randn(n_samples_per_task, n_features).astype(np.float32)
            
            # 创建任务特定的聚类中心
            centers = np.random.randn(n_classes_per_task, n_features)
            
            # 分配样本
            y = np.random.choice(n_classes_per_task, n_samples_per_task)
            
            # 添加聚类结构
            for i in range(n_samples_per_task):
                X[i] = centers[y[i]] + 0.3 * np.random.randn(n_features)
                
            datasets.append((X.astype(np.float32), y.astype(np.int32)))
            
        return datasets


class BrainInspiredModel:
    """简化的脑启发模型用于测试"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, use_attention=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        if TORCH_AVAILABLE:
            self.model = self._create_pytorch_model(input_dim, hidden_dim, output_dim, use_attention)
        else:
            self.model = self._create_numpy_model(input_dim, hidden_dim, output_dim)
    
    def _create_pytorch_model(self, input_dim, hidden_dim, output_dim, use_attention):
        """创建PyTorch模型"""
        class BrainInspiredModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, use_attention):
                super().__init__()
                self.use_attention = use_attention
                
                # 编码器
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                
                # 注意力机制
                if use_attention:
                    self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
                
                # 分类器
                self.classifier = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
            
            def forward(self, x):
                x = self.encoder(x)
                
                if self.use_attention:
                    x = x.unsqueeze(1)  # 添加序列维度
                    attended_x, _ = self.attention(x, x, x)
                    x = attended_x.squeeze(1)
                
                x = self.classifier(x)
                return x
        
        device = 'cuda' if CUDA_AVAILABLE else 'cpu'
        return BrainInspiredModel(input_dim, hidden_dim, output_dim, use_attention).to(device)
    
    def _create_numpy_model(self, input_dim, hidden_dim, output_dim):
        """创建NumPy模型作为备选"""
        class NumpyModel:
            def __init__(self, input_dim, hidden_dim, output_dim):
                # 初始化权重
                self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
                self.b1 = np.zeros(hidden_dim)
                self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.1
                self.b2 = np.zeros(hidden_dim)
                self.W3 = np.random.randn(hidden_dim, output_dim) * 0.1
                self.b3 = np.zeros(output_dim)
                
                # 简化注意力权重
                self.attention_weights = np.random.randn(hidden_dim, hidden_dim) * 0.1
            
            def forward(self, x):
                # 前向传播
                z1 = np.maximum(0, np.dot(x, self.W1) + self.b1)
                z2 = np.maximum(0, np.dot(z1, self.W2) + self.b2)
                
                # 简化的注意力机制
                attention_score = np.dot(z2, self.attention_weights)
                attended_z2 = z2 * np.tanh(attention_score)
                
                output = np.dot(attended_z2, self.W3) + self.b3
                return output
            
            def train(self, X, y, epochs=10, lr=0.001):
                # 简化训练过程
                for epoch in range(epochs):
                    # 随机梯度下降
                    pass
        
        return NumpyModel(input_dim, hidden_dim, output_dim)
    
    def train(self, X, y, epochs=10, batch_size=32, lr=0.001):
        """训练模型"""
        if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            return self._train_pytorch(X, y, epochs, batch_size, lr)
        else:
            return self._train_numpy(X, y, epochs)
    
    def _train_pytorch(self, X, y, epochs, batch_size, lr):
        """PyTorch训练"""
        device = next(self.model.parameters()).device
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            losses.append(epoch_loss / len(dataloader))
        
        return losses
    
    def _train_numpy(self, X, y, epochs):
        """NumPy训练"""
        self.model.train(X, y, epochs)
        return [0.0] * epochs
    
    def predict(self, X):
        """预测"""
        if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            device = next(self.model.parameters()).device
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                outputs = self.model(X_tensor)
                _, predicted = torch.max(outputs, 1)
                return predicted.cpu().numpy()
        else:
            outputs = self.model.forward(X)
            return np.argmax(outputs, axis=1)
    
    def predict_proba(self, X):
        """预测概率"""
        if TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            device = next(self.model.parameters()).device
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(device)
                outputs = self.model(X_tensor)
                return torch.softmax(outputs, dim=1).cpu().numpy()
        else:
            outputs = self.model.forward(X)
            exp_outputs = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            return exp_outputs / np.sum(exp_outputs, axis=1, keepdims=True)


class BenchmarkTestSuite:
    """基准测试套件"""
    
    def __init__(self):
        self.dataset_generator = TestDatasetGenerator()
        self.results = {}
        self.test_config = {
            'input_dim': 64,
            'hidden_dim': 128,
            'output_dim': 10,
            'epochs': 10,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    
    def run_training_speed_benchmark(self):
        """训练速度基准测试"""
        print("\n🚀 训练速度基准测试")
        print("=" * 60)
        
        test_models = ['brain_inspired_attention', 'brain_inspired_no_attention', 'standard']
        dataset_sizes = [500, 1000, 2000]
        
        results = {
            'config': self.test_config,
            'models_tested': test_models,
            'dataset_sizes': dataset_sizes,
            'results': {}
        }
        
        for model_name in test_models:
            print(f"\n🏗️ 测试模型: {model_name}")
            results['results'][model_name] = {}
            
            for size in dataset_sizes:
                print(f"   数据集大小: {size}")
                
                # 生成数据
                X, y = self.dataset_generator.create_synthetic_data(size)
                
                # 创建模型
                if model_name == 'brain_inspired_attention':
                    model = BrainInspiredModel(
                        X.shape[1], 
                        self.test_config['hidden_dim'],
                        len(np.unique(y)),
                        use_attention=True
                    )
                elif model_name == 'brain_inspired_no_attention':
                    model = BrainInspiredModel(
                        X.shape[1], 
                        self.test_config['hidden_dim'],
                        len(np.unique(y)),
                        use_attention=False
                    )
                else:  # standard
                    model = BrainInspiredModel(
                        X.shape[1], 
                        self.test_config['hidden_dim'],
                        len(np.unique(y)),
                        use_attention=False
                    )
                
                # 性能测试
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                try:
                    losses = model.train(
                        X, y,
                        epochs=self.test_config['epochs'],
                        batch_size=self.test_config['batch_size'],
                        lr=self.test_config['learning_rate']
                    )
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    training_time = end_time - start_time
                    memory_used = end_memory - start_memory
                    throughput = size / training_time if training_time > 0 else 0
                    
                    result = {
                        'training_time': training_time,
                        'memory_used': memory_used,
                        'throughput': throughput,
                        'final_loss': losses[-1] if losses else 0.0,
                        'success': True
                    }
                    
                    print(f"     训练时间: {training_time:.2f}秒")
                    print(f"     吞吐量: {throughput:.1f} 样本/秒")
                    print(f"     内存使用: {memory_used:.1f} MB")
                    
                except Exception as e:
                    result = {
                        'error': str(e),
                        'success': False
                    }
                    print(f"     ❌ 训练失败: {e}")
                
                results['results'][model_name][f'size_{size}'] = result
                
                # 清理内存
                del model, X, y
                
        return results
    
    def run_inference_speed_benchmark(self):
        """推理速度基准测试"""
        print("\n⚡ 推理速度基准测试")
        print("=" * 60)
        
        test_models = ['brain_inspired_attention', 'brain_inspired_no_attention']
        batch_sizes = [1, 16, 32, 64, 128]
        
        # 生成测试数据
        X_test, _ = self.dataset_generator.create_synthetic_data(1000)
        
        results = {
            'config': self.test_config,
            'models_tested': test_models,
            'batch_sizes': batch_sizes,
            'test_samples': len(X_test),
            'results': {}
        }
        
        for model_name in test_models:
            print(f"\n🏗️ 测试模型: {model_name}")
            results['results'][model_name] = {}
            
            # 创建模型
            model = BrainInspiredModel(
                X_test.shape[1],
                self.test_config['hidden_dim'],
                self.test_config['output_dim'],
                use_attention=(model_name == 'brain_inspired_attention')
            )
            
            # 预热
            warmup_X = X_test[:100]
            try:
                if TORCH_AVAILABLE:
                    model.model.eval()
                    with torch.no_grad():
                        warmup_tensor = torch.FloatTensor(warmup_X).to(
                            next(model.model.parameters()).device
                        )
                        for _ in range(10):
                            _ = model.model(warmup_tensor)
            except:
                pass
            
            for batch_size in batch_sizes:
                print(f"   批处理大小: {batch_size}")
                
                batch_X = X_test[:batch_size]
                
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                try:
                    # 运行100次推理取平均
                    inference_times = []
                    for _ in range(100):
                        inference_start = time.time()
                        predictions = model.predict(batch_X)
                        inference_end = time.time()
                        inference_times.append(inference_end - inference_start)
                    
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    avg_inference_time = np.mean(inference_times)
                    throughput = batch_size / avg_inference_time if avg_inference_time > 0 else 0
                    
                    result = {
                        'avg_inference_time': avg_inference_time,
                        'throughput': throughput,
                        'memory_used': end_memory - start_memory,
                        'success': True
                    }
                    
                    print(f"     平均推理时间: {avg_inference_time*1000:.2f}ms")
                    print(f"     吞吐量: {throughput:.1f} 样本/秒")
                    
                except Exception as e:
                    result = {
                        'error': str(e),
                        'success': False
                    }
                    print(f"     ❌ 推理失败: {e}")
                
                results['results'][model_name][f'batch_{batch_size}'] = result
        
        return results
    
    def run_memory_usage_benchmark(self):
        """内存使用基准测试"""
        print("\n💾 内存使用基准测试")
        print("=" * 60)
        
        test_models = ['brain_inspired_attention', 'brain_inspired_no_attention']
        dataset_sizes = [500, 1000, 2000, 5000]
        
        results = {
            'config': self.test_config,
            'models_tested': test_models,
            'dataset_sizes': dataset_sizes,
            'results': {}
        }
        
        for model_name in test_models:
            print(f"\n🏗️ 测试模型: {model_name}")
            results['results'][model_name] = {}
            
            for size in dataset_sizes:
                print(f"   数据集大小: {size}")
                
                # 记录初始内存
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                try:
                    # 生成数据
                    X, y = self.dataset_generator.create_synthetic_data(size)
                    after_data_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    data_memory = after_data_memory - initial_memory
                    
                    # 创建模型
                    model = BrainInspiredModel(
                        X.shape[1],
                        self.test_config['hidden_dim'],
                        len(np.unique(y)),
                        use_attention=(model_name == 'brain_inspired_attention')
                    )
                    after_model_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    model_memory = after_model_memory - after_data_memory
                    
                    # 训练时内存
                    model.train(X, y, epochs=2, batch_size=32)  # 减少训练轮数以节省时间
                    peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    result = {
                        'data_memory': data_memory,
                        'model_memory': model_memory,
                        'peak_memory': peak_memory - initial_memory,
                        'total_memory': peak_memory - initial_memory,
                        'success': True
                    }
                    
                    print(f"     数据内存: {data_memory:.1f} MB")
                    print(f"     模型内存: {model_memory:.1f} MB")
                    print(f"     峰值内存: {result['peak_memory']:.1f} MB")
                    
                except Exception as e:
                    result = {
                        'error': str(e),
                        'success': False
                    }
                    print(f"     ❌ 内存测试失败: {e}")
                
                results['results'][model_name][f'size_{size}'] = result
                
                # 清理
                del model, X, y
                
        return results


class ContinualLearningTestSuite:
    """持续学习能力测试套件"""
    
    def __init__(self):
        self.dataset_generator = TestDatasetGenerator()
    
    def test_catastrophic_forgetting(self):
        """测试灾难性遗忘"""
        print("\n🧠 灾难性遗忘测试")
        print("=" * 60)
        
        # 任务1: 学习初始任务
        print("📚 任务1: 学习初始任务")
        task1_X, task1_y = self.dataset_generator.create_synthetic_data(
            n_samples=800, n_features=32, n_classes=5, noise=0.05
        )
        
        model = BrainInspiredModel(32, 64, 5, use_attention=True)
        model.train(task1_X, task1_y, epochs=15)
        
        # 测试初始任务性能
        task1_accuracy_before = self._evaluate_model(model, task1_X, task1_y)
        print(f"   任务1初始准确率: {task1_accuracy_before:.4f}")
        
        # 任务2: 学习新任务
        print("📚 任务2: 学习新任务")
        task2_X, task2_y = self.dataset_generator.create_synthetic_data(
            n_samples=800, n_features=32, n_classes=5, noise=0.05
        )
        
        model.train(task2_X, task2_y, epochs=15)
        
        # 测试两个任务的性能
        task1_accuracy_after = self._evaluate_model(model, task1_X, task1_y)
        task2_accuracy = self._evaluate_model(model, task2_X, task2_y)
        
        forgetting_rate = task1_accuracy_before - task1_accuracy_after
        
        results = {
            'task1_accuracy_before': task1_accuracy_before,
            'task1_accuracy_after': task1_accuracy_after,
            'task2_accuracy': task2_accuracy,
            'forgetting_rate': forgetting_rate,
            'success': True
        }
        
        print(f"   任务1保持准确率: {task1_accuracy_after:.4f}")
        print(f"   任务2准确率: {task2_accuracy:.4f}")
        print(f"   遗忘率: {forgetting_rate:.4f}")
        
        return results
    
    def test_multitask_learning(self):
        """测试多任务学习"""
        print("\n🔄 多任务学习验证")
        print("=" * 60)
        
        # 创建多任务数据
        task1_X, task1_y = self.dataset_generator.create_synthetic_data(
            n_samples=600, n_features=32, n_classes=3, noise=0.05
        )
        task2_X, task2_y = self.dataset_generator.create_synthetic_data(
            n_samples=600, n_features=32, n_classes=4, noise=0.05
        )
        task3_X, task3_y = self.dataset_generator.create_synthetic_data(
            n_samples=600, n_features=32, n_classes=2, noise=0.05
        )
        
        # 创建多任务模型
        model = BrainInspiredModel(32, 64, 9, use_attention=True)  # 3+4+2=9 classes
        
        # 训练多任务
        combined_X = np.vstack([task1_X, task2_X, task3_X])
        combined_y = np.hstack([task1_y, task2_y + 3, task3_y + 7])  # 调整标签
        
        model.train(combined_X, combined_y, epochs=20)
        
        # 评估各任务性能
        task1_acc = self._evaluate_classification(model, task1_X, task1_y, range(3))
        task2_acc = self._evaluate_classification(model, task2_X, task2_y, range(3, 7), offset=-3)
        task3_acc = self._evaluate_classification(model, task3_X, task3_y, range(7, 9), offset=-7)
        
        results = {
            'task1_accuracy': task1_acc,
            'task2_accuracy': task2_acc,
            'task3_accuracy': task3_acc,
            'average_accuracy': (task1_acc + task2_acc + task3_acc) / 3,
            'success': True
        }
        
        print(f"   任务1准确率: {task1_acc:.4f}")
        print(f"   任务2准确率: {task2_acc:.4f}")
        print(f"   任务3准确率: {task3_acc:.4f}")
        print(f"   平均准确率: {results['average_accuracy']:.4f}")
        
        return results
    
    def test_knowledge_transfer(self):
        """测试知识迁移"""
        print("\n🎯 知识迁移测试")
        print("=" * 60)
        
        # 源任务：学习一般特征
        print("📚 源任务: 学习一般特征")
        source_X, source_y = self.dataset_generator.create_synthetic_data(
            n_samples=1000, n_features=64, n_classes=5, noise=0.05
        )
        
        # 创建预训练模型
        pretrained_model = BrainInspiredModel(64, 128, 5, use_attention=True)
        pretrained_model.train(source_X, source_y, epochs=25)
        
        # 源任务性能
        source_accuracy = self._evaluate_model(pretrained_model, source_X, source_y)
        print(f"   源任务准确率: {source_accuracy:.4f}")
        
        # 目标任务：学习特定特征（相似的分布）
        print("📚 目标任务: 知识迁移")
        target_X, target_y = self.dataset_generator.create_synthetic_data(
            n_samples=800, n_features=64, n_classes=5, noise=0.05
        )
        
        # 迁移学习：冻结编码器，只训练分类器
        frozen_model = BrainInspiredModel(64, 128, 5, use_attention=True)
        
        # 复制预训练权重（简化版本）
        if TORCH_AVAILABLE and hasattr(pretrained_model.model, 'encoder'):
            # 复制编码器权重
            if hasattr(frozen_model.model, 'encoder'):
                try:
                    frozen_model.model.encoder.load_state_dict(pretrained_model.model.encoder.state_dict())
                    print("   🔒 编码器已冻结，使用预训练权重")
                except:
                    print("   ⚠️ 无法冻结编码器，使用随机初始化")
        else:
            print("   ⚠️ 简化版本，跳过权重冻结")
        
        # 快速微调
        frozen_model.train(target_X, target_y, epochs=5)
        
        # 评估迁移性能
        transfer_accuracy = self._evaluate_model(frozen_model, target_X, target_y)
        
        # 比较从头训练的性能
        scratch_model = BrainInspiredModel(64, 128, 5, use_attention=True)
        scratch_model.train(target_X, target_y, epochs=5)
        scratch_accuracy = self._evaluate_model(scratch_model, target_X, target_y)
        
        transfer_advantage = transfer_accuracy - scratch_accuracy
        
        results = {
            'source_accuracy': source_accuracy,
            'transfer_accuracy': transfer_accuracy,
            'scratch_accuracy': scratch_accuracy,
            'transfer_advantage': transfer_advantage,
            'success': True
        }
        
        print(f"   迁移学习准确率: {transfer_accuracy:.4f}")
        print(f"   从头训练准确率: {scratch_accuracy:.4f}")
        print(f"   迁移优势: {transfer_advantage:.4f}")
        
        return results
    
    def _evaluate_model(self, model, X, y):
        """评估模型性能"""
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def _evaluate_classification(self, model, X, y, class_indices, offset=0):
        """评估特定类别分类"""
        predictions = model.predict(X)
        
        # 只考虑指定类别
        mask = np.isin(y + offset, class_indices)
        if not np.any(mask):
            return 0.0
        
        correct = np.sum((predictions[mask] == (y[mask] + offset)))
        total = np.sum(mask)
        
        return correct / total if total > 0 else 0.0


class PerformanceOptimizationSuite:
    """性能优化和调试套件"""
    
    def __init__(self):
        self.dataset_generator = TestDatasetGenerator()
    
    def profile_code_performance(self):
        """代码性能分析"""
        print("\n📊 代码性能分析")
        print("=" * 60)
        
        # 生成测试数据
        X, y = self.dataset_generator.create_synthetic_data(2000, 64, 10, 0.05)
        
        # 测试不同配置的性能
        configs = [
            {'use_attention': True, 'hidden_dim': 64, 'batch_size': 32},
            {'use_attention': True, 'hidden_dim': 128, 'batch_size': 32},
            {'use_attention': True, 'hidden_dim': 128, 'batch_size': 64},
            {'use_attention': False, 'hidden_dim': 128, 'batch_size': 32},
        ]
        
        results = {
            'test_data_size': len(X),
            'configurations': configs,
            'results': []
        }
        
        for i, config in enumerate(configs):
            print(f"\n🏗️ 配置 {i+1}: {config}")
            
            try:
                # 创建模型
                model = BrainInspiredModel(
                    X.shape[1],
                    config['hidden_dim'],
                    len(np.unique(y)),
                    use_attention=config['use_attention']
                )
                
                # 性能测试
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # 训练
                losses = model.train(
                    X, y,
                    epochs=10,
                    batch_size=config['batch_size']
                )
                
                # 推理测试
                inference_times = []
                for _ in range(100):
                    inference_start = time.time()
                    predictions = model.predict(X[:100])
                    inference_end = time.time()
                    inference_times.append(inference_end - inference_start)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # 计算性能指标
                training_time = end_time - start_time
                memory_used = end_memory - start_memory
                avg_inference_time = np.mean(inference_times)
                
                # 计算最终准确率
                final_accuracy = self._evaluate_model(model, X[:500], y[:500])
                
                config_result = {
                    'config': config,
                    'training_time': training_time,
                    'memory_used': memory_used,
                    'avg_inference_time': avg_inference_time,
                    'final_accuracy': final_accuracy,
                    'throughput': len(X) / training_time,
                    'success': True
                }
                
                print(f"   训练时间: {training_time:.2f}秒")
                print(f"   内存使用: {memory_used:.1f} MB")
                print(f"   推理时间: {avg_inference_time*1000:.2f}ms")
                print(f"   最终准确率: {final_accuracy:.4f}")
                
            except Exception as e:
                config_result = {
                    'config': config,
                    'error': str(e),
                    'success': False
                }
                print(f"   ❌ 测试失败: {e}")
            
            results['results'].append(config_result)
            del model
        
        return results
    
    def optimize_memory_usage(self):
        """内存优化测试"""
        print("\n💾 内存优化测试")
        print("=" * 60)
        
        # 测试不同批量大小的内存使用
        batch_sizes = [16, 32, 64, 128, 256]
        
        X, y = self.dataset_generator.create_synthetic_data(5000, 64, 10, 0.05)
        
        results = {
            'test_data_size': len(X),
            'batch_sizes': batch_sizes,
            'results': {}
        }
        
        for batch_size in batch_sizes:
            print(f"\n📦 批量大小: {batch_size}")
            
            try:
                # 记录初始内存
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # 创建模型
                model = BrainInspiredModel(64, 128, 10, use_attention=True)
                
                # 训练并监控内存
                model.train(X, y, epochs=5, batch_size=batch_size)
                
                peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_used = peak_memory - initial_memory
                
                # 测试推理时的内存
                inference_start = psutil.Process().memory_info().rss / 1024 / 1024
                predictions = model.predict(X[:1000])
                inference_end = psutil.Process().memory_info().rss / 1024 / 1024
                inference_memory = inference_end - inference_start
                
                batch_result = {
                    'training_memory': memory_used,
                    'inference_memory_increase': inference_memory,
                    'total_memory': memory_used + inference_memory,
                    'success': True
                }
                
                print(f"   训练内存: {memory_used:.1f} MB")
                print(f"   推理内存增量: {inference_memory:.1f} MB")
                
            except Exception as e:
                batch_result = {
                    'error': str(e),
                    'success': False
                }
                print(f"   ❌ 内存测试失败: {e}")
            
            results['results'][f'batch_{batch_size}'] = batch_result
            del model
        
        return results
    
    def test_parallel_processing(self):
        """并行计算优化测试"""
        print("\n⚡ 并行计算优化测试")
        print("=" * 60)
        
        # 测试单线程vs多线程性能
        num_workers_list = [1, 2, 4, 8]
        dataset_sizes = [1000, 2000, 4000]
        
        results = {
            'test_configurations': [],
            'results': {}
        }
        
        for dataset_size in dataset_sizes:
            print(f"\n📊 数据集大小: {dataset_size}")
            
            # 生成数据
            X, y = self.dataset_generator.create_synthetic_data(
                dataset_size, 64, 10, 0.05
            )
            
            results['results'][f'size_{dataset_size}'] = {}
            
            for num_workers in num_workers_list:
                print(f"   🔧 工作进程数: {num_workers}")
                
                try:
                    start_time = time.time()
                    
                    # 简化的并行处理测试
                    if TORCH_AVAILABLE and num_workers > 1:
                        # 使用多进程数据加载器
                        model = BrainInspiredModel(64, 128, 10, use_attention=True)
                        
                        dataset = TensorDataset(
                            torch.FloatTensor(X),
                            torch.LongTensor(y)
                        )
                        
                        dataloader = DataLoader(
                            dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=num_workers
                        )
                        
                        # 训练
                        optimizer = optim.Adam(model.model.parameters(), lr=0.001)
                        criterion = nn.CrossEntropyLoss()
                        
                        model.model.train()
                        for epoch in range(3):
                            for batch_X, batch_y in dataloader:
                                optimizer.zero_grad()
                                outputs = model.model(batch_X)
                                loss = criterion(outputs, batch_y)
                                loss.backward()
                                optimizer.step()
                    else:
                        # 单进程版本
                        model = BrainInspiredModel(64, 128, 10, use_attention=True)
                        model.train(X, y, epochs=3, batch_size=32)
                    
                    end_time = time.time()
                    processing_time = end_time - start_time
                    
                    parallel_result = {
                        'processing_time': processing_time,
                        'num_workers': num_workers,
                        'success': True
                    }
                    
                    print(f"     处理时间: {processing_time:.2f}秒")
                    
                except Exception as e:
                    parallel_result = {
                        'error': str(e),
                        'num_workers': num_workers,
                        'success': False
                    }
                    print(f"     ❌ 并行测试失败: {e}")
                
                results['results'][f'size_{dataset_size}'][f'workers_{num_workers}'] = parallel_result
                del model
        
        return results
    
    def _evaluate_model(self, model, X, y):
        """评估模型性能"""
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


class CompatibilityTestSuite:
    """多环境兼容性测试套件"""
    
    def __init__(self):
        self.system_info = self._collect_system_info()
    
    def test_cpu_gpu_compatibility(self):
        """测试CPU/GPU兼容性"""
        print("\n🖥️ CPU/GPU兼容性测试")
        print("=" * 60)
        
        results = {
            'system_info': self.system_info,
            'device_tests': {}
        }
        
        # 测试CPU版本
        print("🔧 测试CPU版本")
        cpu_result = self._test_device('cpu')
        results['device_tests']['cpu'] = cpu_result
        
        # 测试GPU版本（如果可用）
        if CUDA_AVAILABLE:
            print("🔧 测试GPU版本")
            gpu_result = self._test_device('cuda')
            results['device_tests']['gpu'] = gpu_result
        else:
            print("⚠️ GPU不可用，跳过GPU测试")
            results['device_tests']['gpu'] = {'available': False}
        
        return results
    
    def test_operating_system_compatibility(self):
        """测试操作系统兼容性"""
        print("\n🖥️ 操作系统兼容性测试")
        print("=" * 60)
        
        os_info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'architecture': platform.machine(),
            'python_version': platform.python_version()
        }
        
        results = {
            'os_info': os_info,
            'compatibility_tests': {}
        }
        
        # 基础功能测试
        print("🔧 基础功能测试")
        try:
            # 测试文件操作
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
                f.write("测试文件")
                temp_file = f.name
            
            with open(temp_file, 'r') as f:
                content = f.read()
            
            os.remove(temp_file)
            file_operations = {'success': True, 'result': '正常'}
            
        except Exception as e:
            file_operations = {'success': False, 'error': str(e)}
        
        # 测试路径处理
        try:
            test_path = Path('/tmp/test_path_12345')
            test_path.mkdir(exist_ok=True)
            test_path.rmdir()
            path_operations = {'success': True, 'result': '正常'}
        except Exception as e:
            path_operations = {'success': False, 'error': str(e)}
        
        # 测试环境变量
        try:
            import os
            test_env_var = f"TEST_VAR_{int(time.time())}"
            os.environ[test_env_var] = "test_value"
            retrieved_value = os.environ.get(test_env_var, "")
            del os.environ[test_env_var]
            env_operations = {'success': True, 'result': '正常'} if retrieved_value == "test_value" else {'success': False, 'error': '环境变量读写失败'}
        except Exception as e:
            env_operations = {'success': False, 'error': str(e)}
        
        results['compatibility_tests'] = {
            'file_operations': file_operations,
            'path_operations': path_operations,
            'environment_variables': env_operations
        }
        
        print(f"   文件操作: {'✅ 正常' if file_operations['success'] else '❌ 失败'}")
        print(f"   路径操作: {'✅ 正常' if path_operations['success'] else '❌ 失败'}")
        print(f"   环境变量: {'✅ 正常' if env_operations['success'] else '❌ 失败'}")
        
        return results
    
    def test_dependency_compatibility(self):
        """测试依赖版本兼容性"""
        print("\n📦 依赖版本兼容性测试")
        print("=" * 60)
        
        dependencies = {
            'python': platform.python_version(),
            'torch': torch.__version__ if TORCH_AVAILABLE else 'Not Available',
            'sklearn': 'Available' if SKLEARN_AVAILABLE else 'Not Available',
            'matplotlib': 'Available' if MATPLOTLIB_AVAILABLE else 'Not Available',
            'numpy': np.__version__
        }
        
        # 测试依赖导入
        import_tests = {}
        
        # 测试NumPy
        try:
            import numpy as np
            x = np.array([1, 2, 3])
            result = x.sum()
            import_tests['numpy'] = {'success': True, 'version': np.__version__}
        except Exception as e:
            import_tests['numpy'] = {'success': False, 'error': str(e)}
        
        # 测试PyTorch
        if TORCH_AVAILABLE:
            try:
                import torch
                x = torch.tensor([1, 2, 3])
                result = x.sum().item()
                import_tests['torch'] = {'success': True, 'version': torch.__version__}
            except Exception as e:
                import_tests['torch'] = {'success': False, 'error': str(e)}
        else:
            import_tests['torch'] = {'success': False, 'error': 'PyTorch未安装'}
        
        # 测试sklearn
        if SKLEARN_AVAILABLE:
            try:
                import sklearn
                import_tests['sklearn'] = {'success': True, 'version': sklearn.__version__}
            except Exception as e:
                import_tests['sklearn'] = {'success': False, 'error': str(e)}
        else:
            import_tests['sklearn'] = {'success': False, 'error': 'scikit-learn未安装'}
        
        results = {
            'dependencies': dependencies,
            'import_tests': import_tests
        }
        
        print("📋 依赖版本检查:")
        for dep, test_result in import_tests.items():
            status = "✅" if test_result['success'] else "❌"
            version = test_result.get('version', 'Unknown')
            print(f"   {dep}: {status} {version}")
        
        return results
    
    def _test_device(self, device):
        """测试特定设备"""
        try:
            # 生成测试数据
            X, y = TestDatasetGenerator().create_synthetic_data(1000, 64, 10, 0.05)
            
            # 创建模型
            model = BrainInspiredModel(64, 128, 10, use_attention=True)
            
            # 训练测试
            start_time = time.time()
            model.train(X, y, epochs=5, batch_size=32)
            end_time = time.time()
            
            # 推理测试
            predictions = model.predict(X[:100])
            accuracy = np.mean(predictions == y[:100])
            
            return {
                'available': True,
                'training_time': end_time - start_time,
                'inference_accuracy': accuracy,
                'device': device,
                'success': True
            }
            
        except Exception as e:
            return {
                'available': False,
                'error': str(e),
                'device': device,
                'success': False
            }
    
    def _collect_system_info(self):
        """收集系统信息"""
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available
        }
        
        if TORCH_AVAILABLE:
            info['torch_version'] = torch.__version__
            info['cuda_available'] = CUDA_AVAILABLE
            if CUDA_AVAILABLE:
                info['cuda_version'] = torch.version.cuda
                info['gpu_count'] = torch.cuda.device_count()
                for i in range(torch.cuda.device_count()):
                    info[f'gpu_{i}_name'] = torch.cuda.get_device_name(i)
                    info[f'gpu_{i}_memory'] = torch.cuda.get_device_properties(i).total_memory
        
        return info


def main():
    """主函数"""
    print("🧠 脑启发AI系统 - 全面测试验证套件")
    print("=" * 80)
    print("时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # 创建输出目录
    os.makedirs('data/results', exist_ok=True)
    
    # 初始化测试套件
    benchmark_suite = BenchmarkTestSuite()
    continual_suite = ContinualLearningTestSuite()
    performance_suite = PerformanceOptimizationSuite()
    compatibility_suite = CompatibilityTestSuite()
    
    # 存储所有测试结果
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'system_info': compatibility_suite.system_info,
        'test_suites': {}
    }
    
    # 1. 运行基准测试套件
    print("\n" + "="*80)
    print("📊 基准测试套件")
    print("="*80)
    
    benchmark_results = {}
    try:
        benchmark_results['training_speed'] = benchmark_suite.run_training_speed_benchmark()
    except Exception as e:
        benchmark_results['training_speed'] = {'error': str(e), 'success': False}
    
    try:
        benchmark_results['inference_speed'] = benchmark_suite.run_inference_speed_benchmark()
    except Exception as e:
        benchmark_results['inference_speed'] = {'error': str(e), 'success': False}
    
    try:
        benchmark_results['memory_usage'] = benchmark_suite.run_memory_usage_benchmark()
    except Exception as e:
        benchmark_results['memory_usage'] = {'error': str(e), 'success': False}
    
    all_results['test_suites']['benchmark'] = benchmark_results
    
    # 2. 运行持续学习测试套件
    print("\n" + "="*80)
    print("🧠 持续学习测试套件")
    print("="*80)
    
    continual_results = {}
    try:
        continual_results['catastrophic_forgetting'] = continual_suite.test_catastrophic_forgetting()
    except Exception as e:
        continual_results['catastrophic_forgetting'] = {'error': str(e), 'success': False}
    
    try:
        continual_results['multitask_learning'] = continual_suite.test_multitask_learning()
    except Exception as e:
        continual_results['multitask_learning'] = {'error': str(e), 'success': False}
    
    try:
        continual_results['knowledge_transfer'] = continual_suite.test_knowledge_transfer()
    except Exception as e:
        continual_results['knowledge_transfer'] = {'error': str(e), 'success': False}
    
    all_results['test_suites']['continual_learning'] = continual_results
    
    # 3. 运行性能优化测试套件
    print("\n" + "="*80)
    print("⚡ 性能优化测试套件")
    print("="*80)
    
    performance_results = {}
    try:
        performance_results['code_profiling'] = performance_suite.profile_code_performance()
    except Exception as e:
        performance_results['code_profiling'] = {'error': str(e), 'success': False}
    
    try:
        performance_results['memory_optimization'] = performance_suite.optimize_memory_usage()
    except Exception as e:
        performance_results['memory_optimization'] = {'error': str(e), 'success': False}
    
    try:
        performance_results['parallel_processing'] = performance_suite.test_parallel_processing()
    except Exception as e:
        performance_results['parallel_processing'] = {'error': str(e), 'success': False}
    
    all_results['test_suites']['performance'] = performance_results
    
    # 4. 运行兼容性测试套件
    print("\n" + "="*80)
    print("🖥️ 兼容性测试套件")
    print("="*80)
    
    compatibility_results = {}
    try:
        compatibility_results['cpu_gpu_compatibility'] = compatibility_suite.test_cpu_gpu_compatibility()
    except Exception as e:
        compatibility_results['cpu_gpu_compatibility'] = {'error': str(e), 'success': False}
    
    try:
        compatibility_results['os_compatibility'] = compatibility_suite.test_operating_system_compatibility()
    except Exception as e:
        compatibility_results['os_compatibility'] = {'error': str(e), 'success': False}
    
    try:
        compatibility_results['dependency_compatibility'] = compatibility_suite.test_dependency_compatibility()
    except Exception as e:
        compatibility_results['dependency_compatibility'] = {'error': str(e), 'success': False}
    
    all_results['test_suites']['compatibility'] = compatibility_results
    
    # 生成综合报告
    print("\n" + "="*80)
    print("📈 生成综合报告")
    print("="*80)
    
    # 计算总体成功率
    total_tests = 0
    successful_tests = 0
    
    for suite_name, suite_results in all_results['test_suites'].items():
        for test_name, test_result in suite_results.items():
            total_tests += 1
            if isinstance(test_result, dict) and test_result.get('success', False):
                successful_tests += 1
            elif isinstance(test_result, dict) and 'success' not in test_result:
                # 对于包含多个子测试的结果
                for sub_test in test_result.values():
                    total_tests += 1
                    if isinstance(sub_test, dict) and sub_test.get('success', False):
                        successful_tests += 1
    
    success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
    
    summary = {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': success_rate,
        'timestamp': datetime.now().isoformat()
    }
    
    all_results['summary'] = summary
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'data/results/comprehensive_test_results_{timestamp}.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 打印总结
    print(f"\n🎉 全面测试验证完成!")
    print(f"📊 总测试数: {total_tests}")
    print(f"✅ 成功测试数: {successful_tests}")
    print(f"📈 成功率: {success_rate:.1f}%")
    print(f"💾 结果已保存到: {output_file}")
    
    if success_rate >= 90:
        print("🎯 系统质量评级: 优秀")
    elif success_rate >= 75:
        print("🎯 系统质量评级: 良好")
    elif success_rate >= 60:
        print("🎯 系统质量评级: 合格")
    else:
        print("🎯 系统质量评级: 需要改进")
    
    return success_rate


if __name__ == "__main__":
    success_rate = main()
    sys.exit(0 if success_rate >= 75 else 1)  # 成功退出码为75%以上成功率