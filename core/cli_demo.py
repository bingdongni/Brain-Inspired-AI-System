#!/usr/bin/env python3
"""
脑启发AI演示系统 - 交互式命令行界面
Brain-Inspired AI Demo System - Interactive Command Line Interface

提供完整的系统演示功能：
- 系统初始化和配置
- 数据输入和预处理
- 训练控制和监控
- 结果展示和分析
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，部分功能将不可用")

class BrainInspiredAISystem:
    """脑启发AI演示系统主类"""
    
    def __init__(self):
        self.config = None
        self.models = {}
        self.data_loaders = {}
        self.results = {}
        self.initialized = False
        
    def initialize_system(self, config_path: str = "config.yaml"):
        """初始化系统"""
        print("🧠 正在初始化脑启发AI系统...")
        
        # 检查依赖
        self._check_dependencies()
        
        # 加载配置
        self.config = self._load_config(config_path)
        
        # 创建必要目录
        self._create_directories()
        
        self.initialized = True
        print("✅ 系统初始化完成!")
        return True
        
    def _check_dependencies(self):
        """检查系统依赖"""
        print("📋 检查系统依赖...")
        
        dependencies = {
            'torch': TORCH_AVAILABLE,
            'numpy': True,
            'pathlib': True
        }
        
        missing = [name for name, available in dependencies.items() if not available]
        
        if missing:
            print(f"❌ 缺少依赖: {', '.join(missing)}")
            print("请安装缺少的依赖后重新运行")
            return False
        
        print("✅ 所有依赖检查通过")
        return True
        
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        default_config = {
            'model': {
                'name': 'BrainInspiredNet',
                'hidden_size': 256,
                'num_layers': 4,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'early_stopping': True,
                'patience': 10
            },
            'hippocampus': {
                'memory_capacity': 1000,
                'encoding_dim': 128,
                'retrieval_threshold': 0.7
            },
            'neocortex': {
                'layers': 6,
                'abstraction_levels': 3
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✅ 配置文件加载成功: {config_path}")
                return {**default_config, **config}
            except Exception as e:
                print(f"⚠️ 配置文件加载失败，使用默认配置: {e}")
        
        return default_config
        
    def _create_directories(self):
        """创建必要的目录结构"""
        directories = [
            'data/datasets',
            'data/models',
            'data/results',
            'logs',
            'visualizations'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        print("📁 目录结构创建完成")
        
    def generate_sample_data(self, dataset_name: str = "synthetic"):
        """生成示例数据集"""
        print(f"📊 生成示例数据集: {dataset_name}")
        
        if dataset_name == "synthetic":
            return self._generate_synthetic_data()
        elif dataset_name == "mnist":
            return self._generate_mnist_like_data()
        elif dataset_name == "patterns":
            return self._generate_pattern_data()
        else:
            raise ValueError(f"未知数据集类型: {dataset_name}")
            
    def _generate_synthetic_data(self):
        """生成合成数据集"""
        np.random.seed(42)
        
        # 生成训练数据
        train_size = 1000
        input_dim = 20
        output_dim = 5
        
        X_train = np.random.randn(train_size, input_dim).astype(np.float32)
        y_train = np.random.randint(0, output_dim, train_size)
        
        # 生成测试数据
        test_size = 200
        X_test = np.random.randn(test_size, input_dim).astype(np.float32)
        y_test = np.random.randint(0, output_dim, test_size)
        
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'input_dim': input_dim,
            'output_dim': output_dim
        }
        
        print(f"✅ 合成数据生成完成 - 训练样本: {train_size}, 测试样本: {test_size}")
        return data
        
    def _generate_mnist_like_data(self):
        """生成MNIST风格的数据"""
        np.random.seed(42)
        
        train_size = 1000
        test_size = 200
        image_size = 28 * 28
        num_classes = 10
        
        X_train = np.random.randn(train_size, image_size).astype(np.float32)
        y_train = np.random.randint(0, num_classes, train_size)
        
        X_test = np.random.randn(test_size, image_size).astype(np.float32)
        y_test = np.random.randint(0, num_classes, test_size)
        
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'input_dim': image_size,
            'output_dim': num_classes,
            'dataset_type': 'mnist_like'
        }
        
        print(f"✅ MNIST风格数据生成完成")
        return data
        
    def _generate_pattern_data(self):
        """生成模式识别数据"""
        np.random.seed(42)
        
        # 生成具有明确模式的分类数据
        train_size = 800
        test_size = 200
        input_dim = 10
        num_classes = 4
        
        X_train = []
        y_train = []
        
        for i in range(train_size):
            class_id = i % num_classes
            pattern = np.zeros(input_dim)
            
            # 为每个类别创建特定模式
            if class_id == 0:
                pattern[:3] = np.random.normal(2, 0.5, 3)
            elif class_id == 1:
                pattern[3:6] = np.random.normal(2, 0.5, 3)
            elif class_id == 2:
                pattern[6:9] = np.random.normal(2, 0.5, 3)
            else:
                pattern[9] = np.random.normal(2, 0.5, 1)
                
            X_train.append(pattern)
            y_train.append(class_id)
            
        X_train = np.array(X_train).astype(np.float32)
        y_train = np.array(y_train).astype(np.int64)
        
        # 生成测试数据
        X_test = []
        y_test = []
        
        for i in range(test_size):
            class_id = i % num_classes
            pattern = np.zeros(input_dim)
            
            if class_id == 0:
                pattern[:3] = np.random.normal(2, 0.5, 3)
            elif class_id == 1:
                pattern[3:6] = np.random.normal(2, 0.5, 3)
            elif class_id == 2:
                pattern[6:9] = np.random.normal(2, 0.5, 3)
            else:
                pattern[9] = np.random.normal(2, 0.5, 1)
                
            X_test.append(pattern)
            y_test.append(class_id)
            
        X_test = np.array(X_test).astype(np.float32)
        y_test = np.array(y_test).astype(np.int64)
        
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'input_dim': input_dim,
            'output_dim': num_classes,
            'dataset_type': 'patterns'
        }
        
        print(f"✅ 模式识别数据生成完成")
        return data
        
    def create_models(self, model_type: str = "brain_inspired"):
        """创建模型"""
        print(f"🏗️ 创建模型: {model_type}")
        
        if not TORCH_AVAILABLE:
            print("❌ PyTorch不可用，无法创建深度学习模型")
            return {}
            
        if model_type == "brain_inspired":
            return self._create_brain_inspired_model()
        elif model_type == "hippocampus_only":
            return self._create_hippocampus_model()
        elif model_type == "neocortex_only":
            return self._create_neocortex_model()
        else:
            raise ValueError(f"未知模型类型: {model_type}")
            
    def _create_brain_inspired_model(self):
        """创建脑启发模型"""
        class BrainInspiredNet(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4):
                super().__init__()
                self.input_dim = input_dim
                self.hidden_dim = hidden_dim
                self.output_dim = output_dim
                self.num_layers = num_layers
                
                # 输入层
                self.input_layer = nn.Linear(input_dim, hidden_dim)
                
                # 隐藏层
                self.hidden_layers = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
                ])
                
                # 海马体记忆层（简化的注意力机制）
                self.hippocampus_layer = nn.MultiheadAttention(
                    hidden_dim, num_heads=8, batch_first=True
                )
                
                # 新皮层抽象层
                self.neocortex_layers = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.Linear(hidden_dim // 2, hidden_dim // 4)
                ])
                
                # 输出层
                self.output_layer = nn.Linear(hidden_dim // 4, output_dim)
                
                # 激活函数
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                # 输入层
                x = self.activation(self.input_layer(x))
                x = self.dropout(x)
                
                # 隐藏层
                for layer in self.hidden_layers:
                    x = self.activation(layer(x))
                    x = self.dropout(x)
                
                # 海马体层（注意力机制）
                x = x.unsqueeze(1)  # 添加序列维度
                attended_x, _ = self.hippocampus_layer(x, x, x)
                x = attended_x.squeeze(1)
                
                # 新皮层抽象
                for layer in self.neocortex_layers:
                    x = self.activation(layer(x))
                    x = self.dropout(x)
                
                # 输出层
                output = self.output_layer(x)
                return output
                
        # 获取数据维度
        if hasattr(self, 'current_data'):
            input_dim = self.current_data['input_dim']
            output_dim = self.current_data['output_dim']
        else:
            input_dim = 20
            output_dim = 5
            
        hidden_dim = self.config['model']['hidden_size']
        num_layers = self.config['model']['num_layers']
        
        model = BrainInspiredNet(input_dim, hidden_dim, output_dim, num_layers)
        
        print(f"✅ 脑启发模型创建完成")
        print(f"   - 输入维度: {input_dim}")
        print(f"   - 隐藏维度: {hidden_dim}")
        print(f"   - 输出维度: {output_dim}")
        print(f"   - 层数: {num_layers}")
        
        return {'brain_inspired': model}
        
    def _create_hippocampus_model(self):
        """创建海马体专用模型"""
        class HippocampusModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, output_dim)
                )
                
            def forward(self, x):
                x = self.encoder(x)
                x = x.unsqueeze(1)
                attended_x, _ = self.attention(x, x, x)
                x = attended_x.squeeze(1)
                output = self.decoder(x)
                return output
                
        input_dim = self.current_data['input_dim']
        output_dim = self.current_data['output_dim']
        hidden_dim = self.config['model']['hidden_size']
        
        model = HippocampusModel(input_dim, hidden_dim, output_dim)
        
        print(f"✅ 海马体模型创建完成")
        return {'hippocampus': model}
        
    def _create_neocortex_model(self):
        """创建新皮层专用模型"""
        class NeocortexModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.abstraction_layers = nn.ModuleList([
                    nn.Linear(hidden_dim, hidden_dim // (i + 2)) for i in range(3)
                ])
                self.classifier = nn.Linear(hidden_dim // 3, output_dim)
                self.activation = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
                
                # 预编码器
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
            def forward(self, x):
                x = self.encoder(x)
                
                for layer in self.abstraction_layers:
                    x = self.activation(layer(x))
                    x = self.dropout(x)
                
                output = self.classifier(x)
                return output
                
        input_dim = self.current_data['input_dim']
        output_dim = self.current_data['output_dim']
        hidden_dim = self.config['model']['hidden_size']
        
        model = NeocortexModel(input_dim, hidden_dim, output_dim)
        
        print(f"✅ 新皮层模型创建完成")
        return {'neocortex': model}
        
    def train_model(self, model_name: str, data: Dict, epochs: Optional[int] = None):
        """训练模型"""
        if not TORCH_AVAILABLE:
            print("❌ PyTorch不可用，无法训练模型")
            return {}
            
        print(f"🚀 开始训练模型: {model_name}")
        
        if model_name not in self.models:
            print(f"❌ 模型 {model_name} 不存在")
            return {}
            
        model = self.models[model_name]
        if not hasattr(model, 'train'):
            print(f"❌ 对象 {model_name} 不是PyTorch模型")
            return {}
            
        # 准备数据
        X_train = torch.FloatTensor(data['X_train'])
        y_train = torch.LongTensor(data['y_train'])
        X_test = torch.FloatTensor(data['X_test'])
        y_test = torch.LongTensor(data['y_test'])
        
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.config['training']['batch_size'], shuffle=True)
        
        # 设置训练参数
        epochs = epochs or self.config['training']['epochs']
        learning_rate = self.config['training']['learning_rate']
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练循环
        model.train()
        train_losses = []
        test_accuracies = []
        
        print(f"📈 训练参数: epochs={epochs}, batch_size={self.config['training']['batch_size']}, lr={learning_rate}")
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            # 评估
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                _, predicted = torch.max(test_outputs.data, 1)
                accuracy = (predicted == y_test).float().mean().item()
                test_accuracies.append(accuracy)
            model.train()
            
            # 打印进度
            if (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"   Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
        
        # 保存训练结果
        results = {
            'model_name': model_name,
            'epochs': epochs,
            'final_loss': train_losses[-1],
            'final_accuracy': test_accuracies[-1],
            'train_losses': train_losses,
            'test_accuracies': test_accuracies,
            'training_time': time.time()
        }
        
        self.results[model_name] = results
        
        print(f"✅ 训练完成!")
        print(f"   - 最终损失: {train_losses[-1]:.4f}")
        print(f"   - 最终准确率: {test_accuracies[-1]:.4f}")
        
        return results
        
    def evaluate_model(self, model_name: str, data: Dict):
        """评估模型"""
        if not TORCH_AVAILABLE:
            print("❌ PyTorch不可用，无法评估模型")
            return {}
            
        print(f"📊 评估模型: {model_name}")
        
        if model_name not in self.models:
            print(f"❌ 模型 {model_name} 不存在")
            return {}
            
        model = self.models[model_name]
        
        X_test = torch.FloatTensor(data['X_test'])
        y_test = torch.LongTensor(data['y_test'])
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test).float().mean().item()
            
            # 计算分类报告
            from sklearn.metrics import classification_report, confusion_matrix
            
            y_pred = predicted.numpy()
            y_true = y_test.numpy()
            
            report = classification_report(y_true, y_pred, output_dict=True)
            confusion = confusion_matrix(y_true, y_pred)
            
        evaluation_results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion.tolist(),
            'num_test_samples': len(y_test)
        }
        
        print(f"✅ 评估完成!")
        print(f"   - 准确率: {accuracy:.4f}")
        print(f"   - 测试样本数: {len(y_test)}")
        
        return evaluation_results
        
    def save_results(self, filename: str = None):
        """保存结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/demo_results_{timestamp}.json"
            
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
            
        print(f"💾 结果已保存到: {filename}")
        
    def load_results(self, filename: str):
        """加载结果"""
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            print(f"📂 结果已从 {filename} 加载")
            return True
        else:
            print(f"❌ 文件不存在: {filename}")
            return False
            
    def visualize_results(self, save_plots: bool = True):
        """可视化结果"""
        try:
            import matplotlib.pyplot as plt
            
            print("📊 生成可视化图表...")
            
            if not self.results:
                print("❌ 没有结果可以可视化")
                return
                
            # 训练曲线
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('脑启发AI演示结果', fontsize=16)
            
            for i, (model_name, result) in enumerate(self.results.items()):
                if 'train_losses' in result and 'test_accuracies' in result:
                    # 损失曲线
                    axes[0, 0].plot(result['train_losses'], label=f'{model_name} Loss')
                    axes[0, 0].set_title('训练损失')
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Loss')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True)
                    
                    # 准确率曲线
                    axes[0, 1].plot(result['test_accuracies'], label=f'{model_name} Accuracy')
                    axes[0, 1].set_title('测试准确率')
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Accuracy')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True)
            
            # 性能对比
            model_names = list(self.results.keys())
            accuracies = [self.results[name].get('final_accuracy', 0) for name in model_names]
            
            axes[1, 0].bar(model_names, accuracies)
            axes[1, 0].set_title('模型性能对比')
            axes[1, 0].set_ylabel('准确率')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # 模型参数统计
            param_counts = []
            for model_name in model_names:
                if model_name in self.models:
                    model = self.models[model_name]
                    if hasattr(model, 'parameters'):
                        param_count = sum(p.numel() for p in model.parameters())
                        param_counts.append(param_count)
                    else:
                        param_counts.append(0)
                else:
                    param_counts.append(0)
            
            axes[1, 1].bar(model_names, param_counts)
            axes[1, 1].set_title('模型参数数量')
            axes[1, 1].set_ylabel('参数数量')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_plots:
                os.makedirs('visualizations', exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"visualizations/demo_results_{timestamp}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"📈 图表已保存到: {plot_filename}")
            
            plt.show()
            
        except ImportError:
            print("⚠️ matplotlib未安装，跳过可视化")
        except Exception as e:
            print(f"❌ 可视化失败: {e}")


def interactive_demo():
    """交互式演示模式"""
    print("🧠 脑启发AI演示系统")
    print("=" * 50)
    
    system = BrainInspiredAISystem()
    
    # 系统初始化
    print("\n1️⃣ 系统初始化")
    if not system.initialize_system():
        return
        
    while True:
        print("\n" + "=" * 50)
        print("选择演示功能:")
        print("1. 数据生成")
        print("2. 模型创建") 
        print("3. 模型训练")
        print("4. 模型评估")
        print("5. 结果可视化")
        print("6. 保存/加载结果")
        print("7. 运行完整演示")
        print("0. 退出")
        
        choice = input("\n请输入选择 (0-7): ").strip()
        
        if choice == "0":
            print("👋 感谢使用脑启发AI演示系统!")
            break
        elif choice == "1":
            print("\n📊 数据生成选项:")
            print("1. 合成数据")
            print("2. MNIST风格数据") 
            print("3. 模式识别数据")
            
            data_choice = input("选择数据类型 (1-3): ").strip()
            
            if data_choice == "1":
                data = system.generate_sample_data("synthetic")
            elif data_choice == "2":
                data = system.generate_sample_data("mnist")
            elif data_choice == "3":
                data = system.generate_sample_data("patterns")
            else:
                print("❌ 无效选择")
                continue
                
            system.current_data = data
            print(f"✅ 数据生成完成 - 训练样本: {len(data['X_train'])}, 测试样本: {len(data['X_test'])}")
            
        elif choice == "2":
            print("\n🏗️ 模型创建选项:")
            print("1. 脑启发完整模型")
            print("2. 海马体专用模型")
            print("3. 新皮层专用模型")
            
            model_choice = input("选择模型类型 (1-3): ").strip()
            
            if model_choice == "1":
                models = system.create_models("brain_inspired")
            elif model_choice == "2":
                models = system.create_models("hippocampus_only")
            elif model_choice == "3":
                models = system.create_models("neocortex_only")
            else:
                print("❌ 无效选择")
                continue
                
            system.models.update(models)
            print(f"✅ 模型创建完成: {list(models.keys())}")
            
        elif choice == "3":
            if not system.models:
                print("❌ 请先创建模型")
                continue
                
            print("\n训练模型选择:")
            for i, model_name in enumerate(system.models.keys(), 1):
                print(f"{i}. {model_name}")
                
            train_choice = input(f"选择要训练的模型 (1-{len(system.models)}): ").strip()
            
            try:
                model_idx = int(train_choice) - 1
                model_name = list(system.models.keys())[model_idx]
                
                epochs_input = input("训练轮数 (默认100): ").strip()
                epochs = int(epochs_input) if epochs_input else 100
                
                if not hasattr(system, 'current_data'):
                    print("❌ 请先生成数据")
                    continue
                    
                result = system.train_model(model_name, system.current_data, epochs)
                print(f"✅ {model_name} 训练完成")
                
            except (ValueError, IndexError):
                print("❌ 无效选择")
                
        elif choice == "4":
            if not system.models:
                print("❌ 请先创建模型")
                continue
                
            print("\n评估模型选择:")
            for i, model_name in enumerate(system.models.keys(), 1):
                print(f"{i}. {model_name}")
                
            eval_choice = input(f"选择要评估的模型 (1-{len(system.models)}): ").strip()
            
            try:
                model_idx = int(eval_choice) - 1
                model_name = list(system.models.keys())[model_idx]
                
                if not hasattr(system, 'current_data'):
                    print("❌ 请先生成数据")
                    continue
                    
                evaluation = system.evaluate_model(model_name, system.current_data)
                print(f"✅ {model_name} 评估完成")
                
            except (ValueError, IndexError):
                print("❌ 无效选择")
                
        elif choice == "5":
            system.visualize_results()
            
        elif choice == "6":
            print("\n保存/加载选项:")
            print("1. 保存结果")
            print("2. 加载结果")
            
            save_choice = input("选择操作 (1-2): ").strip()
            
            if save_choice == "1":
                filename = input("保存文件名 (可选): ").strip()
                if not filename:
                    system.save_results()
                else:
                    system.save_results(filename)
            elif save_choice == "2":
                filename = input("加载文件名: ").strip()
                system.load_results(filename)
            else:
                print("❌ 无效选择")
                
        elif choice == "7":
            print("\n🚀 运行完整演示...")
            
            # 生成数据
            print("1. 生成合成数据")
            data = system.generate_sample_data("synthetic")
            system.current_data = data
            
            # 创建模型
            print("2. 创建脑启发模型")
            models = system.create_models("brain_inspired")
            system.models.update(models)
            
            # 训练模型
            print("3. 训练模型")
            result = system.train_model("brain_inspired", data, epochs=50)
            
            # 评估模型
            print("4. 评估模型")
            evaluation = system.evaluate_model("brain_inspired", data)
            
            # 可视化结果
            print("5. 生成可视化")
            system.visualize_results()
            
            # 保存结果
            print("6. 保存结果")
            system.save_results()
            
            print("🎉 完整演示完成!")
            
        else:
            print("❌ 无效选择，请重新输入")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='脑启发AI演示系统')
    parser.add_argument('--mode', choices=['interactive', 'demo', 'batch'], default='interactive',
                       help='运行模式: interactive(交互式), demo(自动演示), batch(批处理)')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--dataset', default='synthetic', help='数据集类型')
    parser.add_argument('--model', default='brain_inspired', help='模型类型')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--output', help='结果输出文件')
    
    args = parser.parse_args()
    
    system = BrainInspiredAISystem()
    
    if args.mode == 'interactive':
        interactive_demo()
    elif args.mode == 'demo':
        # 自动演示模式
        print("🎭 自动演示模式")
        
        # 初始化系统
        system.initialize_system(args.config)
        
        # 生成数据
        data = system.generate_sample_data(args.dataset)
        system.current_data = data
        
        # 创建并训练模型
        models = system.create_models(args.model)
        system.models.update(models)
        
        result = system.train_model(args.model, data, args.epochs)
        
        # 评估模型
        evaluation = system.evaluate_model(args.model, data)
        
        # 保存结果
        if args.output:
            system.save_results(args.output)
        else:
            system.save_results()
            
        print("✅ 自动演示完成!")
        
    elif args.mode == 'batch':
        # 批处理模式 - 运行多个实验
        print("📦 批处理模式")
        
        system.initialize_system(args.config)
        datasets = ['synthetic', 'mnist', 'patterns']
        models = ['brain_inspired', 'hippocampus_only', 'neocortex_only']
        
        for dataset in datasets:
            print(f"\n🔄 数据集: {dataset}")
            data = system.generate_sample_data(dataset)
            system.current_data = data
            
            for model_name in models:
                print(f"   模型: {model_name}")
                
                try:
                    # 创建模型
                    models_dict = system.create_models(model_name)
                    system.models.update(models_dict)
                    
                    # 训练模型
                    result = system.train_model(model_name, data, args.epochs)
                    
                    # 评估模型
                    evaluation = system.evaluate_model(model_name, data)
                    
                    print(f"   ✅ {model_name} 在 {dataset} 上完成")
                    
                except Exception as e:
                    print(f"   ❌ {model_name} 在 {dataset} 上失败: {e}")
        
        # 保存所有结果
        system.save_results("results/batch_results.json")
        print("📦 批处理完成!")


if __name__ == "__main__":
    main()