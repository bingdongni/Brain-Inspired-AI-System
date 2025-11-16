#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化工具
=========

提供各种可视化功能，包括:
- 训练曲线绘制
- 模型架构可视化
- 数据分布分析
- 注意力热图
- 混淆矩阵
- 性能对比
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        """
        初始化训练可视化器
        
        Args:
            save_dir: 图像保存目录
        """
        self.save_dir = Path(save_dir) if save_dir else Path("plots")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_curves(self, train_history: Dict[str, List[float]], 
                           val_history: Optional[Dict[str, List[float]]] = None,
                           metrics: Optional[List[str]] = None,
                           figsize: Tuple[int, int] = (15, 10)):
        """
        绘制训练曲线
        
        Args:
            train_history: 训练历史字典
            val_history: 验证历史字典
            metrics: 要绘制的指标列表
            figsize: 图像大小
        """
        if metrics is None:
            metrics = [key for key in train_history.keys() if 'loss' in key.lower() or 'acc' in key.lower()]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize)
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # 绘制训练曲线
            if metric in train_history:
                ax.plot(train_history[metric], label=f'Train {metric}', color='blue', linewidth=2)
            
            # 绘制验证曲线
            if val_history and metric in val_history:
                ax.plot(val_history[metric], label=f'Validation {metric}', color='red', linewidth=2)
            
            ax.set_title(f'{metric.capitalize()} Training Curve', fontsize=14, fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.save_dir / "training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {save_path}")
        
        plt.show()
    
    def plot_learning_rate_schedule(self, lrs: List[float], 
                                   schedule_type: str = 'cosine',
                                   figsize: Tuple[int, int] = (10, 6)):
        """绘制学习率调度"""
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = range(len(lrs))
        ax.plot(epochs, lrs, linewidth=2, color='green')
        ax.set_title(f'Learning Rate Schedule ({schedule_type.capitalize()})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = self.save_dir / f"lr_schedule_{schedule_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学习率调度图已保存到: {save_path}")
        
        plt.show()
    
    def plot_model_performance_comparison(self, model_results: Dict[str, Dict[str, float]],
                                        metrics: Optional[List[str]] = None,
                                        figsize: Tuple[int, int] = (12, 8)):
        """绘制模型性能对比"""
        if metrics is None:
            metrics = list(next(iter(model_results.values())).keys())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(metrics))
        width = 0.35
        
        model_names = list(model_results.keys())
        values1 = [model_results[model_names[0]][metric] for metric in metrics]
        values2 = [model_results[model_names[1]][metric] for metric in metrics]
        
        ax.bar(x - width/2, values1, width, label=model_names[0], alpha=0.8)
        ax.bar(x + width/2, values2, width, label=model_names[1], alpha=0.8)
        
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = self.save_dir / "model_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存到: {save_path}")
        
        plt.show()

class ModelArchitectureVisualizer:
    """模型架构可视化器"""
    
    @staticmethod
    def plot_model_graph(model, input_shape: Tuple, 
                        save_path: Optional[Union[str, Path]] = None):
        """绘制模型架构图"""
        try:
            from torchviz import make_dot
            
            # 创建示例输入
            dummy_input = torch.randn(input_shape)
            
            # 生成模型图
            dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                dot.render(save_path.stem, save_path.parent.as_posix(), format='png')
                print(f"模型架构图已保存到: {save_path}")
            
            return dot
            
        except ImportError:
            print("请安装torchviz来生成模型架构图: pip install torchviz")
            return None
    
    @staticmethod
    def plot_layer_activations(model, input_data: np.ndarray, 
                              layer_names: Optional[List[str]] = None,
                              save_dir: Optional[Union[str, Path]] = None):
        """绘制层激活可视化"""
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        # 注册钩子
        handles = []
        if layer_names:
            for name in layer_names:
                layer = dict(model.named_modules())[name]
                handle = layer.register_forward_hook(get_activation(name))
                handles.append(handle)
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            _ = model(torch.FloatTensor(input_data))
        
        # 清理钩子
        for handle in handles:
            handle.remove()
        
        # 可视化激活
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            for name, activation in activations.items():
                fig, axes = plt.subplots(1, min(8, activation.shape[-1]), figsize=(15, 2))
                if activation.shape[-1] == 1:
                    axes = [axes]
                
                for i in range(min(8, activation.shape[-1])):
                    axes[i].imshow(activation[0, i], cmap='viridis', aspect='auto')
                    axes[i].set_title(f'{name}\nChannel {i}')
                    axes[i].axis('off')
                
                plt.tight_layout()
                plt.savefig(save_dir / f'{name}_activations.png', dpi=300, bbox_inches='tight')
                plt.close()
        
        return activations

class DataVisualizer:
    """数据可视化器"""
    
    def __init__(self, save_dir: Optional[Union[str, Path]] = None):
        self.save_dir = Path(save_dir) if save_dir else Path("plots")
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_data_distribution(self, data: np.ndarray, 
                             labels: Optional[np.ndarray] = None,
                             plot_type: str = 'histogram',
                             figsize: Tuple[int, int] = (12, 8)):
        """绘制数据分布"""
        if len(data.shape) > 2:
            # 对于图像数据，显示几个样本
            n_samples = min(8, data.shape[0])
            fig, axes = plt.subplots(2, 4, figsize=figsize)
            axes = axes.flatten()
            
            for i in range(n_samples):
                if len(data.shape) == 3:  # 灰度图像
                    axes[i].imshow(data[i], cmap='gray')
                else:  # 彩色图像
                    axes[i].imshow(data[i])
                axes[i].set_title(f'Sample {i+1}')
                axes[i].axis('off')
            
            plt.tight_layout()
        else:
            # 对于数值数据，显示分布
            if plot_type == 'histogram':
                plt.figure(figsize=figsize)
                plt.hist(data.flatten(), bins=50, alpha=0.7, edgecolor='black')
                plt.title('Data Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('Value', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.grid(True, alpha=0.3)
            elif plot_type == 'box':
                plt.figure(figsize=figsize)
                plt.boxplot(data.flatten())
                plt.title('Data Box Plot', fontsize=14, fontweight='bold')
                plt.ylabel('Value', fontsize=12)
                plt.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = self.save_dir / f"data_distribution_{plot_type}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"数据分布图已保存到: {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, importances: np.ndarray, 
                              feature_names: Optional[List[str]] = None,
                              top_k: int = 20,
                              figsize: Tuple[int, int] = (10, 8)):
        """绘制特征重要性"""
        # 获取top-k特征
        indices = np.argsort(importances)[::-1][:top_k]
        top_importances = importances[indices]
        
        if feature_names:
            top_names = [feature_names[i] for i in indices]
        else:
            top_names = [f'Feature {i+1}' for i in indices]
        
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_importances)), top_importances)
        plt.yticks(range(len(top_importances)), top_names)
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_k} Feature Importance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = self.save_dir / "feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {save_path}")
        
        plt.show()

class AttentionVisualizer:
    """注意力可视化器"""
    
    @staticmethod
    def plot_attention_heatmap(attention_weights: np.ndarray,
                             tokens: Optional[List[str]] = None,
                             save_path: Optional[Union[str, Path]] = None):
        """绘制注意力热图"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 绘制热图
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # 设置标签
        if tokens:
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
        
        # 添加颜色条
        plt.colorbar(im, ax=ax)
        
        # 添加数值标签
        for i in range(attention_weights.shape[0]):
            for j in range(attention_weights.shape[1]):
                text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Tokens', fontsize=12)
        ax.set_ylabel('Tokens', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力热图已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_multi_head_attention(attention_weights: Dict[int, np.ndarray],
                                tokens: Optional[List[str]] = None,
                                save_dir: Optional[Union[str, Path]] = None):
        """绘制多头注意力"""
        n_heads = len(attention_weights)
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if n_heads == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (head, weights) in enumerate(attention_weights.items()):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            im = ax.imshow(weights, cmap='Blues', aspect='auto')
            
            if tokens:
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)
            
            ax.set_title(f'Head {head+1}', fontsize=12)
            plt.colorbar(im, ax=ax)
        
        # 隐藏多余的子图
        for i in range(n_heads, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
        
        plt.suptitle('Multi-Head Attention Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "multi_head_attention.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"多头注意力图已保存到: {save_path}")
        
        plt.show()

class MetricVisualizer:
    """指标可视化器"""
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                            class_names: Optional[List[str]] = None,
                            normalize: bool = False,
                            save_path: Optional[Union[str, Path]] = None):
        """绘制混淆矩阵"""
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', square=True,
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray,
                      n_classes: int = 2,
                      save_path: Optional[Union[str, Path]] = None):
        """绘制ROC曲线"""
        if n_classes == 2:
            # 二分类
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('Receiver Operating Characteristic (ROC) Curve', 
                     fontsize=14, fontweight='bold')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
        else:
            # 多分类
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i in range(min(4, n_classes)):
                y_true_binary = (y_true == i).astype(int)
                y_scores_binary = y_scores[:, i] if y_scores.ndim > 1 else y_scores
                
                fpr, tpr, _ = roc_curve(y_true_binary, y_scores_binary)
                roc_auc = auc(fpr, tpr)
                
                axes[i].plot(fpr, tpr, lw=2, 
                           label=f'Class {i} (AUC = {roc_auc:.2f})')
                axes[i].plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
                axes[i].set_xlim([0.0, 1.0])
                axes[i].set_ylim([0.0, 1.05])
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'ROC Curve - Class {i}')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存到: {save_path}")
        
        plt.show()

class InteractiveVisualizer:
    """交互式可视化器"""
    
    @staticmethod
    def create_interactive_training_dashboard(train_history: Dict[str, List[float]],
                                            val_history: Optional[Dict[str, List[float]]] = None):
        """创建交互式训练仪表板"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Training Accuracy', 
                          'Validation Loss', 'Validation Accuracy'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        epochs = range(1, len(train_history['loss']) + 1)
        
        # 训练损失
        fig.add_trace(
            go.Scatter(x=epochs, y=train_history['loss'], 
                      name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 训练准确率
        if 'accuracy' in train_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=train_history['accuracy'], 
                          name='Training Accuracy', line=dict(color='green')),
                row=1, col=2
            )
        
        # 验证损失
        if val_history and 'loss' in val_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=val_history['loss'], 
                          name='Validation Loss', line=dict(color='red')),
                row=2, col=1
            )
        
        # 验证准确率
        if val_history and 'accuracy' in val_history:
            fig.add_trace(
                go.Scatter(x=epochs, y=val_history['accuracy'], 
                          name='Validation Accuracy', line=dict(color='orange')),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Training Dashboard",
            showlegend=True,
            height=600
        )
        
        return fig
    
    @staticmethod
    def create_model_comparison_radar(models_data: Dict[str, Dict[str, float]]):
        """创建模型对比雷达图"""
        categories = list(next(iter(models_data.values())).keys())
        
        fig = go.Figure()
        
        for model_name, metrics in models_data.items():
            values = [metrics[metric] for metric in categories]
            values += values[:1]  # 闭合雷达图
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=model_name
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison"
        )
        
        return fig

# 工具函数
def save_plot(fig_or_ax, save_path: Union[str, Path], dpi: int = 300):
    """保存图像"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(fig_or_ax, 'savefig'):
        fig_or_ax.savefig(save_path, dpi=dpi, bbox_inches='tight')
    else:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    print(f"图像已保存到: {save_path}")

def set_plot_style(style: str = 'seaborn'):
    """设置绘图样式"""
    try:
        plt.style.use(style)
    except OSError:
        print(f"样式 {style} 不可用，使用默认样式")

def create_video_from_images(image_paths: List[Union[str, Path]], 
                           output_path: Union[str, Path],
                           fps: int = 1):
    """从图像序列创建视频"""
    try:
        import cv2
        
        # 获取图像尺寸
        first_image = cv2.imread(str(image_paths[0]))
        height, width, layers = first_image.shape
        
        # 定义视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # 写入图像
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            video_writer.write(frame)
        
        video_writer.release()
        print(f"视频已保存到: {output_path}")
        
    except ImportError:
        print("请安装opencv-python来创建视频: pip install opencv-python")