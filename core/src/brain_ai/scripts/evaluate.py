#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
评估脚本
=======

提供统一的模型评估功能，包括:
- 模型加载
- 测试评估
- 性能指标计算
- 报告生成
- 结果可视化
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import pickle

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brain_ai.utils import Logger, MetricsCollector, ModelUtils
from brain_ai.utils.visualization import MetricVisualizer, ModelArchitectureVisualizer
from brain_ai.utils.metrics_collector import ModelMetrics

class Evaluator:
    """模型评估器"""
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        初始化评估器
        
        Args:
            model: 要评估的模型
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        self.logger = Logger("evaluator")
        self.metrics = MetricsCollector()
        self.model_metrics = ModelMetrics()
        
        self.logger.info(f"评估器初始化完成，设备: {device}")
    
    def evaluate_model(self, test_loader, save_results: bool = True, 
                      results_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        评估模型
        
        Args:
            test_loader: 测试数据加载器
            save_results: 是否保存结果
            results_dir: 结果保存目录
            
        Returns:
            评估结果字典
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0
        
        self.logger.info("开始模型评估...")
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(test_loader, desc="评估")):
                data, target = data.to(self.device), target.to(self.device)
                
                # 前向传播
                output = self.model(data)
                
                # 计算损失
                if hasattr(self.model, 'criterion'):
                    loss = self.model.criterion(output, target)
                else:
                    loss = nn.functional.cross_entropy(output, target)
                
                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)
                
                # 收集预测结果
                if hasattr(output, 'argmax'):
                    predictions = output.argmax(dim=1).cpu().numpy()
                else:
                    predictions = output.cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())
        
        # 计算指标
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        results = {
            'test_loss': total_loss / total_samples,
            'samples': total_samples,
            'predictions': all_predictions.tolist(),
            'targets': all_targets.tolist()
        }
        
        # 计算分类指标
        if len(np.unique(all_targets)) > 1:  # 多分类或二分类
            results.update(self._compute_classification_metrics(all_predictions, all_targets))
        
        # 保存结果
        if save_results and results_dir:
            self._save_evaluation_results(results, results_dir)
        
        # 生成报告
        self._generate_evaluation_report(results)
        
        self.logger.info("模型评估完成！")
        return results
    
    def _compute_classification_metrics(self, predictions: np.ndarray, 
                                      targets: np.ndarray) -> Dict[str, float]:
        """计算分类指标"""
        metrics = {}
        
        try:
            # 基本指标
            metrics['accuracy'] = self.model_metrics.accuracy(predictions, targets)
            metrics['precision'] = self.model_metrics.precision(predictions, targets)
            metrics['recall'] = self.model_metrics.recall(predictions, targets)
            metrics['f1_score'] = self.model_metrics.f1_score(predictions, targets)
            
            # 混淆矩阵
            cm = self.model_metrics.confusion_matrix(predictions, targets)
            metrics['confusion_matrix'] = cm.tolist()
            
            # 分类报告
            classification_report = self.model_metrics.classification_report(predictions, targets)
            metrics['classification_report'] = classification_report
            
        except Exception as e:
            self.logger.warning(f"计算分类指标时出错: {e}")
        
        return metrics
    
    def _save_evaluation_results(self, results: Dict[str, Any], 
                               results_dir: str):
        """保存评估结果"""
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存详细结果
        results_file = results_dir / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # 处理numpy数组
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[key] = value.tolist()
                else:
                    serializable_results[key] = value
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        # 保存预测结果
        predictions_file = results_dir / "predictions.pkl"
        with open(predictions_file, 'wb') as f:
            pickle.dump({
                'predictions': results['predictions'],
                'targets': results['targets']
            }, f)
        
        # 保存模型信息
        model_info = ModelUtils.get_model_info(self.model)
        model_info_file = results_dir / "model_info.json"
        with open(model_info_file, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"评估结果已保存到: {results_dir}")
    
    def _generate_evaluation_report(self, results: Dict[str, Any]):
        """生成评估报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("           模型评估报告")
        report_lines.append("=" * 60)
        report_lines.append(f"测试样本数: {results['samples']}")
        report_lines.append(f"测试损失: {results['test_loss']:.4f}")
        
        # 添加分类指标
        if 'accuracy' in results:
            report_lines.append("\n分类性能指标:")
            report_lines.append(f"准确率 (Accuracy): {results['accuracy']:.4f}")
            report_lines.append(f"精确率 (Precision): {results['precision']:.4f}")
            report_lines.append(f"召回率 (Recall): {results['recall']:.4f}")
            report_lines.append(f"F1分数: {results['f1_score']:.4f}")
        
        # 添加分类报告详情
        if 'classification_report' in results:
            report_lines.append("\n详细分类报告:")
            report_data = results['classification_report']
            
            for class_name, metrics in report_data.items():
                if isinstance(metrics, dict):
                    report_lines.append(f"{class_name}:")
                    for metric_name, value in metrics.items():
                        if isinstance(value, float):
                            report_lines.append(f"  {metric_name}: {value:.4f}")
        
        # 添加模型信息
        model_info = ModelUtils.get_model_info(self.model)
        report_lines.append("\n模型信息:")
        report_lines.append(f"模型类型: {model_info['model_class']}")
        report_lines.append(f"参数数量: {model_info['total_parameters']:,}")
        report_lines.append(f"模型大小: {model_info['model_size_mb']:.2f} MB")
        report_lines.append(f"计算设备: {model_info['device']}")
        
        # 打印报告
        report_text = "\n".join(report_lines)
        print("\n" + report_text)
        
        # 保存报告
        if hasattr(self, 'save_dir'):
            report_file = self.save_dir / "evaluation_report.txt"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
        
        return report_text
    
    def plot_results(self, results: Dict[str, Any], 
                    save_plots: bool = True,
                    plots_dir: Optional[str] = None):
        """绘制评估结果"""
        if 'predictions' not in results or 'targets' not in results:
            self.logger.warning("预测结果不完整，无法绘制图表")
            return
        
        predictions = np.array(results['predictions'])
        targets = np.array(results['targets'])
        
        if save_plots and plots_dir:
            plots_dir = Path(plots_dir)
            plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 绘制混淆矩阵
        if 'confusion_matrix' in results:
            cm = np.array(results['confusion_matrix'])
            class_names = [f"Class {i}" for i in range(cm.shape[0])]
            
            MetricVisualizer.plot_confusion_matrix(
                targets, predictions, class_names,
                save_path=plots_dir / "confusion_matrix.png" if save_plots else None
            )
        
        # 绘制ROC曲线
        if hasattr(self.model, 'predict_proba') or hasattr(self.model, 'output'):
            try:
                # 获取预测概率
                self.model.eval()
                with torch.no_grad():
                    # 这里需要根据具体模型结构调整
                    pass
                
                MetricVisualizer.plot_roc_curve(
                    targets, predictions,
                    save_path=plots_dir / "roc_curve.png" if save_plots else None
                )
            except Exception as e:
                self.logger.warning(f"绘制ROC曲线时出错: {e}")
        
        # 绘制预测分布
        self._plot_prediction_distribution(
            targets, predictions,
            save_path=plots_dir / "prediction_distribution.png" if save_plots else None
        )
    
    def _plot_prediction_distribution(self, targets: np.ndarray, predictions: np.ndarray,
                                    save_path: Optional[Path] = None):
        """绘制预测分布"""
        try:
            from brain_ai.utils.visualization import DataVisualizer
            
            visualizer = DataVisualizer()
            
            # 创建预测概率分布（简化版本）
            correct_preds = (predictions == targets).astype(int)
            accuracy_per_class = {}
            
            for class_id in np.unique(targets):
                class_mask = targets == class_id
                class_accuracy = np.mean(correct_preds[class_mask])
                accuracy_per_class[class_id] = class_accuracy
            
            # 绘制每类准确率
            classes = list(accuracy_per_class.keys())
            accuracies = list(accuracy_per_class.values())
            
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.bar(classes, accuracies, alpha=0.7)
            plt.xlabel('Class')
            plt.ylabel('Accuracy')
            plt.title('Accuracy per Class')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            
            for i, acc in enumerate(accuracies):
                plt.text(classes[i], acc + 0.01, f'{acc:.3f}', 
                        ha='center', va='bottom')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"预测分布图已保存到: {save_path}")
            
            plt.show()
            
        except Exception as e:
            self.logger.warning(f"绘制预测分布时出错: {e}")

def load_model_for_evaluation(model_path: Union[str, Path],
                            model_class: Optional[type] = None,
                            **kwargs) -> nn.Module:
    """
    加载模型用于评估
    
    Args:
        model_path: 模型文件路径
        model_class: 模型类
        kwargs: 传递给模型构造函数的参数
        
    Returns:
        加载的模型
    """
    logger = Logger("evaluator")
    logger.info(f"加载模型: {model_path}")
    
    return ModelUtils.load_model(model_path, model_class, **kwargs)

def evaluate_model(model_path: Union[str, Path],
                  test_loader,
                  model_class: Optional[type] = None,
                  device: str = 'cpu',
                  results_dir: Optional[str] = None,
                  **model_kwargs) -> Dict[str, Any]:
    """
    评估模型的主函数
    
    Args:
        model_path: 模型文件路径
        test_loader: 测试数据加载器
        model_class: 模型类
        device: 计算设备
        results_dir: 结果保存目录
        **model_kwargs: 模型构造函数参数
        
    Returns:
        评估结果
    """
    # 加载模型
    model = load_model_for_evaluation(model_path, model_class, **model_kwargs)
    
    # 创建评估器
    evaluator = Evaluator(model, device)
    
    # 执行评估
    results = evaluator.evaluate_model(test_loader, save_results=True, results_dir=results_dir)
    
    # 绘制结果
    if results_dir:
        evaluator.plot_results(results, save_plots=True, plots_dir=results_dir)
    
    return results

def compare_models(model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    比较多个模型的评估结果
    
    Args:
        model_results: 模型结果字典 {model_name: results}
        
    Returns:
        比较结果DataFrame
    """
    comparison_data = []
    
    for model_name, results in model_results.items():
        row = {'Model': model_name}
        
        # 基本指标
        if 'test_loss' in results:
            row['Test Loss'] = results['test_loss']
        if 'accuracy' in results:
            row['Accuracy'] = results['accuracy']
        if 'precision' in results:
            row['Precision'] = results['precision']
        if 'recall' in results:
            row['Recall'] = results['recall']
        if 'f1_score' in results:
            row['F1 Score'] = results['f1_score']
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估Brain-Inspired AI模型')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径')
    parser.add_argument('--test_data', type=str, required=True, help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    
    args = parser.parse_args()
    
    # 示例用法
    print("评估脚本需要根据具体数据格式进行实现")
    print("请在集成时实现具体的数据加载逻辑")
    
    # 伪代码示例:
    # test_dataset = YourDataset(args.test_data)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    # 
    # results = evaluate_model(
    #     model_path=args.model_path,
    #     test_loader=test_loader,
    #     device=args.device,
    #     results_dir=args.output_dir
    # )