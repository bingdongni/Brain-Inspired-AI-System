"""
预测性早退算法
基于机器学习的智能路由提前退出机制
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class EarlyExitMetrics:
    """早退指标"""
    confidence: float
    accuracy_estimate: float
    latency_saved: float
    energy_saved: float
    correctness_probability: float


class EarlyExitPredictor(nn.Module):
    """早退预测网络"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3)  # confidence, accuracy, early_exit
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        outputs = self.fc3(x)
        
        # 输出归一化
        confidence = self.sigmoid(outputs[:, 0])
        accuracy = self.sigmoid(outputs[:, 1])
        early_exit_prob = self.sigmoid(outputs[:, 2])
        
        return confidence, accuracy, early_exit_prob


class PredictiveEarlyExit:
    """预测性早退路由器"""
    
    def __init__(self,
                 num_modules: int = 8,
                 state_dim: int = 32,
                 confidence_threshold: float = 0.85,
                 learning_rate: float = 1e-3,
                 device: str = 'cpu'):
        
        self.num_modules = num_modules
        self.state_dim = state_dim
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # 早退预测网络
        self.predictor = EarlyExitPredictor(state_dim).to(device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=learning_rate)
        
        # 性能记录
        self.performance_history = deque(maxlen=10000)
        self.prediction_history = deque(maxlen=10000)
        
        # 统计信息
        self.total_requests = 0
        self.early_exits = 0
        self.correct_predictions = 0
        self.total_latency_saved = 0.0
        self.total_energy_saved = 0.0
        
        # 模块信息
        self.modules = self._initialize_modules()
        
        # 预测器参数
        self.prediction_cooldown = 10  # 预测冷却时间
        self.last_prediction_step = 0
        
    def _initialize_modules(self) -> List[Dict]:
        """初始化模块信息"""
        modules = []
        for i in range(self.num_modules):
            modules.append({
                'id': f"module_{i}",
                'base_latency': np.random.uniform(0.1, 1.0),
                'base_energy': np.random.uniform(1.0, 5.0),
                'processing_capacity': np.random.uniform(0.8, 1.2),
                'current_load': 0.0,
                'accuracy_model': np.random.uniform(0.85, 0.98)
            })
        return modules
    
    def should_early_exit(self, 
                         state: np.ndarray,
                         current_module_idx: int,
                         step: int) -> Tuple[bool, EarlyExitMetrics]:
        """判断是否应该早退"""
        # 检查冷却时间
        if step - self.last_prediction_step < self.prediction_cooldown:
            return False, EarlyExitMetrics(0, 0, 0, 0, 0)
        
        # 生成预测
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            confidence, accuracy, early_exit_prob = self.predictor(state_tensor)
        
        confidence_val = confidence.item()
        accuracy_val = accuracy.item()
        early_exit_val = early_exit_prob.item()
        
        # 记录预测
        self.prediction_history.append({
            'step': step,
            'confidence': confidence_val,
            'accuracy': accuracy_val,
            'early_exit_prob': early_exit_val,
            'module_idx': current_module_idx
        })
        
        self.last_prediction_step = step
        
        # 判断早退条件
        should_exit = (confidence_val >= self.confidence_threshold and 
                      early_exit_val >= 0.5 and
                      self._evaluate_exit_benefit(state, current_module_idx))
        
        # 计算预期节省
        if should_exit:
            latency_saved = self._calculate_latency_saved(current_module_idx)
            energy_saved = self._calculate_energy_saved(current_module_idx)
            correctness_prob = min(confidence_val, accuracy_val)
        else:
            latency_saved = 0
            energy_saved = 0
            correctness_prob = 0
        
        metrics = EarlyExitMetrics(
            confidence=confidence_val,
            accuracy_estimate=accuracy_val,
            latency_saved=latency_saved,
            energy_saved=energy_saved,
            correctness_probability=correctness_prob
        )
        
        return should_exit, metrics
    
    def _evaluate_exit_benefit(self, state: np.ndarray, module_idx: int) -> bool:
        """评估早退的效益"""
        # 检查模块状态
        module = self.modules[module_idx]
        
        # 如果模块负载很高，早退更有利
        high_load_bonus = 1.0 if module['current_load'] > 0.8 else 0.5
        
        # 检查历史性能
        recent_performance = self._get_recent_performance(module_idx)
        if recent_performance > 0.9:  # 最近性能很好，可以早退
            performance_bonus = 1.2
        elif recent_performance > 0.7:
            performance_bonus = 1.0
        else:
            performance_bonus = 0.8
        
        # 计算综合效益
        benefit_score = high_load_bonus * performance_bonus
        
        return benefit_score > 1.0
    
    def _calculate_latency_saved(self, module_idx: int) -> float:
        """计算延迟节省"""
        module = self.modules[module_idx]
        
        # 基于当前负载和基础延迟计算节省
        load_factor = module['current_load']
        base_latency = module['base_latency']
        
        # 负载越高，早退节省越多
        saved_latency = base_latency * load_factor * 0.5
        return min(saved_latency, base_latency * 0.8)  # 最多节省80%
    
    def _calculate_energy_saved(self, module_idx: int) -> float:
        """计算能耗节省"""
        module = self.modules[module_idx]
        
        # 基于负载和基础能耗计算节省
        load_factor = module['current_load']
        base_energy = module['base_energy']
        
        # 早退节省的能量
        saved_energy = base_energy * load_factor * 0.6
        return min(saved_energy, base_energy * 0.7)  # 最多节省70%
    
    def _get_recent_performance(self, module_idx: int, window: int = 50) -> float:
        """获取最近性能"""
        recent_entries = [entry for entry in self.performance_history 
                         if entry['module_idx'] == module_idx][-window:]
        
        if not recent_entries:
            return 0.5  # 默认性能
        
        success_rate = np.mean([entry['success'] for entry in recent_entries])
        return success_rate
    
    def update_module_load(self, module_idx: int, load_change: float):
        """更新模块负载"""
        if 0 <= module_idx < len(self.modules):
            self.modules[module_idx]['current_load'] = max(0, min(1, 
                self.modules[module_idx]['current_load'] + load_change))
    
    def record_performance(self, 
                          module_idx: int,
                          actual_accuracy: float,
                          actual_latency: float,
                          actual_energy: float,
                          success: bool,
                          step: int):
        """记录实际性能"""
        # 查找对应的预测记录
        matching_prediction = None
        for pred in reversed(self.prediction_history):
            if pred['step'] <= step and pred['module_idx'] == module_idx:
                matching_prediction = pred
                break
        
        performance_entry = {
            'module_idx': module_idx,
            'step': step,
            'actual_accuracy': actual_accuracy,
            'actual_latency': actual_latency,
            'actual_energy': actual_energy,
            'success': success,
            'prediction': matching_prediction
        }
        
        self.performance_history.append(performance_entry)
        
        # 更新统计
        if success:
            self.correct_predictions += 1
        
        if matching_prediction:
            # 累加节省
            self.total_latency_saved += self._calculate_latency_saved(module_idx)
            self.total_energy_saved += self._calculate_energy_saved(module_idx)
    
    def train_predictor(self, batch_size: int = 32):
        """训练预测器"""
        if len(self.performance_history) < batch_size:
            return 0.0
        
        # 采样训练数据
        recent_history = list(self.performance_history)[-batch_size:]
        
        # 准备训练数据
        states = []
        targets = []
        
        for entry in recent_history:
            if entry['prediction']:
                # 简化状态构建
                state = np.random.randn(self.state_dim)  # 实际中需要真实状态
                target_confidence = min(1.0, entry['actual_accuracy'])
                target_accuracy = entry['actual_accuracy']
                target_early_exit = 1.0 if entry['success'] else 0.0
                
                states.append(state)
                targets.append([target_confidence, target_accuracy, target_early_exit])
        
        if not states:
            return 0.0
        
        # 转换为张量
        states_tensor = torch.FloatTensor(states).to(self.device)
        targets_tensor = torch.FloatTensor(targets).to(self.device)
        
        # 前向传播
        confidence_pred, accuracy_pred, early_exit_pred = self.predictor(states_tensor)
        
        # 计算损失
        loss_confidence = nn.MSELoss()(confidence_pred, targets_tensor[:, 0])
        loss_accuracy = nn.MSELoss()(accuracy_pred, targets_tensor[:, 1])
        loss_early_exit = nn.BCELoss()(early_exit_pred, targets_tensor[:, 2])
        
        total_loss = loss_confidence + loss_accuracy + loss_early_exit
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
    
    def get_early_exit_statistics(self) -> Dict:
        """获取早退统计信息"""
        if self.total_requests == 0:
            return {
                'total_requests': 0,
                'early_exit_rate': 0.0,
                'prediction_accuracy': 0.0,
                'avg_latency_saved': 0.0,
                'avg_energy_saved': 0.0
            }
        
        early_exit_rate = self.early_exits / self.total_requests
        prediction_accuracy = self.correct_predictions / self.total_requests
        avg_latency_saved = self.total_latency_saved / max(self.early_exits, 1)
        avg_energy_saved = self.total_energy_saved / max(self.early_exits, 1)
        
        return {
            'total_requests': self.total_requests,
            'early_exits': self.early_exits,
            'early_exit_rate': early_exit_rate,
            'prediction_accuracy': prediction_accuracy,
            'total_latency_saved': self.total_latency_saved,
            'total_energy_saved': self.total_energy_saved,
            'avg_latency_saved': avg_latency_saved,
            'avg_energy_saved': avg_energy_saved,
            'confidence_threshold': self.confidence_threshold
        }
    
    def get_module_early_exit_analysis(self, module_idx: int) -> Dict:
        """获取模块的早退分析"""
        if module_idx >= len(self.modules):
            return {}
        
        module = self.modules[module_idx]
        
        # 统计该模块的早退情况
        module_performance = [p for p in self.performance_history 
                            if p['module_idx'] == module_idx]
        module_predictions = [p for p in self.prediction_history 
                            if p['module_idx'] == module_idx]
        
        if not module_performance:
            return {'error': 'No performance data available'}
        
        # 计算分析指标
        success_rate = np.mean([p['success'] for p in module_performance])
        early_exit_count = sum(1 for p in module_performance if p['success'])
        early_exit_rate_module = early_exit_count / len(module_performance)
        
        # 性能趋势
        recent_performance = module_performance[-10:] if len(module_performance) >= 10 else module_performance
        recent_accuracy = np.mean([p['actual_accuracy'] for p in recent_performance])
        
        # 预测准确性
        if module_predictions:
            avg_confidence = np.mean([p['confidence'] for p in module_predictions])
            avg_predicted_accuracy = np.mean([p['accuracy'] for p in module_predictions])
        else:
            avg_confidence = 0.0
            avg_predicted_accuracy = 0.0
        
        return {
            'module_id': module['id'],
            'current_load': module['current_load'],
            'base_latency': module['base_latency'],
            'base_energy': module['base_energy'],
            'success_rate': success_rate,
            'early_exit_rate': early_exit_rate_module,
            'early_exit_count': early_exit_count,
            'recent_accuracy': recent_accuracy,
            'avg_confidence': avg_confidence,
            'avg_predicted_accuracy': avg_predicted_accuracy,
            'performance_entries': len(module_performance),
            'prediction_entries': len(module_predictions)
        }
    
    def adjust_confidence_threshold(self, target_early_exit_rate: float = 0.2):
        """动态调整置信度阈值"""
        current_rate = self.early_exits / max(self.total_requests, 1)
        
        # 如果早退率过高，提高阈值；如果过低，降低阈值
        adjustment = 0.05  # 调整步长
        max_adjustment = 0.3
        
        if current_rate > target_early_exit_rate * 1.5:
            self.confidence_threshold = min(0.95, self.confidence_threshold + adjustment)
        elif current_rate < target_early_exit_rate * 0.5:
            self.confidence_threshold = max(0.5, self.confidence_threshold - adjustment)
        
        # 限制调整范围
        self.confidence_threshold = max(0.5, min(0.95, self.confidence_threshold))
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'confidence_threshold': self.confidence_threshold,
            'modules': self.modules,
            'statistics': self.get_early_exit_statistics()
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.confidence_threshold = checkpoint['confidence_threshold']
        self.modules = checkpoint['modules']