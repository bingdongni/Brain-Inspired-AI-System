"""
经验重放模块

实现多种经验重放策略，包括优先重放、分类重放、压缩重放和生成式重放。
支持不同类型的数据和任务，用于持续学习中的经验回放。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from collections import deque, defaultdict
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """
    经验数据结构
    """
    state: torch.Tensor
    action: Optional[torch.Tensor] = None
    reward: Optional[torch.Tensor] = None
    next_state: Optional[torch.Tensor] = None
    done: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    importance: float = 1.0
    age: int = 0
    task_id: Optional[int] = None


class BaseReplayStrategy(ABC):
    """
    重放策略基类
    """
    
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
    
    @abstractmethod
    def add_experience(self, experience: Experience) -> None:
        """添加经验"""
        pass
    
    @abstractmethod
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """采样批量经验"""
        pass
    
    @abstractmethod
    def update_priorities(self, experiences: List[Experience], priorities: List[float]) -> None:
        """更新优先级"""
        pass


class UniformReplay(BaseReplayStrategy):
    """
    均匀重放策略
    
    简单的均匀采样策略，所有经验具有相等重要性。
    """
    
    def __init__(self, buffer_size: int):
        super().__init__(buffer_size)
        self.buffer = deque(maxlen=buffer_size)
    
    def add_experience(self, experience: Experience) -> None:
        """添加经验"""
        self.buffer.append(experience)
    
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """采样批量经验"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        return random.sample(list(self.buffer), batch_size)
    
    def update_priorities(self, experiences: List[Experience], priorities: List[float]) -> None:
        """均匀重放不需要更新优先级"""
        pass


class PrioritizedReplay(BaseReplayStrategy):
    """
    优先重放策略
    
    基于TD误差和重要性的优先重放，重要经验被更频繁地采样。
    """
    
    def __init__(self, buffer_size: int, alpha: float = 0.6, beta: float = 0.4, epsilon: float = 1e-6):
        """
        初始化优先重放
        
        Args:
            buffer_size: 缓冲区大小
            alpha: 优先级强度参数
            beta: 重要性采样校正参数
            epsilon: 优先级偏置
        """
        super().__init__(buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
        self.buffer = []
        self.priorities = []
        self.max_priority = 1.0
    
    def add_experience(self, experience: Experience) -> None:
        """添加经验"""
        if len(self.buffer) >= self.buffer_size:
            # 移除优先级最低的经验
            min_idx = np.argmin(self.priorities)
            self.buffer.pop(min_idx)
            self.priorities.pop(min_idx)
        
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """采样批量经验"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # 计算采样概率
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 更新经验的权重
        sampled_experiences = [self.buffer[i] for i in indices]
        for exp, weight in zip(sampled_experiences, weights):
            exp.importance = weight.item()
        
        return sampled_experiences
    
    def update_priorities(self, experiences: List[Experience], priorities: List[float]) -> None:
        """更新优先级"""
        for experience, priority in zip(experiences, priorities):
            if experience in self.buffer:
                idx = self.buffer.index(experience)
                self.priorities[idx] = priority + self.epsilon
                self.max_priority = max(self.max_priority, self.priorities[idx])


class CategoricalReplay(BaseReplayStrategy):
    """
    分类重放策略
    
    基于任务或类别的分类重放，确保每个类别都有代表性样本。
    """
    
    def __init__(self, buffer_size: int, num_categories: int, category_key: str = 'task_id'):
        """
        初始化分类重放
        
        Args:
            buffer_size: 缓冲区大小
            num_categories: 类别数量
            category_key: 用于分类的键名
        """
        super().__init__(buffer_size)
        self.num_categories = num_categories
        self.category_key = category_key
        
        # 每个类别的缓冲区
        self.category_buffers = {i: deque(maxlen=buffer_size // num_categories) 
                               for i in range(num_categories)}
        self.category_priorities = {i: [] for i in range(num_categories)}
    
    def add_experience(self, experience: Experience) -> None:
        """添加经验"""
        category = getattr(experience, self.category_key, 0)
        if category >= self.num_categories:
            category = 0
        
        self.category_buffers[category].append(experience)
        if len(self.category_priorities[category]) >= self.category_buffers[category].maxlen:
            self.category_priorities[category].pop(0)
        self.category_priorities[category].append(experience.importance)
    
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """采样批量经验"""
        experiences = []
        
        # 每个类别采样数量
        samples_per_category = max(1, batch_size // self.num_categories)
        
        for category in range(self.num_categories):
            if len(self.category_buffers[category]) > 0:
                num_samples = min(samples_per_category, 
                                len(self.category_buffers[category]),
                                batch_size - len(experiences))
                
                category_samples = random.sample(
                    list(self.category_buffers[category]), num_samples)
                experiences.extend(category_samples)
        
        # 如果不足，补充采样
        while len(experiences) < batch_size:
            all_experiences = []
            for category in self.category_buffers.values():
                all_experiences.extend(category)
            
            if not all_experiences:
                break
            
            experiences.append(random.choice(all_experiences))
        
        return experiences[:batch_size]
    
    def update_priorities(self, experiences: List[Experience], priorities: List[float]) -> None:
        """更新优先级"""
        for experience, priority in zip(experiences, priorities):
            for category, buffer in self.category_buffers.items():
                if experience in buffer:
                    idx = list(buffer).index(experience)
                    if idx < len(self.category_priorities[category]):
                        self.category_priorities[category][idx] = priority
                    break


class CompressedReplay(BaseReplayStrategy):
    """
    压缩重放策略
    
    使用压缩表示存储经验，减少存储开销。
    """
    
    def __init__(self, buffer_size: int, compression_ratio: float = 0.1):
        """
        初始化压缩重放
        
        Args:
            buffer_size: 缓冲区大小
            compression_ratio: 压缩比例
        """
        super().__init__(buffer_size)
        self.compression_ratio = compression_ratio
        
        self.buffer = deque(maxlen=buffer_size)
        self.compressed_buffer = deque(maxlen=int(buffer_size * compression_ratio))
        self.compression_stats = defaultdict(int)
    
    def add_experience(self, experience: Experience) -> None:
        """添加经验"""
        # 正常添加
        self.buffer.append(experience)
        
        # 压缩概率
        if random.random() < self.compression_ratio:
            compressed_exp = self._compress_experience(experience)
            self.compressed_buffer.append(compressed_exp)
    
    def _compress_experience(self, experience: Experience) -> Experience:
        """压缩经验"""
        compressed_exp = Experience(
            state=self._compress_tensor(experience.state),
            action=experience.action,
            reward=experience.reward,
            next_state=experience.next_state,
            done=experience.done,
            label=experience.label,
            metadata=experience.metadata,
            importance=experience.importance,
            task_id=experience.task_id
        )
        
        self.compression_stats['compressed'] += 1
        return compressed_exp
    
    def _compress_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """压缩张量"""
        if tensor.numel() < 100:  # 小张量不压缩
            return tensor
        
        # 使用主成分分析进行压缩（简化版本）
        # 这里实现一个简单的降采样
        original_shape = tensor.shape
        
        if len(original_shape) >= 3:  # 图像数据
            # 简单的池化压缩
            kernel_size = max(1, min(original_shape[-2:]) // 4)
            if kernel_size > 1:
                # 调整张量维度用于池化
                if len(original_shape) == 4:  # NCHW
                    tensor = tensor.mean(dim=(-1, -2), keepdim=True)
                elif len(original_shape) == 3:  # CHW
                    tensor = tensor.mean(dim=(-1, -2), keepdim=True)
        
        return tensor
    
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """采样批量经验"""
        experiences = []
        
        # 从原始缓冲区采样
        if len(self.buffer) > 0:
            original_samples = min(batch_size // 2, len(self.buffer))
            experiences.extend(random.sample(list(self.buffer), original_samples))
        
        # 从压缩缓冲区采样
        if len(self.compressed_buffer) > 0 and len(experiences) < batch_size:
            compressed_samples = min(batch_size - len(experiences), len(self.compressed_buffer))
            experiences.extend(random.sample(list(self.compressed_buffer), compressed_samples))
        
        return experiences[:batch_size]
    
    def update_priorities(self, experiences: List[Experience], priorities: List[float]) -> None:
        """更新优先级"""
        for experience, priority in zip(experiences, priorities):
            experience.importance = priority
    
    def get_compression_stats(self) -> Dict[str, int]:
        """获取压缩统计信息"""
        return dict(self.compression_stats)


class GenerativeReplay(BaseReplayStrategy):
    """
    生成式重放策略
    
    结合生成模型和真实数据的混合重放。
    """
    
    def __init__(self, buffer_size: int, generator, real_ratio: float = 0.7):
        """
        初始化生成式重放
        
        Args:
            buffer_size: 缓冲区大小
            generator: 生成模型
            real_ratio: 真实数据比例
        """
        super().__init__(buffer_size)
        self.generator = generator
        self.real_ratio = real_ratio
        
        self.real_buffer = deque(maxlen=int(buffer_size * real_ratio))
        self.synthetic_buffer = deque(maxlen=int(buffer_size * (1 - real_ratio)))
    
    def add_experience(self, experience: Experience, is_synthetic: bool = False) -> None:
        """添加经验"""
        if is_synthetic:
            self.synthetic_buffer.append(experience)
        else:
            self.real_buffer.append(experience)
    
    def generate_synthetic_data(self, 
                              num_samples: int, 
                              task_id: Optional[int] = None,
                              labels: Optional[torch.Tensor] = None) -> List[Experience]:
        """生成合成数据"""
        synthetic_experiences = []
        
        with torch.no_grad():
            generated_samples = self.generator.generate_samples(
                num_samples=num_samples,
                labels=labels
            )
        
        for i, sample in enumerate(generated_samples):
            synthetic_exp = Experience(
                state=sample,
                label=labels[i] if labels is not None else None,
                task_id=task_id,
                metadata={'synthetic': True}
            )
            synthetic_experiences.append(synthetic_exp)
            self.add_experience(synthetic_exp, is_synthetic=True)
        
        return synthetic_experiences
    
    def sample_batch(self, batch_size: int) -> List[Experience]:
        """采样批量经验"""
        experiences = []
        
        # 确定真实和合成样本数量
        real_samples = int(batch_size * self.real_ratio)
        synthetic_samples = batch_size - real_samples
        
        # 采样真实数据
        if len(self.real_buffer) > 0:
            num_real = min(real_samples, len(self.real_buffer))
            experiences.extend(random.sample(list(self.real_buffer), num_real))
        
        # 采样合成数据
        if len(self.synthetic_buffer) > 0:
            num_synthetic = min(synthetic_samples, len(self.synthetic_buffer))
            experiences.extend(random.sample(list(self.synthetic_buffer), num_synthetic))
        
        # 如果还是不够，生成更多合成数据
        while len(experiences) < batch_size:
            synthetic_exp = self.generate_synthetic_data(1)[0]
            experiences.append(synthetic_exp)
        
        return experiences[:batch_size]
    
    def update_priorities(self, experiences: List[Experience], priorities: List[float]) -> None:
        """更新优先级"""
        for experience, priority in zip(experiences, priorities):
            experience.importance = priority


class ExperienceReplayBuffer:
    """
    综合经验重放缓冲区
    
    支持多种重放策略的组合使用。
    """
    
    def __init__(self,
                 buffer_size: int,
                 strategies: Optional[List[str]] = None,
                 strategy_configs: Optional[Dict[str, Dict]] = None):
        """
        初始化重放缓冲区
        
        Args:
            buffer_size: 缓冲区大小
            strategies: 重放策略列表
            strategy_configs: 策略配置字典
        """
        self.buffer_size = buffer_size
        self.strategies = strategies or ['uniform']
        self.strategy_configs = strategy_configs or {}
        
        # 创建策略实例
        self.replay_strategies = {}
        for strategy_name in self.strategies:
            config = self.strategy_configs.get(strategy_name, {})
            
            if strategy_name == 'uniform':
                self.replay_strategies[strategy_name] = UniformReplay(buffer_size)
            elif strategy_name == 'prioritized':
                self.replay_strategies[strategy_name] = PrioritizedReplay(buffer_size, **config)
            elif strategy_name == 'categorical':
                self.replay_strategies[strategy_name] = CategoricalReplay(buffer_size, **config)
            elif strategy_name == 'compressed':
                self.replay_strategies[strategy_name] = CompressedReplay(buffer_size, **config)
            elif strategy_name == 'generative':
                self.replay_strategies[strategy_name] = GenerativeReplay(buffer_size, **config)
            else:
                logger.warning(f"未知策略: {strategy_name}，使用均匀重放")
                self.replay_strategies[strategy_name] = UniformReplay(buffer_size)
        
        # 选择主要策略
        self.main_strategy = list(self.replay_strategies.keys())[0]
        
        # 统计信息
        self.add_count = 0
        self.sample_count = 0
    
    def add_experience(self, 
                      experience: Experience,
                      strategy: Optional[str] = None) -> None:
        """添加经验"""
        strategy = strategy or self.main_strategy
        
        if strategy in self.replay_strategies:
            self.replay_strategies[strategy].add_experience(experience)
            self.add_count += 1
        else:
            logger.warning(f"策略 {strategy} 不存在，使用默认策略")
            self.replay_strategies[self.main_strategy].add_experience(experience)
    
    def add_experiences(self,
                       experiences: List[Experience],
                       strategy: Optional[str] = None) -> None:
        """批量添加经验"""
        for experience in experiences:
            self.add_experience(experience, strategy)
    
    def sample_batch(self, 
                    batch_size: int,
                    strategy: Optional[str] = None) -> List[Experience]:
        """采样批量经验"""
        strategy = strategy or self.main_strategy
        
        if strategy in self.replay_strategies:
            experiences = self.replay_strategies[strategy].sample_batch(batch_size)
            self.sample_count += 1
            return experiences
        else:
            logger.warning(f"策略 {strategy} 不存在，使用默认策略")
            return self.replay_strategies[self.main_strategy].sample_batch(batch_size)
    
    def update_priorities(self, 
                         experiences: List[Experience],
                         priorities: List[float],
                         strategy: Optional[str] = None) -> None:
        """更新优先级"""
        strategy = strategy or self.main_strategy
        
        if strategy in self.replay_strategies:
            if hasattr(self.replay_strategies[strategy], 'update_priorities'):
                self.replay_strategies[strategy].update_priorities(experiences, priorities)
        else:
            logger.warning(f"策略 {strategy} 不支持优先级更新")
    
    def get_buffer_size(self, strategy: Optional[str] = None) -> int:
        """获取缓冲区大小"""
        strategy = strategy or self.main_strategy
        
        if strategy in self.replay_strategies:
            if hasattr(self.replay_strategies[strategy], 'buffer'):
                return len(self.replay_strategies[strategy].buffer)
            elif hasattr(self.replay_strategies[strategy], 'category_buffers'):
                return sum(len(buf) for buf in self.replay_strategies[strategy].category_buffers.values())
        
        return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        stats = {
            'add_count': self.add_count,
            'sample_count': self.sample_count,
            'main_strategy': self.main_strategy,
            'buffer_size': self.get_buffer_size(),
            'strategies': list(self.replay_strategies.keys())
        }
        
        # 添加各策略的统计信息
        for name, strategy in self.replay_strategies.items():
            if hasattr(strategy, 'get_compression_stats'):
                stats[f'{name}_compression'] = strategy.get_compression_stats()
        
        return stats
    
    def clear(self, strategy: Optional[str] = None) -> None:
        """清空缓冲区"""
        strategy = strategy or self.main_strategy
        
        if strategy in self.replay_strategies:
            if hasattr(strategy, 'buffer'):
                strategy.buffer.clear()
            elif hasattr(strategy, 'category_buffers'):
                for buf in strategy.category_buffers.values():
                    buf.clear()


class ValueBasedReplay(BaseReplayStrategy):
    """
    基于价值的重放策略
    
    基于经验的TD误差或奖励值进行重放。
    """
    
    def __init__(self, buffer_size: int, value_threshold: float = 0.1):
        super().__init__(buffer_size)
        self.value_threshold = value_threshold
        self.buffer = deque(maxlen=buffer_size)
        self.values = deque(maxlen=buffer_size)
    
    def add_experience(self, experience: Experience) -> None:
        value = abs(experience.reward.item() if experience.reward is not None else 0.0)
        self.buffer.append(experience)
        self.values.append(value)
    
    def sample_batch(self, batch_size: int) -> List[Experience]:
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # 基于价值概率采样
        values = np.array(self.values)
        probs = np.maximum(values, self.value_threshold)
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[i] for i in indices]
    
    def update_priorities(self, experiences: List[Experience], priorities: List[float]) -> None:
        for experience, priority in zip(experiences, priorities):
            if experience in self.buffer:
                idx = self.buffer.index(experience)
                self.values[idx] = priority