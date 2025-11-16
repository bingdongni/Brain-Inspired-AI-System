"""
巩固引擎
实现记忆从海马体向新皮层的巩固过程
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import asyncio
import time
from dataclasses import dataclass, field
from collections import deque, defaultdict

from ..attention_mechanism import AttentionMemory


@dataclass
class ConsolidationConfig:
    """巩固配置"""
    consolidation_threshold: float = 0.7
    replay_frequency: float = 0.1  # 每秒重放概率
    strengthening_rate: float = 0.05
    forgetting_rate: float = 0.02
    sequence_length: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    temporal_scaling: float = 1.0
    pattern_associations: bool = True


@dataclass
class MemoryTrace:
    """记忆痕迹"""
    content: torch.Tensor
    timestamp: float
    importance: float
    access_count: int
    associations: List[str] = field(default_factory=list)
    consolidation_stage: int = 0  # 0=新记忆, 1=部分巩固, 2=完全巩固
    strength: float = 1.0


class ConsolidationEngine:
    """
    记忆巩固引擎
    管理从海马体向新皮层的记忆巩固过程
    """
    
    def __init__(self, config: ConsolidationConfig):
        self.config = config
        
        # 记忆存储
        self.episodic_traces = {}  # 情景记忆痕迹
        self.consolidated_memories = {}  # 已巩固记忆
        self.association_graph = defaultdict(set)  # 关联图
        
        # 巩固网络
        self.consolidation_network = ConsolidationNetwork(config)
        
        # 重放系统
        self.replay_buffer = deque(maxlen=10000)
        self.replay_priorities = {}
        
        # 统计信息
        self.stats = {
            'total_consolidations': 0,
            'successful_consolidations': 0,
            'replay_events': 0,
            'association_formations': 0,
            'strengthening_events': 0,
            'forgetting_events': 0,
            'average_consolidation_time': 0.0
        }
        
        # 当前时间步
        self.current_time_step = 0
        
        # 巩固调度
        self.consolidation_schedule = defaultdict(list)
        self.priority_queue = []
        
    async def initialize(self):
        """初始化巩固引擎"""
        print("巩固引擎初始化完成")
    
    async def register_episodic_memory(
        self,
        memory_id: str,
        content: torch.Tensor,
        importance: float = 0.5,
        context: torch.Tensor = None,
        associations: List[str] = None
    ) -> bool:
        """
        注册情景记忆
        
        Args:
            memory_id: 记忆ID
            content: 记忆内容
            importance: 重要性
            context: 上下文
            associations: 关联ID列表
            
        Returns:
            注册是否成功
        """
        try:
            memory_trace = MemoryTrace(
                content=content.clone(),
                timestamp=self.current_time_step,
                importance=importance,
                access_count=0,
                associations=associations or []
            )
            
            self.episodic_traces[memory_id] = memory_trace
            
            # 记录关联
            if associations:
                for assoc_id in associations:
                    self.association_graph[memory_id].add(assoc_id)
                    self.association_graph[assoc_id].add(memory_id)
                self.stats['association_formations'] += 1
            
            # 加入巩固调度
            self._schedule_consolidation(memory_id, importance)
            
            # 如果是高重要性记忆，立即开始巩固
            if importance >= self.config.consolidation_threshold:
                await self._initiate_consolidation(memory_id)
            
            return True
            
        except Exception as e:
            print(f"注册情景记忆失败: {e}")
            return False
    
    async def consolidate_memory(
        self,
        memory_id: str,
        consolidation_type: str = "standard"
    ) -> Dict:
        """
        执行记忆巩固
        
        Args:
            memory_id: 记忆ID
            consolidation_type: 巩固类型
            
        Returns:
            巩固结果
        """
        start_time = time.time()
        
        try:
            if memory_id not in self.episodic_traces:
                return {'error': f'记忆 {memory_id} 不存在'}
            
            memory_trace = self.episodic_traces[memory_id]
            
            # 检查是否满足巩固条件
            if not self._check_consolidation_criteria(memory_trace):
                return {'status': 'not_ready_for_consolidation'}
            
            # 执行巩固
            consolidation_result = await self._perform_consolidation(
                memory_trace, consolidation_type
            )
            
            # 更新记忆状态
            memory_trace.consolidation_stage = min(memory_trace.consolidation_stage + 1, 2)
            memory_trace.strength *= (1 + self.config.strengthening_rate)
            
            # 如果完全巩固，移动到巩固记忆库
            if memory_trace.consolidation_stage >= 2:
                self._move_to_consolidated(memory_id, memory_trace)
            
            consolidation_time = time.time() - start_time
            
            # 更新统计
            self.stats['total_consolidations'] += 1
            self.stats['successful_consolidations'] += 1
            self._update_average_consolidation_time(consolidation_time)
            
            return {
                'success': True,
                'memory_id': memory_id,
                'consolidation_stage': memory_trace.consolidation_stage,
                'strength': memory_trace.strength,
                'consolidation_time': consolidation_time,
                'result': consolidation_result
            }
            
        except Exception as e:
            print(f"记忆巩固失败: {e}")
            self.stats['total_consolidations'] += 1
            return {'error': str(e)}
    
    async def replay_memory(
        self,
        memory_id: str,
        replay_mode: str = "pattern_completion"
    ) -> Dict:
        """
        重放记忆
        
        Args:
            memory_id: 记忆ID
            replay_mode: 重放模式
            
        Returns:
            重放结果
        """
        try:
            if memory_id not in self.episodic_traces and memory_id not in self.consolidated_memories:
                return {'error': f'记忆 {memory_id} 不存在'}
            
            # 获取记忆内容
            if memory_id in self.episodic_traces:
                memory_content = self.episodic_traces[memory_id].content
            else:
                memory_content = self.consolidated_memories[memory_id]['content']
            
            # 执行重放
            if replay_mode == "pattern_completion":
                replayed_content = await self._pattern_completion_replay(memory_content)
            elif replay_mode == "sequence_replay":
                replayed_content = await self._sequence_replay(memory_id, memory_content)
            elif replay_mode == "association_replay":
                replayed_content = await self._association_replay(memory_id, memory_content)
            else:
                replayed_content = memory_content
            
            # 加强记忆
            await self._strengthen_memory(memory_id)
            
            # 更新重放优先级
            self._update_replay_priority(memory_id, replayed_content)
            
            # 添加到重放缓冲区
            self.replay_buffer.append({
                'memory_id': memory_id,
                'content': replayed_content,
                'timestamp': self.current_time_step,
                'replay_mode': replay_mode
            })
            
            self.stats['replay_events'] += 1
            
            return {
                'success': True,
                'memory_id': memory_id,
                'replay_mode': replay_mode,
                'replayed_content': replayed_content,
                'strength': self._get_memory_strength(memory_id)
            }
            
        except Exception as e:
            print(f"记忆重放失败: {e}")
            return {'error': str(e)}
    
    async def batch_consolidation(
        self,
        time_window: int = 100,
        batch_size: int = None
    ) -> Dict:
        """
        批量巩固
        
        Args:
            time_window: 时间窗口
            batch_size: 批处理大小
            
        Returns:
            批量巩固结果
        """
        batch_size = batch_size or self.config.batch_size
        current_time = self.current_time_step
        
        # 选择需要巩固的记忆
        candidates = []
        for memory_id, trace in self.episodic_traces.items():
            if (trace.consolidation_stage < 2 and
                current_time - trace.timestamp <= time_window and
                trace.importance >= self.config.consolidation_threshold):
                candidates.append((memory_id, trace.importance))
        
        # 按重要性排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        candidates = candidates[:batch_size]
        
        # 执行批量巩固
        results = []
        for memory_id, _ in candidates:
            result = await self.consolidate_memory(memory_id)
            results.append(result)
        
        # 统计结果
        successful = sum(1 for r in results if r.get('success', False))
        
        return {
            'total_candidates': len(candidates),
            'successful_consolidations': successful,
            'consolidation_results': results,
            'time_window': time_window,
            'batch_size': batch_size
        }
    
    async def automatic_consolidation(self) -> Dict:
        """
        自动巩固调度
        
        Returns:
            自动巩固结果
        """
        try:
            consolidation_events = []
            
            # 检查需要巩固的记忆
            current_time = self.current_time_step
            
            for memory_id, trace in self.episodic_traces.items():
                if trace.consolidation_stage < 2:
                    # 计算巩固时机
                    consolidation_likelihood = self._calculate_consolidation_likelihood(trace)
                    
                    # 如果满足条件，执行巩固
                    if np.random.random() < consolidation_likelihood * self.config.replay_frequency:
                        result = await self.consolidate_memory(memory_id)
                        consolidation_events.append(result)
            
            return {
                'consolidation_events': consolidation_events,
                'total_events': len(consolidation_events),
                'timestamp': current_time
            }
            
        except Exception as e:
            print(f"自动巩固失败: {e}")
            return {'error': str(e)}
    
    def get_consolidation_status(self) -> Dict:
        """获取巩固状态"""
        return {
            'episodic_memory_count': len(self.episodic_traces),
            'consolidated_memory_count': len(self.consolidated_memories),
            'consolidation_schedule_size': sum(len(schedule) for schedule in self.consolidation_schedule.values()),
            'replay_buffer_size': len(self.replay_buffer),
            'association_graph_size': len(self.association_graph),
            'statistics': self.stats,
            'current_consolidation_stage_distribution': self._get_stage_distribution(),
            'memory_strength_distribution': self._get_strength_distribution()
        }
    
    def _schedule_consolidation(self, memory_id: str, importance: float):
        """调度巩固"""
        # 基于重要性确定巩固时间
        consolidation_delay = int((1 - importance) * 100)  # 重要性越高，延迟越短
        schedule_time = self.current_time_step + consolidation_delay
        
        self.consolidation_schedule[schedule_time].append(memory_id)
        
        # 维护优先级队列
        priority = importance * 1000 + consolidation_delay
        self.priority_queue.append((priority, memory_id, schedule_time))
        self.priority_queue.sort(key=lambda x: x[0])
    
    def _check_consolidation_criteria(self, memory_trace: MemoryTrace) -> bool:
        """检查巩固条件"""
        # 访问次数要求
        if memory_trace.access_count < 3:
            return False
        
        # 时间要求
        time_since_creation = self.current_time_step - memory_trace.timestamp
        if time_since_creation < 10:  # 至少10个时间步
            return False
        
        # 重要性要求
        if memory_trace.importance < self.config.consolidation_threshold * 0.8:
            return False
        
        return True
    
    async def _perform_consolidation(
        self,
        memory_trace: MemoryTrace,
        consolidation_type: str
    ) -> Dict:
        """执行巩固过程"""
        # 使用巩固网络处理记忆
        consolidated_representation = self.consolidation_network.consolidate(
            memory_trace.content,
            memory_trace.importance,
            memory_trace.timestamp
        )
        
        # 应用时间缩放
        if self.config.temporal_scaling != 1.0:
            time_factor = np.exp(-self.config.temporal_scaling * 
                               (self.current_time_step - memory_trace.timestamp))
            consolidated_representation *= time_factor
        
        # 存储巩固结果
        consolidated_data = {
            'original_content': memory_trace.content,
            'consolidated_representation': consolidated_representation,
            'importance': memory_trace.importance,
            'consolidation_timestamp': self.current_time_step,
            'strength': memory_trace.strength
        }
        
        # 如果是完全巩固，存储到巩固记忆库
        if memory_trace.consolidation_stage >= 1:
            self.consolidated_memories[f"consolidated_{id(memory_trace)}"] = consolidated_data
        
        return {
            'consolidated_representation': consolidated_representation,
            'strength_factor': memory_trace.strength,
            'consolidation_type': consolidation_type
        }
    
    def _move_to_consolidated(self, memory_id: str, memory_trace: MemoryTrace):
        """移动到巩固记忆库"""
        consolidated_id = f"consolidated_{memory_id}"
        
        self.consolidated_memories[consolidated_id] = {
            'original_memory_id': memory_id,
            'content': memory_trace.content,
            'importance': memory_trace.importance,
            'strength': memory_trace.strength,
            'creation_timestamp': memory_trace.timestamp,
            'consolidation_timestamp': self.current_time_step,
            'associations': list(self.association_graph.get(memory_id, set()))
        }
        
        # 从情景记忆中移除
        if memory_id in self.episodic_traces:
            del self.episodic_traces[memory_id]
    
    async def _pattern_completion_replay(self, memory_content: torch.Tensor) -> torch.Tensor:
        """模式完成重放"""
        # 基于内容的一部分预测完整模式
        partial_content = memory_content[:len(memory_content)//2]
        completed_content = self.consolidation_network.complete_pattern(partial_content)
        
        return completed_content
    
    async def _sequence_replay(self, memory_id: str, memory_content: torch.Tensor) -> torch.Tensor:
        """序列重放"""
        # 重放时间序列模式
        sequence_data = self._get_sequence_data(memory_id)
        
        if sequence_data:
            # 使用序列预测模型
            replayed_sequence = self.consolidation_network.replay_sequence(sequence_data)
            return replayed_sequence[-1]  # 返回序列的最后一个元素
        else:
            return memory_content
    
    async def _association_replay(self, memory_id: str, memory_content: torch.Tensor) -> torch.Tensor:
        """关联重放"""
        # 基于关联网络重放
        associated_memories = list(self.association_graph.get(memory_id, set()))
        
        if associated_memories:
            # 聚合关联记忆
            associated_content = torch.zeros_like(memory_content)
            for assoc_id in associated_memories:
                if assoc_id in self.episodic_traces:
                    associated_content += self.episodic_traces[assoc_id].content
            
            # 结合原始内容和关联内容
            replayed_content = 0.7 * memory_content + 0.3 * associated_content
            return replayed_content
        else:
            return memory_content
    
    async def _strengthen_memory(self, memory_id: str):
        """加强记忆"""
        if memory_id in self.episodic_traces:
            self.episodic_traces[memory_id].strength *= (1 + self.config.strengthening_rate)
            self.episodic_traces[memory_id].access_count += 1
            self.stats['strengthening_events'] += 1
    
    def _update_replay_priority(self, memory_id: str, replayed_content: torch.Tensor):
        """更新重放优先级"""
        # 基于重放内容和强度计算优先级
        if memory_id in self.episodic_traces:
            strength = self.episodic_traces[memory_id].strength
            activation = torch.norm(replayed_content).item()
            
            priority = strength * activation
            self.replay_priorities[memory_id] = priority
    
    def _get_memory_strength(self, memory_id: str) -> float:
        """获取记忆强度"""
        if memory_id in self.episodic_traces:
            return self.episodic_traces[memory_id].strength
        elif memory_id in self.consolidated_memories:
            return self.consolidated_memories[memory_id]['strength']
        else:
            return 0.0
    
    def _calculate_consolidation_likelihood(self, memory_trace: MemoryTrace) -> float:
        """计算巩固可能性"""
        # 基于重要性、访问次数、时间等因素计算
        importance_factor = memory_trace.importance
        access_factor = min(memory_trace.access_count / 10.0, 1.0)
        time_factor = np.exp(-0.01 * (self.current_time_step - memory_trace.timestamp))
        
        return importance_factor * access_factor * time_factor
    
    def _update_average_consolidation_time(self, consolidation_time: float):
        """更新平均巩固时间"""
        current_avg = self.stats['average_consolidation_time']
        total_events = self.stats['total_consolidations']
        
        if total_events > 1:
            self.stats['average_consolidation_time'] = (
                (current_avg * (total_events - 1) + consolidation_time) / total_events
            )
        else:
            self.stats['average_consolidation_time'] = consolidation_time
    
    def _get_sequence_data(self, memory_id: str) -> Optional[List[torch.Tensor]]:
        """获取序列数据"""
        # 从重放缓冲区中查找相关序列
        sequence_data = []
        
        for replay_item in list(self.replay_buffer)[-10:]:  # 最近10个重放项目
            if replay_item['memory_id'] == memory_id:
                sequence_data.append(replay_item['content'])
        
        return sequence_data if sequence_data else None
    
    def _get_stage_distribution(self) -> Dict[str, int]:
        """获取巩固阶段分布"""
        distribution = {
            'new': 0,      # 阶段0
            'partial': 0,  # 阶段1
            'complete': 0  # 阶段2
        }
        
        for trace in self.episodic_traces.values():
            if trace.consolidation_stage == 0:
                distribution['new'] += 1
            elif trace.consolidation_stage == 1:
                distribution['partial'] += 1
            elif trace.consolidation_stage == 2:
                distribution['complete'] += 1
        
        return distribution
    
    def _get_strength_distribution(self) -> Dict[str, int]:
        """获取记忆强度分布"""
        distribution = {
            'weak': 0,      # 0-0.3
            'medium': 0,    # 0.3-0.7
            'strong': 0     # 0.7-1.0
        }
        
        for trace in self.episodic_traces.values():
            strength = trace.strength
            if strength < 0.3:
                distribution['weak'] += 1
            elif strength < 0.7:
                distribution['medium'] += 1
            else:
                distribution['strong'] += 1
        
        return distribution
    
    async def _initiate_consolidation(self, memory_id: str):
        """立即开始巩固"""
        # 高重要性记忆立即开始巩固过程
        await self.consolidate_memory(memory_id)


class ConsolidationNetwork(nn.Module):
    """巩固网络"""
    
    def __init__(self, config: ConsolidationConfig):
        super().__init__()
        self.config = config
        
        # 巩固编码器
        self.encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        
        # 巩固解码器
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.Tanh()
        )
        
        # 时间处理
        self.time_processor = nn.Linear(1, 64)
        self.fusion_layer = nn.Linear(128 + 64, 128)
        
        # 模式完成网络
        self.pattern_completer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Tanh()
        )
        
        # 序列预测网络
        self.sequence_predictor = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
    
    def consolidate(
        self,
        content: torch.Tensor,
        importance: float,
        timestamp: float
    ) -> torch.Tensor:
        """巩固记忆"""
        # 编码内容
        encoded = self.encoder(content)
        
        # 处理时间信息
        time_info = torch.tensor([[timestamp]], dtype=torch.float32)
        time_features = self.time_processor(time_info)
        
        # 融合内容和时间信息
        combined = torch.cat([encoded, time_features], dim=-1)
        consolidated = self.fusion_layer(combined)
        
        # 解码为最终表示
        final_representation = self.decoder(consolidated)
        
        # 应用重要性权重
        importance_weight = torch.tensor([importance])
        final_representation = final_representation * importance_weight
        
        return final_representation
    
    def complete_pattern(self, partial_pattern: torch.Tensor) -> torch.Tensor:
        """完成模式"""
        # 使用编码器处理部分模式
        encoded_partial = self.encoder(partial_pattern)
        
        # 补全缺失部分
        completed = self.pattern_completer(encoded_partial)
        
        return completed
    
    def replay_sequence(self, sequence_data: List[torch.Tensor]) -> torch.Tensor:
        """重放序列"""
        if len(sequence_data) < 2:
            return sequence_data[0] if sequence_data else torch.zeros(512)
        
        # 准备序列数据
        sequence_tensor = torch.stack(sequence_data).unsqueeze(0)  # [1, seq_len, 512]
        
        # 预测序列
        output, _ = self.sequence_predictor(sequence_tensor)
        
        return output.squeeze(0)  # [seq_len, 512]


if __name__ == "__main__":
    import asyncio
    
    # 创建配置
    config = ConsolidationConfig(
        consolidation_threshold=0.6,
        strengthening_rate=0.1,
        temporal_scaling=0.01
    )
    
    # 创建引擎
    engine = ConsolidationEngine(config)
    
    # 初始化
    asyncio.run(engine.initialize())
    
    # 测试注册记忆
    test_content = torch.randn(512)
    memory_id = "test_memory_001"
    
    success = asyncio.run(engine.register_episodic_memory(
        memory_id=memory_id,
        content=test_content,
        importance=0.8,
        associations=["assoc_001", "assoc_002"]
    ))
    
    print(f"记忆注册成功: {success}")
    
    # 执行巩固
    consolidation_result = asyncio.run(engine.consolidate_memory(memory_id))
    print(f"巩固结果: {consolidation_result}")
    
    # 执行重放
    replay_result = asyncio.run(engine.replay_memory(memory_id))
    print(f"重放结果: {replay_result}")
    
    # 批量巩固
    batch_result = asyncio.run(engine.batch_consolidation(time_window=50))
    print(f"批量巩固结果: {batch_result}")
    
    # 自动巩固
    auto_result = asyncio.run(engine.automatic_consolidation())
    print(f"自动巩固结果: {auto_result}")
    
    # 获取状态
    status = engine.get_consolidation_status()
    print(f"巩固状态: {status}")