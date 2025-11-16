"""
大脑系统核心模块
===============

实现了基于生物启发的AI大脑系统，包括大脑区域建模、突触连接、
记忆机制和神经可塑性等核心功能。

主要特性:
- 大脑区域分层架构
- 突触可塑性机制
- 记忆形成与检索
- 神经振荡与节律
- 注意力与意识模型

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import numpy as np
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import uuid
import math
from threading import Lock

from .base_module import BaseModule, ModuleConfig, ModuleState


class BrainRegion(Enum):
    """大脑区域枚举"""
    HIPPOCAMPUS = "hippocampus"           # 海马体：记忆巩固与检索
    ENTORHINAL = "entorhinal"             # 内嗅皮层：空间导航与记忆编码
    PREFRONTAL = "prefrontal"             # 前额叶：执行控制与工作记忆
    CORTEX = "cortex"                     # 皮层：感觉处理与感知
    AMYGDALA = "amygdala"                 # 杏仁核：情感处理与恐惧学习
    THALAMUS = "thalamus"                 # 丘脑：信息中继与意识门控
    CEREBELLUM = "cerebellum"             # 小脑：运动协调与程序化记忆
    STRIATAL = "striatal"                 # 纹状体：习惯学习与奖励处理


class SynapticPlasticity(Enum):
    """突触可塑性类型"""
    LTP = "long_term_potentiation"        # 长时程增强
    LTD = "long_term_depression"          # 长时程抑制
    STDP = "spike_timing_dependent"       # 时序依赖性突触可塑性
    HOMEPLASTICITY = "homeostatic"        # 稳态可塑性


class MemoryType(Enum):
    """记忆类型"""
    EPISODIC = "episodic"                 # 情景记忆
    SEMANTIC = "semantic"                 # 语义记忆
    PROCEDURAL = "procedural"             # 程序性记忆
    WORKING = "working"                   # 工作记忆
    EMOTIONAL = "emotional"               # 情感记忆
    SPATIAL = "spatial"                   # 空间记忆


@dataclass
class Neuron:
    """神经元模型"""
    neuron_id: str
    region: BrainRegion
    position: Tuple[float, float, float]
    activation_level: float = 0.0
    threshold: float = 0.5
    refractory_period: float = 0.0
    connections: Set[str] = field(default_factory=set)
    plasticity_history: List[float] = field(default_factory=list)


@dataclass
class Synapse:
    """突触连接模型"""
    synapse_id: str
    pre_neuron_id: str
    post_neuron_id: str
    weight: float
    plasticity_type: SynapticPlasticity
    last_activity: float = 0.0
    eligibility_trace: float = 0.0
    neuromodulator_level: float = 1.0
    active: bool = True


@dataclass
class MemoryTrace:
    """记忆痕迹"""
    trace_id: str
    memory_type: MemoryType
    activation_pattern: np.ndarray
    strength: float
    formation_time: float
    last_retrieval: float = 0.0
    retrieval_count: int = 0
    associated_regions: List[BrainRegion] = field(default_factory=list)


class BrainOscillation:
    """大脑振荡器"""
    
    def __init__(self, frequency: float, phase: float = 0.0):
        self.frequency = frequency  # Hz
        self.phase = phase          # 弧度
        self.amplitude = 1.0
    
    def sample(self, time: float) -> float:
        """在指定时间采样振荡值"""
        return self.amplitude * math.sin(2 * math.pi * self.frequency * time + self.phase)
    
    def update_phase(self, dt: float):
        """更新相位"""
        self.phase += 2 * math.pi * self.frequency * dt


class BrainRegionModule(BaseModule):
    """大脑区域模块"""
    
    def __init__(self, region: BrainRegion, config: ModuleConfig):
        super().__init__(config)
        self.region = region
        self.neurons: Dict[str, Neuron] = {}
        self.synapses: Dict[str, Synapse] = {}
        self.oscillations: Dict[str, BrainOscillation] = {}
        self.memory_traces: Dict[str, MemoryTrace] = {}
        self.input_buffer = deque(maxlen=1000)
        self.output_buffer = deque(maxlen=1000)
        self.activation_map = np.array([])
        self._activity_history = deque(maxlen=1000)
        
    def add_neuron(self, position: Tuple[float, float, float], 
                   threshold: float = 0.5) -> str:
        """添加神经元"""
        neuron_id = str(uuid.uuid4())
        neuron = Neuron(
            neuron_id=neuron_id,
            region=self.region,
            position=position,
            threshold=threshold
        )
        self.neurons[neuron_id] = neuron
        return neuron_id
    
    def connect_neurons(self, pre_id: str, post_id: str, 
                       initial_weight: float = 0.1,
                       plasticity_type: SynapticPlasticity = SynapticPlasticity.LTP) -> str:
        """连接神经元"""
        if pre_id not in self.neurons or post_id not in self.neurons:
            raise ValueError("神经元不存在")
        
        synapse_id = str(uuid.uuid4())
        synapse = Synapse(
            synapse_id=synapse_id,
            pre_neuron_id=pre_id,
            post_neuron_id=post_id,
            weight=initial_weight,
            plasticity_type=plasticity_type
        )
        self.synapses[synapse_id] = synapse
        
        # 更新神经元的连接集合
        self.neurons[pre_id].connections.add(post_id)
        
        return synapse_id
    
    def process_input(self, input_pattern: np.ndarray) -> np.ndarray:
        """处理输入模式"""
        current_time = time.time()
        
        # 添加到输入缓冲区
        self.input_buffer.append((current_time, input_pattern))
        
        # 更新激活水平
        self._update_neuron_activations(input_pattern)
        
        # 突触可塑性更新
        self._update_synaptic_plasticity(current_time)
        
        # 振荡器更新
        self._update_oscillations(dt=0.001)  # 1ms 时间步长
        
        # 计算输出
        output = self._compute_output()
        
        # 添加到输出缓冲区
        self.output_buffer.append((current_time, output))
        
        return output
    
    def _update_neuron_activations(self, input_pattern: np.ndarray):
        """更新神经元激活水平"""
        # 简化的神经元激活模型
        for neuron_id, neuron in self.neurons.items():
            # 计算输入加权求和
            input_sum = 0.0
            for synapse in self.synapses.values():
                if synapse.post_neuron_id == neuron_id and synapse.active:
                    pre_neuron = self.neurons[synapse.pre_neuron_id]
                    input_sum += pre_neuron.activation_level * synapse.weight
            
            # 添加外部输入
            if len(self.neurons) > 0:
                input_idx = hash(neuron_id) % len(input_pattern)
                input_sum += input_pattern[input_idx % len(input_pattern)]
            
            # 激活函数（ReLU + 阈值）
            neuron.activation_level = max(0.0, input_sum - neuron.threshold)
            
            # 记录到历史
            self._activity_history.append((time.time(), neuron.activation_level))
    
    def _update_synaptic_plasticity(self, current_time: float):
        """更新突触可塑性"""
        for synapse in self.synapses.values():
            if not synapse.active:
                continue
            
            # 获取前后神经元
            pre_neuron = self.neurons.get(synapse.pre_neuron_id)
            post_neuron = self.neurons.get(synapse.post_neuron_id)
            
            if pre_neuron is None or post_neuron is None:
                continue
            
            # STDP (时序依赖性突触可塑性)
            if synapse.plasticity_type == SynapticPlasticity.STDP:
                self._stdp_update(synapse, pre_neuron, post_neuron, current_time)
            
            # 赫布学习规则
            elif synapse.plasticity_type == SynapticPlasticity.LTP:
                if pre_neuron.activation_level > 0.5 and post_neuron.activation_level > 0.5:
                    synapse.weight = min(1.0, synapse.weight + 0.01)
            
            elif synapse.plasticity_type == SynapticPlasticity.LTD:
                if pre_neuron.activation_level < 0.3 and post_neuron.activation_level < 0.3:
                    synapse.weight = max(0.0, synapse.weight - 0.005)
    
    def _stdp_update(self, synapse: Synapse, pre_neuron: Neuron, 
                    post_neuron: Neuron, current_time: float):
        """STDP更新规则"""
        dt = current_time - synapse.last_activity
        if abs(dt) < 0.001:  # 1ms 时间窗口
            
            # 前突触先于后突触激活 → LTP
            if pre_neuron.activation_level > 0.5 and post_neuron.activation_level > 0.5:
                synapse.weight = min(1.0, synapse.weight + 0.1 * math.exp(-dt * 1000))
            
            # 后突触先于前突触激活 → LTD  
            elif pre_neuron.activation_level > 0.5 and post_neuron.activation_level < 0.5:
                synapse.weight = max(0.0, synapse.weight - 0.1 * math.exp(dt * 1000))
        
        synapse.last_activity = current_time
    
    def _update_oscillations(self, dt: float):
        """更新大脑振荡"""
        for oscillation in self.oscillations.values():
            oscillation.update_phase(dt)
    
    def _compute_output(self) -> np.ndarray:
        """计算区域输出"""
        if not self.neurons:
            return np.array([])
        
        # 获取所有神经元激活水平
        activations = np.array([
            neuron.activation_level for neuron in self.neurons.values()
        ])
        
        return activations
    
    def add_oscillation(self, frequency: float, amplitude: float = 1.0) -> str:
        """添加振荡"""
        oscillation_id = str(uuid.uuid4())
        self.oscillations[oscillation_id] = BrainOscillation(frequency, amplitude)
        return oscillation_id
    
    def get_activity_summary(self) -> Dict[str, Any]:
        """获取区域活动摘要"""
        total_neurons = len(self.neurons)
        active_neurons = sum(1 for n in self.neurons.values() if n.activation_level > 0.1)
        avg_activation = np.mean([n.activation_level for n in self.neurons.values()]) if self.neurons else 0.0
        
        return {
            'region': self.region.value,
            'total_neurons': total_neurons,
            'active_neurons': active_neurons,
            'average_activation': avg_activation,
            'total_synapses': len(self.synapses),
            'active_synapses': sum(1 for s in self.synapses.values() if s.active),
            'memory_traces': len(self.memory_traces)
        }
    
    def initialize(self) -> bool:
        """初始化大脑区域"""
        self.state = ModuleState.INITIALIZING
        
        # 创建基本神经元网络
        if not self.neurons:
            # 创建一个简化的大脑区域结构
            for i in range(100):  # 100个神经元
                x = np.random.uniform(-1, 1)
                y = np.random.uniform(-1, 1) 
                z = np.random.uniform(-1, 1)
                self.add_neuron((x, y, z))
        
        # 建立随机连接
        neuron_ids = list(self.neurons.keys())
        for i in range(200):  # 200个随机连接
            pre_id = np.random.choice(neuron_ids)
            post_id = np.random.choice(neuron_ids)
            if pre_id != post_id:
                weight = np.random.uniform(0.1, 0.3)
                self.connect_neurons(pre_id, post_id, weight)
        
        # 添加振荡器（模拟不同频段的脑电波）
        self.add_oscillation(frequency=8.0, amplitude=0.3)   # Theta波段 (4-8Hz)
        self.add_oscillation(frequency=12.0, amplitude=0.5)  # Alpha波段 (8-12Hz)
        self.add_oscillation(frequency=40.0, amplitude=0.1)  # Gamma波段 (30-100Hz)
        
        self.state = ModuleState.INITIALIZED
        return True
    
    def cleanup(self) -> bool:
        """清理区域资源"""
        self.neurons.clear()
        self.synapses.clear()
        self.oscillations.clear()
        self.memory_traces.clear()
        self.input_buffer.clear()
        self.output_buffer.clear()
        return True


class BrainSystem(BaseModule):
    """
    统一的大脑系统类
    
    整合多个大脑区域模块，提供全局的协调与控制机制。
    支持模块间的通信、记忆共享和全局状态管理。
    """
    
    def __init__(self, config: ModuleConfig):
        super().__init__(config)
        self.regions: Dict[BrainRegion, BrainRegionModule] = {}
        self.global_memory: Dict[str, MemoryTrace] = {}
        self.attention_focus: Optional[str] = None
        self.global_oscillations: Dict[str, BrainOscillation] = {}
        self.consciousness_level: float = 0.0
        self.metacognitive_state: Dict[str, Any] = {}
        self._communication_matrix = np.array([])
        self._lock = Lock()
        
    def add_region(self, region: BrainRegion, 
                   module_config: ModuleConfig) -> BrainRegionModule:
        """添加大脑区域模块"""
        with self._lock:
            region_module = BrainRegionModule(region, module_config)
            self.regions[region] = region_module
            
            # 更新通信矩阵
            self._update_communication_matrix()
            
            return region_module
    
    def remove_region(self, region: BrainRegion) -> bool:
        """移除大脑区域"""
        with self._lock:
            if region in self.regions:
                region_module = self.regions[region]
                region_module.cleanup()
                del self.regions[region]
                self._update_communication_matrix()
                return True
            return False
    
    def process_global_input(self, input_pattern: np.ndarray) -> Dict[BrainRegion, np.ndarray]:
        """处理全局输入，分配到各区域"""
        outputs = {}
        
        # 根据区域功能分发输入
        for region, region_module in self.regions.items():
            if region_module.state == ModuleState.ACTIVE:
                # 不同区域接收不同类型的输入
                processed_input = self._distribute_input_by_region(region, input_pattern)
                output = region_module.process_input(processed_input)
                outputs[region] = output
        
        return outputs
    
    def _distribute_input_by_region(self, region: BrainRegion, 
                                   global_input: np.ndarray) -> np.ndarray:
        """根据区域功能分发输入"""
        if region == BrainRegion.PREFRONTAL:
            # 前额叶：执行控制，输入高层认知信息
            return global_input * 0.8
        
        elif region == BrainRegion.CORTEX:
            # 皮层：感觉处理，接收原始输入
            return global_input
        
        elif region == BrainRegion.HIPPOCAMPUS:
            # 海马体：记忆检索，输入情境信息
            return global_input * 0.6
        
        else:
            # 其他区域：分配部分输入
            return global_input * 0.3
    
    def create_memory_trace(self, memory_type: MemoryType, 
                          activation_pattern: np.ndarray,
                          associated_regions: List[BrainRegion]) -> str:
        """创建记忆痕迹"""
        with self._lock:
            trace_id = str(uuid.uuid4())
            
            memory_trace = MemoryTrace(
                trace_id=trace_id,
                memory_type=memory_type,
                activation_pattern=activation_pattern.copy(),
                strength=0.1,
                formation_time=time.time(),
                associated_regions=associated_regions.copy()
            )
            
            self.global_memory[trace_id] = memory_trace
            
            # 在相关区域存储记忆
            for region in associated_regions:
                if region in self.regions:
                    self.regions[region].memory_traces[trace_id] = memory_trace
            
            return trace_id
    
    def retrieve_memory(self, query_pattern: np.ndarray, 
                       memory_type: Optional[MemoryType] = None,
                       threshold: float = 0.7) -> List[MemoryTrace]:
        """检索记忆"""
        with self._lock:
            candidates = []
            
            for trace in self.global_memory.values():
                # 类型过滤
                if memory_type and trace.memory_type != memory_type:
                    continue
                
                # 计算相似度（余弦相似度）
                similarity = self._compute_pattern_similarity(
                    query_pattern, trace.activation_pattern
                )
                
                if similarity > threshold:
                    trace.last_retrieval = time.time()
                    trace.retrieval_count += 1
                    
                    # 强化记忆（重复检索会加强记忆）
                    trace.strength = min(1.0, trace.strength + 0.01)
                    
                    candidates.append((trace, similarity))
            
            # 按相似度排序
            candidates.sort(key=lambda x: x[1], reverse=True)
            return [trace for trace, _ in candidates[:10]]  # 返回前10个最相似的
    
    def _compute_pattern_similarity(self, pattern1: np.ndarray, 
                                   pattern2: np.ndarray) -> float:
        """计算模式相似度"""
        if len(pattern1) == 0 or len(pattern2) == 0:
            return 0.0
        
        # 确保两个向量长度相同
        min_len = min(len(pattern1), len(pattern2))
        p1 = pattern1[:min_len]
        p2 = pattern2[:min_len]
        
        # 计算余弦相似度
        dot_product = np.dot(p1, p2)
        norm1 = np.linalg.norm(p1)
        norm2 = np.linalg.norm(p2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def update_attention(self, focus_region: BrainRegion):
        """更新注意力焦点"""
        self.attention_focus = focus_region.value
        
        # 增强焦点区域的神经活动
        if focus_region in self.regions:
            region_module = self.regions[focus_region]
            
            # 注意力增强机制
            for neuron in region_module.neurons.values():
                neuron.activation_level *= 1.2  # 注意力增益
    
    def compute_consciousness_level(self) -> float:
        """计算意识水平"""
        if not self.regions:
            return 0.0
        
        # 整合所有区域的激活水平
        total_activation = 0.0
        active_regions = 0
        
        for region_module in self.regions.values():
            if region_module.neurons:
                region_avg = np.mean([
                    neuron.activation_level 
                    for neuron in region_module.neurons.values()
                ])
                total_activation += region_avg
                active_regions += 1
        
        if active_regions == 0:
            return 0.0
        
        # 意识水平 = 区域间协调程度 + 平均激活水平
        coordination_factor = self._compute_inter_region_coordination()
        avg_activation = total_activation / active_regions
        
        consciousness = 0.6 * coordination_factor + 0.4 * avg_activation
        self.consciousness_level = max(0.0, min(1.0, consciousness))
        
        return self.consciousness_level
    
    def _compute_inter_region_coordination(self) -> float:
        """计算区域间协调程度"""
        if len(self.regions) < 2:
            return 0.0
        
        # 计算区域间的相关性
        region_activities = []
        for region_module in self.regions.values():
            if region_module.neurons:
                activity = np.mean([
                    neuron.activation_level 
                    for neuron in region_module.neurons.values()
                ])
                region_activities.append(activity)
        
        if len(region_activities) < 2:
            return 0.0
        
        # 计算标准差作为协调性指标（标准差小表示协调）
        std_dev = np.std(region_activities)
        coordination = 1.0 / (1.0 + std_dev)  # 转换为0-1范围
        
        return coordination
    
    def _update_communication_matrix(self):
        """更新区域间通信矩阵"""
        n_regions = len(self.regions)
        if n_regions > 0:
            self._communication_matrix = np.random.uniform(0.1, 0.5, (n_regions, n_regions))
            # 设置对角线为0（区域不与自己通信）
            np.fill_diagonal(self._communication_matrix, 0)
    
    def get_brain_state(self) -> Dict[str, Any]:
        """获取大脑系统当前状态"""
        region_states = {}
        for region, module in self.regions.items():
            region_states[region.value] = module.get_activity_summary()
        
        return {
            'consciousness_level': self.consciousness_level,
            'attention_focus': self.attention_focus,
            'global_memory_size': len(self.global_memory),
            'region_states': region_states,
            'metacognitive_state': self.metacognitive_state.copy()
        }
    
    def initialize(self) -> bool:
        """初始化大脑系统"""
        self.state = ModuleState.INITIALIZING
        
        # 创建标准的大脑区域配置
        region_configs = {
            BrainRegion.PREFRONTAL: ModuleConfig("prefrontal", version="1.0", priority=1),
            BrainRegion.HIPPOCAMPUS: ModuleConfig("hippocampus", version="1.0", priority=2),
            BrainRegion.CORTEX: ModuleConfig("cortex", version="1.0", priority=3),
            BrainRegion.ENTORHINAL: ModuleConfig("entorhinal", version="1.0", priority=4),
            BrainRegion.AMYGDALA: ModuleConfig("amygdala", version="1.0", priority=5),
        }
        
        # 添加所有区域
        for region, config in region_configs.items():
            self.add_region(region, config)
        
        # 初始化所有区域
        for region, module in self.regions.items():
            if not module.initialize():
                self.logger.error(f"区域 {region.value} 初始化失败")
                return False
        
        # 设置全局振荡器
        self.global_oscillations["alpha"] = BrainOscillation(frequency=10.0)
        self.global_oscillations["gamma"] = BrainOscillation(frequency=40.0)
        
        self.state = ModuleState.INITIALIZED
        return True
    
    def cleanup(self) -> bool:
        """清理大脑系统"""
        for region_module in self.regions.values():
            region_module.cleanup()
        
        self.regions.clear()
        self.global_memory.clear()
        self.global_oscillations.clear()
        
        return True


# 创建系统实例管理器
_brain_system_instance = None

def get_brain_system() -> BrainSystem:
    """获取全局大脑系统实例"""
    global _brain_system_instance
    if _brain_system_instance is None:
        config = ModuleConfig("brain_system", version="1.0.0", priority=0)
        _brain_system_instance = BrainSystem(config)
        _brain_system_instance.initialize()
    return _brain_system_instance