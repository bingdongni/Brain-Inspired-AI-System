"""
抽象算法配置
============

定义知识抽象算法的配置参数和类型定义。
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple


class AbstractionType(Enum):
    """抽象类型"""
    CLUSTERING = "clustering"           # 聚类抽象
    PROBABILISTIC = "probabilistic"     # 概率抽象
    NEURAL = "neural"                  # 神经网络抽象
    HIERARCHICAL = "hierarchical"       # 层次抽象
    SYMBOLIC = "symbolic"              # 符号抽象
    COMBINED = "combined"              # 组合抽象


class KnowledgeRepresentationType(Enum):
    """知识表示类型"""
    DISTRIBUTED = "distributed"         # 分布式表示
    LOCALIST = "localist"              # 局部表示
    SYMBOLIC = "symbolic"              # 符号表示
    HYBRID = "hybrid"                  # 混合表示
    GRAPH = "graph"                    # 图结构表示


class ConceptType(Enum):
    """概念类型"""
    OBJECT = "object"                  # 物体概念
    ACTION = "action"                  # 动作概念
    PROPERTY = "property"              # 属性概念
    SPATIAL = "spatial"                # 空间概念
    TEMPORAL = "temporal"              # 时间概念
    CAUSAL = "causal"                  # 因果概念
    ABSTRACT = "abstract"              # 抽象概念
    SOCIAL = "social"                  # 社会概念
    EMOTIONAL = "emotional"            # 情感概念


class AbstractionLevel(Enum):
    """抽象层级"""
    PERCEPTUAL = "perceptual"          # 感知层
    CATEGORICAL = "categorical"        # 类别层
    CONCEPTUAL = "conceptual"          # 概念层
    SCHEMATIC = "schematic"            # 模式层
    META_CONCEPTUAL = "meta_conceptual" # 元概念层


@dataclass
class AbstractionConfig:
    """抽象配置"""
    
    # 基本参数
    feature_dim: int                    # 特征维度
    abstraction_type: AbstractionType = AbstractionType.CLUSTERING
    representation_type: KnowledgeRepresentationType = KnowledgeRepresentationType.DISTRIBUTED
    
    # 概念参数
    max_concepts: int = 1000            # 最大概念数
    min_activation_threshold: float = 0.1  # 激活阈值
    concept_formation_threshold: float = 0.7  # 概念形成阈值
    
    # 抽象层级
    num_abstraction_levels: int = 4     # 抽象层级数
    hierarchical_abstraction: bool = True  # 层次抽象
    
    # 学习参数
    learning_rate: float = 1e-4         # 学习率
    concept_stability: float = 0.95     # 概念稳定性
    forgetting_rate: float = 0.01       # 遗忘率
    
    # 记忆参数
    working_memory_size: int = 100      # 工作记忆大小
    long_term_memory_size: int = 10000  # 长期记忆大小
    episodic_memory: bool = True        # 情景记忆
    
    # 推理参数
    analogical_reasoning: bool = True    # 类比推理
    causal_reasoning: bool = True       # 因果推理
    temporal_reasoning: bool = True     # 时间推理
    
    # 评估参数
    concept_quality_threshold: float = 0.6  # 概念质量阈值
    abstraction_coherence: float = 0.8     # 抽象连贯性


@dataclass  
class ConceptConfig:
    """概念配置"""
    
    concept_id: str                     # 概念ID
    concept_type: ConceptType           # 概念类型
    abstraction_level: AbstractionLevel # 抽象层级
    
    # 概念属性
    activation_threshold: float = 0.5   # 激活阈值
    stability_factor: float = 0.9       # 稳定性因子
    generality: float = 0.5             # 普遍性
    
    # 关联属性
    associations: List[str] = None      # 关联概念
    properties: Dict[str, Any] = None   # 概念属性
    
    # 时间属性
    formation_time: Optional[float] = None  # 形成时间
    last_accessed: Optional[float] = None   # 最后访问时间
    
    def __post_init__(self):
        if self.associations is None:
            self.associations = []
        if self.properties is None:
            self.properties = {}


class ConceptQuality:
    """概念质量评估"""
    
    def __init__(self):
        self.coherence_score = 0.0      # 连贯性分数
        self.coverage_score = 0.0       # 覆盖性分数
        self.distinctiveness_score = 0.0  # 独特性分数
        self.stability_score = 0.0      # 稳定性分数
        self.generality_score = 0.0     # 普遍性分数
    
    def compute_overall_quality(self) -> float:
        """计算整体质量分数"""
        scores = [
            self.coherence_score,
            self.coverage_score, 
            self.distinctiveness_score,
            self.stability_score,
            self.generality_score
        ]
        return sum(scores) / len(scores)
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'coherence': self.coherence_score,
            'coverage': self.coverage_score,
            'distinctiveness': self.distinctiveness_score,
            'stability': self.stability_score,
            'generality': self.generality_score,
            'overall': self.compute_overall_quality()
        }


class AbstractionMetrics:
    """抽象指标"""
    
    def __init__(self):
        self.total_concepts = 0
        self.active_concepts = 0
        self.abstracted_instances = 0
        self.abstraction_accuracy = 0.0
        self.concept_diversity = 0.0
        self.hierarchy_depth = 0
        
        # 质量指标
        self.avg_concept_quality = 0.0
        self.abstract_reasoning_success = 0.0
        
        # 效率指标
        self.abstraction_speed = 0.0
        self.concept_formation_rate = 0.0
    
    def update(self, new_concepts: int, active_concepts: int, 
               abstracted_instances: int, accuracy: float):
        """更新指标"""
        self.total_concepts += new_concepts
        self.active_concepts = active_concepts
        self.abstracted_instances = abstracted_instances
        self.abstraction_accuracy = accuracy
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要统计"""
        return {
            'total_concepts': self.total_concepts,
            'active_concepts': self.active_concepts,
            'abstraction_rate': self.abstracted_instances / max(self.total_concepts, 1),
            'accuracy': self.abstraction_accuracy,
            'concept_utilization': self.active_concepts / max(self.total_concepts, 1),
            'avg_quality': self.avg_concept_quality,
            'reasoning_success': self.abstract_reasoning_success,
            'abstraction_speed': self.abstraction_speed
        }


class ConceptMemory:
    """概念记忆系统"""
    
    def __init__(self, config: AbstractionConfig):
        self.config = config
        
        # 概念存储
        self.concept_storage = {}       # 概念ID -> 概念数据
        self.concept_activations = {}   # 概念ID -> 激活值
        self.concept_timestamps = {}    # 概念ID -> 时间戳
        
        # 概念关联
        self.concept_associations = {}  # 概念ID -> 关联列表
        self.association_strengths = {} # 关联对 -> 强度
        
        # 记忆层次
        self.working_memory = {}        # 工作记忆中的概念
        self.episodic_memory = []       # 情景记忆
        
        # 统计信息
        self.access_count = {}          # 概念访问计数
        self.formation_count = 0        # 概念形成计数
    
    def store_concept(self, concept_data: Dict[str, Any], 
                     concept_id: str, activation: float = 0.0) -> bool:
        """存储概念"""
        try:
            self.concept_storage[concept_id] = concept_data
            self.concept_activations[concept_id] = activation
            self.concept_timestamps[concept_id] = torch.tensor(0.0)  # 简化的时间戳
            
            # 更新访问计数
            self.access_count[concept_id] = self.access_count.get(concept_id, 0) + 1
            self.formation_count += 1
            
            return True
        except Exception as e:
            print(f"概念存储失败: {e}")
            return False
    
    def retrieve_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """检索概念"""
        if concept_id in self.concept_storage:
            concept_data = self.concept_storage[concept_id]
            
            # 更新访问计数和最后访问时间
            self.access_count[concept_id] = self.access_count.get(concept_id, 0) + 1
            self.concept_timestamps[concept_id] = torch.tensor(1.0)  # 更新访问时间
            
            return concept_data
        return None
    
    def activate_concept(self, concept_id: str, activation: float) -> bool:
        """激活概念"""
        if concept_id in self.concept_storage:
            self.concept_activations[concept_id] = activation
            
            # 如果激活值超过阈值，加入工作记忆
            if activation > self.config.min_activation_threshold:
                self.working_memory[concept_id] = activation
                
                # 管理工作记忆大小
                if len(self.working_memory) > self.config.working_memory_size:
                    # 移除激活最低的概念
                    min_activation_concept = min(self.working_memory.keys(), 
                                               key=lambda x: self.working_memory[x])
                    del self.working_memory[min_activation_concept]
            
            return True
        return False
    
    def get_active_concepts(self, threshold: float = 0.5) -> List[str]:
        """获取激活的概念"""
        active_concepts = []
        for concept_id, activation in self.concept_activations.items():
            if activation > threshold:
                active_concepts.append(concept_id)
        return active_concepts
    
    def update_associations(self, concept1: str, concept2: str, strength: float) -> None:
        """更新概念关联"""
        # 添加双向关联
        if concept1 not in self.concept_associations:
            self.concept_associations[concept1] = []
        if concept2 not in self.concept_associations:
            self.concept_associations[concept2] = []
            
        if concept2 not in self.concept_associations[concept1]:
            self.concept_associations[concept1].append(concept2)
        if concept1 not in self.concept_associations[concept2]:
            self.concept_associations[concept2].append(concept1)
        
        # 更新关联强度
        association_key = tuple(sorted([concept1, concept2]))
        self.association_strengths[association_key] = strength
    
    def prune_inactive_concepts(self, inactivity_threshold: int = 100) -> int:
        """剪枝不活跃概念"""
        inactive_concepts = []
        
        for concept_id, access_count in self.access_count.items():
            if access_count < inactivity_threshold:
                inactive_concepts.append(concept_id)
        
        # 移除不活跃概念
        for concept_id in inactive_concepts:
            self.remove_concept(concept_id)
        
        return len(inactive_concepts)
    
    def remove_concept(self, concept_id: str) -> bool:
        """移除概念"""
        try:
            # 从所有存储中移除
            keys_to_remove = [
                self.concept_storage, self.concept_activations, 
                self.concept_timestamps, self.access_count
            ]
            
            for storage in keys_to_remove:
                storage.pop(concept_id, None)
            
            # 从关联中移除
            self.concept_associations.pop(concept_id, None)
            for associations in self.concept_associations.values():
                if concept_id in associations:
                    associations.remove(concept_id)
            
            # 移除关联强度
            for association_key in list(self.association_strengths.keys()):
                if concept_id in association_key:
                    self.association_strengths.pop(association_key)
            
            return True
        except Exception as e:
            print(f"概念移除失败: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        return {
            'total_concepts': len(self.concept_storage),
            'active_concepts': len(self.get_active_concepts()),
            'working_memory_size': len(self.working_memory),
            'total_associations': sum(len(assocs) for assocs in self.concept_associations.values()) // 2,
            'avg_activation': sum(self.concept_activations.values()) / max(len(self.concept_activations), 1),
            'most_accessed': max(self.access_count.items(), key=lambda x: x[1])[0] if self.access_count else None,
            'formation_count': self.formation_count
        }


class AbstractionLogger:
    """抽象过程日志记录器"""
    
    def __init__(self, max_entries: int = 10000):
        self.max_entries = max_entries
        self.entries = []
        self.current_session = 0
    
    def log_concept_formation(self, concept_id: str, concept_type: ConceptType, 
                            abstraction_level: AbstractionLevel, quality: float) -> None:
        """记录概念形成"""
        entry = {
            'type': 'concept_formation',
            'timestamp': len(self.entries),
            'concept_id': concept_id,
            'concept_type': concept_type.value,
            'abstraction_level': abstraction_level.value,
            'quality': quality,
            'session': self.current_session
        }
        self.entries.append(entry)
        
        # 限制日志大小
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
    
    def log_abstraction_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """记录抽象事件"""
        entry = {
            'type': 'abstraction_event',
            'timestamp': len(self.entries),
            'event_type': event_type,
            'details': details,
            'session': self.current_session
        }
        self.entries.append(entry)
        
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
    
    def get_recent_entries(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的日志条目"""
        return self.entries[-count:] if len(self.entries) >= count else self.entries
    
    def get_concept_formation_stats(self) -> Dict[str, Any]:
        """获取概念形成统计"""
        formation_entries = [e for e in self.entries if e['type'] == 'concept_formation']
        
        if not formation_entries:
            return {'total_formations': 0}
        
        type_counts = {}
        level_counts = {}
        quality_scores = []
        
        for entry in formation_entries:
            # 统计类型
            concept_type = entry['concept_type']
            type_counts[concept_type] = type_counts.get(concept_type, 0) + 1
            
            # 统计层级
            level = entry['abstraction_level']
            level_counts[level] = level_counts.get(level, 0) + 1
            
            # 收集质量分数
            quality_scores.append(entry['quality'])
        
        return {
            'total_formations': len(formation_entries),
            'type_distribution': type_counts,
            'level_distribution': level_counts,
            'avg_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            'quality_std': (sum((q - sum(quality_scores)/len(quality_scores))**2 for q in quality_scores) / len(quality_scores))**0.5 if len(quality_scores) > 1 else 0.0
        }