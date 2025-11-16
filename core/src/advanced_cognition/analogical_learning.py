"""
类比学习和创造性问题解决能力
=========================

实现类比学习、相似性计算和创新思维模拟，包括：
- 相似性计算算法
- 类比推理引擎
- 创造性问题解决
- 模式识别与创新
- 联想记忆系统

主要特性：
- 多维度相似性计算
- 结构化类比推理
- 创造性思维模拟
- 跨领域知识迁移
- 创新解决方案生成

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import numpy as np
import logging
import time
import json
import math
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import itertools
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_module import BaseModule, ModuleConfig, ModuleState


class SimilarityType(Enum):
    """相似性类型"""
    SEMANTIC = "semantic"        # 语义相似性
    STRUCTURAL = "structural"    # 结构相似性
    FUNCTIONAL = "functional"    # 功能相似性
    TEMPORAL = "temporal"        # 时间相似性
    SPATIAL = "spatial"          # 空间相似性
    BEHAVIORAL = "behavioral"    # 行为相似性


class AnalogyType(Enum):
    """类比类型"""
    DIRECT = "direct"            # 直接类比
    STRUCTURAL = "structural"    # 结构类比
    FUNCTIONAL = "functional"    # 功能类比
    CONCEPTUAL = "conceptual"    # 概念类比
    METAPHORICAL = "metaphorical" # 隐喻类比


class CreativityLevel(Enum):
    """创造性水平"""
    ROUTINE = "routine"          # 常规水平
    ADAPTIVE = "adaptive"        # 适应水平
    INNOVATIVE = "innovative"    # 创新水平
    BREAKTHROUGH = "breakthrough" # 突破水平


@dataclass
class KnowledgeConcept:
    """知识概念"""
    concept_id: str
    name: str
    description: str
    properties: Dict[str, Any]
    relations: Dict[str, List[str]]  # 关系类型 -> 相关概念列表
    examples: List[str] = field(default_factory=list)
    domain: str = "general"
    abstractness: float = 0.5  # 抽象程度 0-1
    
    def to_vector(self) -> np.ndarray:
        """转换为向量表示"""
        # 简化的向量化
        vector = np.zeros(100)  # 假设100维向量空间
        
        # 基于名称和描述的特征
        name_chars = [ord(c) for c in self.name.lower()[:20]]
        for i, char_code in enumerate(name_chars):
            vector[i] = char_code / 256.0
        
        # 基于属性的特征
        for i, (key, value) in enumerate(self.properties.items()):
            vector[20 + i] = hash(str(key)) % 1000 / 1000.0
            if i < 10:  # 前10个属性
                vector[30 + i] = hash(str(value)) % 1000 / 1000.0
        
        return vector


@dataclass
class AnalogyMapping:
    """类比映射"""
    source_concept: KnowledgeConcept
    target_concept: KnowledgeConcept
    mapping_type: AnalogyType
    similarity_score: float
    mapping_details: Dict[str, Any]
    confidence: float
    created_time: float = field(default_factory=time.time)
    
    def is_valid(self) -> bool:
        """验证映射是否有效"""
        return self.similarity_score > 0.3 and self.confidence > 0.5


@dataclass
class CreativeSolution:
    """创造性解决方案"""
    solution_id: str
    description: str
    creativity_level: CreativityLevel
    novelty_score: float
    feasibility_score: float
    effectiveness_score: float
    source_analogies: List[AnalogyMapping]
    innovation_features: List[str]
    implementation_steps: List[str]
    created_time: float = field(default_factory=time.time)
    
    def get_overall_score(self) -> float:
        """获取总体评分"""
        return (self.novelty_score + self.feasibility_score + self.effectiveness_score) / 3.0


class SimilarityCalculator:
    """相似性计算器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.similarity_weights = {
            SimilarityType.SEMANTIC: 0.3,
            SimilarityType.STRUCTURAL: 0.25,
            SimilarityType.FUNCTIONAL: 0.25,
            SimilarityType.BEHAVIORAL: 0.2
        }
    
    def calculate_similarity(self, concept1: KnowledgeConcept, concept2: KnowledgeConcept,
                           similarity_types: List[SimilarityType] = None) -> Dict[SimilarityType, float]:
        """计算概念间相似性"""
        if similarity_types is None:
            similarity_types = list(SimilarityType)
        
        similarities = {}
        
        for sim_type in similarity_types:
            if sim_type == SimilarityType.SEMANTIC:
                similarities[sim_type] = self._calculate_semantic_similarity(concept1, concept2)
            elif sim_type == SimilarityType.STRUCTURAL:
                similarities[sim_type] = self._calculate_structural_similarity(concept1, concept2)
            elif sim_type == SimilarityType.FUNCTIONAL:
                similarities[sim_type] = self._calculate_functional_similarity(concept1, concept2)
            elif sim_type == SimilarityType.TEMPORAL:
                similarities[sim_type] = self._calculate_temporal_similarity(concept1, concept2)
            elif sim_type == SimilarityType.SPATIAL:
                similarities[sim_type] = self._calculate_spatial_similarity(concept1, concept2)
            elif sim_type == SimilarityType.BEHAVIORAL:
                similarities[sim_type] = self._calculate_behavioral_similarity(concept1, concept2)
        
        return similarities
    
    def calculate_weighted_similarity(self, concept1: KnowledgeConcept, concept2: KnowledgeConcept,
                                    similarity_types: List[SimilarityType] = None) -> float:
        """计算加权相似性"""
        similarities = self.calculate_similarity(concept1, concept2, similarity_types)
        
        if not similarities:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for sim_type, similarity in similarities.items():
            weight = self.similarity_weights.get(sim_type, 0.2)
            weighted_sum += similarity * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_semantic_similarity(self, concept1: KnowledgeConcept, concept2: KnowledgeConcept) -> float:
        """计算语义相似性"""
        # 向量相似性
        vec1 = concept1.to_vector()
        vec2 = concept2.to_vector()
        
        cosine_sim = self._cosine_similarity(vec1, vec2)
        
        # 名称相似性
        name_sim = self._name_similarity(concept1.name, concept2.name)
        
        # 属性相似性
        prop_sim = self._property_similarity(concept1.properties, concept2.properties)
        
        return (cosine_sim * 0.4 + name_sim * 0.3 + prop_sim * 0.3)
    
    def _calculate_structural_similarity(self, concept1: KnowledgeConcept, concept2: KnowledgeConcept) -> float:
        """计算结构相似性"""
        # 关系结构相似性
        rel1 = set(concept1.relations.keys())
        rel2 = set(concept2.relations.keys())
        
        if not rel1 and not rel2:
            return 1.0
        if not rel1 or not rel2:
            return 0.0
        
        intersection = len(rel1.intersection(rel2))
        union = len(rel1.union(rel2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # 关系强度相似性
        strength_sim = self._relation_strength_similarity(concept1.relations, concept2.relations)
        
        return (jaccard_similarity * 0.6 + strength_sim * 0.4)
    
    def _calculate_functional_similarity(self, concept1: KnowledgeConcept, concept2: KnowledgeConcept) -> float:
        """计算功能相似性"""
        # 基于属性中描述功能的字段
        func_keys = ['function', 'purpose', 'goal', 'objective', 'capability']
        
        func1 = {}
        func2 = {}
        
        for key in func_keys:
            for prop_key, prop_val in concept1.properties.items():
                if key.lower() in prop_key.lower():
                    func1[prop_key] = prop_val
            for prop_key, prop_val in concept2.properties.items():
                if key.lower() in prop_key.lower():
                    func2[prop_key] = prop_val
        
        if not func1 and not func2:
            return 0.5  # 中性相似性
        
        return self._property_similarity(func1, func2)
    
    def _calculate_temporal_similarity(self, concept1: KnowledgeConcept, concept2: KnowledgeConcept) -> float:
        """计算时间相似性"""
        # 基于抽象程度的时间特征
        time_diff = abs(concept1.abstractness - concept2.abstractness)
        return 1.0 - time_diff
    
    def _calculate_spatial_similarity(self, concept1: KnowledgeConcept, concept2: KnowledgeConcept) -> float:
        """计算空间相似性"""
        # 基于概念域的相似性
        if concept1.domain == concept2.domain:
            return 1.0
        elif self._are_related_domains(concept1.domain, concept2.domain):
            return 0.7
        else:
            return 0.3
    
    def _calculate_behavioral_similarity(self, concept1: KnowledgeConcept, concept2: KnowledgeConcept) -> float:
        """计算行为相似性"""
        # 基于示例的行为模式
        examples1 = set(concept1.examples)
        examples2 = set(concept2.examples)
        
        if not examples1 and not examples2:
            return 0.5
        
        if not examples1 or not examples2:
            return 0.3
        
        intersection = len(examples1.intersection(examples2))
        union = len(examples1.union(examples2))
        
        return intersection / union if union > 0 else 0.0
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """余弦相似性"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """名称相似性"""
        # 简单的编辑距离
        if name1 == name2:
            return 1.0
        
        # 使用字符级别的相似性
        chars1 = set(name1.lower())
        chars2 = set(name2.lower())
        
        intersection = len(chars1.intersection(chars2))
        union = len(chars1.union(chars2))
        
        return intersection / union if union > 0 else 0.0
    
    def _property_similarity(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> float:
        """属性相似性"""
        if not props1 and not props2:
            return 1.0
        if not props1 or not props2:
            return 0.3
        
        # 属性名称相似性
        keys1 = set(props1.keys())
        keys2 = set(props2.keys())
        
        key_intersection = len(keys1.intersection(keys2))
        key_union = len(keys1.union(keys2))
        
        key_similarity = key_intersection / key_union if key_union > 0 else 0.0
        
        # 属性值相似性
        value_similarities = []
        for key in keys1.intersection(keys2):
            val1 = props1[key]
            val2 = props2[key]
            val_sim = self._value_similarity(val1, val2)
            value_similarities.append(val_sim)
        
        avg_value_similarity = np.mean(value_similarities) if value_similarities else 0.0
        
        return (key_similarity * 0.4 + avg_value_similarity * 0.6)
    
    def _value_similarity(self, val1: Any, val2: Any) -> float:
        """值相似性"""
        if val1 == val2:
            return 1.0
        
        # 数值相似性
        try:
            num1 = float(val1)
            num2 = float(val2)
            diff = abs(num1 - num2)
            max_val = max(abs(num1), abs(num2), 1.0)  # 避免除零
            return 1.0 - min(diff / max_val, 1.0)
        except (ValueError, TypeError):
            pass
        
        # 字符串相似性
        if isinstance(val1, str) and isinstance(val2, str):
            return self._name_similarity(val1, val2)
        
        return 0.3  # 默认低相似性
    
    def _relation_strength_similarity(self, rel1: Dict[str, List[str]], rel2: Dict[str, List[str]]) -> float:
        """关系强度相似性"""
        all_relations = set(rel1.keys()).union(set(rel2.keys()))
        
        if not all_relations:
            return 1.0
        
        similarities = []
        for rel_type in all_relations:
            count1 = len(rel1.get(rel_type, []))
            count2 = len(rel2.get(rel_type, []))
            
            if count1 == 0 and count2 == 0:
                similarities.append(1.0)
            else:
                max_count = max(count1, count2)
                if max_count == 0:
                    similarities.append(0.0)
                else:
                    similarities.append(min(count1, count2) / max_count)
        
        return np.mean(similarities)
    
    def _are_related_domains(self, domain1: str, domain2: str) -> bool:
        """判断是否为相关域"""
        # 简化的域关系判断
        related_pairs = [
            ('science', 'technology'),
            ('medicine', 'biology'),
            ('mathematics', 'physics'),
            ('art', 'design'),
            ('business', 'economics')
        ]
        
        pair1 = (domain1, domain2)
        pair2 = (domain2, domain1)
        
        return pair1 in related_pairs or pair2 in related_pairs


class AnalogyEngine:
    """类比推理引擎"""
    
    def __init__(self, similarity_calculator: SimilarityCalculator):
        self.similarity_calculator = similarity_calculator
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 类比知识库
        self.analogy_knowledge_base: Dict[str, List[AnalogyMapping]] = defaultdict(list)
        self.concept_cache: Dict[str, KnowledgeConcept] = {}
    
    def find_analogies(self, source_concept: KnowledgeConcept, target_domain: str = None,
                      min_similarity: float = 0.5) -> List[AnalogyMapping]:
        """寻找类比"""
        analogies = []
        
        # 在概念库中寻找相似概念
        for concept_id, target_concept in self.concept_cache.items():
            if target_domain and target_concept.domain != target_domain:
                continue
            
            similarity = self.similarity_calculator.calculate_weighted_similarity(
                source_concept, target_concept
            )
            
            if similarity >= min_similarity:
                analogy = self._create_analogy_mapping(source_concept, target_concept, similarity)
                analogies.append(analogy)
        
        # 按相似性排序
        analogies.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return analogies
    
    def create_analogical_reasoning(self, source_problem: str, source_solution: str,
                                  target_problem: str) -> Optional[str]:
        """创建类比推理"""
        # 解析问题和解决方案
        source_concept = self._extract_concept_from_text(source_problem)
        target_concept = self._extract_concept_from_text(target_problem)
        
        if not source_concept or not target_concept:
            return None
        
        # 寻找类比映射
        analogies = self.find_analogies(source_concept, target_concept.domain)
        
        if not analogies:
            return None
        
        # 应用类比
        best_analogy = analogies[0]
        return self._apply_analogical_reasoning(source_solution, best_analogy)
    
    def learn_analogy(self, source_concept: KnowledgeConcept, target_concept: KnowledgeConcept,
                     mapping_type: AnalogyType, success: bool = True):
        """学习新的类比"""
        similarity = self.similarity_calculator.calculate_weighted_similarity(source_concept, target_concept)
        
        mapping = AnalogyMapping(
            source_concept=source_concept,
            target_concept=target_concept,
            mapping_type=mapping_type,
            similarity_score=similarity,
            mapping_details={'learning_success': success},
            confidence=0.8 if success else 0.4
        )
        
        concept_key = f"{source_concept.concept_id}_{target_concept.concept_id}"
        self.analogy_knowledge_base[concept_key].append(mapping)
        
        # 缓存概念
        self.concept_cache[source_concept.concept_id] = source_concept
        self.concept_cache[target_concept.concept_id] = target_concept
    
    def _create_analogy_mapping(self, source: KnowledgeConcept, target: KnowledgeConcept,
                              similarity: float) -> AnalogyMapping:
        """创建类比映射"""
        mapping_type = self._determine_analogy_type(source, target)
        
        return AnalogyMapping(
            source_concept=source,
            target_concept=target,
            mapping_type=mapping_type,
            similarity_score=similarity,
            mapping_details=self._analyze_mapping_details(source, target),
            confidence=min(similarity * 1.2, 1.0)
        )
    
    def _determine_analogy_type(self, source: KnowledgeConcept, target: KnowledgeConcept) -> AnalogyType:
        """确定类比类型"""
        # 基于相似性类型确定类比类型
        similarities = self.similarity_calculator.calculate_similarity(source, target)
        
        structural_sim = similarities.get(SimilarityType.STRUCTURAL, 0)
        functional_sim = similarities.get(SimilarityType.FUNCTIONAL, 0)
        
        if structural_sim > functional_sim:
            return AnalogyType.STRUCTURAL
        elif functional_sim > 0.7:
            return AnalogyType.FUNCTIONAL
        else:
            return AnalogyType.DIRECT
    
    def _analyze_mapping_details(self, source: KnowledgeConcept, target: KnowledgeConcept) -> Dict[str, Any]:
        """分析映射细节"""
        return {
            'shared_properties': list(set(source.properties.keys()).intersection(set(target.properties.keys()))),
            'shared_relations': list(set(source.relations.keys()).intersection(set(target.relations.keys()))),
            'domain_transfer': source.domain != target.domain,
            'abstractness_match': abs(source.abstractness - target.abstractness) < 0.3
        }
    
    def _extract_concept_from_text(self, text: str) -> Optional[KnowledgeConcept]:
        """从文本提取概念"""
        # 简化的概念提取
        words = text.split()[:10]  # 取前10个词
        
        return KnowledgeConcept(
            concept_id=f"extracted_{hash(text)}",
            name=" ".join(words),
            description=text,
            properties={'text_length': len(text), 'word_count': len(words)},
            relations={},
            examples=[text]
        )
    
    def _apply_analogical_reasoning(self, source_solution: str, analogy: AnalogyMapping) -> str:
        """应用类比推理"""
        # 简化的类比应用
        source = analogy.source_concept
        target = analogy.target_concept
        
        reasoning = f"""
        基于类比推理：
        源概念: {source.name}
        目标概念: {target.name}
        相似性: {analogy.similarity_score:.3f}
        
        源解决方案: {source_solution}
        
        类比应用：
        由于{source.name}和{target.name}在{analogy.mapping_type.value}方面相似，
        可以考虑将源解决方案的思路应用到目标问题上。
        """
        
        return reasoning.strip()


class PatternRecognizer:
    """模式识别器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.pattern_cache: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def recognize_patterns(self, data: List[Any], pattern_types: List[str] = None) -> List[Dict[str, Any]]:
        """识别模式"""
        if pattern_types is None:
            pattern_types = ['sequential', 'numerical', 'structural', 'behavioral']
        
        patterns = []
        
        for pattern_type in pattern_types:
            if pattern_type == 'sequential':
                patterns.extend(self._recognize_sequential_patterns(data))
            elif pattern_type == 'numerical':
                patterns.extend(self._recognize_numerical_patterns(data))
            elif pattern_type == 'structural':
                patterns.extend(self._recognize_structural_patterns(data))
            elif pattern_type == 'behavioral':
                patterns.extend(self._recognize_behavioral_patterns(data))
        
        return patterns
    
    def _recognize_sequential_patterns(self, data: List[Any]) -> List[Dict[str, Any]]:
        """识别序列模式"""
        patterns = []
        
        # 检查递增/递减序列
        if self._is_monotonic_sequence(data):
            patterns.append({
                'type': 'sequential',
                'subtype': 'monotonic',
                'direction': 'increasing' if data[-1] > data[0] else 'decreasing',
                'confidence': 0.9,
                'pattern': '单调序列'
            })
        
        # 检查周期性模式
        if self._has_periodic_pattern(data):
            patterns.append({
                'type': 'sequential',
                'subtype': 'periodic',
                'confidence': 0.7,
                'pattern': '周期性序列'
            })
        
        return patterns
    
    def _recognize_numerical_patterns(self, data: List[Any]) -> List[Dict[str, Any]]:
        """识别数值模式"""
        patterns = []
        numeric_data = []
        
        # 提取数值
        for item in data:
            try:
                numeric_data.append(float(item))
            except (ValueError, TypeError):
                continue
        
        if len(numeric_data) < 2:
            return patterns
        
        # 算术级数
        if self._is_arithmetic_progression(numeric_data):
            patterns.append({
                'type': 'numerical',
                'subtype': 'arithmetic_progression',
                'common_difference': numeric_data[1] - numeric_data[0],
                'confidence': 0.8,
                'pattern': '算术级数'
            })
        
        # 几何级数
        if self._is_geometric_progression(numeric_data):
            patterns.append({
                'type': 'numerical',
                'subtype': 'geometric_progression',
                'common_ratio': numeric_data[1] / numeric_data[0] if numeric_data[0] != 0 else 1,
                'confidence': 0.8,
                'pattern': '几何级数'
            })
        
        return patterns
    
    def _recognize_structural_patterns(self, data: List[Any]) -> List[Dict[str, Any]]:
        """识别结构模式"""
        patterns = []
        
        # 检查重复结构
        if self._has_repeating_structure(data):
            patterns.append({
                'type': 'structural',
                'subtype': 'repeating',
                'confidence': 0.7,
                'pattern': '重复结构'
            })
        
        return patterns
    
    def _recognize_behavioral_patterns(self, data: List[Any]) -> List[Dict[str, Any]]:
        """识别行为模式"""
        patterns = []
        
        # 检查行为序列
        if len(data) > 3:
            patterns.append({
                'type': 'behavioral',
                'subtype': 'action_sequence',
                'confidence': 0.6,
                'pattern': '行为序列'
            })
        
        return patterns
    
    def _is_monotonic_sequence(self, data: List[Any]) -> bool:
        """检查单调序列"""
        if len(data) < 2:
            return False
        
        try:
            numeric_data = [float(x) for x in data if isinstance(x, (int, float))]
            if len(numeric_data) < 2:
                return False
            
            increasing = all(numeric_data[i] <= numeric_data[i+1] for i in range(len(numeric_data)-1))
            decreasing = all(numeric_data[i] >= numeric_data[i+1] for i in range(len(numeric_data)-1))
            
            return increasing or decreasing
        except (ValueError, TypeError):
            return False
    
    def _has_periodic_pattern(self, data: List[Any]) -> bool:
        """检查周期性模式"""
        # 简化的周期性检测
        if len(data) < 6:
            return False
        
        # 检查是否每3个元素重复
        period = 3
        for i in range(len(data) - period):
            if data[i] == data[i + period] and data[i + 1] == data[i + period + 1]:
                return True
        
        return False
    
    def _is_arithmetic_progression(self, data: List[float]) -> bool:
        """检查算术级数"""
        if len(data) < 3:
            return False
        
        diff = data[1] - data[0]
        tolerance = abs(diff) * 0.1  # 10%容忍度
        
        for i in range(2, len(data)):
            if abs((data[i] - data[i-1]) - diff) > tolerance:
                return False
        
        return True
    
    def _is_geometric_progression(self, data: List[float]) -> bool:
        """检查几何级数"""
        if len(data) < 3 or any(x == 0 for x in data):
            return False
        
        ratio = data[1] / data[0]
        tolerance = abs(ratio) * 0.1  # 10%容忍度
        
        for i in range(2, len(data)):
            if data[i-1] == 0 or abs((data[i] / data[i-1]) - ratio) > tolerance:
                return False
        
        return True
    
    def _has_repeating_structure(self, data: List[Any]) -> bool:
        """检查重复结构"""
        if len(data) < 4:
            return False
        
        # 检查前一半是否重复后一半
        half = len(data) // 2
        return data[:half] == data[half:half*2]


class InnovationEngine:
    """创新引擎"""
    
    def __init__(self, analogy_engine: AnalogyEngine, pattern_recognizer: PatternRecognizer):
        self.analogy_engine = analogy_engine
        self.pattern_recognizer = pattern_recognizer
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 创新模板
        self.innovation_templates = [
            "结合{concept1}和{concept2}的{feature}特性",
            "将{domain1}的{method}应用于{domain2}",
            "用{technology}改造{traditional_approach}",
            "通过{mechanism}实现{goal}的新方法",
            "借鉴{bio_system}的{mechanism}设计{artificial_system}"
        ]
        
        self.creativity_strategies = [
            "类比创新",
            "组合创新",
            "分解重组",
            "逆向思考",
            "跨界迁移"
        ]
    
    def generate_creative_solutions(self, problem: str, context: Dict[str, Any]) -> List[CreativeSolution]:
        """生成创造性解决方案"""
        solutions = []
        
        # 分析问题
        problem_analysis = self._analyze_problem(problem, context)
        
        # 应用不同创新策略
        for strategy in self.creativity_strategies:
            strategy_solutions = self._apply_innovation_strategy(strategy, problem_analysis)
            solutions.extend(strategy_solutions)
        
        # 评估和排序
        for solution in solutions:
            solution.novelty_score = self._evaluate_novelty(solution, context)
            solution.feasibility_score = self._evaluate_feasibility(solution, context)
            solution.effectiveness_score = self._evaluate_effectiveness(solution, problem_analysis)
        
        # 按总体评分排序
        solutions.sort(key=lambda x: x.get_overall_score(), reverse=True)
        
        return solutions
    
    def _analyze_problem(self, problem: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """分析问题"""
        analysis = {
            'problem_type': self._classify_problem_type(problem),
            'complexity_level': self._assess_problem_complexity(problem),
            'domain': context.get('domain', 'general'),
            'constraints': context.get('constraints', []),
            'resources': context.get('available_resources', [])
        }
        
        # 识别关键概念
        key_concepts = self._extract_key_concepts(problem)
        analysis['key_concepts'] = key_concepts
        
        # 识别相关模式
        patterns = self.pattern_recognizer.recognize_patterns(key_concepts)
        analysis['patterns'] = patterns
        
        return analysis
    
    def _apply_innovation_strategy(self, strategy: str, problem_analysis: Dict[str, Any]) -> List[CreativeSolution]:
        """应用创新策略"""
        if strategy == "类比创新":
            return self._analogical_innovation(problem_analysis)
        elif strategy == "组合创新":
            return self._combinatorial_innovation(problem_analysis)
        elif strategy == "分解重组":
            return self._decompositional_innovation(problem_analysis)
        elif strategy == "逆向思考":
            return self._reverse_thinking_innovation(problem_analysis)
        elif strategy == "跨界迁移":
            return self._cross_domain_innovation(problem_analysis)
        else:
            return []
    
    def _analogical_innovation(self, analysis: Dict[str, Any]) -> List[CreativeSolution]:
        """类比创新"""
        solutions = []
        
        key_concepts = analysis['key_concepts']
        
        # 为每个关键概念寻找类比
        for concept in key_concepts[:2]:  # 限制数量
            analogies = self.analogy_engine.find_analogys(concept, min_similarity=0.3)
            
            for analogy in analogies[:3]:  # 取前3个类比
                solution = CreativeSolution(
                    solution_id=f"analogy_{len(solutions)}",
                    description=f"基于{analogy.source_concept.name}到{analogy.target_concept.name}的类比创新方案",
                    creativity_level=CreativityLevel.INNOVATIVE,
                    novelty_score=0.8,
                    feasibility_score=analogy.confidence,
                    effectiveness_score=0.7,
                    source_analogies=[analogy],
                    innovation_features=[f"跨域类比：{analogy.mapping_type.value}"],
                    implementation_steps=["识别类比映射", "转换解决思路", "适应目标领域", "验证有效性"]
                )
                solutions.append(solution)
        
        return solutions
    
    def _combinatorial_innovation(self, analysis: Dict[str, Any]) -> List[CreativeSolution]:
        """组合创新"""
        solutions = []
        key_concepts = analysis['key_concepts']
        
        if len(key_concepts) >= 2:
            # 概念组合
            for i in range(len(key_concepts)):
                for j in range(i+1, len(key_concepts)):
                    concept1 = key_concepts[i]
                    concept2 = key_concepts[j]
                    
                    solution = CreativeSolution(
                        solution_id=f"combo_{len(solutions)}",
                        description=f"结合{concept1.name}和{concept2.name}的混合解决方案",
                        creativity_level=CreativityLevel.ADAPTIVE,
                        novelty_score=0.6,
                        feasibility_score=0.8,
                        effectiveness_score=0.7,
                        source_analogies=[],
                        innovation_features=["多概念融合", "协同效应"],
                        implementation_steps=["识别组合机会", "设计融合机制", "优化组合效果", "测试验证"]
                    )
                    solutions.append(solution)
        
        return solutions
    
    def _decompositional_innovation(self, analysis: Dict[str, Any]) -> List[CreativeSolution]:
        """分解重组创新"""
        solutions = []
        
        solution = CreativeSolution(
            solution_id="decomp_0",
            description="通过分解问题组件并重新组织来解决",
            creativity_level=CreativityLevel.ADAPTIVE,
            novelty_score=0.5,
            feasibility_score=0.9,
            effectiveness_score=0.8,
            source_analogies=[],
            innovation_features=["结构重组", "流程优化"],
            implementation_steps=["分解复杂问题", "识别关键组件", "重新组织流程", "整合解决方案"]
        )
        solutions.append(solution)
        
        return solutions
    
    def _reverse_thinking_innovation(self, analysis: Dict[str, Any]) -> List[CreativeSolution]:
        """逆向思考创新"""
        solutions = []
        
        solution = CreativeSolution(
            solution_id="reverse_0",
            description="采用逆向思维方法，从结果反推解决方案",
            creativity_level=CreativityLevel.INNOVATIVE,
            novelty_score=0.7,
            feasibility_score=0.6,
            effectiveness_score=0.7,
            source_analogies=[],
            innovation_features=["逆向推理", "颠覆性思考"],
            implementation_steps=["定义理想结果", "逆向推理过程", "识别关键步骤", "实施方案"]
        )
        solutions.append(solution)
        
        return solutions
    
    def _cross_domain_innovation(self, analysis: Dict[str, Any]) -> List[CreativeSolution]:
        """跨界迁移创新"""
        solutions = []
        
        source_domain = analysis['domain']
        target_domains = ['technology', 'biology', 'nature', 'art', 'business']
        
        for target_domain in target_domains[:2]:  # 限制数量
            if target_domain != source_domain:
                solution = CreativeSolution(
                    solution_id=f"cross_{len(solutions)}",
                    description=f"从{source_domain}领域借鉴方法到{target_domain}领域",
                    creativity_level=CreativityLevel.BREAKTHROUGH,
                    novelty_score=0.9,
                    feasibility_score=0.5,
                    effectiveness_score=0.6,
                    source_analogies=[],
                    innovation_features=["跨界迁移", "领域融合"],
                    implementation_steps=["研究源领域方法", "分析适配性", "调整适应新领域", "验证效果"]
                )
                solutions.append(solution)
        
        return solutions
    
    def _classify_problem_type(self, problem: str) -> str:
        """分类问题类型"""
        if 'how' in problem.lower() or '如何' in problem:
            return "methodological"
        elif 'why' in problem.lower() or '为什么' in problem:
            return "causal"
        elif 'what' in problem.lower() or '什么' in problem:
            return "descriptive"
        else:
            return "general"
    
    def _assess_problem_complexity(self, problem: str) -> str:
        """评估问题复杂度"""
        word_count = len(problem.split())
        if word_count <= 10:
            return "simple"
        elif word_count <= 30:
            return "moderate"
        else:
            return "complex"
    
    def _extract_key_concepts(self, problem: str) -> List[KnowledgeConcept]:
        """提取关键概念"""
        words = problem.split()
        concepts = []
        
        for i, word in enumerate(words[:10]):  # 限制前10个词
            concept = KnowledgeConcept(
                concept_id=f"concept_{i}",
                name=word,
                description=f"概念: {word}",
                properties={'position': i, 'word_length': len(word)},
                relations={},
                examples=[word]
            )
            concepts.append(concept)
        
        return concepts
    
    def _evaluate_novelty(self, solution: CreativeSolution, context: Dict[str, Any]) -> float:
        """评估新颖性"""
        # 基于创意水平和创新特征
        base_score = {
            CreativityLevel.ROUTINE: 0.3,
            CreativityLevel.ADAPTIVE: 0.5,
            CreativityLevel.INNOVATIVE: 0.8,
            CreativityLevel.BREAKTHROUGH: 0.95
        }.get(solution.creativity_level, 0.5)
        
        # 基于创新特征数量
        feature_bonus = min(len(solution.innovation_features) * 0.1, 0.3)
        
        return min(base_score + feature_bonus, 1.0)
    
    def _evaluate_feasibility(self, solution: CreativeSolution, context: Dict[str, Any]) -> float:
        """评估可行性"""
        # 基于实现步骤复杂度
        step_complexity = len(solution.implementation_steps) / 10.0
        
        # 基于类比映射的可信度
        analogy_confidence = 0.0
        if solution.source_analogies:
            analogy_confidence = np.mean([a.confidence for a in solution.source_analogies])
        
        return max(0.1, 0.8 - step_complexity + analogy_confidence * 0.2)
    
    def _evaluate_effectiveness(self, solution: CreativeSolution, analysis: Dict[str, Any]) -> float:
        """评估有效性"""
        # 基于问题类型匹配
        problem_match = 0.8  # 默认匹配度
        
        # 基于资源可用性
        available_resources = analysis.get('resources', [])
        required_resources = len(solution.implementation_steps)
        
        resource_ratio = len(available_resources) / max(required_resources, 1)
        
        return min(problem_match * resource_ratio, 1.0)


class AnalogicalLearner(BaseModule):
    """类比学习器"""
    
    def __init__(self):
        module_config = ModuleConfig("analogical_learner", version="1.0")
        super().__init__(module_config)
        
        # 核心组件
        self.similarity_calculator = SimilarityCalculator()
        self.analogy_engine = AnalogyEngine(self.similarity_calculator)
        self.pattern_recognizer = PatternRecognizer()
        self.innovation_engine = InnovationEngine(self.analogy_engine, self.pattern_recognizer)
        
        # 学习历史
        self.learning_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            'analogy_success_rate': 0.0,
            'creativity_score': 0.0,
            'innovation_count': 0
        }
    
    def learn_from_example(self, source_problem: str, source_solution: str,
                          success: bool = True) -> bool:
        """从示例中学习"""
        self.state = ModuleState.ACTIVE
        
        try:
            # 提取源概念
            source_concept = self.analogy_engine._extract_concept_from_text(source_problem)
            
            if not source_concept:
                return False
            
            # 学习（简化实现）
            learning_record = {
                'timestamp': time.time(),
                'source_problem': source_problem,
                'source_solution': source_solution,
                'success': success,
                'learning_type': 'example_based'
            }
            
            self.learning_history.append(learning_record)
            
            # 更新性能指标
            self._update_performance_metrics(success)
            
            self.logger.info(f"从示例学习: {source_problem[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"学习失败: {e}")
            return False
    
    def solve_problem_creatively(self, problem: str, context: Dict[str, Any]) -> List[CreativeSolution]:
        """创造性解决问题"""
        self.state = ModuleState.ACTIVE
        
        try:
            self.logger.info(f"开始创造性解决问题: {problem[:50]}...")
            
            # 生成解决方案
            solutions = self.innovation_engine.generate_creative_solutions(problem, context)
            
            # 记录解决过程
            solving_record = {
                'timestamp': time.time(),
                'problem': problem,
                'context': context,
                'solutions_generated': len(solutions),
                'avg_novelty': np.mean([s.novelty_score for s in solutions]) if solutions else 0,
                'best_solution_score': solutions[0].get_overall_score() if solutions else 0
            }
            
            self.learning_history.append(solving_record)
            
            self.state = ModuleState.COMPLETED
            self.logger.info(f"生成了 {len(solutions)} 个创造性解决方案")
            
            return solutions
            
        except Exception as e:
            self.logger.error(f"创造性解决问题失败: {e}")
            self.state = ModuleState.ERROR
            return []
    
    def find_analogies(self, concept: KnowledgeConcept, target_domain: str = None) -> List[AnalogyMapping]:
        """寻找类比"""
        return self.analogy_engine.find_analogies(concept, target_domain)
    
    def recognize_patterns(self, data: List[Any]) -> List[Dict[str, Any]]:
        """识别模式"""
        return self.pattern_recognizer.recognize_patterns(data)
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计"""
        if not self.learning_history:
            return {'total_learning_events': 0}
        
        recent_history = self.learning_history[-10:]  # 最近10次
        
        return {
            'total_learning_events': len(self.learning_history),
            'recent_success_rate': np.mean([1 if h.get('success', False) else 0 for h in recent_history]),
            'avg_creativity_score': np.mean([h.get('best_solution_score', 0) for h in recent_history if 'solutions_generated' in h]),
            'total_innovations': sum([h.get('solutions_generated', 0) for h in self.learning_history]),
            'performance_metrics': self.performance_metrics.copy()
        }
    
    def _update_performance_metrics(self, success: bool):
        """更新性能指标"""
        # 更新类比成功率
        success_count = sum([1 for h in self.learning_history if h.get('success', False)])
        total_count = len(self.learning_history)
        
        if total_count > 0:
            self.performance_metrics['analogy_success_rate'] = success_count / total_count
        
        # 更新创新数量
        innovation_count = sum([h.get('solutions_generated', 0) for h in self.learning_history])
        self.performance_metrics['innovation_count'] = innovation_count
        
        # 计算创造力分数
        creativity_scores = [h.get('best_solution_score', 0) for h in self.learning_history if 'solutions_generated' in h]
        self.performance_metrics['creativity_score'] = np.mean(creativity_scores) if creativity_scores else 0.0
    
    def initialize(self) -> bool:
        """初始化类比学习器"""
        self.state = ModuleState.INITIALIZED
        return True
    
    def cleanup(self) -> bool:
        """清理类比学习器"""
        self.learning_history = []
        self.performance_metrics = {
            'analogy_success_rate': 0.0,
            'creativity_score': 0.0,
            'innovation_count': 0
        }
        return True


# 便利函数
def create_analogical_learner() -> AnalogicalLearner:
    """创建类比学习器"""
    return AnalogicalLearner()


def create_creative_problem_solver() -> InnovationEngine:
    """创建创造性问题解决器"""
    similarity_calc = SimilarityCalculator()
    analogy_engine = AnalogyEngine(similarity_calc)
    pattern_recognizer = PatternRecognizer()
    return InnovationEngine(analogy_engine, pattern_recognizer)