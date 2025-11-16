"""
知识抽象算法 (Abstraction Algorithms)
====================================

实现新皮层的知识抽象和概念形成机制：
- 稀疏显式单元（概念单元）
- 分布式群体表征
- 语义抽象与泛化
- 概念整合与迁移

基于论文：
- "Representation of abstract semantic knowledge in populations of human single neurons"
- "Concept cells: the building blocks of declarative memory functions"
- "Dual coding of knowledge in the human brain"
- "Learning-associated astrocyte ensembles regulate memory recall"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math
import random


class AbstractionLevel(Enum):
    """抽象层次"""
    SENSORY = "sensory"          # 感觉层面
    PERCEPTUAL = "perceptual"    # 感知层面
    CONCEPTUAL = "conceptual"    # 概念层面
    SEMANTIC = "semantic"        # 语义层面
    ABSTRACT = "abstract"        # 抽象层面


class ConceptType(Enum):
    """概念类型"""
    OBJECT = "object"            # 物体概念
    ACTION = "action"            # 动作概念
    PROPERTY = "property"        # 属性概念
    RELATION = "relation"        # 关系概念
    EMOTION = "emotion"          # 情感概念
    SOCIAL = "social"            # 社会概念


@dataclass
class ConceptConfig:
    """概念单元配置"""
    concept_id: str
    concept_type: ConceptType
    abstraction_level: AbstractionLevel
    
    # 概念单元参数
    activation_threshold: float = 0.7    # 激活阈值
    selectivity_sharpness: float = 3.0   # 选择性锐度
    sparsity_level: float = 0.1          # 稀疏程度
    
    # 学习参数
    learning_rate: float = 0.01
    consolidation_rate: float = 0.005
    forgetting_rate: float = 0.001
    
    # 关联参数
    association_strength: float = 0.8
    association_decay: float = 0.95
    
    # 泛化参数
    generalization_range: float = 0.3
    transfer_weight: float = 0.5


class ConceptUnit(nn.Module):
    """
    概念单元 (Concept Cell)
    
    模拟新皮层中的稀疏显式概念编码单元，对特定概念表现出高度选择性响应。
    基于MTL中概念单元的发现：稀疏而明确的编码，支持记忆和泛化。
    """
    
    def __init__(self, config: ConceptConfig, input_dim: int):
        super().__init__()
        self.config = config
        self.concept_id = config.concept_id
        self.concept_type = config.concept_type
        self.abstraction_level = config.abstraction_level
        
        # 概念选择性权重（决定单元对什么概念响应）
        self.selectivity_weights = nn.Parameter(torch.randn(input_dim))
        
        # 概念强度编码器
        self.concept_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 选择性调制器（控制响应的选择性）
        self.selectivity_modulator = nn.Sequential(
            nn.Linear(1, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim),
            nn.Sigmoid()
        )
        
        # 稀疏性调节器（实现稀疏激活）
        self.sparsity_regulator = SparsityController(config.sparsity_level)
        
        # 概念巩固器（长期记忆巩固）
        self.consolidator = ConceptConsolidator(config)
        
    def forward(self, input_features: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        概念单元前向传播
        
        Args:
            input_features: 输入特征 [batch, input_dim]
            context: 上下文信息
            
        Returns:
            概念响应和相关信息
        """
        batch_size = input_features.shape[0]
        
        # 计算概念选择性响应
        selective_response = F.linear(input_features, self.selectivity_weights)
        
        # 概念强度编码
        concept_strength = self.concept_encoder(input_features)  # [batch, 1]
        
        # 选择性调制
        selectivity_modulation = self.selectivity_modulator(concept_strength)
        modulated_response = selective_response * selectivity_modulation.sum(dim=-1, keepdim=True)
        
        # 应用稀疏性调节
        sparse_response = self.sparsity_regulator(modulated_response)
        
        # 激活判断（超过阈值才激活）
        activation = (sparse_response > self.config.activation_threshold).float()
        
        # 概念巩固（如果单元被激活）
        consolidated_response = self.consolidator(sparse_response, concept_strength)
        
        # 计算概念置信度
        confidence = torch.sigmoid(sparse_response * self.config.selectivity_sharpness)
        
        # 泛化能力评估
        generalization_score = self._compute_generalization(
            input_features, concept_strength
        )
        
        return {
            'concept_response': consolidated_response,
            'activation': activation,
            'confidence': confidence,
            'concept_strength': concept_strength,
            'selectivity': selectivity_modulation,
            'sparse_response': sparse_response,
            'generalization_score': generalization_score,
            'concept_id': self.concept_id,
            'concept_type': self.concept_type.value,
            'abstraction_level': self.abstraction_level.value
        }
    
    def _compute_generalization(self, features: torch.Tensor, 
                              strength: torch.Tensor) -> torch.Tensor:
        """计算概念单元的泛化能力"""
        # 基于概念强度的泛化评估
        # 强度越高，泛化能力越强（概念更稳定）
        base_generalization = torch.sigmoid(strength * 2 - 1)
        
        # 结合输入特征的统计多样性
        feature_diversity = 1.0 / (1.0 + torch.var(features, dim=-1, keepdim=True))
        
        # 综合泛化分数
        generalization = base_generalization * feature_diversity
        
        return generalization


class SparsityController(nn.Module):
    """稀疏性控制器 - 实现神经元的稀疏激活特性"""
    
    def __init__(self, target_sparsity: float):
        super().__init__()
        self.target_sparsity = target_sparsity
        
        # 稀疏性门控
        self.sparsity_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, responses: torch.Tensor) -> torch.Tensor:
        """应用稀疏性控制"""
        # 计算当前响应的稀疏性
        current_sparsity = self._compute_sparsity(responses)
        
        # 生成稀疏性调节信号
        sparsity_adjustment = self.sparsity_gate(current_sparsity.unsqueeze(-1))
        
        # 应用稀疏性调节
        # 目标稀疏性越高，响应越稀疏
        adjustment_factor = 1.0 - self.target_sparsity
        sparse_responses = responses * adjustment_factor * sparsity_adjustment.squeeze(-1)
        
        return sparse_responses
    
    def _compute_sparsity(self, responses: torch.Tensor) -> torch.Tensor:
        """计算响应向量的稀疏性（基于零元素比例）"""
        zero_mask = (responses.abs() < 0.1).float()
        sparsity = zero_mask.mean(dim=-1)
        return sparsity


class ConceptConsolidator(nn.Module):
    """概念巩固器 - 模拟长期记忆巩固机制"""
    
    def __init__(self, config: ConceptConfig):
        super().__init__()
        self.consolidation_rate = config.consolidation_rate
        self.forgetting_rate = config.forgetting_rate
        
        # 巩固强度调节器
        self.consolidation_strengthener = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 遗忘门控器
        self.forgetting_gater = nn.Sequential(
            nn.Linear(2, 8),  # 输入特征和概念强度
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
    def forward(self, response: torch.Tensor, 
                concept_strength: torch.Tensor) -> torch.Tensor:
        """概念巩固前向传播"""
        # 计算巩固强度
        consolidation_signal = self.consolidation_strengthener(concept_strength)
        
        # 计算遗忘信号
        forgetting_input = torch.cat([response.abs(), concept_strength], dim=-1)
        forgetting_signal = self.forgetting_gater(forgetting_input)
        
        # 巩固：增强响应
        consolidated = response + self.consolidation_rate * consolidation_signal.squeeze(-1) * response
        
        # 遗忘：衰减响应
        consolidated = consolidated * (1.0 - self.forgetting_rate * forgetting_signal.squeeze(-1))
        
        return consolidated


class SemanticAbstraction(nn.Module):
    """
    语义抽象引擎
    
    实现从具体表征到抽象概念的转换：
    - 概念聚类与抽象
    - 语义关系发现
    - 抽象层次构建
    - 概念迁移与泛化
    """
    
    def __init__(self, input_dim: int, num_abstraction_levels: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_abstraction_levels = num_abstraction_levels
        
        # 各抽象层次的编码器
        self.abstraction_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else input_dim // 2, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, input_dim // 4),
                nn.LayerNorm(input_dim // 4)
            ) for i in range(num_abstraction_levels)
        ])
        
        # 概念聚类器
        self.concept_cluster = ConceptCluster(input_dim // 4, num_abstraction_levels)
        
        # 语义关系建模器
        self.relation_modeler = SemanticRelationModeler(input_dim // 4)
        
        # 抽象层次控制器
        self.abstraction_controller = AbstractionController(num_abstraction_levels)
        
        # 迁移学习器
        self.transfer_learner = TransferLearner(input_dim // 4)
        
    def forward(self, concrete_features: torch.Tensor, 
                concept_context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        语义抽象前向传播
        
        Args:
            concrete_features: 具体特征 [batch, input_dim]
            concept_context: 概念上下文信息
            
        Returns:
            各抽象层次的结果
        """
        batch_size = concrete_features.shape[0]
        
        # 多层次抽象编码
        abstraction_levels = {}
        current_features = concrete_features
        
        for level in range(self.num_abstraction_levels):
            # 当前层次的抽象编码
            abstracted = self.abstraction_encoders[level](current_features)
            abstraction_levels[f"level_{level}"] = {
                'features': abstracted,
                'level': level
            }
            
            # 为下一层准备输入（可能是当前层特征或其他组合）
            current_features = abstracted
        
        # 概念聚类（在最抽象层次）
        top_level_features = abstraction_levels[f"level_{self.num_abstraction_levels-1}"]['features']
        cluster_results = self.concept_cluster(top_level_features)
        
        # 语义关系建模
        relation_results = self.relation_modeler(top_level_features)
        
        # 抽象层次控制
        control_signals = self.abstraction_controller(abstraction_levels)
        
        # 迁移学习
        transfer_results = self.transfer_learner(
            concrete_features, top_level_features, concept_context
        )
        
        # 整合所有结果
        return {
            'abstraction_levels': abstraction_levels,
            'concept_clusters': cluster_results,
            'semantic_relations': relation_results,
            'control_signals': control_signals,
            'transfer_learning': transfer_results,
            'final_abstract_representation': top_level_features,
            'abstraction_strength': control_signals['abstraction_strength']
        }


class ConceptCluster(nn.Module):
    """概念聚类器 - 将相似的概念进行聚类"""
    
    def __init__(self, feature_dim: int, max_clusters: int = 50):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_clusters = max_clusters
        
        # 聚类中心学习器
        self.cluster_center_learner = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 相似性度量器
        self.similarity_measure = nn.CosineSimilarity(dim=-1)
        
        # 聚类分配器
        self.cluster_assign = nn.Sequential(
            nn.Linear(feature_dim, max_clusters),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """概念聚类"""
        batch_size = features.shape[0]
        
        # 学习聚类中心
        cluster_centers = self.cluster_center_learner(features.mean(dim=0, keepdim=True))
        
        # 计算相似性
        similarities = self.similarity_measure(
            features.unsqueeze(1), 
            cluster_centers.unsqueeze(0)
        )
        
        # 分配到最相似的聚类
        cluster_assignments = self.cluster_assign(features)
        
        # 计算聚类质量
        cluster_quality = torch.max(similarities, dim=-1)[0]
        
        # 更新聚类中心（在线学习）
        updated_centers = self._update_centers(features, cluster_assignments, similarities)
        
        return {
            'cluster_centers': updated_centers,
            'similarities': similarities,
            'assignments': cluster_assignments,
            'cluster_quality': cluster_quality,
            'num_active_clusters': torch.sum(cluster_quality > 0.5)
        }
    
    def _update_centers(self, features: torch.Tensor, 
                       assignments: torch.Tensor, 
                       similarities: torch.Tensor) -> torch.Tensor:
        """在线更新聚类中心"""
        # 基于相似性和分配权重更新中心
        weights = F.softmax(similarities, dim=-1)
        weighted_features = features.unsqueeze(1) * weights.unsqueeze(-1)
        
        # 新的聚类中心
        new_centers = weighted_features.sum(dim=0) / weights.sum(dim=0, keepdim=True)
        
        return new_centers


class SemanticRelationModeler(nn.Module):
    """语义关系建模器 - 发现概念间的语义关系"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 关系类型识别器
        self.relation_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 10),  # 10种关系类型
            nn.LogSoftmax(dim=-1)
        )
        
        # 关系强度计算器
        self.relation_strength_calc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """语义关系建模"""
        batch_size, feature_dim = features.shape
        
        # 计算所有概念对之间的关系
        relation_pairs = []
        relation_strengths = []
        
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                pair_features = torch.cat([features[i], features[j]], dim=-1)
                relation_pairs.append(pair_features)
                
                strength = self.relation_strength_calc(pair_features)
                relation_strengths.append(strength)
        
        if len(relation_pairs) > 0:
            relation_pairs = torch.stack(relation_pairs)
            relation_strengths = torch.stack(relation_strengths)
            
            # 分类关系类型
            relation_types = self.relation_classifier(relation_pairs)
            
            # 计算最可能的语义关系
            most_likely_relations = torch.argmax(relation_types, dim=-1)
            
        else:
            relation_types = torch.zeros(0, 10)
            most_likely_relations = torch.zeros(0, dtype=torch.long)
            relation_strengths = torch.zeros(0, 1)
        
        return {
            'relation_types': relation_types,
            'most_likely_relations': most_likely_relations,
            'relation_strengths': relation_strengths,
            'num_relations': len(relation_pairs)
        }


class AbstractionController(nn.Module):
    """抽象层次控制器 - 控制在不同抽象层次上的处理"""
    
    def __init__(self, num_levels: int):
        super().__init__()
        self.num_levels = num_levels
        
        # 抽象强度控制器
        self.abstraction_strengthener = nn.Sequential(
            nn.Linear(num_levels, num_levels * 2),
            nn.ReLU(),
            nn.Linear(num_levels * 2, 1),
            nn.Sigmoid()
        )
        
        # 层次注意力生成器
        self.level_attention = nn.Parameter(torch.ones(num_levels) / num_levels)
        
    def forward(self, abstraction_levels: Dict) -> Dict[str, torch.Tensor]:
        """抽象层次控制"""
        # 计算抽象强度
        level_features = torch.stack([
            level['features'].mean(dim=0) for level in abstraction_levels.values()
        ])
        
        abstraction_strength = self.abstraction_strengthener(level_features)
        
        # 生成层次注意力
        level_weights = F.softmax(self.level_attention, dim=0)
        
        return {
            'abstraction_strength': abstraction_strength,
            'level_weights': level_weights,
            'controlled_features': level_features
        }


class TransferLearner(nn.Module):
    """迁移学习器 - 实现概念在不同领域间的迁移"""
    
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        
        # 迁移适配器
        self.transfer_adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 领域泛化器
        self.domain_generalizer = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 迁移质量评估器
        self.transfer_evaluator = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, concrete_features: torch.Tensor, 
                abstract_features: torch.Tensor,
                context: Optional[Dict]) -> Dict[str, torch.Tensor]:
        """迁移学习"""
        # 迁移适配
        adapted_features = self.transfer_adapter(abstract_features)
        
        # 领域泛化
        if context is not None and 'source_domain' in context:
            domain_context = context['source_domain']
            generalized = self.domain_generalizer(
                torch.cat([adapted_features, domain_context], dim=-1)
            )
        else:
            generalized = adapted_features
            
        # 评估迁移质量
        transfer_input = torch.cat([concrete_features, generalized], dim=-1)
        transfer_quality = self.transfer_evaluator(transfer_input)
        
        return {
            'adapted_features': adapted_features,
            'generalized_features': generalized,
            'transfer_quality': transfer_quality
        }


class AbstractionEngine(nn.Module):
    """
    抽象引擎 (Abstraction Engine)
    
    整合所有抽象组件的主引擎，实现完整的信息抽象流水线：
    - 概念单元激活
    - 语义抽象处理
    - 抽象层次构建
    - 概念迁移与泛化
    """
    
    def __init__(self, input_dim: int, concept_configs: List[ConceptConfig]):
        super().__init__()
        self.input_dim = input_dim
        self.concept_configs = concept_configs
        
        # 概念单元库
        self.concept_units = nn.ModuleList([
            ConceptUnit(config, input_dim) for config in concept_configs
        ])
        
        # 语义抽象引擎
        self.semantic_abstraction = SemanticAbstraction(input_dim)
        
        # 双编码系统（显式+分布）
        self.dual_coding = DualCodingSystem(input_dim, len(concept_configs))
        
        # 概念整合器
        self.concept_integrator = ConceptIntegrator(len(concept_configs))
        
        # 抽象记忆系统
        self.abstract_memory = AbstractMemorySystem(len(concept_configs))
        
    def forward(self, input_features: torch.Tensor,
                context: Optional[Dict] = None) -> Dict[str, Union[torch.Tensor, List, Dict]]:
        """
        抽象引擎前向传播
        
        Args:
            input_features: 输入特征 [batch, input_dim]
            context: 上下文信息
            
        Returns:
            完整的抽象处理结果
        """
        batch_size = input_features.shape[0]
        
        # 步骤1：概念单元激活
        concept_responses = []
        for concept_unit in self.concept_units:
            response = concept_unit(input_features, context)
            concept_responses.append(response)
        
        # 步骤2：语义抽象处理
        semantic_results = self.semantic_abstraction(input_features, context)
        
        # 步骤3：双编码系统处理
        dual_coding_results = self.dual_coding(concept_responses, semantic_results)
        
        # 步骤4：概念整合
        integration_results = self.concept_integrator(concept_responses)
        
        # 步骤5：抽象记忆处理
        memory_results = self.abstract_memory(concept_responses)
        
        # 整合最终结果
        final_abstraction = self._integrate_all_results(
            concept_responses, semantic_results, dual_coding_results,
            integration_results, memory_results
        )
        
        return {
            'concept_responses': concept_responses,
            'semantic_abstraction': semantic_results,
            'dual_coding': dual_coding_results,
            'integration': integration_results,
            'memory': memory_results,
            'final_abstraction': final_abstraction,
            'abstraction_summary': self._generate_summary(concept_responses, semantic_results)
        }
    
    def _integrate_all_results(self, concept_responses: List[Dict],
                             semantic_results: Dict,
                             dual_coding_results: Dict,
                             integration_results: Dict,
                             memory_results: Dict) -> Dict[str, torch.Tensor]:
        """整合所有抽象结果"""
        
        # 提取关键特征
        concept_activations = torch.stack([
            resp['concept_response'] for resp in concept_responses
        ])  # [num_concepts, batch_size]
        
        abstract_features = semantic_results['final_abstract_representation']
        
        # 整合显式概念和分布式表征
        integrated_representation = (
            0.3 * concept_activations.mean(dim=0) +  # 显式概念
            0.4 * abstract_features +               # 分布式抽象
            0.3 * dual_coding_results['integrated_representation']  # 双编码整合
        )
        
        # 计算抽象质量指标
        abstraction_quality = self._compute_abstraction_quality(
            concept_activations, abstract_features
        )
        
        return {
            'integrated_representation': integrated_representation,
            'abstraction_quality': abstraction_quality,
            'concept_strength': concept_activations.max(dim=0)[0],
            'semantic_coherence': semantic_results['abstraction_strength']
        }
    
    def _compute_abstraction_quality(self, concept_activations: torch.Tensor,
                                   abstract_features: torch.Tensor) -> torch.Tensor:
        """计算抽象质量"""
        # 概念激活的集中度（概念越稀疏，抽象质量越高）
        concept_sparsity = 1.0 / (1.0 + torch.var(concept_activations, dim=0).mean())
        
        # 抽象特征的稳定性
        feature_stability = torch.sigmoid(torch.norm(abstract_features, dim=-1))
        
        # 综合质量分数
        quality = (concept_sparsity + feature_stability) / 2
        
        return quality.unsqueeze(-1)
    
    def _generate_summary(self, concept_responses: List[Dict],
                         semantic_results: Dict) -> Dict[str, any]:
        """生成抽象处理摘要"""
        
        # 激活的概念数量
        active_concepts = sum(
            1 for resp in concept_responses 
            if resp['activation'].mean() > 0.5
        )
        
        # 主要概念类型
        concept_types = [resp['concept_type'] for resp in concept_responses
                        if resp['activation'].mean() > 0.5]
        
        # 抽象层次
        abstraction_level = semantic_results['abstraction_strength'].item()
        
        return {
            'num_active_concepts': active_concepts,
            'active_concept_types': list(set(concept_types)),
            'abstraction_level': abstraction_level,
            'generalization_score': semantic_results['transfer_learning']['transfer_quality'].mean().item()
        }


class DualCodingSystem(nn.Module):
    """双编码系统 - 显式概念单元 + 分布式群体表征"""
    
    def __init__(self, input_dim: int, num_concepts: int):
        super().__init__()
        self.input_dim = input_dim
        self.num_concepts = num_concepts
        
        # 分布式表征编码器
        self.distributed_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        
        # 显式-分布整合器
        self.explicit_distributed_integrator = nn.Sequential(
            nn.Linear(input_dim + num_concepts, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim)
        )
        
    def forward(self, concept_responses: List[Dict],
                semantic_results: Dict) -> Dict[str, torch.Tensor]:
        """双编码处理"""
        # 提取显式概念激活
        explicit_activations = torch.stack([
            resp['concept_response'] for resp in concept_responses
        ])  # [num_concepts, batch_size]
        
        # 分布式表征
        distributed_repr = self.distributed_encoder(
            semantic_results['final_abstract_representation']
        )
        
        # 整合显式和分布式表征
        combined_input = torch.cat([
            distributed_repr,
            explicit_activations.transpose(0, 1)
        ], dim=-1)
        
        integrated_representation = self.explicit_distributed_integrator(combined_input)
        
        return {
            'explicit_activations': explicit_activations,
            'distributed_representation': distributed_repr,
            'integrated_representation': integrated_representation
        }


class ConceptIntegrator(nn.Module):
    """概念整合器 - 整合多个概念单元的输出"""
    
    def __init__(self, num_concepts: int):
        super().__init__()
        self.num_concepts = num_concepts
        
        # 概念注意力
        self.concept_attention = nn.MultiheadAttention(
            embed_dim=1,  # 每个概念单元输出一个值
            num_heads=min(8, num_concepts),
            batch_first=True
        )
        
        # 概念权重学习器
        self.concept_weight_learner = nn.Sequential(
            nn.Linear(num_concepts, num_concepts),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, concept_responses: List[Dict]) -> Dict[str, torch.Tensor]:
        """概念整合"""
        batch_size = concept_responses[0]['concept_response'].shape[0]
        
        # 收集所有概念响应
        all_responses = torch.stack([
            resp['concept_response'] for resp in concept_responses
        ])  # [num_concepts, batch_size]
        
        # 概念注意力整合
        attended_responses, attention_weights = self.concept_attention(
            all_responses.unsqueeze(-1),  # query
            all_responses.unsqueeze(-1),  # key, value
            all_responses.unsqueeze(-1)
        )
        
        # 学习概念权重
        concept_weights = self.concept_weight_learner(all_responses.mean(dim=1))
        
        # 加权整合
        integrated_response = (all_responses * concept_weights.unsqueeze(-1)).sum(dim=0)
        
        return {
            'integrated_response': integrated_response,
            'concept_weights': concept_weights,
            'attention_weights': attention_weights.squeeze(-1),
            'response_variance': torch.var(all_responses, dim=0)
        }


class AbstractMemorySystem(nn.Module):
    """抽象记忆系统 - 管理抽象概念的存储和检索"""
    
    def __init__(self, num_concepts: int):
        super().__init__()
        self.num_concepts = num_concepts
        
        # 记忆编码器
        self.memory_encoder = nn.Sequential(
            nn.Linear(num_concepts, num_concepts // 2),
            nn.ReLU(),
            nn.Linear(num_concepts // 2, num_concepts // 4)
        )
        
        # 记忆检索器
        self.memory_retriever = nn.Sequential(
            nn.Linear(num_concepts // 4, num_concepts),
            nn.Sigmoid()
        )
        
    def forward(self, concept_responses: List[Dict]) -> Dict[str, torch.Tensor]:
        """抽象记忆处理"""
        # 编码当前概念状态
        concept_states = torch.stack([
            resp['concept_response'] for resp in concept_responses
        ])  # [num_concepts, batch_size]
        
        encoded_memory = self.memory_encoder(concept_states)
        
        # 检索相关记忆
        retrieved_memory = self.memory_retriever(encoded_memory)
        
        # 计算记忆相关性
        memory_relevance = F.cosine_similarity(
            concept_states, retrieved_memory, dim=0
        )
        
        return {
            'encoded_memory': encoded_memory,
            'retrieved_memory': retrieved_memory,
            'memory_relevance': memory_relevance,
            'memory_strength': torch.norm(encoded_memory, dim=0)
        }


# 工厂函数

def create_concept_units(input_dim: int, num_concepts: int = 20) -> List[ConceptUnit]:
    """创建概念单元集合"""
    concept_types = list(ConceptType)
    abstraction_levels = list(AbstractionLevel)
    
    concepts = []
    for i in range(num_concepts):
        config = ConceptConfig(
            concept_id=f"concept_{i}",
            concept_type=concept_types[i % len(concept_types)],
            abstraction_level=abstraction_levels[i // 5]  # 均匀分布抽象层次
        )
        concepts.append(ConceptUnit(config, input_dim))
    
    return concepts


def create_abstraction_engine(input_dim: int, num_concepts: int = 20) -> AbstractionEngine:
    """创建抽象引擎"""
    concept_configs = [
        ConceptConfig(
            concept_id=f"concept_{i}",
            concept_type=ConceptType.OBJECT,
            abstraction_level=AbstractionLevel.CONCEPTUAL
        )
        for i in range(num_concepts)
    ]
    
    return AbstractionEngine(input_dim, concept_configs)


if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建抽象引擎
    abstraction_engine = create_abstraction_engine(256, 20).to(device)
    
    # 测试输入
    test_features = torch.randn(4, 256).to(device)
    
    # 前向传播
    results = abstraction_engine(test_features)
    
    print(f"抽象引擎测试:")
    print(f"输入特征形状: {test_features.shape}")
    print(f"激活概念数量: {results['abstraction_summary']['num_active_concepts']}")
    print(f"抽象层次: {results['abstraction_summary']['abstraction_level']:.3f}")
    print(f"泛化分数: {results['abstraction_summary']['generalization_score']:.3f}")
    print(f"最终抽象表示形状: {results['final_abstraction']['integrated_representation'].shape}")
    print(f"抽象质量: {results['final_abstraction']['abstraction_quality'].mean().item():.3f}")
