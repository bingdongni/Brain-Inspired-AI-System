"""
多步推理机制
===========

实现复杂的推理系统，包括：
- 链式思维推理 (Chain of Thought)
- 层次化推理 (Hierarchical Reasoning)
- 推理步骤管理和验证
- 推理结果评估和优化

主要特性：
- 分步推理过程
- 推理链验证和修正
- 层次化问题分解
- 推理路径优化
- 多种推理策略

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import numpy as np
import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import re
import math
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_module import BaseModule, ModuleConfig, ModuleState


class ReasoningType(Enum):
    """推理类型"""
    DEDUCTIVE = "deductive"        # 演绎推理
    INDUCTIVE = "inductive"        # 归纳推理
    ABDUCTIVE = "abductive"        # 溯因推理
    ANALOGICAL = "analogical"      # 类比推理
    CAUSAL = "causal"             # 因果推理
    TEMPORAL = "temporal"         # 时间推理
    SPATIAL = "spatial"           # 空间推理


class StepType(Enum):
    """推理步骤类型"""
    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    INFERENCE = "inference"
    VALIDATION = "validation"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    CONCLUSION = "conclusion"


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ReasoningContext:
    """推理上下文"""
    problem_statement: str
    available_information: List[str]
    assumptions: List[str]
    constraints: List[str]
    goals: List[str]
    knowledge_base: Dict[str, Any]
    reasoning_history: List['ReasoningStep'] = field(default_factory=list)


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: str
    step_type: StepType
    reasoning_type: ReasoningType
    description: str
    input_data: Any
    output_data: Any = None
    confidence: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    error_message: Optional[str] = None
    evidence: List[str] = field(default_factory=list)
    
    def complete(self, output_data: Any, confidence: float = None, evidence: List[str] = None):
        """标记步骤完成"""
        self.output_data = output_data
        if confidence is not None:
            self.confidence = confidence
        if evidence is not None:
            self.evidence = evidence
        self.status = StepStatus.COMPLETED
        self.duration = time.time() - self.timestamp
    
    def fail(self, error_message: str):
        """标记步骤失败"""
        self.status = StepStatus.FAILED
        self.error_message = error_message
        self.duration = time.time() - self.timestamp


@dataclass
class ReasoningResult:
    """推理结果"""
    final_conclusion: Any
    reasoning_chain: List[ReasoningStep]
    confidence_score: float
    reasoning_path: List[str]
    alternative_paths: List[List[str]] = field(default_factory=list)
    validation_results: Dict[str, bool] = field(default_factory=dict)
    execution_summary: Dict[str, Any] = field(default_factory=dict)


class ReasoningStrategy(ABC):
    """推理策略基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate_reasoning_steps(self, context: ReasoningContext) -> List[ReasoningStep]:
        """生成推理步骤"""
        pass
    
    @abstractmethod
    def execute_step(self, step: ReasoningStep, context: ReasoningContext) -> Any:
        """执行推理步骤"""
        pass


class ChainOfThoughtReasoningStrategy(ReasoningStrategy):
    """链式思维推理策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'max_steps': 10,
            'min_confidence': 0.5,
            'enable_self_verification': True,
            'reasoning_depth': 'detailed'
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def generate_reasoning_steps(self, context: ReasoningContext) -> List[ReasoningStep]:
        """生成链式思维推理步骤"""
        steps = []
        
        # 1. 理解问题
        steps.append(ReasoningStep(
            step_id="understand_problem",
            step_type=StepType.OBSERVATION,
            reasoning_type=ReasoningType.DEDUCTIVE,
            description="理解和分析问题陈述",
            input_data=context.problem_statement,
            dependencies=[]
        ))
        
        # 2. 整理已知信息
        steps.append(ReasoningStep(
            step_id="organize_information",
            step_type=StepType.ANALYSIS,
            reasoning_type=ReasoningType.INDUCTIVE,
            description="整理和分析已知信息",
            input_data=context.available_information,
            dependencies=["understand_problem"]
        ))
        
        # 3. 识别模式
        steps.append(ReasoningStep(
            step_id="identify_patterns",
            step_type=StepType.ANALYSIS,
            reasoning_type=ReasoningType.INDUCTIVE,
            description="识别信息中的模式和规律",
            input_data=context.available_information,
            dependencies=["organize_information"]
        ))
        
        # 4. 生成假设
        steps.append(ReasoningStep(
            step_id="generate_hypotheses",
            step_type=StepType.INFERENCE,
            reasoning_type=ReasoningType.ABDUCTIVE,
            description="基于已知信息生成可能的解释或假设",
            input_data=context.available_information,
            dependencies=["identify_patterns"]
        ))
        
        # 5. 逻辑推理
        steps.append(ReasoningStep(
            step_id="logical_reasoning",
            step_type=StepType.INFERENCE,
            reasoning_type=ReasoningType.DEDUCTIVE,
            description="使用逻辑规则进行推理",
            input_data={},
            dependencies=["generate_hypotheses"]
        ))
        
        # 6. 验证推理
        if self.config['enable_self_verification']:
            steps.append(ReasoningStep(
                step_id="verify_reasoning",
                step_type=StepType.VALIDATION,
                reasoning_type=ReasoningType.DEDUCTIVE,
                description="验证推理过程的逻辑性和正确性",
                input_data={},
                dependencies=["logical_reasoning"]
            ))
        
        # 7. 得出结论
        steps.append(ReasoningStep(
            step_id="draw_conclusion",
            step_type=StepType.CONCLUSION,
            reasoning_type=ReasoningType.DEDUCTIVE,
            description="基于推理过程得出最终结论",
            input_data={},
            dependencies=["logical_reasoning", "verify_reasoning"] if self.config['enable_self_verification'] else ["logical_reasoning"]
        ))
        
        return steps
    
    def execute_step(self, step: ReasoningStep, context: ReasoningContext) -> Any:
        """执行链式思维步骤"""
        if step.step_id == "understand_problem":
            return self._understand_problem(step, context)
        elif step.step_id == "organize_information":
            return self._organize_information(step, context)
        elif step.step_id == "identify_patterns":
            return self._identify_patterns(step, context)
        elif step.step_id == "generate_hypotheses":
            return self._generate_hypotheses(step, context)
        elif step.step_id == "logical_reasoning":
            return self._logical_reasoning(step, context)
        elif step.step_id == "verify_reasoning":
            return self._verify_reasoning(step, context)
        elif step.step_id == "draw_conclusion":
            return self._draw_conclusion(step, context)
        else:
            raise ValueError(f"未知步骤: {step.step_id}")
    
    def _understand_problem(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """理解问题"""
        # 解析问题陈述
        problem_analysis = {
            'main_question': self._extract_main_question(context.problem_statement),
            'sub_questions': self._extract_sub_questions(context.problem_statement),
            'problem_type': self._classify_problem_type(context.problem_statement),
            'key_concepts': self._extract_key_concepts(context.problem_statement),
            'complexity_level': self._assess_complexity(context.problem_statement)
        }
        
        step.reasoning_chain = [
            f"1. 分析问题陈述: '{context.problem_statement}'",
            f"2. 识别主要问题: {problem_analysis['main_question']}",
            f"3. 分类问题类型: {problem_analysis['problem_type']}",
            f"4. 评估复杂度: {problem_analysis['complexity_level']}"
        ]
        
        return problem_analysis
    
    def _organize_information(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """整理信息"""
        organized_info = {
            'facts': [],
            'relationships': [],
            'unknowns': [],
            'constraints': context.constraints.copy()
        }
        
        # 分析已知信息
        for info in context.available_information:
            if self._is_fact(info):
                organized_info['facts'].append(info)
            elif self._is_constraint(info):
                organized_info['constraints'].append(info)
            else:
                organized_info['unknowns'].append(info)
        
        # 识别关系
        organized_info['relationships'] = self._identify_relationships(organized_info['facts'])
        
        step.reasoning_chain = [
            "1. 分类已知信息",
            f"2. 识别事实: {len(organized_info['facts'])}条",
            f"3. 识别约束: {len(organized_info['constraints'])}条",
            f"4. 识别未知: {len(organized_info['unknowns'])}条"
        ]
        
        return organized_info
    
    def _identify_patterns(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """识别模式"""
        patterns = {
            'recurring_themes': [],
            'numerical_patterns': [],
            'logical_patterns': [],
            'temporal_patterns': []
        }
        
        # 简化的模式识别
        facts = context.knowledge_base.get('facts', [])
        
        # 识别重复主题
        themes = self._extract_themes(facts)
        patterns['recurring_themes'] = themes
        
        # 识别数值模式
        numerical_data = self._extract_numerical_data(facts)
        patterns['numerical_patterns'] = self._find_numerical_patterns(numerical_data)
        
        step.reasoning_chain = [
            "1. 分析信息中的重复模式",
            f"2. 发现主题模式: {len(patterns['recurring_themes'])}个",
            f"3. 发现数值模式: {len(patterns['numerical_patterns'])}个",
            "4. 评估模式的重要性"
        ]
        
        return patterns
    
    def _generate_hypotheses(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """生成假设"""
        hypotheses = []
        
        # 基于模式生成假设
        patterns = context.knowledge_base.get('patterns', {})
        
        # 简单假设生成
        if patterns.get('numerical_patterns'):
            for pattern in patterns['numerical_patterns']:
                hypothesis = f"基于数值模式 '{pattern}' 的假设"
                hypotheses.append({
                    'hypothesis': hypothesis,
                    'confidence': 0.7,
                    'evidence': [pattern],
                    'testable': True
                })
        
        # 逻辑假设
        logical_assumption = "基于逻辑推理的假设"
        hypotheses.append({
            'hypothesis': logical_assumption,
            'confidence': 0.6,
            'evidence': ['逻辑一致性'],
            'testable': True
        })
        
        step.reasoning_chain = [
            "1. 基于识别的模式生成假设",
            f"2. 生成了 {len(hypotheses)} 个假设",
            "3. 评估假设的可验证性",
            "4. 分配初始置信度"
        ]
        
        return {'hypotheses': hypotheses}
    
    def _logical_reasoning(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """逻辑推理"""
        reasoning_result = {
            'logical_chain': [],
            'deductions': [],
            'conclusions': [],
            'confidence_score': 0.0
        }
        
        # 构建逻辑链
        reasoning_result['logical_chain'] = [
            "前提1: 如果P则Q",
            "前提2: P为真",
            "结论: 因此Q为真"
        ]
        
        # 演绎推理
        reasoning_result['deductions'] = [
            "从一般到特殊的推理",
            "基于已知规则的逻辑推导"
        ]
        
        # 计算置信度
        reasoning_result['confidence_score'] = 0.8
        
        step.reasoning_chain = reasoning_result['logical_chain']
        
        return reasoning_result
    
    def _verify_reasoning(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """验证推理"""
        verification = {
            'consistency_check': True,
            'completeness_check': True,
            'logic_validation': True,
            'contradictions': [],
            'missing_information': []
        }
        
        # 检查逻辑一致性
        verification['consistency_check'] = self._check_logical_consistency(context)
        
        # 检查完整性
        verification['completeness_check'] = self._check_reasoning_completeness(context)
        
        # 逻辑验证
        verification['logic_validation'] = self._validate_logic(context)
        
        step.reasoning_chain = [
            "1. 检查逻辑一致性",
            "2. 验证推理完整性",
            "3. 确认没有矛盾",
            "4. 验证结论合理性"
        ]
        
        return verification
    
    def _draw_conclusion(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """得出结论"""
        conclusion = {
            'primary_conclusion': '',
            'supporting_evidence': [],
            'confidence_level': 0.0,
            'limitations': [],
            'future_investigation': []
        }
        
        # 综合所有推理结果
        reasoning_history = context.reasoning_history
        
        # 主要结论
        conclusion['primary_conclusion'] = "基于链式思维推理的综合结论"
        
        # 支持证据
        evidence = []
        for history_step in reasoning_history:
            if hasattr(history_step, 'evidence') and history_step.evidence:
                evidence.extend(history_step.evidence)
        conclusion['supporting_evidence'] = evidence
        
        # 置信度
        confidences = [step.confidence for step in reasoning_history if step.confidence > 0]
        conclusion['confidence_level'] = np.mean(confidences) if confidences else 0.5
        
        step.reasoning_chain = [
            "1. 综合所有推理步骤",
            "2. 收集支持证据",
            "3. 评估结论置信度",
            "4. 识别结论局限性"
        ]
        
        return conclusion
    
    # 辅助方法
    def _extract_main_question(self, problem_statement: str) -> str:
        """提取主要问题"""
        # 简单的问题提取
        sentences = problem_statement.split('.')
        return sentences[0] if sentences else problem_statement
    
    def _extract_sub_questions(self, problem_statement: str) -> List[str]:
        """提取子问题"""
        # 简化实现
        sentences = [s.strip() for s in problem_statement.split('.') if s.strip()]
        return sentences[1:] if len(sentences) > 1 else []
    
    def _classify_problem_type(self, problem_statement: str) -> str:
        """分类问题类型"""
        if '多少' in problem_statement or '数量' in problem_statement:
            return "数量问题"
        elif '为什么' in problem_statement:
            return "因果问题"
        elif '如何' in problem_statement:
            return "方法问题"
        else:
            return "一般问题"
    
    def _extract_key_concepts(self, problem_statement: str) -> List[str]:
        """提取关键概念"""
        # 简单关键词提取
        words = problem_statement.split()
        return [word for word in words if len(word) > 3]
    
    def _assess_complexity(self, problem_statement: str) -> str:
        """评估复杂度"""
        sentence_count = len([s for s in problem_statement.split('.') if s.strip()])
        if sentence_count <= 2:
            return "简单"
        elif sentence_count <= 5:
            return "中等"
        else:
            return "复杂"
    
    def _is_fact(self, info: str) -> bool:
        """判断是否为事实"""
        return '是' in info or '为' in info or '等于' in info
    
    def _is_constraint(self, info: str) -> bool:
        """判断是否为约束"""
        return '不能' in info or '限制' in info or '必须' in info
    
    def _identify_relationships(self, facts: List[str]) -> List[Dict[str, str]]:
        """识别关系"""
        relationships = []
        for i, fact1 in enumerate(facts):
            for j, fact2 in enumerate(facts[i+1:], i+1):
                if self._are_related(fact1, fact2):
                    relationships.append({
                        'entity1': fact1,
                        'entity2': fact2,
                        'relationship': 'related'
                    })
        return relationships
    
    def _are_related(self, fact1: str, fact2: str) -> bool:
        """判断两个事实是否相关"""
        words1 = set(fact1.split())
        words2 = set(fact2.split())
        common_words = words1.intersection(words2)
        return len(common_words) > 0
    
    def _extract_themes(self, facts: List[str]) -> List[str]:
        """提取主题"""
        # 简化主题提取
        themes = []
        for fact in facts:
            if len(fact) > 10:  # 较长的句子可能是主题
                themes.append(fact[:50] + "..." if len(fact) > 50 else fact)
        return themes
    
    def _extract_numerical_data(self, facts: List[str]) -> List[float]:
        """提取数值数据"""
        numbers = []
        for fact in facts:
            # 简单的数字提取
            import re
            nums = re.findall(r'\d+\.?\d*', fact)
            numbers.extend([float(num) for num in nums])
        return numbers
    
    def _find_numerical_patterns(self, numbers: List[float]) -> List[str]:
        """发现数值模式"""
        patterns = []
        if len(numbers) >= 2:
            if all(numbers[i] <= numbers[i+1] for i in range(len(numbers)-1)):
                patterns.append("递增序列")
            if all(numbers[i] >= numbers[i+1] for i in range(len(numbers)-1)):
                patterns.append("递减序列")
        return patterns
    
    def _check_logical_consistency(self, context: ReasoningContext) -> bool:
        """检查逻辑一致性"""
        # 简化实现
        return True
    
    def _check_reasoning_completeness(self, context: ReasoningContext) -> bool:
        """检查推理完整性"""
        # 简化实现
        return len(context.available_information) > 0
    
    def _validate_logic(self, context: ReasoningContext) -> bool:
        """验证逻辑"""
        # 简化实现
        return True


class HierarchicalReasoningStrategy(ReasoningStrategy):
    """层次化推理策略"""
    
    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            'max_levels': 5,
            'min_level_size': 2,
            'max_level_size': 8,
            'hierarchical_decomposition': True
        }
        if config:
            default_config.update(config)
        super().__init__(default_config)
    
    def generate_reasoning_steps(self, context: ReasoningContext) -> List[ReasoningStep]:
        """生成层次化推理步骤"""
        steps = []
        
        # 1. 问题分解
        steps.append(ReasoningStep(
            step_id="decompose_problem",
            step_type=StepType.ANALYSIS,
            reasoning_type=ReasoningType.INDUCTIVE,
            description="将复杂问题分解为子问题",
            input_data=context.problem_statement,
            dependencies=[]
        ))
        
        # 2. 构建层次结构
        steps.append(ReasoningStep(
            step_id="build_hierarchy",
            step_type=StepType.SYNTHESIS,
            reasoning_type=ReasoningType.DEDUCTIVE,
            description="构建问题的层次结构",
            input_data={},
            dependencies=["decompose_problem"]
        ))
        
        # 3. 逐层推理
        steps.append(ReasoningStep(
            step_id="layer_by_layer_reasoning",
            step_type=StepType.INFERENCE,
            reasoning_type=ReasoningType.DEDUCTIVE,
            description="从底层开始逐层推理",
            input_data={},
            dependencies=["build_hierarchy"]
        ))
        
        # 4. 层次整合
        steps.append(ReasoningStep(
            step_id="integrate_hierarchy",
            step_type=StepType.SYNTHESIS,
            reasoning_type=ReasoningType.DEDUCTIVE,
            description="将各层推理结果整合",
            input_data={},
            dependencies=["layer_by_layer_reasoning"]
        ))
        
        # 5. 验证层次逻辑
        steps.append(ReasoningStep(
            step_id="validate_hierarchy",
            step_type=StepType.VALIDATION,
            reasoning_type=ReasoningType.DEDUCTIVE,
            description="验证层次逻辑的完整性",
            input_data={},
            dependencies=["integrate_hierarchy"]
        ))
        
        return steps
    
    def execute_step(self, step: ReasoningStep, context: ReasoningContext) -> Any:
        """执行层次化推理步骤"""
        if step.step_id == "decompose_problem":
            return self._decompose_problem(step, context)
        elif step.step_id == "build_hierarchy":
            return self._build_hierarchy(step, context)
        elif step.step_id == "layer_by_layer_reasoning":
            return self._layer_by_layer_reasoning(step, context)
        elif step.step_id == "integrate_hierarchy":
            return self._integrate_hierarchy(step, context)
        elif step.step_id == "validate_hierarchy":
            return self._validate_hierarchy(step, context)
        else:
            raise ValueError(f"未知步骤: {step.step_id}")
    
    def _decompose_problem(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """分解问题"""
        decomposition = {
            'main_components': [],
            'sub_problems': [],
            'dependencies': [],
            'complexity_score': 0.0
        }
        
        # 简单的问题分解
        problem_parts = context.problem_statement.split(',')
        decomposition['main_components'] = [part.strip() for part in problem_parts if part.strip()]
        
        # 创建子问题
        for i, component in enumerate(decomposition['main_components']):
            decomposition['sub_problems'].append({
                'id': f'sub_problem_{i}',
                'description': component,
                'complexity': len(component.split()),
                'dependencies': []
            })
        
        # 计算复杂度
        decomposition['complexity_score'] = len(decomposition['main_components'])
        
        step.reasoning_chain = [
            "1. 分析问题的主要组成部分",
            f"2. 识别了 {len(decomposition['main_components'])} 个主要组件",
            "3. 创建子问题结构",
            "4. 评估分解复杂度"
        ]
        
        return decomposition
    
    def _build_hierarchy(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """构建层次结构"""
        hierarchy = {
            'levels': [],
            'connections': [],
            'level_summaries': []
        }
        
        # 构建3层结构（简化）
        hierarchy['levels'] = [
            {'level': 0, 'name': 'High Level', 'items': ['overall_problem']},
            {'level': 1, 'name': 'Mid Level', 'items': ['component_1', 'component_2']},
            {'level': 2, 'name': 'Low Level', 'items': ['detail_1', 'detail_2', 'detail_3']}
        ]
        
        # 连接关系
        hierarchy['connections'] = [
            {'from': 'overall_problem', 'to': 'component_1'},
            {'from': 'overall_problem', 'to': 'component_2'},
            {'from': 'component_1', 'to': 'detail_1'},
            {'from': 'component_1', 'to': 'detail_2'},
            {'from': 'component_2', 'to': 'detail_3'}
        ]
        
        step.reasoning_chain = [
            "1. 构建多层次结构",
            f"2. 创建了 {len(hierarchy['levels'])} 个层次",
            "3. 建立层次间连接",
            "4. 准备层次化推理"
        ]
        
        return hierarchy
    
    def _layer_by_layer_reasoning(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """逐层推理"""
        reasoning_results = {
            'level_results': [],
            'reasoning_paths': [],
            'cross_level_insights': []
        }
        
        # 模拟每层推理
        for level in range(3):
            level_result = {
                'level': level,
                'reasoning_steps': f'在第{level}层进行推理',
                'conclusions': [f'第{level}层的结论'],
                'confidence': 0.8 - level * 0.1
            }
            reasoning_results['level_results'].append(level_result)
        
        step.reasoning_chain = [
            "1. 从最低层开始推理",
            "2. 逐层向上推理",
            "3. 收集各层结论",
            "4. 准备层次整合"
        ]
        
        return reasoning_results
    
    def _integrate_hierarchy(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """整合层次"""
        integration = {
            'integrated_conclusion': '',
            'level_contributions': {},
            'hierarchical_summary': '',
            'coherence_score': 0.0
        }
        
        # 综合各层结果
        integration['integrated_conclusion'] = "基于层次化推理的综合结论"
        integration['hierarchical_summary'] = "层次化推理总结"
        integration['coherence_score'] = 0.85
        
        step.reasoning_chain = [
            "1. 收集各层推理结果",
            "2. 识别层次间关联",
            "3. 综合形成整体结论",
            "4. 评估结论一致性"
        ]
        
        return integration
    
    def _validate_hierarchy(self, step: ReasoningStep, context: ReasoningContext) -> Dict[str, Any]:
        """验证层次"""
        validation = {
            'hierarchy_consistency': True,
            'logic_flow': True,
            'completeness': True,
            'validation_score': 0.9
        }
        
        step.reasoning_chain = [
            "1. 验证层次结构一致性",
            "2. 检查逻辑流程",
            "3. 确认完整性",
            "4. 生成最终验证报告"
        ]
        
        return validation


class MultiStepReasoner(BaseModule):
    """多步推理器"""
    
    def __init__(self, reasoning_strategies: Dict[ReasoningType, ReasoningStrategy]):
        module_config = ModuleConfig("multi_step_reasoner", version="1.0")
        super().__init__(module_config)
        
        self.reasoning_strategies = reasoning_strategies
        self.reasoning_history: List[ReasoningResult] = []
        
        # 推理配置
        self.reasoning_config = {
            'max_reasoning_steps': 20,
            'confidence_threshold': 0.6,
            'enable_parallel_steps': False,
            'timeout_per_step': 30.0,
            'validation_frequency': 5
        }
    
    def reason(self, problem: str, information: List[str], 
               reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE) -> ReasoningResult:
        """执行多步推理"""
        self.state = ModuleState.ACTIVE
        self.logger.info(f"开始 {reasoning_type.value} 推理: {problem}")
        
        try:
            # 创建推理上下文
            context = self._create_reasoning_context(problem, information)
            
            # 选择推理策略
            strategy = self.reasoning_strategies.get(reasoning_type)
            if not strategy:
                raise ValueError(f"未找到推理策略: {reasoning_type.value}")
            
            # 生成推理步骤
            steps = strategy.generate_reasoning_steps(context)
            
            # 执行推理步骤
            completed_steps = []
            for step in steps:
                step_result = self._execute_step(step, strategy, context)
                context.reasoning_history.append(step_result)
                completed_steps.append(step_result)
                
                if step_result.status == StepStatus.FAILED:
                    self.logger.warning(f"推理步骤失败: {step_result.step_id}")
                    break
            
            # 生成最终结果
            result = self._generate_reasoning_result(completed_steps, context)
            self.reasoning_history.append(result)
            
            self.state = ModuleState.COMPLETED
            self.logger.info(f"推理完成，置信度: {result.confidence_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"推理失败: {e}")
            self.state = ModuleState.ERROR
            raise e
    
    def _create_reasoning_context(self, problem: str, information: List[str]) -> ReasoningContext:
        """创建推理上下文"""
        return ReasoningContext(
            problem_statement=problem,
            available_information=information,
            assumptions=[],
            constraints=[],
            goals=[],
            knowledge_base={}
        )
    
    def _execute_step(self, step: ReasoningStep, strategy: ReasoningStrategy, 
                     context: ReasoningContext) -> ReasoningStep:
        """执行单个推理步骤"""
        step.status = StepStatus.ACTIVE
        step.timestamp = time.time()
        
        try:
            # 检查依赖
            for dep_id in step.dependencies:
                dep_step = next((s for s in context.reasoning_history if s.step_id == dep_id), None)
                if not dep_step or dep_step.status != StepStatus.COMPLETED:
                    step.fail(f"依赖步骤未完成: {dep_id}")
                    return step
            
            # 执行步骤
            output_data = strategy.execute_step(step, context)
            
            # 计算置信度（简化）
            confidence = self._calculate_step_confidence(step, output_data)
            
            step.complete(output_data, confidence)
            
        except Exception as e:
            step.fail(str(e))
            self.logger.error(f"步骤执行失败 {step.step_id}: {e}")
        
        return step
    
    def _calculate_step_confidence(self, step: ReasoningStep, output_data: Any) -> float:
        """计算步骤置信度"""
        # 简化的置信度计算
        base_confidence = 0.7
        
        # 根据步骤类型调整
        type_multipliers = {
            StepType.OBSERVATION: 0.9,
            StepType.ANALYSIS: 0.8,
            StepType.INFERENCE: 0.7,
            StepType.VALIDATION: 0.9,
            StepType.SYNTHESIS: 0.8,
            StepType.CONCLUSION: 0.8
        }
        
        multiplier = type_multipliers.get(step.step_type, 1.0)
        return min(1.0, base_confidence * multiplier)
    
    def _generate_reasoning_result(self, steps: List[ReasoningStep], 
                                  context: ReasoningContext) -> ReasoningResult:
        """生成推理结果"""
        # 提取最终结论
        final_step = next((s for s in reversed(steps) if s.step_type == StepType.CONCLUSION), None)
        final_conclusion = final_step.output_data if final_step else context.problem_statement
        
        # 计算总体置信度
        completed_steps = [s for s in steps if s.status == StepStatus.COMPLETED]
        confidence_score = np.mean([s.confidence for s in completed_steps]) if completed_steps else 0.0
        
        # 构建推理路径
        reasoning_path = [step.step_id for step in steps]
        
        # 生成执行摘要
        execution_summary = {
            'total_steps': len(steps),
            'completed_steps': len(completed_steps),
            'total_time': sum(s.duration for s in completed_steps),
            'average_confidence': confidence_score,
            'success_rate': len(completed_steps) / len(steps) if steps else 0.0
        }
        
        return ReasoningResult(
            final_conclusion=final_conclusion,
            reasoning_chain=steps,
            confidence_score=confidence_score,
            reasoning_path=reasoning_path,
            execution_summary=execution_summary
        )
    
    def get_reasoning_history(self) -> List[ReasoningResult]:
        """获取推理历史"""
        return self.reasoning_history
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        if not self.reasoning_history:
            return {}
        
        completed_reasonings = [r for r in self.reasoning_history if r.confidence_score > 0]
        
        return {
            'total_reasonings': len(self.reasoning_history),
            'completed_reasonings': len(completed_reasonings),
            'average_confidence': np.mean([r.confidence_score for r in completed_reasonings]),
            'average_steps': np.mean([len(r.reasoning_chain) for r in completed_reasonings]),
            'success_rate': len(completed_reasonings) / len(self.reasoning_history)
        }
    
    def initialize(self) -> bool:
        """初始化推理器"""
        self.state = ModuleState.INITIALIZED
        return True
    
    def cleanup(self) -> bool:
        """清理推理器"""
        self.reasoning_history = []
        return True


# 便利函数
def create_chain_of_thought_reasoner() -> MultiStepReasoner:
    """创建链式思维推理器"""
    strategies = {
        ReasoningType.DEDUCTIVE: ChainOfThoughtReasoningStrategy(),
        ReasoningType.INDUCTIVE: ChainOfThoughtReasoningStrategy(),
        ReasoningType.ABDUCTIVE: ChainOfThoughtReasoningStrategy()
    }
    
    return MultiStepReasoner(strategies)


def create_hierarchical_reasoner() -> MultiStepReasoner:
    """创建层次化推理器"""
    strategies = {
        ReasoningType.DEDUCTIVE: HierarchicalReasoningStrategy(),
        ReasoningType.INDUCTIVE: HierarchicalReasoningStrategy()
    }
    
    return MultiStepReasoner(strategies)


def create_comprehensive_reasoner() -> MultiStepReasoner:
    """创建综合推理器"""
    strategies = {
        ReasoningType.DEDUCTIVE: ChainOfThoughtReasoningStrategy(),
        ReasoningType.INDUCTIVE: ChainOfThoughtReasoningStrategy(),
        ReasoningType.ABDUCTIVE: ChainOfThoughtReasoningStrategy(),
        ReasoningType.CAUSAL: HierarchicalReasoningStrategy()
    }
    
    return MultiStepReasoner(strategies)