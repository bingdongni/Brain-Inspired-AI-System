"""
系统集成与认知功能协调器
====================

实现各个认知功能模块之间的协调工作，包括：
- 认知系统集成器
- 模块协调器
- 系统编排器
- 集成框架管理

主要特性：
- 模块间协调
- 工作流编排
- 资源共享管理
- 性能监控协调
- 错误处理与恢复

作者: Brain-Inspired AI Team
创建时间: 2025-11-16
"""

import numpy as np
import logging
import time
import json
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import queue
import asyncio
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.base_module import BaseModule, ModuleConfig, ModuleState
from advanced_cognition.end_to_end_pipeline import EndToEndTrainingPipeline, PipelineConfig, PipelineStage
from advanced_cognition.performance_optimization import PerformanceOptimizer, OptimizationConfig, PerformanceMetric
from advanced_cognition.multi_step_reasoning import MultiStepReasoner, ReasoningType
from advanced_cognition.analogical_learning import AnalogicalLearner, CreativeSolution


class IntegrationMode(Enum):
    """集成模式"""
    SEQUENTIAL = "sequential"      # 顺序执行
    PARALLEL = "parallel"          # 并行执行
    PIPELINE = "pipeline"          # 流水线
    ADAPTIVE = "adaptive"          # 自适应


class ModuleState(Enum):
    """模块状态"""
    INACTIVE = "inactive"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class IntegrationWorkflow:
    """集成工作流"""
    workflow_id: str
    name: str
    description: str
    modules: List[str]  # 模块ID列表
    execution_order: List[str]  # 执行顺序
    dependencies: Dict[str, List[str]]  # 模块依赖关系
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModuleExecutionResult:
    """模块执行结果"""
    module_id: str
    status: ModuleState
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    output_data: Any = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """工作流执行"""
    workflow_id: str
    execution_id: str
    start_time: float
    status: ModuleState = ModuleState.ACTIVE
    module_results: Dict[str, ModuleExecutionResult] = field(default_factory=dict)
    total_duration: Optional[float] = None
    success_rate: float = 0.0
    
    def complete(self):
        """标记完成"""
        self.status = ModuleState.COMPLETED
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time
        
        # 计算成功率
        total_modules = len(self.module_results)
        if total_modules > 0:
            successful_modules = sum(1 for result in self.module_results.values() 
                                   if result.status == ModuleState.COMPLETED)
            self.success_rate = successful_modules / total_modules
    
    def fail(self, error_message: str):
        """标记失败"""
        self.status = ModuleState.ERROR
        self.end_time = time.time()
        self.total_duration = self.end_time - self.start_time


class ModuleCoordinator:
    """模块协调器"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.modules: Dict[str, BaseModule] = {}
        self.module_states: Dict[str, ModuleState] = {}
        self.execution_queue = queue.Queue()
        self.resource_pool: Dict[str, Any] = {}
        
        # 协调状态
        self.is_coordinating = False
        self.coordination_thread = None
        
    def register_module(self, module_id: str, module: BaseModule):
        """注册模块"""
        self.modules[module_id] = module
        self.module_states[module_id] = ModuleState.INACTIVE
        self.logger.info(f"模块 {module_id} 已注册")
    
    def unregister_module(self, module_id: str):
        """注销模块"""
        if module_id in self.modules:
            # 停止模块
            module = self.modules[module_id]
            if hasattr(module, 'cleanup'):
                module.cleanup()
            
            del self.modules[module_id]
            del self.module_states[module_id]
            
            self.logger.info(f"模块 {module_id} 已注销")
    
    def execute_module(self, module_id: str, input_data: Any, 
                      timeout: float = 300.0) -> ModuleExecutionResult:
        """执行模块"""
        if module_id not in self.modules:
            raise ValueError(f"模块 {module_id} 未注册")
        
        module = self.modules[module_id]
        start_time = time.time()
        
        result = ModuleExecutionResult(
            module_id=module_id,
            status=ModuleState.PROCESSING,
            start_time=start_time
        )
        
        try:
            # 检查模块状态
            if self.module_states[module_id] not in [ModuleState.INACTIVE, ModuleState.COMPLETED]:
                if hasattr(module, 'initialize'):
                    module.initialize()
                self.module_states[module_id] = ModuleState.INITIALIZING
            
            # 执行模块
            self.module_states[module_id] = ModuleState.PROCESSING
            
            if hasattr(module, 'execute'):
                # 对于有execute方法的模块
                if asyncio.iscoroutinefunction(module.execute):
                    # 异步执行
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result.output_data = loop.run_until_complete(
                        module.execute(input_data)
                    )
                    loop.close()
                else:
                    # 同步执行
                    result.output_data = module.execute(input_data)
            elif hasattr(module, 'reason'):
                # 推理模块
                if isinstance(input_data, dict) and 'problem' in input_data:
                    reasoning_result = module.reason(
                        input_data['problem'],
                        input_data.get('information', []),
                        input_data.get('reasoning_type', ReasoningType.DEDUCTIVE)
                    )
                    result.output_data = reasoning_result
            else:
                # 其他模块的通用执行方法
                result.output_data = self._execute_generic_module(module, input_data)
            
            result.status = ModuleState.COMPLETED
            self.module_states[module_id] = ModuleState.COMPLETED
            
        except Exception as e:
            result.status = ModuleState.ERROR
            result.error_message = str(e)
            self.module_states[module_id] = ModuleState.ERROR
            self.logger.error(f"模块 {module_id} 执行失败: {e}")
        
        finally:
            result.end_time = time.time()
            result.duration = result.end_time - start_time
        
        return result
    
    def _execute_generic_module(self, module: BaseModule, input_data: Any) -> Any:
        """通用模块执行"""
        # 根据模块类型选择执行方法
        module_name = module.__class__.__name__
        
        if 'TrainingPipeline' in module_name:
            # 训练管道
            if isinstance(input_data, tuple):
                return module.execute(input_data[0])
            else:
                return module.execute(input_data)
        
        elif 'PerformanceOptimizer' in module_name:
            # 性能优化器
            if callable(input_data):
                return module.start_optimization(input_data)
            else:
                return module.start_optimization(lambda x: x)  # 默认目标函数
        
        elif 'AnalogicalLearner' in module_name:
            # 类比学习器
            if isinstance(input_data, dict) and 'problem' in input_data:
                return module.solve_problem_creatively(
                    input_data['problem'],
                    input_data.get('context', {})
                )
            else:
                return module
        
        else:
            # 其他模块，返回模块实例
            return module
    
    def get_module_status(self, module_id: str) -> Optional[ModuleState]:
        """获取模块状态"""
        return self.module_states.get(module_id)
    
    def get_all_module_status(self) -> Dict[str, ModuleState]:
        """获取所有模块状态"""
        return self.module_states.copy()
    
    def shutdown_all_modules(self):
        """关闭所有模块"""
        self.logger.info("开始关闭所有模块...")
        
        for module_id, module in self.modules.items():
            try:
                self.logger.info(f"关闭模块 {module_id}...")
                if hasattr(module, 'cleanup'):
                    module.cleanup()
                self.module_states[module_id] = ModuleState.SHUTTING_DOWN
            except Exception as e:
                self.logger.error(f"关闭模块 {module_id} 失败: {e}")
        
        self.logger.info("所有模块已关闭")


class SystemOrchestrator:
    """系统编排器"""
    
    def __init__(self, module_coordinator: ModuleCoordinator):
        self.coordinator = module_coordinator
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 工作流管理
        self.workflows: Dict[str, IntegrationWorkflow] = {}
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        
        # 编排配置
        self.orchestration_config = {
            'max_concurrent_executions': 5,
            'default_timeout': 1800,  # 30分钟
            'retry_attempts': 3,
            'enable_monitoring': True
        }
    
    def register_workflow(self, workflow: IntegrationWorkflow):
        """注册工作流"""
        self.workflows[workflow.workflow_id] = workflow
        self.logger.info(f"工作流 {workflow.workflow_id} 已注册: {workflow.name}")
    
    def execute_workflow(self, workflow_id: str, input_data: Any) -> WorkflowExecution:
        """执行工作流"""
        if workflow_id not in self.workflows:
            raise ValueError(f"工作流 {workflow_id} 未注册")
        
        workflow = self.workflows[workflow_id]
        execution_id = f"{workflow_id}_{int(time.time())}"
        
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            start_time=time.time()
        )
        
        self.active_executions[execution_id] = execution
        self.logger.info(f"开始执行工作流 {workflow_id} (ID: {execution_id})")
        
        try:
            # 检查并发限制
            if len(self.active_executions) > self.orchestration_config['max_concurrent_executions']:
                raise RuntimeError("超过最大并发执行数")
            
            # 执行工作流
            if workflow.dependencies:
                result = self._execute_workflow_with_dependencies(workflow, input_data, execution)
            else:
                result = self._execute_workflow_sequential(workflow, input_data, execution)
            
            execution.complete()
            self.execution_history.append(execution)
            
            # 从活跃执行中移除
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            self.logger.info(f"工作流 {workflow_id} 执行完成，成功率: {execution.success_rate:.2%}")
            
            return execution
            
        except Exception as e:
            execution.fail(str(e))
            self.execution_history.append(execution)
            
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            
            self.logger.error(f"工作流 {workflow_id} 执行失败: {e}")
            raise e
    
    def _execute_workflow_sequential(self, workflow: IntegrationWorkflow, 
                                   input_data: Any, execution: WorkflowExecution) -> Any:
        """顺序执行工作流"""
        current_data = input_data
        
        for module_id in workflow.execution_order:
            self.logger.info(f"执行模块: {module_id}")
            
            module_result = self.coordinator.execute_module(module_id, current_data)
            execution.module_results[module_id] = module_result
            
            if module_result.status == ModuleState.ERROR:
                self.logger.error(f"模块 {module_id} 执行失败，停止工作流")
                break
            
            current_data = module_result.output_data
        
        return current_data
    
    def _execute_workflow_with_dependencies(self, workflow: IntegrationWorkflow,
                                          input_data: Any, execution: WorkflowExecution) -> Any:
        """带依赖的工作流执行"""
        # 构建依赖图
        dependency_graph = self._build_dependency_graph(workflow)
        
        # 拓扑排序
        execution_order = self._topological_sort(dependency_graph)
        
        # 执行模块
        module_outputs = {}
        
        for module_id in execution_order:
            # 准备输入数据
            module_input = self._prepare_module_input(module_id, input_data, module_outputs, workflow)
            
            # 执行模块
            module_result = self.coordinator.execute_module(module_id, module_input)
            execution.module_results[module_id] = module_result
            
            if module_result.status == ModuleState.ERROR:
                self.logger.error(f"模块 {module_id} 执行失败")
                break
            
            module_outputs[module_id] = module_result.output_data
        
        return module_outputs
    
    def _build_dependency_graph(self, workflow: IntegrationWorkflow) -> Dict[str, List[str]]:
        """构建依赖图"""
        graph = {module_id: [] for module_id in workflow.modules}
        
        for module_id, dependencies in workflow.dependencies.items():
            for dep in dependencies:
                if dep in workflow.modules:
                    graph[dep].append(module_id)  # dep -> module_id
        
        return graph
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """拓扑排序"""
        in_degree = {node: 0 for node in graph}
        
        # 计算入度
        for node in graph:
            for neighbor in graph[node]:
                in_degree[neighbor] += 1
        
        # 找到所有入度为0的节点
        queue = [node for node in in_degree if in_degree[node] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # 减少相邻节点的入度
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _prepare_module_input(self, module_id: str, input_data: Any, 
                            module_outputs: Dict[str, Any], 
                            workflow: IntegrationWorkflow) -> Any:
        """准备模块输入"""
        dependencies = workflow.dependencies.get(module_id, [])
        
        if not dependencies:
            return input_data
        
        # 如果有依赖，组合依赖输出
        if len(dependencies) == 1:
            return module_outputs.get(dependencies[0], input_data)
        else:
            # 多依赖组合
            return {
                'primary_input': input_data,
                'dependency_outputs': {dep: module_outputs.get(dep) for dep in dependencies}
            }
    
    def get_workflow_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """获取工作流状态"""
        return self.active_executions.get(execution_id)
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        if not self.execution_history:
            return {'total_executions': 0}
        
        recent_executions = self.execution_history[-10:]  # 最近10次
        
        return {
            'total_executions': len(self.execution_history),
            'active_executions': len(self.active_executions),
            'recent_success_rate': np.mean([e.success_rate for e in recent_executions]),
            'avg_duration': np.mean([e.total_duration for e in recent_executions if e.total_duration]),
            'workflows_count': len(self.workflows),
            'most_successful_workflow': self._get_most_successful_workflow()
        }
    
    def _get_most_successful_workflow(self) -> Optional[str]:
        """获取最成功的工作流"""
        if not self.execution_history:
            return None
        
        workflow_success = defaultdict(list)
        for execution in self.execution_history:
            workflow_success[execution.workflow_id].append(execution.success_rate)
        
        best_workflow = max(workflow_success.items(), 
                          key=lambda x: np.mean(x[1]))
        
        return best_workflow[0] if np.mean(best_workflow[1]) > 0.5 else None


class CognitiveSystemIntegrator(BaseModule):
    """认知系统集成器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        module_config = ModuleConfig("cognitive_system_integrator", version="1.0")
        super().__init__(module_config)
        
        self.config = config or {}
        
        # 核心组件
        self.coordinator = ModuleCoordinator()
        self.orchestrator = SystemOrchestrator(self.coordinator)
        
        # 集成统计
        self.integration_metrics = {
            'total_workflows': 0,
            'successful_executions': 0,
            'total_execution_time': 0.0,
            'module_utilization': {}
        }
        
        # 自适应配置
        self.adaptive_config = {
            'enable_self_optimization': True,
            'performance_threshold': 0.8,
            'optimization_interval': 300,  # 5分钟
            'load_balancing': True
        }
        
    def initialize_cognitive_system(self, modules_config: Dict[str, Any]) -> bool:
        """初始化认知系统"""
        self.state = ModuleState.INITIALIZING
        self.logger.info("初始化认知系统...")
        
        try:
            # 注册模块
            for module_id, module_info in modules_config.items():
                if 'module' in module_info:
                    self.coordinator.register_module(module_id, module_info['module'])
            
            # 注册标准工作流
            self._register_standard_workflows()
            
            self.state = ModuleState.INITIALIZED
            self.logger.info("认知系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"认知系统初始化失败: {e}")
            self.state = ModuleState.ERROR
            return False
    
    def _register_standard_workflows(self):
        """注册标准工作流"""
        
        # 1. 端到端机器学习工作流
        ml_workflow = IntegrationWorkflow(
            workflow_id="end_to_end_ml",
            name="端到端机器学习",
            description="从数据预处理到模型部署的完整流程",
            modules=["data_preprocessor", "feature_extractor", "model_trainer", "model_evaluator", "deployment_manager"],
            execution_order=["data_preprocessor", "feature_extractor", "model_trainer", "model_evaluator", "deployment_manager"],
            dependencies={}
        )
        self.orchestrator.register_workflow(ml_workflow)
        
        # 2. 性能优化工作流
        optimization_workflow = IntegrationWorkflow(
            workflow_id="performance_optimization",
            name="性能优化",
            description="自动化的模型性能优化流程",
            modules=["performance_monitor", "parameter_tuner", "resource_manager", "optimization_engine"],
            execution_order=["performance_monitor", "parameter_tuner", "resource_manager", "optimization_engine"],
            dependencies={}
        )
        self.orchestrator.register_workflow(optimization_workflow)
        
        # 3. 智能推理工作流
        reasoning_workflow = IntegrationWorkflow(
            workflow_id="intelligent_reasoning",
            name="智能推理",
            description="多步推理和类比学习工作流",
            modules=["multi_step_reasoner", "analogical_learner", "pattern_recognizer", "innovation_engine"],
            execution_order=["multi_step_reasoner", "analogical_learner", "pattern_recognizer", "innovation_engine"],
            dependencies={}
        )
        self.orchestrator.register_workflow(reasoning_workflow)
        
        # 4. 综合认知工作流
        cognitive_workflow = IntegrationWorkflow(
            workflow_id="comprehensive_cognition",
            name="综合认知",
            description="集成所有认知功能的完整工作流",
            modules=["data_preprocessor", "multi_step_reasoner", "analogical_learner", "performance_optimizer", "model_trainer"],
            execution_order=["data_preprocessor", "multi_step_reasoner", "analogical_learner", "performance_optimizer", "model_trainer"],
            dependencies={
                "performance_optimizer": ["multi_step_reasoner", "analogical_learner"],
                "model_trainer": ["data_preprocessor", "performance_optimizer"]
            }
        )
        self.orchestrator.register_workflow(cognitive_workflow)
        
        self.integration_metrics['total_workflows'] = 4
    
    def execute_cognitive_task(self, task_type: str, task_data: Any, 
                             context: Dict[str, Any] = None) -> Any:
        """执行认知任务"""
        self.state = ModuleState.ACTIVE
        context = context or {}
        
        self.logger.info(f"执行认知任务: {task_type}")
        
        try:
            # 根据任务类型选择工作流
            workflow_id = self._select_workflow_for_task(task_type)
            
            if not workflow_id:
                raise ValueError(f"没有找到适合任务类型 {task_type} 的工作流")
            
            # 准备任务数据
            prepared_data = self._prepare_task_data(task_data, context)
            
            # 执行工作流
            execution = self.orchestrator.execute_workflow(workflow_id, prepared_data)
            
            # 更新统计
            self._update_integration_metrics(execution)
            
            self.state = ModuleState.COMPLETED
            
            return {
                'execution_result': execution,
                'workflow_id': workflow_id,
                'task_output': self._extract_task_output(execution, task_type),
                'performance_summary': self._generate_performance_summary(execution)
            }
            
        except Exception as e:
            self.logger.error(f"认知任务执行失败: {e}")
            self.state = ModuleState.ERROR
            raise e
    
    def _select_workflow_for_task(self, task_type: str) -> Optional[str]:
        """为任务选择工作流"""
        workflow_mapping = {
            'ml_training': 'end_to_end_ml',
            'performance_tuning': 'performance_optimization',
            'reasoning': 'intelligent_reasoning',
            'creative_problem_solving': 'intelligent_reasoning',
            'full_cognitive': 'comprehensive_cognition'
        }
        
        return workflow_mapping.get(task_type)
    
    def _prepare_task_data(self, task_data: Any, context: Dict[str, Any]) -> Any:
        """准备任务数据"""
        # 添加上下文信息
        if isinstance(task_data, dict):
            prepared_data = task_data.copy()
            prepared_data['context'] = context
        else:
            prepared_data = {
                'input_data': task_data,
                'context': context
            }
        
        return prepared_data
    
    def _extract_task_output(self, execution: WorkflowExecution, task_type: str) -> Any:
        """提取任务输出"""
        if not execution.module_results:
            return None
        
        # 根据任务类型提取不同输出
        if task_type in ['ml_training', 'performance_tuning']:
            # 返回最后的模块输出
            last_module = execution.workflow_id
            return execution.module_results.get(last_module + '_end') or list(execution.module_results.values())[-1].output_data
        elif task_type in ['reasoning', 'creative_problem_solving']:
            # 返回推理结果
            reasoner_results = [r for r in execution.module_results.values() 
                              if 'reasoner' in r.module_id or 'learner' in r.module_id]
            return reasoner_results[-1].output_data if reasoner_results else None
        else:
            # 默认返回所有输出
            return {module_id: result.output_data for module_id, result in execution.module_results.items()}
    
    def _generate_performance_summary(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """生成性能摘要"""
        return {
            'execution_time': execution.total_duration,
            'success_rate': execution.success_rate,
            'modules_executed': len(execution.module_results),
            'failed_modules': len([r for r in execution.module_results.values() 
                                 if r.status == ModuleState.ERROR]),
            'module_performance': {
                module_id: {
                    'duration': result.duration,
                    'status': result.status.value,
                    'error': result.error_message
                } for module_id, result in execution.module_results.items()
            }
        }
    
    def _update_integration_metrics(self, execution: WorkflowExecution):
        """更新集成指标"""
        self.integration_metrics['total_execution_time'] += execution.total_duration or 0
        
        if execution.success_rate > 0.8:
            self.integration_metrics['successful_executions'] += 1
        
        # 更新模块利用率
        for module_id in execution.module_results:
            if module_id not in self.integration_metrics['module_utilization']:
                self.integration_metrics['module_utilization'][module_id] = 0
            self.integration_metrics['module_utilization'][module_id] += 1
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'module_states': self.coordinator.get_all_module_status(),
            'active_executions': len(self.orchestrator.active_executions),
            'integration_metrics': self.integration_metrics,
            'execution_statistics': self.orchestrator.get_execution_statistics(),
            'system_health': self._assess_system_health()
        }
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """评估系统健康状态"""
        health_score = 1.0
        
        # 检查模块状态
        module_states = self.coordinator.get_all_module_status()
        error_modules = [m for m, state in module_states.items() if state == ModuleState.ERROR]
        if error_modules:
            health_score *= (len(module_states) - len(error_modules)) / len(module_states)
        
        # 检查执行统计
        exec_stats = self.orchestrator.get_execution_statistics()
        recent_success = exec_stats.get('recent_success_rate', 1.0)
        health_score *= recent_success
        
        return {
            'overall_health': health_score,
            'error_modules': error_modules,
            'recent_success_rate': recent_success,
            'active_workflows': len(self.orchestrator.workflows),
            'status': 'healthy' if health_score > 0.7 else 'degraded' if health_score > 0.4 else 'unhealthy'
        }
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """优化系统性能"""
        self.logger.info("开始系统性能优化...")
        
        # 获取当前状态
        current_status = self.get_system_status()
        
        # 识别性能瓶颈
        bottlenecks = self._identify_performance_bottlenecks(current_status)
        
        # 应用优化策略
        optimizations = []
        for bottleneck in bottlenecks:
            optimization = self._apply_optimization_strategy(bottleneck)
            optimizations.append(optimization)
        
        return {
            'bottlenecks_identified': len(bottlenecks),
            'optimizations_applied': optimizations,
            'performance_improvement': self._estimate_performance_improvement(bottlenecks)
        }
    
    def _identify_performance_bottlenecks(self, status: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 检查模块错误率
        error_modules = status.get('system_health', {}).get('error_modules', [])
        if error_modules:
            bottlenecks.append({
                'type': 'module_errors',
                'affected_modules': error_modules,
                'severity': 'high',
                'description': f"模块 {error_modules} 出现错误"
            })
        
        # 检查模块利用率不均
        utilization = status.get('integration_metrics', {}).get('module_utilization', {})
        if utilization:
            max_util = max(utilization.values()) if utilization.values() else 0
            min_util = min(utilization.values()) if utilization.values() else 0
            
            if max_util > 0 and (max_util - min_util) / max_util > 0.5:
                bottlenecks.append({
                    'type': 'load_imbalance',
                    'utilization_stats': utilization,
                    'severity': 'medium',
                    'description': "模块负载不均衡"
                })
        
        return bottlenecks
    
    def _apply_optimization_strategy(self, bottleneck: Dict[str, Any]) -> Dict[str, Any]:
        """应用优化策略"""
        bottleneck_type = bottleneck['type']
        
        if bottleneck_type == 'module_errors':
            # 重启错误模块
            affected_modules = bottleneck['affected_modules']
            for module_id in affected_modules:
                module = self.coordinator.modules.get(module_id)
                if module and hasattr(module, 'initialize'):
                    try:
                        module.initialize()
                        self.coordinator.module_states[module_id] = ModuleState.INITIALIZED
                    except Exception as e:
                        self.logger.error(f"重启模块 {module_id} 失败: {e}")
            
            return {'action': 'restart_error_modules', 'modules': affected_modules}
        
        elif bottleneck_type == 'load_imbalance':
            # 负载均衡优化（简化实现）
            return {'action': 'load_balancing', 'description': "实施负载均衡策略"}
        
        else:
            return {'action': 'monitor', 'description': "继续监控"}
    
    def _estimate_performance_improvement(self, bottlenecks: List[Dict[str, Any]]) -> float:
        """估算性能改进"""
        improvement = 0.0
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'module_errors':
                improvement += 0.3  # 修复错误模块可提升30%性能
            elif bottleneck['type'] == 'load_imbalance':
                improvement += 0.2  # 负载均衡可提升20%性能
        
        return min(improvement, 0.5)  # 最多提升50%
    
    def shutdown_system(self):
        """关闭系统"""
        self.logger.info("开始关闭认知系统...")
        
        self.state = ModuleState.SHUTTING_DOWN
        
        try:
            # 等待活跃执行完成
            while self.orchestrator.active_executions:
                self.logger.info(f"等待 {len(self.orchestrator.active_executions)} 个活跃执行完成...")
                time.sleep(1)
            
            # 关闭协调器
            self.coordinator.shutdown_all_modules()
            
            self.state = ModuleState.COMPLETED
            self.logger.info("认知系统已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭系统时出错: {e}")
            self.state = ModuleState.ERROR
    
    def initialize(self) -> bool:
        """初始化系统集成器"""
        self.state = ModuleState.INITIALIZED
        return True
    
    def cleanup(self) -> bool:
        """清理系统集成器"""
        self.shutdown_system()
        return True


# 便利函数
def create_cognitive_system_integrator(config: Dict[str, Any] = None) -> CognitiveSystemIntegrator:
    """创建认知系统集成器"""
    return CognitiveSystemIntegrator(config)


def create_integration_framework(modules: Dict[str, BaseModule]) -> CognitiveSystemIntegrator:
    """创建集成框架"""
    integrator = CognitiveSystemIntegrator()
    
    # 注册模块
    modules_config = {}
    for module_id, module in modules.items():
        modules_config[module_id] = {'module': module}
    
    integrator.initialize_cognitive_system(modules_config)
    
    return integrator


def setup_complete_cognitive_system(training_pipeline, optimizer, reasoner, learner) -> CognitiveSystemIntegrator:
    """设置完整的认知系统"""
    modules = {
        'training_pipeline': training_pipeline,
        'performance_optimizer': optimizer,
        'multi_step_reasoner': reasoner,
        'analogical_learner': learner
    }
    
    return create_integration_framework(modules)