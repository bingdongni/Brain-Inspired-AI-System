"""
渐进式神经网络模块

实现渐进式神经网络(Progressive Neural Networks, PNN)，支持横向连接和跨任务知识迁移。
每个新任务创建新的网络列，通过横向连接复用之前学习的知识。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
from collections import OrderedDict
import copy

logger = logging.getLogger(__name__)


class ProgressiveColumn(nn.Module):
    """
    渐进式网络列
    
    表示单个任务的网络列，包含新任务特定的参数和与其他列的连接。
    """
    
    def __init__(self,
                 base_network: nn.Module,
                 task_id: int,
                 lateral_connections: Optional[Dict[str, nn.Module]] = None,
                 adapter_type: str = 'linear'):
        """
        初始化渐进式列
        
        Args:
            base_network: 基础网络架构
            task_id: 任务ID
            lateral_connections: 横向连接模块字典
            adapter_type: 适配器类型 ('linear', 'mlp', 'gate')
        """
        super().__init__()
        
        self.task_id = task_id
        self.base_network = copy.deepcopy(base_network)
        self.lateral_connections = lateral_connections or {}
        self.adapter_type = adapter_type
        
        # 冻结基础网络的参数
        for param in self.base_network.parameters():
            param.requires_grad = False
        
        # 创建适配器层
        self.adapters = nn.ModuleDict()
        self._create_adapters()
        
        # 任务特定的最终层
        self.task_specific_head = None
        
        logger.info(f"创建渐进式列 {task_id}")
    
    def _create_adapters(self):
        """创建适配器层"""
        if not self.lateral_connections:
            return
        
        for name, connection in self.lateral_connections.items():
            # 获取目标层的输入维度
            if hasattr(connection, 'out_features'):
                input_dim = connection.out_features
            elif hasattr(connection, 'out_channels'):
                input_dim = connection.out_channels
            else:
                # 尝试从forward推断
                try:
                    dummy_input = torch.randn(1, connection.in_features if hasattr(connection, 'in_features') else 64)
                    dummy_output = connection(dummy_input)
                    input_dim = dummy_output.numel()
                except:
                    input_dim = 64  # 默认值
            
            # 创建适配器
            if self.adapter_type == 'linear':
                self.adapters[name] = nn.Linear(input_dim, input_dim)
            elif self.adapter_type == 'mlp':
                self.adapters[name] = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 2),
                    nn.ReLU(),
                    nn.Linear(input_dim // 2, input_dim)
                )
            elif self.adapter_type == 'gate':
                self.adapters[name] = nn.Sequential(
                    nn.Linear(input_dim, input_dim // 4),
                    nn.ReLU(),
                    nn.Linear(input_dim // 4, input_dim),
                    nn.Sigmoid()
                )
    
    def forward(self, x: torch.Tensor, prev_columns_output: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入数据
            prev_columns_output: 之前列的输出列表
            
        Returns:
            当前列的输出
        """
        # 基础网络前向传播
        base_output = self.base_network(x)
        
        # 如果没有之前的列输出，直接返回基础输出
        if not prev_columns_output or not self.lateral_connections:
            return base_output
        
        # 应用横向连接和适配器
        enhanced_output = base_output
        
        # 假设横向连接在中间层应用，这里简化处理
        if len(prev_columns_output) > 0:
            # 平均融合所有之前列的输出
            prev_output = torch.stack(prev_columns_output, dim=0).mean(dim=0)
            
            # 应用适配器
            if 'output' in self.adapters:
                adapted_prev = self.adapters['output'](prev_output)
                # 残差连接
                enhanced_output = enhanced_output + 0.1 * adapted_prev
        
        return enhanced_output


class ProgressiveLateralConnection(nn.Module):
    """
    渐进式横向连接
    
    实现不同列之间的知识共享机制。
    """
    
    def __init__(self,
                 source_dim: int,
                 target_dim: int,
                 connection_type: str = 'attention',
                 dropout_rate: float = 0.1):
        """
        初始化横向连接
        
        Args:
            source_dim: 源维度
            target_dim: 目标维度
            connection_type: 连接类型 ('attention', 'bilinear', 'projection')
            dropout_rate: Dropout率
        """
        super().__init__()
        
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.connection_type = connection_type
        self.dropout_rate = dropout_rate
        
        if connection_type == 'attention':
            # 注意力机制连接
            self.attention = nn.MultiheadAttention(
                embed_dim=source_dim,
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.output_proj = nn.Linear(source_dim, target_dim)
            
        elif connection_type == 'bilinear':
            # 双线性连接
            self.bilinear = nn.Bilinear(source_dim, source_dim, target_dim)
            
        elif connection_type == 'projection':
            # 投影连接
            self.projection = nn.Sequential(
                nn.Linear(source_dim, source_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(source_dim // 2, target_dim)
            )
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, source_output: torch.Tensor, target_input: torch.Tensor) -> torch.Tensor:
        """
        横向连接前向传播
        
        Args:
            source_output: 源列的输出
            target_input: 目标列的输入
            
        Returns:
            连接后的输出
        """
        source_output = self.dropout(source_output)
        
        if self.connection_type == 'attention':
            # 注意力机制
            batch_size = source_output.size(0)
            source_features = source_output.view(batch_size, 1, -1)
            target_features = target_input.view(batch_size, 1, -1)
            
            attended_output, _ = self.attention(source_features, target_features, target_features)
            connected_output = attended_output.squeeze(1)
            
            return self.output_proj(connected_output)
            
        elif self.connection_type == 'bilinear':
            # 双线性连接
            return self.bilinear(source_output, target_input)
            
        elif self.connection_type == 'projection':
            # 投影连接
            projected_source = self.projection(source_output)
            return projected_source + target_input  # 残差连接


class ProgressiveNeuralNetwork(nn.Module):
    """
    渐进式神经网络
    
    实现完整的渐进式神经网络架构，支持多个任务的知识共享和迁移。
    """
    
    def __init__(self,
                 base_network: nn.Module,
                 max_tasks: int = 10,
                 connection_type: str = 'attention',
                 adapter_type: str = 'linear',
                 sharing_strategy: str = 'progressive',
                 gate_mechanism: bool = True):
        """
        初始化渐进式神经网络
        
        Args:
            base_network: 基础网络架构
            max_tasks: 最大任务数量
            connection_type: 横向连接类型
            adapter_type: 适配器类型
            sharing_strategy: 共享策略 ('progressive', 'selective', 'adaptive')
            gate_mechanism: 是否使用门控机制
        """
        super().__init__()
        
        self.base_network = base_network
        self.max_tasks = max_tasks
        self.connection_type = connection_type
        self.adapter_type = adapter_type
        self.sharing_strategy = sharing_strategy
        self.gate_mechanism = gate_mechanism
        
        # 网络列
        self.columns = nn.ModuleDict()
        self.num_columns = 0
        
        # 横向连接
        self.lateral_connections = nn.ModuleDict()
        
        # 任务特定的分类头
        self.task_heads = nn.ModuleDict()
        
        # 门控机制
        if gate_mechanism:
            self.gate_networks = nn.ModuleDict()
        
        # 知识共享权重
        self.sharing_weights = nn.Parameter(torch.ones(max_tasks, max_tasks))
        
        # 冻结的基础网络
        self.frozen_base = copy.deepcopy(base_network)
        for param in self.frozen_base.parameters():
            param.requires_grad = False
        
        logger.info(f"初始化渐进式网络，最大任务数: {max_tasks}")
    
    def add_new_task(self,
                    input_dim: Optional[Tuple[int, ...]] = None,
                    output_dim: Optional[int] = None,
                    base_head: Optional[nn.Module] = None) -> int:
        """
        添加新任务列
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            base_head: 基础头部网络
            
        Returns:
            新任务ID
        """
        task_id = self.num_columns
        
        # 创建新的网络列
        if task_id > 0:
            # 创建横向连接
            self._create_lateral_connections(task_id)
            
            # 创建列特定的适配器
            adapters = self._create_column_adapters(task_id)
        else:
            adapters = None
        
        # 创建新列
        new_column = ProgressiveColumn(
            base_network=self.frozen_base,
            task_id=task_id,
            lateral_connections=adapters,
            adapter_type=self.adapter_type
        )
        
        self.columns[str(task_id)] = new_column
        
        # 创建任务特定的分类头
        if base_head is not None:
            task_head = copy.deepcopy(base_head)
        else:
            # 创建默认分类头
            if input_dim is not None:
                if len(input_dim) == 1:  # 全连接
                    task_head = nn.Linear(input_dim[0], output_dim or 10)
                else:  # 卷积
                    task_head = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Linear(input_dim[0], 256),
                        nn.ReLU(),
                        nn.Linear(256, output_dim or 10)
                    )
            else:
                # 尝试从基础网络推断
                try:
                    dummy_input = torch.randn(1, *self._get_input_shape())
                    dummy_output = self.frozen_base(dummy_input)
                    feature_dim = dummy_output.numel() // dummy_output.size(0)
                    task_head = nn.Linear(feature_dim, output_dim or 10)
                except:
                    task_head = nn.Linear(512, output_dim or 10)
        
        self.task_heads[str(task_id)] = task_head
        
        # 创建门控网络
        if self.gate_mechanism:
            self.gate_networks[str(task_id)] = self._create_gate_network(task_id)
        
        self.num_columns += 1
        
        logger.info(f"添加新任务列 {task_id}")
        return task_id
    
    def _create_lateral_connections(self, new_task_id: int):
        """为新任务创建横向连接"""
        for prev_task_id in range(new_task_id):
            connection_name = f"{prev_task_id}_to_{new_task_id}"
            
            # 创建横向连接
            lateral_conn = ProgressiveLateralConnection(
                source_dim=512,  # 假设特征维度
                target_dim=512,
                connection_type=self.connection_type
            )
            
            self.lateral_connections[connection_name] = lateral_conn
    
    def _create_column_adapters(self, task_id: int) -> Dict[str, nn.Module]:
        """为新列创建适配器"""
        adapters = {}
        
        # 为每个之前的任务创建适配器
        for prev_task_id in range(task_id):
            adapter_name = f"adapter_{prev_task_id}"
            
            if self.adapter_type == 'linear':
                adapters[adapter_name] = nn.Linear(512, 512)
            elif self.adapter_type == 'mlp':
                adapters[adapter_name] = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512)
                )
        
        return adapters
    
    def _create_gate_network(self, task_id: int) -> nn.Module:
        """创建门控网络"""
        return nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Sigmoid()
        )
    
    def _get_input_shape(self) -> Tuple[int, ...]:
        """推断输入形状"""
        # 尝试从基础网络获取输入形状
        if hasattr(self.frozen_base, 'input_shape'):
            return self.frozen_base.input_shape
        else:
            # 默认形状
            return (3, 224, 224)
    
    def forward(self, 
               x: torch.Tensor, 
               task_id: Optional[int] = None,
               return_all_outputs: bool = False) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入数据
            task_id: 指定任务ID，None表示所有任务
            return_all_outputs: 是否返回所有任务的输出
            
        Returns:
            指定任务的输出或所有任务的输出字典
        """
        if self.num_columns == 0:
            raise ValueError("网络中没有任务列，请先添加任务")
        
        outputs = {}
        
        for col_id, column in self.columns.items():
            task_id_int = int(col_id)
            
            # 获取之前列的输出
            prev_outputs = []
            for prev_id in range(task_id_int):
                if str(prev_id) in outputs:
                    prev_outputs.append(outputs[str(prev_id)])
            
            # 当前列前向传播
            column_output = column(x, prev_outputs)
            
            # 应用门控机制
            if self.gate_mechanism and str(task_id_int) in self.gate_networks:
                gate_weights = self.gate_networks[str(task_id_int)](column_output)
                column_output = column_output * gate_weights
            
            # 任务特定的分类头
            if str(task_id_int) in self.task_heads:
                final_output = self.task_heads[str(task_id_int)](column_output)
            else:
                final_output = column_output
            
            outputs[col_id] = final_output
        
        # 返回结果
        if return_all_outputs:
            return outputs
        elif task_id is not None:
            return outputs[str(task_id)]
        else:
            # 默认返回最新任务
            return outputs[str(self.num_columns - 1)]
    
    def get_task_parameters(self, task_id: int) -> Dict[str, nn.Parameter]:
        """
        获取特定任务的参数
        
        Args:
            task_id: 任务ID
            
        Returns:
            参数字典
        """
        task_id_str = str(task_id)
        params = {}
        
        # 列参数
        if task_id_str in self.columns:
            for name, param in self.columns[task_id_str].named_parameters():
                if param.requires_grad:
                    params[f"column_{name}"] = param
        
        # 分类头参数
        if task_id_str in self.task_heads:
            for name, param in self.task_heads[task_id_str].named_parameters():
                params[f"head_{name}"] = param
        
        # 门控网络参数
        if self.gate_mechanism and task_id_str in self.gate_networks:
            for name, param in self.gate_networks[task_id_str].named_parameters():
                params[f"gate_{name}"] = param
        
        # 横向连接参数
        for conn_name, conn_module in self.lateral_connections.items():
            if f"to_{task_id}" in conn_name or f"{task_id}_to" in conn_name:
                for name, param in conn_module.named_parameters():
                    params[f"conn_{conn_name}_{name}"] = param
        
        return params
    
    def freeze_task_parameters(self, task_id: int) -> None:
        """
        冻结特定任务的参数
        
        Args:
            task_id: 任务ID
        """
        params = self.get_task_parameters(task_id)
        for param in params.values():
            param.requires_grad = False
        
        logger.info(f"冻结任务 {task_id} 的参数")
    
    def unfreeze_task_parameters(self, task_id: int) -> None:
        """
        解冻特定任务的参数
        
        Args:
            task_id: 任务ID
        """
        params = self.get_task_parameters(task_id)
        for param in params.values():
            param.requires_grad = True
        
        logger.info(f"解冻任务 {task_id} 的参数")
    
    def get_knowledge_transfer_matrix(self) -> torch.Tensor:
        """
        获取知识转移矩阵
        
        Returns:
            知识转移权重矩阵
        """
        return self.sharing_weights[:self.num_columns, :self.num_columns]
    
    def update_sharing_weights(self, task_ids: List[int], performance_matrix: torch.Tensor) -> None:
        """
        更新知识共享权重
        
        Args:
            task_ids: 任务ID列表
            performance_matrix: 性能矩阵 [len(task_ids), len(task_ids)]
        """
        # 基于性能矩阵更新共享权重
        for i, task_i in enumerate(task_ids):
            for j, task_j in enumerate(task_ids):
                if task_i < self.num_columns and task_j < self.num_columns:
                    # 基于性能相似度调整权重
                    similarity = performance_matrix[i, j].item()
                    self.sharing_weights[task_i, task_j] = similarity
        
        logger.info(f"更新了 {len(task_ids)} 个任务的知识共享权重")
    
    def prune_redundant_connections(self, threshold: float = 0.01) -> int:
        """
        修剪冗余连接
        
        Args:
            threshold: 权重阈值
            
        Returns:
            修剪的连接数量
        """
        pruned_count = 0
        
        # 修剪小的共享权重
        self.sharing_weights.data[self.sharing_weights.abs() < threshold] = 0
        
        # 统计修剪的连接
        total_weights = self.sharing_weights.numel()
        nonzero_weights = (self.sharing_weights != 0).sum().item()
        pruned_count = total_weights - nonzero_weights
        
        logger.info(f"修剪了 {pruned_count} 个冗余连接")
        
        return pruned_count
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        获取内存使用统计
        
        Returns:
            内存使用信息字典
        """
        total_params = 0
        trainable_params = 0
        
        # 计算所有模块的参数
        for module in [self.columns, self.task_heads, self.lateral_connections]:
            if module:
                param_count = sum(p.numel() for p in module.parameters())
                trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total_params += param_count
                trainable_params += trainable_count
        
        # 包括门控网络
        if self.gate_mechanism and self.gate_networks:
            gate_params = sum(p.numel() for p in self.gate_networks.parameters())
            gate_trainable = sum(p.numel() for p in self.gate_networks.parameters() if p.requires_grad)
            total_params += gate_params
            trainable_params += gate_trainable
        
        # 包括共享权重
        total_params += self.sharing_weights.numel()
        trainable_params += self.sharing_weights.numel()  # 假设共享权重可训练
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'num_columns': self.num_columns,
            'memory_per_column': total_params // max(self.num_columns, 1)
        }
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        保存检查点
        
        Args:
            filepath: 保存路径
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'num_columns': self.num_columns,
            'config': {
                'connection_type': self.connection_type,
                'adapter_type': self.adapter_type,
                'sharing_strategy': self.sharing_strategy,
                'gate_mechanism': self.gate_mechanism,
                'max_tasks': self.max_tasks
            }
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"渐进式网络检查点已保存到: {filepath}")
    
    def load_checkpoint(self, filepath: str, strict: bool = True) -> None:
        """
        加载检查点
        
        Args:
            filepath: 模型路径
            strict: 是否严格匹配
        """
        checkpoint = torch.load(filepath, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            self.load_state_dict(checkpoint['state_dict'], strict=strict)
            self.num_columns = checkpoint.get('num_columns', self.num_columns)
        
        logger.info(f"渐进式网络模型已从 {filepath} 加载")


class AdaptiveProgressiveNetwork(ProgressiveNeuralNetwork):
    """
    自适应渐进式网络
    
    扩展的渐进式网络，支持自适应知识共享和动态架构调整。
    """
    
    def __init__(self,
                 base_network: nn.Module,
                 max_tasks: int = 10,
                 adaptation_threshold: float = 0.1,
                 compression_ratio: float = 0.8,
                 **kwargs):
        super().__init__(base_network, max_tasks, **kwargs)
        
        self.adaptation_threshold = adaptation_threshold
        self.compression_ratio = compression_ratio
        
        # 自适应参数
        self.column_importance = nn.Parameter(torch.ones(max_tasks))
        self.adaptation_history = []
        
    def adaptive_add_task(self,
                        performance_feedback: Optional[Dict[int, float]] = None) -> int:
        """
        自适应添加任务
        
        Args:
            performance_feedback: 性能反馈字典 {task_id: performance}
            
        Returns:
            新任务ID
        """
        # 基于性能反馈决定是否需要新的列
        if performance_feedback:
            # 分析现有列的性能
            avg_performance = np.mean(list(performance_feedback.values()))
            
            if avg_performance > 1.0 - self.adaptation_threshold:
                # 性能良好，可能不需要新列
                logger.info("当前性能良好，尝试压缩现有列")
                self._compress_existing_columns()
                return max(0, self.num_columns - 1)  # 返回现有列
        
        # 添加新任务列
        return self.add_new_task()
    
    def _compress_existing_columns(self) -> None:
        """压缩现有列以节省资源"""
        logger.info("开始压缩现有列")
        
        # 基于重要性评分排序
        importance_scores = self.column_importance[:self.num_columns]
        
        # 保留最重要的列
        important_indices = torch.argsort(importance_scores, descending=True)[:int(self.num_columns * self.compression_ratio)]
        important_indices = sorted(important_indices.tolist())
        
        # 重组织列
        self._reorganize_columns(important_indices)
        
        logger.info(f"压缩完成，保留 {len(important_indices)} 列")
    
    def _reorganize_columns(self, keep_indices: List[int]) -> None:
        """
        重新组织列
        
        Args:
            keep_indices: 保留的列索引
        """
        new_columns = nn.ModuleDict()
        new_heads = nn.ModuleDict()
        new_gates = nn.ModuleDict() if self.gate_mechanism else None
        
        old_to_new_mapping = {}
        
        for new_idx, old_idx in enumerate(keep_indices):
            old_col_key = str(old_idx)
            new_col_key = str(new_idx)
            
            if old_col_key in self.columns:
                new_columns[new_col_key] = self.columns[old_col_key]
            if old_col_key in self.task_heads:
                new_heads[new_col_key] = self.task_heads[old_col_key]
            if self.gate_mechanism and old_col_key in self.gate_networks:
                new_gates[new_col_key] = self.gate_networks[old_col_key]
            
            old_to_new_mapping[old_idx] = new_idx
        
        # 更新模块
        self.columns = new_columns
        self.task_heads = new_heads
        if self.gate_mechanism:
            self.gate_networks = new_gates
        
        self.num_columns = len(keep_indices)
        
        logger.info(f"重新组织列，从 {len(keep_indices)} 列减少到 {self.num_columns} 列")