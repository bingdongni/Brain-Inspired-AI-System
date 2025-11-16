"""
海马体模拟器主系统
整合所有核心模块的完整实现
基于最新神经科学研究成果的生物启发式记忆系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import json
import time

# 导入所有核心模块
from hippocampus.encoders.transformer_encoder import TransformerMemoryEncoder
from hippocampus.encoders.attention_mechanism import EnhancedAttention
from hippocampus.encoders.pattern_completion import PatternCompletionModule
from hippocampus.encoders.temporal_alignment import TemporalAlignmentModule
from memory_cell.neural_dictionary import DifferentiableNeuralDictionary
from pattern_separation.pattern_separator import PatternSeparationNetwork
from hippocampus.fast_learning import OneShotLearner
from hippocampus.episodic_memory import EpisodicMemorySystem


class HippocampusSimulator(nn.Module):
    """
    完整海马体模拟器
    集成所有核心功能模块的生物启发式记忆系统
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 512,
                 vocab_size: int = 30000,
                 max_seq_len: int = 1024,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化海马体模拟器
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            vocab_size: 词汇表大小
            max_seq_len: 最大序列长度
            config: 配置参数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # 加载配置
        self.config = config or self._get_default_config()
        
        # === 核心模块初始化 ===
        
        # 1. 记忆编码器（基于Transformer）
        self.transformer_encoder = TransformerMemoryEncoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=self.config.get('transformer_layers', 8),
            num_heads=self.config.get('num_heads', 8),
            max_seq_len=max_seq_len,
            msb_enhancement=self.config.get('msb_enhancement', True),
            pattern_completion=self.config.get('pattern_completion', True),
            temporal_alignment=self.config.get('temporal_alignment', True)
        )
        
        # 2. 可微分神经字典
        self.neural_dictionary = DifferentiableNeuralDictionary(
            key_dim=hidden_dim,
            value_dim=hidden_dim,
            num_cells=self.config.get('dict_cells', 8),
            capacity_per_cell=self.config.get('dict_capacity', 1000),
            hierarchical_levels=self.config.get('hierarchical_levels', 2)
        )
        
        # 3. 模式分离网络
        self.pattern_separator = PatternSeparationNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_granule_cells=self.config.get('granule_cells', 1000),
            num_ca3_cells=self.config.get('ca3_cells', 200),
            sparsity=self.config.get('sparsity', 0.02)
        )
        
        # 4. 一次性学习器
        self.one_shot_learner = OneShotLearner(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_way=self.config.get('num_way', 5),
            num_shot=self.config.get('num_shot', 1)
        )
        
        # 5. 情景记忆系统
        self.episodic_memory = EpisodicMemorySystem(
            content_dim=hidden_dim,
            temporal_dim=self.config.get('temporal_dim', 64),
            context_dim=hidden_dim,
            num_cells=self.config.get('episodic_cells', 8),
            capacity_per_cell=self.config.get('episodic_capacity', 100)
        )
        
        # === 模块间连接 ===
        
        # 跨模块注意力机制
        self.cross_module_attention = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 整合多个模块的输出
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 记忆路由网络
        self.memory_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5),  # 5个主要模块的路由权重
            nn.Softmax(dim=-1)
        )
        
        # 全局记忆管理器
        self.global_memory_manager = nn.ModuleDict({
            'working_memory': nn.Linear(hidden_dim, hidden_dim),
            'long_term_memory': nn.Linear(hidden_dim, hidden_dim),
            'episodic_buffer': nn.Linear(hidden_dim * 3, hidden_dim),
            'consolidation_gate': nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        })
        
        # 性能监控器
        self.performance_monitor = {
            'total_operations': 0,
            'module_usage': {name: 0 for name in [
                'transformer_encoder', 'neural_dictionary', 'pattern_separator',
                'one_shot_learner', 'episodic_memory'
            ]},
            'response_times': [],
            'memory_usage': [],
            'accuracy_metrics': []
        }
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'transformer_layers': 8,
            'num_heads': 8,
            'msb_enhancement': True,
            'pattern_completion': True,
            'temporal_alignment': True,
            'dict_cells': 8,
            'dict_capacity': 1000,
            'hierarchical_levels': 2,
            'granule_cells': 1000,
            'ca3_cells': 200,
            'sparsity': 0.02,
            'num_way': 5,
            'num_shot': 1,
            'episodic_cells': 8,
            'episodic_capacity': 100,
            'temporal_dim': 64,
            'consolidation_threshold': 0.8,
            'learning_rate': 0.01
        }
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                memory_query: Optional[torch.Tensor] = None,
                context: Optional[torch.Tensor] = None,
                mode: str = 'encoding',
                return_stats: bool = True) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        海马体模拟器主前向传播
        
        Args:
            input_ids: 输入序列
            attention_mask: 注意力掩码
            memory_query: 记忆查询
            context: 上下文信息
            mode: 运行模式 ('encoding', 'retrieval', 'learning', 'consolidation')
            return_stats: 是否返回统计信息
            
        Returns:
            输出结果、统计信息
        """
        start_time = time.time()
        
        # 更新操作计数
        self.performance_monitor['total_operations'] += 1
        
        # === 第一阶段：记忆编码 ===
        if mode in ['encoding', 'learning']:
            # 确保input_ids是Long类型（用于embedding层）
            if input_ids.dtype == torch.float32:
                # 对于浮点输入，将其转换为token索引（简化处理）
                input_ids = (input_ids.abs() * 1000).long() % self.vocab_size
            
            # Transformer编码
            transformer_output, transformer_stats = self.transformer_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                context=context,
                memory_type='episodic',
                return_stats=True
            )
            self.performance_monitor['module_usage']['transformer_encoder'] += 1
            
            # 模式分离
            if input_ids.shape[0] > 1:  # 只有多个输入时才需要分离
                sep_output1, sep_output2, sep_stats = self.pattern_separator(
                    transformer_output[:, 0], transformer_output[:, 1]
                )
                self.performance_monitor['module_usage']['pattern_separator'] += 1
            else:
                sep_output1, sep_output2 = transformer_output[:, 0], None
                sep_stats = {}
                self.performance_monitor['module_usage']['pattern_separator'] += 1
            
            # 存储到神经字典
            dict_result = self.neural_dictionary.write_memory(
                key=sep_output1,
                value=transformer_output.mean(dim=1)
            )
            self.performance_monitor['module_usage']['neural_dictionary'] += 1
            
            # 存储情景记忆
            timestamp = time.time()
            episodic_result = self.episodic_memory.store_episode(
                content=transformer_output.mean(dim=1),
                timestamp=timestamp,
                context=context or torch.zeros_like(transformer_output.mean(dim=1)),
                episode_id=f"episode_{self.performance_monitor['total_operations']}"
            )
            self.performance_monitor['module_usage']['episodic_memory'] += 1
            
            # 整合输出
            combined_output = torch.cat([
                transformer_output.mean(dim=1),
                sep_output1,
                transformer_output.mean(dim=1)
            ], dim=-1)
            
            output = self.cross_module_attention(combined_output)
            
        # === 第二阶段：记忆检索 ===
        elif mode == 'retrieval':
            # 从神经字典检索
            dict_result, retrieval_stats = self.neural_dictionary.retrieve_memory(
                query=memory_query or input_ids.mean(dim=1, keepdim=True),
                top_k=5
            )
            self.performance_monitor['module_usage']['neural_dictionary'] += 1
            
            # 从情景记忆检索
            episodic_output, episodic_stats = self.episodic_memory.retrieve_episodes(
                query_content=memory_query or input_ids.mean(dim=1),
                query_context=context or torch.zeros_like(input_ids.mean(dim=1)),
                retrieval_type='hybrid'
            )
            self.performance_monitor['module_usage']['episodic_memory'] += 1
            
            # 整合检索结果
            combined_output = torch.cat([
                dict_result.mean(dim=1) if dict_result.numel() > 0 else input_ids.mean(dim=1),
                episodic_output.mean(dim=1) if episodic_output.numel() > 0 else input_ids.mean(dim=1),
                memory_query or input_ids.mean(dim=1)
            ], dim=-1)
            
            output = self.cross_module_attention(combined_output)
            
        # === 第三阶段：快速学习 ===
        elif mode == 'learning':
            # Few-shot学习（简化实现）
            support_size = input_ids.shape[0] // 2
            support_x = input_ids[:support_size]
            query_x = input_ids[support_size:]
            
            # 模拟支持集标签（实际应用中需要提供真实标签）
            support_y = torch.randint(0, self.config.get('num_way', 5), (support_size,))
            
            predictions, learning_stats = self.one_shot_learner.few_shot_learning(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x
            )
            self.performance_monitor['module_usage']['one_shot_learner'] += 1
            
            output = predictions.mean(dim=1) if predictions.dim() > 2 else predictions
            
        # === 第四阶段：记忆巩固 ===
        elif mode == 'consolidation':
            # 更新记忆系统
            consolidation_stats = self.episodic_memory.update_memory_system()
            
            # 神经字典压缩
            compression_stats = self.neural_dictionary.compress_memories()
            
            output = input_ids.mean(dim=1)  # 返回输入的均值作为基础输出
        
        # === 全局记忆路由 ===
        memory_routing = self.memory_router(output)
        
        # === 计算响应时间 ===
        response_time = time.time() - start_time
        self.performance_monitor['response_times'].append(response_time)
        
        # 限制响应时间历史记录长度
        if len(self.performance_monitor['response_times']) > 1000:
            self.performance_monitor['response_times'] = self.performance_monitor['response_times'][-1000:]
        
        # === 准备统计信息 ===
        if return_stats:
            stats = {
                'mode': mode,
                'response_time': response_time,
                'total_operations': self.performance_monitor['total_operations'],
                'module_usage': self.performance_monitor['module_usage'].copy(),
                'output_norm': torch.norm(output).item(),
                'input_shape': list(input_ids.shape),
                'memory_routing': memory_routing.detach().cpu().numpy()
            }
            
            # 添加模式特定统计
            if mode == 'encoding':
                stats.update({
                    'transformer_stats': transformer_stats,
                    'separation_stats': sep_stats,
                    'dictionary_result': dict_result,
                    'episodic_result': episodic_result
                })
            elif mode == 'retrieval':
                stats.update({
                    'retrieval_stats': retrieval_stats,
                    'episodic_stats': episodic_stats
                })
            elif mode == 'learning':
                stats['learning_stats'] = learning_stats
            elif mode == 'consolidation':
                stats['consolidation_stats'] = consolidation_stats
                stats['compression_stats'] = compression_stats
            
            return output, stats
        else:
            return output, None
    
    def encode_memory(self,
                     content: torch.Tensor,
                     context: Optional[torch.Tensor] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        专门的记忆编码接口
        
        Args:
            content: 要编码的记忆内容
            context: 上下文信息
            metadata: 元数据
            
        Returns:
            编码结果和相关信息
        """
        with torch.no_grad():
            # 编码记忆
            encoded_output, stats = self.forward(
                input_ids=content.unsqueeze(0),
                context=context,
                mode='encoding',
                return_stats=True
            )
            
            # 存储到各个系统
            neural_dict_result = self.neural_dictionary.write_memory(
                key=content,
                value=encoded_output.squeeze(0)
            )
            
            episodic_result = self.episodic_memory.store_episode(
                content=encoded_output.squeeze(0),
                timestamp=time.time(),
                context=context or torch.zeros_like(encoded_output.squeeze(0)),
                episode_id=metadata.get('id') if metadata else None
            )
            
            encoding_result = {
                'encoded_content': encoded_output.squeeze(0),
                'content_norm': torch.norm(content).item(),
                'encoded_norm': torch.norm(encoded_output).item(),
                'neural_dictionary': neural_dict_result,
                'episodic_memory': episodic_result,
                'encoding_stats': stats,
                'metadata': metadata or {}
            }
            
            return encoding_result
    
    def retrieve_memory(self,
                       query: torch.Tensor,
                       retrieval_type: str = 'hybrid',
                       num_results: int = 5) -> Dict[str, Any]:
        """
        专门的记忆检索接口
        
        Args:
            query: 检索查询
            retrieval_type: 检索类型
            num_results: 返回结果数量
            
        Returns:
            检索结果和相关信息
        """
        with torch.no_grad():
            # 从神经字典检索
            dict_results, dict_stats = self.neural_dictionary.retrieve_memory(
                query=query,
                top_k=num_results,
                fusion_method=retrieval_type
            )
            
            # 从情景记忆检索
            episodic_results, episodic_stats = self.episodic_memory.retrieve_episodes(
                query_content=query,
                query_context=torch.zeros_like(query),
                retrieval_type=retrieval_type
            )
            
            # 整合结果
            similarity_score = F.cosine_similarity(query, dict_results.mean(dim=1)).item() if dict_results.numel() > 0 else 0.0
            
            retrieval_result = {
                'query': query,
                'neural_dictionary_results': dict_results,
                'episodic_memory_results': episodic_results,
                'similarity_score': similarity_score,
                'retrieval_type': retrieval_type,
                'num_results': num_results,
                'retrieval_stats': {
                    'dictionary_stats': dict_stats,
                    'episodic_stats': episodic_stats
                }
            }
            
            return retrieval_result
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """
        执行记忆巩固
        
        Returns:
            巩固结果和统计信息
        """
        consolidation_start = time.time()
        
        # 更新情景记忆
        episodic_update = self.episodic_memory.update_memory_system()
        
        # 压缩神经字典
        dict_compression = self.neural_dictionary.compress_memories()
        
        # 全局统计
        global_stats = self.get_system_statistics()
        
        consolidation_time = time.time() - consolidation_start
        
        consolidation_result = {
            'consolidation_time': consolidation_time,
            'episodic_update': episodic_update,
            'dictionary_compression': dict_compression,
            'global_statistics': global_stats,
            'timestamp': time.time()
        }
        
        return consolidation_result
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        获取完整的系统统计信息
        
        Returns:
            包含所有模块统计信息的字典
        """
        # 各个模块的统计信息
        transformer_stats = self.transformer_encoder.get_memory_statistics()
        dictionary_stats = self.neural_dictionary.get_global_statistics()
        separator_stats = self.pattern_separator.get_network_statistics()
        learner_stats = self.one_shot_learner.get_learning_statistics()
        episodic_stats = self.episodic_memory.get_system_statistics()
        
        # 系统整体统计
        system_stats = {
            'system_info': {
                'total_parameters': sum(p.numel() for p in self.parameters()),
                'total_memory_usage': sum(p.numel() * p.element_size() for p in self.parameters()) / (1024**2),
                'model_size_mb': sum(p.numel() * p.element_size() for p in self.parameters()) / (1024**2)
            },
            'performance_monitor': {
                'total_operations': self.performance_monitor['total_operations'],
                'module_usage': self.performance_monitor['module_usage'].copy(),
                'avg_response_time': np.mean(self.performance_monitor['response_times']) if self.performance_monitor['response_times'] else 0.0,
                'response_time_history_length': len(self.performance_monitor['response_times'])
            },
            'modules': {
                'transformer_encoder': transformer_stats,
                'neural_dictionary': dictionary_stats,
                'pattern_separator': separator_stats,
                'one_shot_learner': learner_stats,
                'episodic_memory': episodic_stats
            },
            'configuration': self.config.copy()
        }
        
        return system_stats
    
    def export_memory_state(self, filepath: str):
        """
        导出记忆状态到文件
        
        Args:
            filepath: 导出文件路径
        """
        memory_state = {
            'system_statistics': self.get_system_statistics(),
            'neural_dictionary_state': {
                'memory_keys': self.neural_dictionary.memory_cells[0].memory_keys.detach().cpu().numpy().tolist(),
                'memory_values': self.neural_dictionary.memory_cells[0].memory_values.detach().cpu().numpy().tolist(),
                'memory_strengths': self.neural_dictionary.memory_cells[0].memory_strengths.detach().cpu().numpy().tolist()
            },
            'episodic_memory_buffer': list(self.episodic_memory.global_memory_manager['short_term_buffer']),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(memory_state, f, indent=2, ensure_ascii=False)
    
    def load_memory_state(self, filepath: str):
        """
        从文件加载记忆状态
        
        Args:
            filepath: 记忆状态文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            memory_state = json.load(f)
        
        # 这里可以实现具体的加载逻辑
        # 注意：在实际应用中需要处理张量维度和设备兼容性问题
        print(f"Loaded memory state from {filepath}")
        print(f"System statistics: {memory_state['system_statistics']['performance_monitor']['total_operations']} operations")
    
    def reset_system(self):
        """重置整个系统"""
        # 清空所有模块
        self.neural_dictionary.clear_memory()
        self.episodic_memory.clear_memory_system()
        
        # 重置性能监控器
        self.performance_monitor = {
            'total_operations': 0,
            'module_usage': {name: 0 for name in [
                'transformer_encoder', 'neural_dictionary', 'pattern_separator',
                'one_shot_learner', 'episodic_memory'
            ]},
            'response_times': [],
            'memory_usage': [],
            'accuracy_metrics': []
        }
        
        print("Hippocampus simulator system reset completed")


# 便捷函数和工厂方法
def create_hippocampus_simulator(input_dim: int = 768,
                                hidden_dim: int = 512,
                                vocab_size: int = 30000,
                                config_path: Optional[str] = None) -> HippocampusSimulator:
    """
    创建海马体模拟器的便捷函数
    
    Args:
        input_dim: 输入维度
        hidden_dim: 隐藏层维度
        vocab_size: 词汇表大小
        config_path: 配置文件路径
        
    Returns:
        配置好的海马体模拟器实例
    """
    # 加载配置文件
    config = None
    if config_path:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    
    simulator = HippocampusSimulator(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        config=config
    )
    
    return simulator


def get_default_config() -> Dict[str, Any]:
    """获取默认配置文件"""
    return {
        'transformer_layers': 8,
        'num_heads': 8,
        'msb_enhancement': True,
        'pattern_completion': True,
        'temporal_alignment': True,
        'dict_cells': 8,
        'dict_capacity': 1000,
        'hierarchical_levels': 2,
        'granule_cells': 1000,
        'ca3_cells': 200,
        'sparsity': 0.02,
        'num_way': 5,
        'num_shot': 1,
        'episodic_cells': 8,
        'episodic_capacity': 100,
        'temporal_dim': 64,
        'consolidation_threshold': 0.8,
        'learning_rate': 0.01
    }


if __name__ == "__main__":
    # 示例使用
    print("初始化海马体模拟器...")
    
    # 创建模拟器
    simulator = create_hippocampus_simulator()
    
    print(f"系统参数数量: {sum(p.numel() for p in simulator.parameters()):,}")
    
    # 创建测试数据
    batch_size = 4
    seq_len = 16
    test_input = torch.randn(batch_size, seq_len, simulator.input_dim)
    
    print(f"测试输入形状: {test_input.shape}")
    
    # 执行前向传播
    output, stats = simulator.forward(
        input_ids=test_input,
        mode='encoding',
        return_stats=True
    )
    
    print(f"输出形状: {output.shape}")
    print(f"响应时间: {stats['response_time']:.4f}s")
    print(f"总操作数: {stats['total_operations']}")
    
    # 获取系统统计
    system_stats = simulator.get_system_statistics()
    print(f"系统内存使用: {system_stats['system_info']['model_size_mb']:.2f} MB")
    
    print("海马体模拟器测试完成！")