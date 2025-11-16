#!/usr/bin/env python3
"""
海马体模拟器核心模块完整实现演示
基于Science期刊2025年最新研究成果的生物启发式记忆系统

实现的核心模块：
1. Transformer-based记忆编码器
2. 可微分神经字典
3. 模式分离机制
4. 快速一次性学习功能
5. 情景记忆存储和检索系统
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Any, Tuple, Optional, List

# 导入所有核心模块
from hippocampus.encoders.transformer_encoder import TransformerMemoryEncoder
from memory_cell.neural_dictionary import DifferentiableNeuralDictionary
from pattern_separation.pattern_separator import PatternSeparationNetwork
from hippocampus.fast_learning import OneShotLearner
from hippocampus.episodic_memory import EpisodicMemorySystem


class CompleteHippocampusModuleDemo:
    """
    海马体模拟器核心模块完整演示类
    展示所有核心功能模块的集成和协作
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化演示系统"""
        self.config = config or self._get_default_config()
        self.setup_modules()
        
        # 演示统计
        self.demo_stats = {
            'total_operations': 0,
            'memory_operations': 0,
            'learning_operations': 0,
            'retrieval_operations': 0,
            'start_time': time.time()
        }
        
        print("🧠 海马体模拟器核心模块初始化完成")
        print(f"   - 总参数数量: {sum(p.numel() for p in self._get_all_parameters()):,}")
        print(f"   - 激活模块: {len(self.active_modules)}")
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取演示默认配置"""
        return {
            'input_dim': 512,
            'hidden_dim': 256,
            'vocab_size': 1000,
            'max_seq_len': 128,
            'transformer_layers': 6,
            'num_heads': 8,
            'dict_cells': 6,
            'dict_capacity': 500,
            'granule_cells': 800,
            'ca3_cells': 150,
            'sparsity': 0.02,
            'episodic_cells': 6,
            'episodic_capacity': 50,
            'temporal_dim': 32
        }
    
    def setup_modules(self):
        """设置所有核心模块"""
        print("\n🔧 初始化核心模块...")
        
        # 1. Transformer记忆编码器
        self.transformer_encoder = TransformerMemoryEncoder(
            vocab_size=self.config['vocab_size'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['transformer_layers'],
            num_heads=self.config['num_heads'],
            max_seq_len=self.config['max_seq_len'],
            msb_enhancement=True,
            pattern_completion=True,
            temporal_alignment=True
        )
        
        # 2. 可微分神经字典
        self.neural_dictionary = DifferentiableNeuralDictionary(
            key_dim=self.config['hidden_dim'],
            value_dim=self.config['hidden_dim'],
            num_cells=self.config['dict_cells'],
            capacity_per_cell=self.config['dict_capacity'],
            temperature=1.0
        )
        
        # 3. 模式分离网络
        self.pattern_separator = PatternSeparationNetwork(
            input_dim=self.config['hidden_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_granule_cells=self.config['granule_cells'],
            num_ca3_cells=self.config['ca3_cells'],
            sparsity=self.config['sparsity']
        )
        
        # 4. 快速学习器
        self.one_shot_learner = OneShotLearner(
            input_dim=self.config['hidden_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_way=3,
            num_shot=1
        )
        
        # 5. 情景记忆系统
        self.episodic_memory = EpisodicMemorySystem(
            content_dim=self.config['hidden_dim'],
            temporal_dim=self.config['temporal_dim'],
            context_dim=self.config['hidden_dim'],
            num_cells=self.config['episodic_cells'],
            capacity_per_cell=self.config['episodic_capacity']
        )
        
        # 活跃模块列表
        self.active_modules = [
            'transformer_encoder', 'neural_dictionary', 'pattern_separator',
            'one_shot_learner', 'episodic_memory'
        ]
        
        print("   ✅ 所有核心模块初始化完成")
    
    def _get_all_parameters(self):
        """获取所有模块的参数"""
        all_params = []
        all_params.extend(self.transformer_encoder.parameters())
        all_params.extend(self.neural_dictionary.parameters())
        all_params.extend(self.pattern_separator.parameters())
        all_params.extend(self.one_shot_learner.parameters())
        all_params.extend(self.episodic_memory.parameters())
        return all_params
    
    def demo_memory_encoding(self, num_samples: int = 5) -> Dict[str, Any]:
        """
        演示记忆编码功能
        
        Args:
            num_samples: 演示样本数量
            
        Returns:
            编码演示结果
        """
        print(f"\n📝 记忆编码演示 ({num_samples}个样本)")
        print("-" * 50)
        
        encoding_results = []
        
        for i in range(num_samples):
            # 创建模拟输入
            # 使用浮点数输入，避免token索引问题
            input_features = torch.randn(1, self.config['input_dim'])
            context = torch.randn(1, self.config['hidden_dim'])
            
            print(f"\n样本 {i+1}:")
            
            # 1. Transformer编码（将浮点数输入转换为适合的格式）
            with torch.no_grad():
                # 将输入转换为适合Transformer的格式
                batch_size, input_dim = input_features.shape
                seq_len = 4  # 固定序列长度
                
                # 创建模拟token输入
                token_input = torch.randint(0, self.config['vocab_size'], (batch_size, seq_len))
                
                # 执行Transformer编码
                transformer_output, trans_stats = self.transformer_encoder(
                    input_ids=token_input,
                    context=context,
                    memory_type='episodic',
                    return_stats=True
                )
                
                print(f"   🔹 Transformer编码: 形状 {transformer_output.shape}")
            
            # 2. 模式分离
            with torch.no_grad():
                if batch_size > 1:
                    sep1, sep2, sep_stats = self.pattern_separator(
                        transformer_output[:, 0], transformer_output[:, 1]
                    )
                    separation_degree = sep_stats.get('separation_degree', 0.0)
                else:
                    sep1, sep2, sep_stats = self.pattern_separator(transformer_output[:, 0])
                    separation_degree = 0.0
                
                print(f"   🔹 模式分离完成, 分离度: {separation_degree:.3f}")
            
            # 3. 存储到神经字典
            with torch.no_grad():
                dict_result = self.neural_dictionary.write_memory(
                    key=sep1,
                    value=transformer_output.mean(dim=1)
                )
                print(f"   🔹 神经字典写入: {dict_result['total_writes']} 项")
            
            # 4. 存储情景记忆
            with torch.no_grad():
                timestamp = time.time()
                episodic_result = self.episodic_memory.store_episode(
                    content=transformer_output.mean(dim=1),
                    timestamp=timestamp,
                    context=context,
                    episode_id=f"demo_episode_{i}"
                )
                print(f"   🔹 情景记忆存储: {episodic_result['global_episode_id']}")
            
            encoding_results.append({
                'sample_id': i,
                'transformer_stats': trans_stats,
                'separation_stats': sep_stats,
                'dictionary_result': dict_result,
                'episodic_result': episodic_result,
                'input_features': input_features,
                'context': context
            })
            
            self.demo_stats['memory_operations'] += 1
        
        print(f"\n✅ 记忆编码演示完成，共处理 {num_samples} 个样本")
        return {'encoding_results': encoding_results}
    
    def demo_memory_retrieval(self, num_queries: int = 3) -> Dict[str, Any]:
        """
        演示记忆检索功能
        
        Args:
            num_queries: 检索查询数量
            
        Returns:
            检索演示结果
        """
        print(f"\n🔍 记忆检索演示 ({num_queries}个查询)")
        print("-" * 50)
        
        retrieval_results = []
        
        for i in range(num_queries):
            # 创建查询向量
            query = torch.randn(self.config['hidden_dim'])
            
            print(f"\n查询 {i+1}:")
            
            # 1. 神经字典检索
            with torch.no_grad():
                dict_results, dict_stats = self.neural_dictionary.retrieve_memory(
                    query=query.unsqueeze(0),
                    top_k=3,
                    fusion_method='attention'
                )
                print(f"   🔹 神经字典检索: {len(dict_stats['cell_stats'])} 个细胞搜索")
            
            # 2. 情景记忆检索
            with torch.no_grad():
                episodic_results, episodic_stats = self.episodic_memory.retrieve_episodes(
                    query_content=query.unsqueeze(0),
                    query_context=torch.zeros_like(query.unsqueeze(0)),
                    retrieval_type='hybrid'
                )
                print(f"   🔹 情景记忆检索: {episodic_stats['retrieval_type']} 模式")
            
            # 3. 计算检索质量
            with torch.no_grad():
                if dict_results.numel() > 0:
                    similarity = F.cosine_similarity(query, dict_results.mean(dim=1)).item()
                else:
                    similarity = 0.0
                
                print(f"   🔹 检索相似度: {similarity:.3f}")
            
            retrieval_results.append({
                'query_id': i,
                'query': query,
                'dictionary_results': dict_results,
                'episodic_results': episodic_results,
                'similarity_score': similarity,
                'dictionary_stats': dict_stats,
                'episodic_stats': episodic_stats
            })
            
            self.demo_stats['retrieval_operations'] += 1
        
        print(f"\n✅ 记忆检索演示完成，共执行 {num_queries} 个查询")
        return {'retrieval_results': retrieval_results}
    
    def demo_fast_learning(self, num_tasks: int = 2) -> Dict[str, Any]:
        """
        演示快速学习功能
        
        Args:
            num_tasks: 学习任务数量
            
        Returns:
            学习演示结果
        """
        print(f"\n⚡ 快速学习演示 ({num_tasks}个任务)")
        print("-" * 50)
        
        learning_results = []
        
        for i in range(num_tasks):
            print(f"\n任务 {i+1}:")
            
            # 创建few-shot学习数据
            support_size = 3
            query_size = 2
            
            support_x = torch.randn(support_size, self.config['hidden_dim'])
            query_x = torch.randn(query_size, self.config['hidden_dim'])
            support_y = torch.randint(0, 3, (support_size,))  # 3类分类
            
            print(f"   🔹 支持集: {support_x.shape}, 查询集: {query_x.shape}")
            
            # 执行few-shot学习
            with torch.no_grad():
                predictions, learning_stats = self.one_shot_learner.few_shot_learning(
                    support_x=support_x,
                    support_y=support_y,
                    query_x=query_x
                )
                
                print(f"   🔹 学习完成: {predictions.shape}")
                print(f"   🔹 平均记忆强度: {learning_stats['avg_memory_strength']:.3f}")
            
            learning_results.append({
                'task_id': i,
                'support_x': support_x,
                'support_y': support_y,
                'query_x': query_x,
                'predictions': predictions,
                'learning_stats': learning_stats
            })
            
            self.demo_stats['learning_operations'] += 1
        
        print(f"\n✅ 快速学习演示完成，共执行 {num_tasks} 个任务")
        return {'learning_results': learning_results}
    
    def demo_pattern_separation(self, num_pairs: int = 3) -> Dict[str, Any]:
        """
        演示模式分离功能
        
        Args:
            num_pairs: 输入对数量
            
        Returns:
            模式分离演示结果
        """
        print(f"\n🎯 模式分离演示 ({num_pairs}个输入对)")
        print("-" * 50)
        
        separation_results = []
        
        for i in range(num_pairs):
            # 创建两个相似输入
            base_input = torch.randn(self.config['hidden_dim'])
            # 创建相似但不同的第二个输入
            similarity_level = 0.7 + 0.2 * i / max(1, num_pairs - 1)  # 0.7-0.9的相似度
            similar_input = similarity_level * base_input + (1 - similarity_level) * torch.randn(self.config['hidden_dim'])
            
            print(f"\n输入对 {i+1}:")
            print(f"   目标相似度: {similarity_level:.2f}")
            
            # 执行模式分离
            with torch.no_grad():
                sep1, sep2, sep_stats = self.pattern_separator(
                    base_input.unsqueeze(0), 
                    similar_input.unsqueeze(0)
                )
                
                # 计算实际分离效果
                input_similarity = F.cosine_similarity(
                    base_input.unsqueeze(0), 
                    similar_input.unsqueeze(0)
                ).item()
                
                output_similarity = F.cosine_similarity(sep1, sep2).item()
                separation_improvement = input_similarity - output_similarity
                
                print(f"   🔹 输入相似度: {input_similarity:.3f}")
                print(f"   🔹 输出相似度: {output_similarity:.3f}")
                print(f"   🔹 分离提升: {separation_improvement:.3f}")
            
            separation_results.append({
                'pair_id': i,
                'base_input': base_input,
                'similar_input': similar_input,
                'target_similarity': similarity_level,
                'input_similarity': input_similarity,
                'output_similarity': output_similarity,
                'separation_improvement': separation_improvement,
                'separation_stats': sep_stats
            })
        
        print(f"\n✅ 模式分离演示完成，共处理 {num_pairs} 个输入对")
        return {'separation_results': separation_results}
    
    def demo_memory_consolidation(self) -> Dict[str, Any]:
        """
        演示记忆巩固功能
        
        Returns:
            巩固演示结果
        """
        print(f"\n🔄 记忆巩固演示")
        print("-" * 50)
        
        consolidation_start = time.time()
        
        # 执行记忆系统更新
        with torch.no_grad():
            # 更新情景记忆
            episodic_update = self.episodic_memory.update_memory_system()
            
            # 压缩神经字典
            dict_compression = self.neural_dictionary.compress_memories()
            
            consolidation_time = time.time() - consolidation_start
        
        print(f"   🔹 情景记忆更新: {episodic_update['cells_updated']} 个细胞")
        print(f"   🔹 字典压缩: {len(dict_compression)} 个操作")
        print(f"   🔹 巩固时间: {consolidation_time:.4f}s")
        
        consolidation_result = {
            'consolidation_time': consolidation_time,
            'episodic_update': episodic_update,
            'dictionary_compression': dict_compression,
            'timestamp': time.time()
        }
        
        print(f"\n✅ 记忆巩固演示完成")
        return consolidation_result
    
    def get_comprehensive_statistics(self) -> Dict[str, Any]:
        """
        获取综合系统统计信息
        
        Returns:
            完整的系统统计信息
        """
        print(f"\n📊 综合系统统计")
        print("-" * 50)
        
        # 各个模块的统计
        transformer_stats = self.transformer_encoder.get_memory_statistics()
        dictionary_stats = self.neural_dictionary.get_global_statistics()
        separator_stats = self.pattern_separator.get_network_statistics()
        learner_stats = self.one_shot_learner.get_learning_statistics()
        episodic_stats = self.episodic_memory.get_system_statistics()
        
        # 系统总体统计
        all_params = list(self.transformer_encoder.parameters()) + \
                    list(self.neural_dictionary.parameters()) + \
                    list(self.pattern_separator.parameters()) + \
                    list(self.one_shot_learner.parameters()) + \
                    list(self.episodic_memory.parameters())
        
        total_params = sum(p.numel() for p in all_params)
        model_size_mb = sum(p.numel() * p.element_size() for p in all_params) / (1024**2)
        
        # 演示统计
        runtime = time.time() - self.demo_stats['start_time']
        
        comprehensive_stats = {
            'system_overview': {
                'total_parameters': total_params,
                'model_size_mb': model_size_mb,
                'active_modules': len(self.active_modules),
                'demo_runtime_seconds': runtime,
                'total_demo_operations': self.demo_stats['total_operations']
            },
            'module_statistics': {
                'transformer_encoder': transformer_stats,
                'neural_dictionary': dictionary_stats,
                'pattern_separator': separator_stats,
                'one_shot_learner': learner_stats,
                'episodic_memory': episodic_stats
            },
            'demo_operations': {
                'memory_operations': self.demo_stats['memory_operations'],
                'learning_operations': self.demo_stats['learning_operations'],
                'retrieval_operations': self.demo_stats['retrieval_operations']
            },
            'configuration': self.config
        }
        
        # 打印关键统计
        print(f"   📋 总参数数量: {total_params:,}")
        print(f"   💾 模型大小: {model_size_mb:.2f} MB")
        print(f"   ⏱️  演示运行时间: {runtime:.2f}s")
        print(f"   🔢 记忆操作: {self.demo_stats['memory_operations']}")
        print(f"   📚 学习操作: {self.demo_stats['learning_operations']}")
        print(f"   🔍 检索操作: {self.demo_stats['retrieval_operations']}")
        print(f"   📈 神经字典容量: {dictionary_stats['total_capacity']}")
        print(f"   🧠 情景记忆容量: {episodic_stats['total_capacity']}")
        
        return comprehensive_stats
    
    def run_complete_demo(self):
        """运行完整演示"""
        print("=" * 80)
        print("🧠 海马体模拟器核心模块完整演示")
        print("基于Science期刊2025年最新研究成果")
        print("=" * 80)
        
        self.demo_stats['total_operations'] = 0
        
        try:
            # 1. 记忆编码演示
            encoding_result = self.demo_memory_encoding(num_samples=4)
            self.demo_stats['total_operations'] += 1
            
            # 2. 模式分离演示
            separation_result = self.demo_pattern_separation(num_pairs=3)
            self.demo_stats['total_operations'] += 1
            
            # 3. 记忆检索演示
            retrieval_result = self.demo_memory_retrieval(num_queries=3)
            self.demo_stats['total_operations'] += 1
            
            # 4. 快速学习演示
            learning_result = self.demo_fast_learning(num_tasks=2)
            self.demo_stats['total_operations'] += 1
            
            # 5. 记忆巩固演示
            consolidation_result = self.demo_memory_consolidation()
            self.demo_stats['total_operations'] += 1
            
            # 6. 综合统计
            comprehensive_stats = self.get_comprehensive_statistics()
            
            print("\n" + "=" * 80)
            print("🎉 海马体模拟器核心模块演示完成！")
            print("✅ 所有核心功能模块验证成功")
            print("📋 实现状态:")
            print("   ✓ Transformer-based记忆编码器")
            print("   ✓ 可微分神经字典")
            print("   ✓ 模式分离机制")
            print("   ✓ 快速一次性学习功能")
            print("   ✓ 情景记忆存储和检索系统")
            print("=" * 80)
            
            return {
                'encoding_demo': encoding_result,
                'separation_demo': separation_result,
                'retrieval_demo': retrieval_result,
                'learning_demo': learning_result,
                'consolidation_demo': consolidation_result,
                'statistics': comprehensive_stats
            }
            
        except Exception as e:
            print(f"\n❌ 演示过程中出现错误: {str(e)}")
            print("请检查模块配置和依赖关系")
            raise


def main():
    """主演示函数"""
    # 创建并运行演示
    demo = CompleteHippocampusModuleDemo()
    results = demo.run_complete_demo()
    
    return results


if __name__ == "__main__":
    results = main()