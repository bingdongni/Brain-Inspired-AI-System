#!/usr/bin/env python3
"""
记忆学习演示 - 海马体记忆机制
Memory Learning Demo - Hippocampus Memory Mechanism

演示海马体的核心功能：
- 序列记忆学习
- 模式补全
- 记忆检索
- 遗忘曲线分析
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Any
import argparse
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch未安装，将使用简化版本演示")

class HippocampusMemorySystem:
    """海马体记忆系统"""
    
    def __init__(self, memory_capacity: int = 1000, encoding_dim: int = 128):
        self.memory_capacity = memory_capacity
        self.encoding_dim = encoding_dim
        
        # 记忆存储
        self.memories = []
        self.memory_strengths = []
        self.memory_patterns = []
        
        # 模式分离和补全参数
        self.pattern_separation_threshold = 0.3
        self.pattern_completion_threshold = 0.7
        self.forgetting_rate = 0.05
        
        # 统计信息
        self.stats = {
            'total_learned': 0,
            'successful_retrievals': 0,
            'pattern_completions': 0,
            'forgetting_events': 0
        }
        
    def encode_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """编码模式到高维空间"""
        if not TORCH_AVAILABLE:
            # 简化的编码机制
            encoded = np.random.randn(self.encoding_dim)
            return encoded / np.linalg.norm(encoded)
            
        # 使用神经网络编码
        encoder = nn.Sequential(
            nn.Linear(len(pattern), 256),
            nn.ReLU(),
            nn.Linear(256, self.encoding_dim),
            nn.Tanh()
        )
        
        with torch.no_grad():
            pattern_tensor = torch.FloatTensor(pattern)
            encoded = encoder(pattern_tensor).numpy()
            
        return encoded / (np.linalg.norm(encoded) + 1e-8)
        
    def store_memory(self, pattern: np.ndarray, strength: float = 1.0):
        """存储记忆"""
        # 编码模式
        encoded_pattern = self.encode_pattern(pattern)
        
        # 检查是否已存在相似记忆
        similarity_threshold = 0.8
        for i, existing_pattern in enumerate(self.memory_patterns):
            similarity = np.dot(encoded_pattern, existing_pattern)
            if similarity > similarity_threshold:
                # 加强现有记忆
                self.memory_strengths[i] += strength * 0.1
                self.stats['total_learned'] += 1
                return i
                
        # 创建新记忆
        if len(self.memories) < self.memory_capacity:
            memory_id = len(self.memories)
            self.memories.append(pattern.copy())
            self.memory_patterns.append(encoded_pattern)
            self.memory_strengths.append(strength)
            self.stats['total_learned'] += 1
            return memory_id
        else:
            # 替换最弱的记忆
            min_strength_idx = np.argmin(self.memory_strengths)
            self.memories[min_strength_idx] = pattern.copy()
            self.memory_patterns[min_strength_idx] = encoded_pattern
            self.memory_strengths[min_strength_idx] = strength
            self.stats['total_learned'] += 1
            return min_strength_idx
            
    def retrieve_memory(self, query_pattern: np.ndarray, 
                       partial_match: bool = False) -> Tuple[Any, float, List[int]]:
        """检索记忆"""
        if not self.memory_patterns:
            return None, 0.0, []
            
        # 编码查询模式
        encoded_query = self.encode_pattern(query_pattern)
        
        # 计算相似度
        similarities = []
        for pattern in self.memory_patterns:
            similarity = np.dot(encoded_query, pattern)
            similarities.append(similarity)
            
        similarities = np.array(similarities)
        
        # 找到最相似的记忆
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # 判断检索成功
        threshold = self.pattern_completion_threshold if partial_match else 0.8
        
        if best_similarity > threshold:
            self.stats['successful_retrievals'] += 1
            retrieved_memory = self.memories[best_match_idx]
            return retrieved_memory, best_similarity, [best_match_idx]
        else:
            return None, best_similarity, []
            
    def complete_pattern(self, partial_pattern: np.ndarray) -> Tuple[Any, float]:
        """模式补全"""
        # 找到最相似的完整模式
        retrieved, similarity, indices = self.retrieve_memory(partial_pattern, partial_match=True)
        
        if retrieved is not None and similarity > self.pattern_completion_threshold:
            self.stats['pattern_completions'] += 1
            
            # 使用相似度加权补全
            if len(indices) > 1:
                # 多重补全
                weighted_completion = np.zeros_like(retrieved)
                total_weight = 0
                
                for idx in indices:
                    weight = similarities[idx]
                    weighted_completion += weight * self.memories[idx]
                    total_weight += weight
                    
                completed_pattern = weighted_completion / (total_weight + 1e-8)
            else:
                completed_pattern = retrieved
                
            return completed_pattern, similarity
            
        return None, 0.0
        
    def simulate_forgetting(self, time_steps: int = 100):
        """模拟遗忘过程"""
        forgetting_curve = []
        
        for step in range(time_steps):
            # 随机遗忘一些记忆
            for i in range(len(self.memory_strengths)):
                if np.random.random() < self.forgetting_rate:
                    self.memory_strengths[i] *= 0.95
                    self.stats['forgetting_events'] += 1
                    
            # 计算平均记忆强度
            avg_strength = np.mean(self.memory_strengths) if self.memory_strengths else 0
            forgetting_curve.append(avg_strength)
            
        return forgetting_curve
        
    def learn_sequence(self, sequence: List[np.ndarray]) -> Dict[str, Any]:
        """学习序列模式"""
        print(f"📚 学习序列模式，长度: {len(sequence)}")
        
        sequence_results = {
            'learned_items': [],
            'retrieval_successes': 0,
            'completion_successes': 0,
            'processing_time': 0
        }
        
        start_time = time.time()
        
        for i, item in enumerate(sequence):
            # 存储当前项目
            memory_id = self.store_memory(item, strength=1.0)
            sequence_results['learned_items'].append(memory_id)
            
            # 测试检索（除了最后一个项目）
            if i < len(sequence) - 1:
                retrieved, similarity, indices = self.retrieve_memory(item)
                if retrieved is not None:
                    sequence_results['retrieval_successes'] += 1
                    
            # 测试模式补全
            if i > 0:
                # 使用部分模式测试补全
                partial = item[:len(item)//2] if len(item) > 2 else item
                completed, similarity = self.complete_pattern(partial)
                if completed is not None:
                    sequence_results['completion_successes'] += 1
                    
            print(f"   步骤 {i+1}: 记忆强度 {self.memory_strengths[-1]:.3f}")
            
        sequence_results['processing_time'] = time.time() - start_time
        
        return sequence_results
        
    def analyze_memory_patterns(self) -> Dict[str, Any]:
        """分析记忆模式统计"""
        if not self.memory_patterns:
            return {}
            
        patterns_array = np.array(self.memory_patterns)
        
        analysis = {
            'num_memories': len(self.memories),
            'avg_strength': np.mean(self.memory_strengths),
            'strength_std': np.std(self.memory_strengths),
            'memory_diversity': np.mean([np.std(pattern) for pattern in self.memory_patterns]),
            'capacity_usage': len(self.memories) / self.memory_capacity,
            'retrieval_success_rate': self.stats['successful_retrievals'] / max(1, self.stats['total_learned']),
            'pattern_completion_rate': self.stats['pattern_completions'] / max(1, self.stats['total_learned']),
            'stats': self.stats.copy()
        }
        
        return analysis


def generate_sequence_data(sequence_type: str = "numbers", length: int = 10) -> List[np.ndarray]:
    """生成序列数据"""
    if sequence_type == "numbers":
        # 数字序列
        sequence = []
        for i in range(length):
            num = i + 1
            # 将数字编码为向量
            pattern = np.zeros(20)
            pattern[:num] = 1.0  # 设置前 num 个位置为 1
            sequence.append(pattern)
            
    elif sequence_type == "patterns":
        # 几何模式序列
        sequence = []
        patterns = [
            np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]),  # 对角线
            np.array([1, 1, 1, 0, 0, 0, 0, 0, 0]),  # 水平线
            np.array([1, 0, 0, 1, 0, 0, 1, 0, 0]),  # 垂直线
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])   # 全部
        ]
        
        for i in range(length):
            pattern_idx = i % len(patterns)
            pattern = np.zeros(20)
            pattern[:len(patterns[pattern_idx])] = patterns[pattern_idx]
            sequence.append(pattern)
            
    elif sequence_type == "letters":
        # 字母序列
        sequence = []
        letters = "HELLOWORLD"
        for i in range(length):
            letter = letters[i % len(letters)]
            # 将字母编码为向量
            pattern = np.zeros(26)
            pattern[ord(letter) - ord('A')] = 1.0
            sequence.append(pattern)
            
    else:
        raise ValueError(f"未知序列类型: {sequence_type}")
        
    return sequence


def run_memory_learning_demo():
    """运行记忆学习演示"""
    print("🧠 海马体记忆学习演示")
    print("=" * 50)
    
    # 创建海马体记忆系统
    hippocampus = HippocampusMemorySystem(
        memory_capacity=1000,
        encoding_dim=128
    )
    
    print("\n1️⃣ 基础记忆功能演示")
    print("-" * 30)
    
    # 基础记忆测试
    test_patterns = [
        np.array([1, 0, 0, 1, 0, 0, 1, 0]),
        np.array([0, 1, 0, 0, 1, 0, 0, 1]),
        np.array([1, 1, 0, 1, 1, 0, 1, 1])
    ]
    
    print("存储测试模式...")
    for i, pattern in enumerate(test_patterns):
        memory_id = hippocampus.store_memory(pattern, strength=1.0)
        print(f"   模式 {i+1}: 存储到记忆单元 {memory_id}")
        
    print("\n检索测试模式...")
    for i, pattern in enumerate(test_patterns):
        retrieved, similarity, indices = hippocampus.retrieve_memory(pattern)
        if retrieved is not None:
            print(f"   模式 {i+1}: 检索成功 (相似度: {similarity:.3f})")
        else:
            print(f"   模式 {i+1}: 检索失败")
    
    print("\n2️⃣ 序列学习演示")
    print("-" * 30)
    
    # 学习数字序列
    number_sequence = generate_sequence_data("numbers", 8)
    print("学习数字序列: [1, 2, 3, 4, 5, 6, 7, 8]")
    
    sequence_results = hippocampus.learn_sequence(number_sequence)
    
    print(f"\n序列学习结果:")
    print(f"   学习项目数: {len(sequence_results['learned_items'])}")
    print(f"   检索成功率: {sequence_results['retrieval_successes']}/{len(number_sequence)-1}")
    print(f"   补全成功率: {sequence_results['completion_successes']}/{len(number_sequence)-1}")
    print(f"   处理时间: {sequence_results['processing_time']:.3f}秒")
    
    print("\n3️⃣ 模式补全演示")
    print("-" * 30)
    
    # 测试模式补全
    test_completion_patterns = [
        number_sequence[0][:4],  # 部分数字1
        number_sequence[3][:3],  # 部分数字4
        number_sequence[6][:5],  # 部分数字7
    ]
    
    completion_results = []
    
    for i, partial_pattern in enumerate(test_completion_patterns):
        completed_pattern, similarity = hippocampus.complete_pattern(partial_pattern)
        
        if completed_pattern is not None:
            print(f"   测试 {i+1}: 补全成功 (相似度: {similarity:.3f})")
            # 检查补全准确性
            original_idx = i * 2  # 对应的完整模式索引
            if original_idx < len(number_sequence):
                original = number_sequence[original_idx]
                accuracy = 1.0 - np.mean(np.abs(completed_pattern - original))
                print(f"         补全准确率: {accuracy:.3f}")
                completion_results.append(similarity)
        else:
            print(f"   测试 {i+1}: 补全失败")
            
    print("\n4️⃣ 遗忘曲线演示")
    print("-" * 30)
    
    # 生成遗忘曲线
    print("生成遗忘曲线 (100个时间步)...")
    forgetting_curve = hippocampus.simulate_forgetting(100)
    
    print(f"遗忘曲线统计:")
    print(f"   初始平均强度: {forgetting_curve[0]:.3f}")
    print(f"   最终平均强度: {forgetting_curve[-1]:.3f}")
    print(f"   遗忘率: {(forgetting_curve[0] - forgetting_curve[-1]) / forgetting_curve[0] * 100:.1f}%")
    
    print("\n5️⃣ 记忆模式分析")
    print("-" * 30)
    
    # 分析记忆模式
    analysis = hippocampus.analyze_memory_patterns()
    
    print("记忆系统分析:")
    print(f"   记忆数量: {analysis['num_memories']}")
    print(f"   平均强度: {analysis['avg_strength']:.3f}")
    print(f"   强度标准差: {analysis['strength_std']:.3f}")
    print(f"   记忆多样性: {analysis['memory_diversity']:.3f}")
    print(f"   容量使用率: {analysis['capacity_usage']:.1%}")
    print(f"   检索成功率: {analysis['retrieval_success_rate']:.1%}")
    print(f"   补全成功率: {analysis['pattern_completion_rate']:.1%}")
    
    print("\n6️⃣ 可视化结果")
    print("-" * 30)
    
    try:
        # 创建可视化图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('海马体记忆学习演示结果', fontsize=16)
        
        # 遗忘曲线
        axes[0, 0].plot(forgetting_curve, 'b-', linewidth=2)
        axes[0, 0].set_title('遗忘曲线')
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('平均记忆强度')
        axes[0, 0].grid(True)
        
        # 记忆强度分布
        if hippocampus.memory_strengths:
            axes[0, 1].hist(hippocampus.memory_strengths, bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title('记忆强度分布')
            axes[0, 1].set_xlabel('记忆强度')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].grid(True)
        
        # 学习性能指标
        metrics = ['检索成功率', '补全成功率']
        values = [analysis['retrieval_success_rate'], analysis['pattern_completion_rate']]
        
        axes[1, 0].bar(metrics, values, color=['blue', 'orange'])
        axes[1, 0].set_title('学习性能指标')
        axes[1, 0].set_ylabel('成功率')
        axes[1, 0].set_ylim(0, 1)
        
        # 补全相似度分布
        if completion_results:
            axes[1, 1].hist(completion_results, bins=10, alpha=0.7, color='red')
            axes[1, 1].set_title('模式补全相似度分布')
            axes[1, 1].set_xlabel('相似度')
            axes[1, 1].set_ylabel('频次')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, '无补全数据', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('模式补全相似度分布')
        
        plt.tight_layout()
        
        # 保存图表
        import os
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/memory_learning_demo.png', dpi=300, bbox_inches='tight')
        print("📊 可视化图表已保存到: visualizations/memory_learning_demo.png")
        
        plt.show()
        
    except ImportError:
        print("⚠️ matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
    
    print("\n7️⃣ 保存演示结果")
    print("-" * 30)
    
    # 保存结果
    results = {
        'demo_type': 'memory_learning',
        'timestamp': time.time(),
        'sequence_results': sequence_results,
        'completion_results': completion_results,
        'forgetting_curve': forgetting_curve,
        'analysis': analysis,
        'hippocampus_stats': hippocampus.stats
    }
    
    import os
    os.makedirs('data/results', exist_ok=True)
    
    with open('data/results/memory_learning_demo_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("💾 演示结果已保存到: data/results/memory_learning_demo_results.json")
    
    print("\n🎉 记忆学习演示完成!")
    print("=" * 50)
    
    # 总结
    print("\n📋 演示总结:")
    print(f"✅ 成功存储了 {len(hippocampus.memories)} 个记忆")
    print(f"✅ 检索成功率达到 {analysis['retrieval_success_rate']:.1%}")
    print(f"✅ 模式补全成功率达到 {analysis['pattern_completion_rate']:.1%}")
    print(f"✅ 记忆系统容量使用率: {analysis['capacity_usage']:.1%}")
    
    if analysis['retrieval_success_rate'] > 0.8:
        print("🎯 海马体记忆机制工作正常!")
    else:
        print("⚠️ 记忆机制可能需要调优")
        
    return results


def run_pattern_separation_demo():
    """运行模式分离演示"""
    print("\n🧩 模式分离演示")
    print("=" * 50)
    
    hippocampus = HippocampusMemorySystem(memory_capacity=500, encoding_dim=64)
    
    # 生成容易混淆的模式
    similar_patterns = []
    
    for i in range(10):
        base_pattern = np.random.randn(16)
        
        # 创建相似的模式
        pattern1 = base_pattern + np.random.randn(16) * 0.1
        pattern2 = base_pattern + np.random.randn(16) * 0.1
        pattern3 = base_pattern + np.random.randn(16) * 0.5  # 更不相似
        
        similar_patterns.extend([pattern1, pattern2, pattern3])
    
    print("测试模式分离能力...")
    
    # 存储所有模式
    for i, pattern in enumerate(similar_patterns):
        hippocampus.store_memory(pattern)
        
    # 测试相似模式的分离
    test_base = np.random.randn(16)
    test_similar1 = test_base + np.random.randn(16) * 0.05
    test_similar2 = test_base + np.random.randn(16) * 0.5
    
    retrieved1, sim1, _ = hippocampus.retrieve_memory(test_similar1)
    retrieved2, sim2, _ = hippocampus.retrieve_memory(test_similar2)
    
    print(f"测试结果:")
    print(f"   非常相似模式检索相似度: {sim1:.3f}")
    print(f"   中等相似模式检索相似度: {sim2:.3f}")
    print(f"   分离效果: {abs(sim1 - sim2):.3f}")
    
    return {
        'high_similarity': sim1,
        'medium_similarity': sim2,
        'separation_effectiveness': abs(sim1 - sim2)
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='海马体记忆学习演示')
    parser.add_argument('--demo', choices=['all', 'memory', 'separation'], default='all',
                       help='演示类型: all(全部), memory(记忆学习), separation(模式分离)')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图表')
    parser.add_argument('--save-results', action='store_true', help='保存结果')
    
    args = parser.parse_args()
    
    if args.demo in ['all', 'memory']:
        results = run_memory_learning_demo()
        
    if args.demo in ['all', 'separation']:
        separation_results = run_pattern_separation_demo()
        
    if args.save_results:
        print("\n💾 所有结果已保存")
        
    if args.visualize:
        print("\n📊 启动交互式可视化...")
        import matplotlib.pyplot as plt
        plt.ion()  # 开启交互模式
        plt.show()


if __name__ == "__main__":
    main()