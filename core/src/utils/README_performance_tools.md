# 性能优化工具使用指南

本目录包含了脑启发AI系统的完整性能优化工具集，用于识别、分析和修复系统中的性能瓶颈和内存使用问题。

## 🛠️ 工具概览

### 核心优化工具

1. **performance_optimizer.py** - 核心性能优化工具
   - 智能缓存机制 (SmartLRUCache)
   - 内存池管理 (MemoryPool)
   - 性能监控 (PerformanceMonitor)
   - 异步批处理器 (AsyncBatchProcessor)

2. **file_memory_optimizer.py** - 文件操作和内存管理
   - 安全文件管理器 (SafeFileManager)
   - 内存泄漏检测器 (MemoryLeakDetector)
   - 资源跟踪器 (ResourceTracker)

3. **loop_optimizer.py** - 循环优化和性能基准测试
   - 循环优化器 (LoopOptimizer)
   - 矢量化操作 (VectorizedOperations)
   - 性能分析器 (PerformanceProfiler)
   - 基准测试运行器 (BenchmarkRunner)

4. **auto_performance_fixer.py** - 自动性能修复
   - 自动检测代码问题
   - 安全应用修复
   - 修复回滚机制

5. **performance_optimizer_main.py** - 主执行脚本
   - 整合所有优化工具
   - 生成综合报告
   - 支持多种运行模式

## 🚀 快速开始

### 1. 运行完整性能分析

```bash
# 运行完整的性能分析
python src/utils/performance_optimizer_main.py --mode full --output-dir ./performance_results

# 扫描项目问题
python src/utils/performance_optimizer_main.py --mode scan --output-dir ./results

# 应用自动修复
python src/utils/performance_optimizer_main.py --mode fix --severity-threshold high

# 运行基准测试
python src/utils/performance_optimizer_main.py --mode benchmark --output-dir ./benchmarks
```

### 2. 在代码中集成优化工具

#### 智能缓存
```python
from src.utils.performance_optimizer import smart_cache

@smart_cache(maxsize=128, ttl=3600)
def expensive_computation(data):
    # 您的计算逻辑
    return processed_data
```

#### 性能监控
```python
from src.utils.performance_optimizer import measure_performance

@measure_performance()
def my_function():
    # 您的代码逻辑
    pass
```

#### 安全文件操作
```python
from src.utils.file_memory_optimizer import get_global_file_manager

file_manager = get_global_file_manager()

# 安全保存数据
success = file_manager.safe_save_pickle(data, "/path/to/file.pkl")

# 安全加载数据
data = file_manager.safe_load_pickle("/path/to/file.pkl")
```

#### 内存泄漏检测
```python
from src.utils.file_memory_optimizer import get_global_memory_detector

detector = get_global_memory_detector()
detector.start_monitoring()

# 在应用中使用
large_data = [list(range(10000)) for _ in range(100)]
detector.watch_object(large_data, "test_large_data")

# 检查内存趋势
trend = detector.get_memory_trend()
print(f"内存趋势: {trend}")
```

#### 循环优化
```python
from src.utils.loop_optimizer import LoopOptimizer

# 优化列表操作
data = list(range(10000))
optimized_data = LoopOptimizer.optimize_range_loop(data, 'multiply', 2.0)

# 向量化操作
import numpy as np
array_data = np.array(data)
vectorized_result = LoopOptimizer.vectorized_operation(array_data, 'multiply', 2.0)
```

#### 性能基准测试
```python
from src.utils.loop_optimizer import BenchmarkRunner

runner = BenchmarkRunner()

# 基准测试函数性能
result = runner.run_benchmark(my_function, iterations=1000)
print(f"执行时间: {result.execution_time:.4f}秒")
print(f"吞吐量: {result.throughput_per_second:.2f} ops/sec")

# 比较两个函数
comparison = runner.compare_functions(slow_function, fast_function, 1000, test_data)
print(f"性能提升: {comparison['improvement_percent']:.1f}%")
```

## 📊 性能优化分析报告

运行完整分析后，会在输出目录生成以下文件：

- `comprehensive_performance_report.md` - 综合性能分析报告
- `performance_analysis_[timestamp].json` - 详细分析数据
- `performance_issues.json` - 发现的问题列表
- `fix_report.md` - 自动修复报告
- `benchmark_report.md` - 基准测试报告
- `performance_chart.png` - 性能可视化图表

## 🔧 关键优化建议

### 1. 立即实施 (高优先级)

#### 循环优化
```python
# ❌ 低效代码
for i in range(len(data)):
    result.append(data[i] * 2)

# ✅ 优化代码
result = [x * 2 for x in data]

# ✅ 向量化代码
import numpy as np
result = np.array(data) * 2
```

#### 内存管理
```python
# 使用内存池管理频繁的小对象
from src.utils.performance_optimizer import MemoryPool

pool = MemoryPool(block_size=1024, max_blocks=100)
block = pool.allocate()
# 使用block
pool.deallocate(block)
```

#### 缓存机制
```python
# 智能缓存自动管理内存使用
@smart_cache(maxsize=1000, ttl=3600)
def expensive_function(data):
    # 计算密集型操作
    return result
```

### 2. 短期优化 (中优先级)

#### 并发处理
```python
from src.utils.performance_optimizer import AsyncBatchProcessor

processor = AsyncBatchProcessor(max_workers=4, batch_size=100)
results = await processor.process_batch(data_list, processing_function)
```

#### 安全文件操作
```python
# 自动处理压缩和大小限制
file_manager = SafeFileManager(max_file_size_mb=50, compression_enabled=True)

# 批量保存
file_manager.batch_save(data_dict, directory_path, file_format='json')

# 安全的pickle操作
file_manager.safe_save_pickle(model_data, model_path)
file_manager.safe_load_pickle(model_path)
```

### 3. 长期优化 (低优先级)

#### 架构重构
- 考虑微服务架构分解大模块
- 实现分布式计算框架
- 添加GPU加速支持

## 📈 性能监控

### 系统性能监控
```python
from src.utils.performance_optimizer import get_global_monitor

monitor = get_global_monitor()
system_metrics = monitor.get_system_metrics()
print(f"CPU使用率: {system_metrics['cpu_percent']:.1f}%")
print(f"内存使用: {system_metrics['memory_used_mb']:.1f} MB")
```

### 内存泄漏检测
```python
detector = get_global_memory_detector()
detector.start_monitoring()

# 监控对象
detector.watch_object(large_object, "monitored_object")

# 检查内存趋势
trend = detector.get_memory_trend()
if trend['trend'] == 'increasing':
    print("⚠️ 检测到内存增长趋势")
```

## 🔄 自动修复

### 应用自动修复
```python
from src.utils.auto_performance_fixer import AutoPerformanceFixer

fixer = AutoPerformanceFixer("/path/to/project")

# 扫描问题
issues = fixer.scan_project()
print(f"发现 {len(issues)} 个问题")

# 应用高优先级修复
applied_fixes = fixer.apply_safe_fixes(severity_threshold='high')

# 生成修复报告
report = fixer.generate_fix_report("/tmp/fix_report.md")
```

### 回滚修复
```python
# 如果需要回滚修复
fixer.rollback_fixes()
```

## 📋 常见问题解决

### 1. 内存泄漏问题
- **问题**: 大量使用pickle但未正确关闭文件
- **解决**: 使用`SafeFileManager`的上下文管理器
- **预防**: 启用内存泄漏检测器

### 2. 循环性能问题
- **问题**: 使用`for i in range(len(x))`模式
- **解决**: 使用列表推导式或向量化操作
- **预防**: 运行循环优化基准测试

### 3. 缓存管理问题
- **问题**: 缓存无大小限制
- **解决**: 使用`SmartLRUCache`设置`maxsize`和`ttl`
- **预防**: 监控缓存命中率

### 4. 全局变量问题
- **问题**: 使用全局变量导致内存泄漏
- **解决**: 使用线程安全的单例模式
- **预防**: 扫描全局变量使用

## 🎯 性能目标

实施这些优化后，预期达到以下性能指标：

- **执行速度提升**: 30-50%
- **内存使用减少**: 20-30%
- **并发处理能力**: 提升50-100%
- **系统稳定性**: 显著改善

## 📞 技术支持

如果在使用性能优化工具时遇到问题：

1. 检查工具日志输出
2. 验证项目路径配置
3. 确保Python环境正确安装依赖
4. 查看生成的性能报告获取详细建议

## 🔄 版本更新

性能优化工具会持续更新，建议定期运行完整分析以获得最新的优化建议。

---

**注意**: 在生产环境中应用修复前，请确保创建完整的项目备份。性能优化工具已内置备份机制，但手动验证仍然重要。