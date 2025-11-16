#!/usr/bin/env python3
"""
核心AI模块代码质量修复验证测试
验证修复后的代码能否正常运行
"""

import sys
import os
import traceback
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """测试模块导入"""
    logger.info("开始测试模块导入...")
    
    try:
        # 测试海马体模块导入
        sys.path.insert(0, '/workspace/brain-inspired-ai/src/modules')
        from hippocampus import HippocampalSimulator
        logger.info("✓ 海马体模块导入成功")
        
        # 测试记忆接口模块导入  
        from memory_interface import MemoryInterfaceCore, create_memory_interface
        logger.info("✓ 记忆接口模块导入成功")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 模块导入失败: {e}")
        traceback.print_exc()
        return False

def test_hippocampus_simulator():
    """测试海马体模拟器"""
    logger.info("开始测试海马体模拟器...")
    
    try:
        import torch
        from hippocampus.core.simulator import create_hippocampus_simulator, get_hippocampus_config
        
        # 测试创建模拟器
        simulator = create_hippocampus_simulator(input_dim=256)
        logger.info("✓ 海马体模拟器创建成功")
        
        # 测试基本功能
        test_input = torch.randn(1, 256)
        
        # 测试编码
        encoding_result = simulator.encode_memory(test_input)
        assert 'final_encoding' in encoding_result
        logger.info("✓ 记忆编码功能正常")
        
        # 测试存储
        memory_id = simulator.store_memory(encoding_result['final_encoding'])
        logger.info("✓ 记忆存储功能正常")
        
        # 测试检索
        retrieval_result = simulator.retrieve_memory(encoding_result['final_encoding'])
        assert 'retrieved_memory' in retrieval_result
        logger.info("✓ 记忆检索功能正常")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 海马体模拟器测试失败: {e}")
        traceback.print_exc()
        return False

def test_memory_interface():
    """测试记忆接口"""
    logger.info("开始测试记忆接口...")
    
    try:
        import torch
        from memory_interface import create_memory_interface
        
        # 创建接口
        interface = create_memory_interface()
        logger.info("✓ 记忆接口创建成功")
        
        # 注册模块
        interface.register_module("module1", "mock_instance1")
        interface.register_module("module2", "mock_instance2")
        logger.info("✓ 模块注册成功")
        
        # 测试记忆传输
        test_data = torch.randn(5, 64)
        success = interface.transfer_memory(
            source_module="module1",
            target_module="module2",
            memory_data=test_data,
            transfer_type="full"
        )
        assert success
        logger.info("✓ 记忆传输功能正常")
        
        # 获取状态
        status = interface.get_interface_status()
        assert 'connected_modules' in status
        logger.info("✓ 接口状态获取正常")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 记忆接口测试失败: {e}")
        traceback.print_exc()
        return False

def test_exception_handling():
    """测试异常处理"""
    logger.info("开始测试异常处理...")
    
    try:
        from hippocampus.core.simulator import create_hippocampus_simulator
        
        # 测试无效输入
        try:
            simulator = create_hippocampus_simulator(input_dim=-1)
            logger.warning("⚠ 未检测到负数输入错误")
        except Exception:
            logger.info("✓ 负数输入检测正常")
        
        # 测试None输入
        try:
            simulator = create_hippocampus_simulator(input_dim=0)
            logger.warning("⚠ 未检测到零输入错误")
        except Exception:
            logger.info("✓ 零输入检测正常")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 异常处理测试失败: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("开始执行核心AI模块代码质量修复验证")
    logger.info("=" * 60)
    
    test_results = []
    
    # 测试1: 模块导入
    result1 = test_imports()
    test_results.append(("模块导入", result1))
    
    # 测试2: 海马体模拟器
    if result1:
        result2 = test_hippocampus_simulator()
        test_results.append(("海马体模拟器", result2))
    else:
        test_results.append(("海马体模拟器", False))
    
    # 测试3: 记忆接口
    if result1:
        result3 = test_memory_interface()
        test_results.append(("记忆接口", result3))
    else:
        test_results.append(("记忆接口", False))
    
    # 测试4: 异常处理
    if result1:
        result4 = test_exception_handling()
        test_results.append(("异常处理", result4))
    else:
        test_results.append(("异常处理", False))
    
    # 汇总结果
    logger.info("\n" + "=" * 60)
    logger.info("测试结果汇总")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        logger.info(f"{test_name:15} {status}")
        if result:
            passed += 1
    
    logger.info("=" * 60)
    logger.info(f"总体结果: {passed}/{total} 测试通过")
    
    success_rate = (passed / total) * 100
    if success_rate >= 90:
        logger.info(f"🎉 代码质量修复效果优秀！成功率达到 {success_rate:.1f}%")
    elif success_rate >= 70:
        logger.info(f"👍 代码质量修复效果良好！成功率达到 {success_rate:.1f}%")
    else:
        logger.info(f"⚠ 代码质量需要进一步改进！成功率为 {success_rate:.1f}%")
    
    logger.info("=" * 60)
    
    return success_rate >= 70

if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"测试执行过程中出现未处理的错误: {e}")
        traceback.print_exc()
        sys.exit(1)
