"""
记忆接口模块
提供不同记忆模块之间的通信和数据交换功能
"""

from .memory_interface_core import (
    MemoryInterfaceCore,
    MemoryInterfaceError,
    create_memory_interface,
    setup_basic_interfaces
)

__version__ = "1.0.0"
__author__ = "Brain-Inspired AI Research Team"

__all__ = [
    "MemoryInterfaceCore",
    "MemoryInterfaceError", 
    "create_memory_interface",
    "setup_basic_interfaces"
]

# 模块信息
MODULE_INFO = {
    "name": "记忆接口模块",
    "version": __version__,
    "description": "提供不同记忆模块间的通信和数据交换",
    "components": {
        "core_interface": "核心接口管理",
        "memory_transfer": "记忆数据传输",
        "module_registry": "模块注册管理",
        "buffer_management": "缓冲管理"
    }
}

def get_module_info():
    """获取模块信息"""
    return MODULE_INFO

# 使用示例
def example_usage():
    """使用示例"""
    
    print("=== 记忆接口模块使用示例 ===")
    
    # 1. 创建接口
    interface = create_memory_interface()
    print("✓ 接口创建成功")
    
    # 2. 注册模块
    interface.register_module("hippocampus", "hippocampus_instance")
    interface.register_module("neocortex", "neocortex_instance")
    print("✓ 模块注册完成")
    
    # 3. 模拟记忆数据传输
    import torch
    test_memory = torch.randn(5, 64)
    
    try:
        success = interface.transfer_memory(
            source_module="hippocampus",
            target_module="neocortex", 
            memory_data=test_memory,
            transfer_type="full"
        )
        print("✓ 记忆传输成功")
        
    except Exception as e:
        print(f"⚠ 传输过程中出现错误: {e}")
    
    # 4. 获取接口状态
    status = interface.get_interface_status()
    print(f"✓ 接口状态: {status['connected_modules']} 模块已连接")
    print(f"✓ 传输统计: {status['transfer_stats']}")
    
    return interface

if __name__ == "__main__":
    example_usage()
