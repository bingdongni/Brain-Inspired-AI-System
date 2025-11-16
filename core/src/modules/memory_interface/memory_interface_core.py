"""
记忆接口模块
负责不同记忆模块之间的通信和数据交换
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MemoryInterfaceError(Exception):
    """记忆接口异常基类"""
    pass

class MemoryInterfaceCore:
    """记忆接口核心类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化记忆接口
        
        Args:
            config: 接口配置参数
        """
        self.config = config or {}
        self.connected_modules = {}
        self.memory_buffer = {}
        self.transfer_stats = {
            'total_transfers': 0,
            'successful_transfers': 0,
            'failed_transfers': 0
        }
        
        logger.info("记忆接口核心模块初始化完成")
    
    def register_module(self, module_name: str, module_instance: Any) -> None:
        """
        注册模块到接口
        
        Args:
            module_name: 模块名称
            module_instance: 模块实例
        """
        if not module_name or not isinstance(module_name, str):
            raise MemoryInterfaceError("模块名称必须是有效的字符串")
        
        self.connected_modules[module_name] = module_instance
        logger.info(f"模块 {module_name} 已注册到记忆接口")
    
    def transfer_memory(
        self, 
        source_module: str, 
        target_module: str, 
        memory_data: Any,
        transfer_type: str = "full"
    ) -> bool:
        """
        在模块间传输记忆数据
        
        Args:
            source_module: 源模块名称
            target_module: 目标模块名称
            memory_data: 要传输的记忆数据
            transfer_type: 传输类型 ("full", "partial", "compressed")
            
        Returns:
            bool: 传输是否成功
        """
        self.transfer_stats['total_transfers'] += 1
        
        try:
            # 验证模块是否存在
            if source_module not in self.connected_modules:
                raise MemoryInterfaceError(f"源模块 {source_module} 未注册")
            
            if target_module not in self.connected_modules:
                raise MemoryInterfaceError(f"目标模块 {target_module} 未注册")
            
            # 执行记忆传输
            if transfer_type == "full":
                transferred_data = memory_data
            elif transfer_type == "partial":
                transferred_data = self._partial_transfer(memory_data)
            elif transfer_type == "compressed":
                transferred_data = self._compress_memory(memory_data)
            else:
                raise MemoryInterfaceError(f"不支持的传输类型: {transfer_type}")
            
            # 存储到缓冲
            self.memory_buffer[f"{source_module}_{target_module}"] = transferred_data
            
            self.transfer_stats['successful_transfers'] += 1
            logger.info(f"记忆传输成功: {source_module} -> {target_module}")
            return True
            
        except Exception as e:
            self.transfer_stats['failed_transfers'] += 1
            logger.error(f"记忆传输失败: {source_module} -> {target_module}, 错误: {e}")
            raise MemoryInterfaceError(f"记忆传输失败: {str(e)}") from e
    
    def _partial_transfer(self, memory_data: Any) -> Any:
        """部分传输记忆数据"""
        if isinstance(memory_data, torch.Tensor):
            return memory_data[:min(memory_data.shape[0], 100)]  # 限制大小
        elif isinstance(memory_data, dict):
            return {k: v for i, (k, v) in enumerate(memory_data.items()) if i < 10}
        else:
            return memory_data
    
    def _compress_memory(self, memory_data: Any) -> Any:
        """压缩记忆数据"""
        if isinstance(memory_data, torch.Tensor):
            # 简单的平均值压缩
            return torch.mean(memory_data, dim=0, keepdim=True)
        else:
            return memory_data
    
    def get_interface_status(self) -> Dict[str, Any]:
        """获取接口状态信息"""
        return {
            'connected_modules': list(self.connected_modules.keys()),
            'buffer_size': len(self.memory_buffer),
            'transfer_stats': self.transfer_stats.copy(),
            'config': self.config.copy()
        }


# 便捷函数
def create_memory_interface(config: Optional[Dict[str, Any]] = None) -> MemoryInterfaceCore:
    """创建记忆接口实例"""
    return MemoryInterfaceCore(config)

def setup_basic_interfaces():
    """设置基本接口连接"""
    logger.info("设置基本记忆接口...")
    # 这里可以设置默认的模块间连接
    pass


if __name__ == "__main__":
    # 测试代码
    logger.info("记忆接口模块测试")
    
    # 创建接口
    interface = create_memory_interface()
    
    # 注册模拟模块
    interface.register_module("hippocampus", "hippocampus_module")
    interface.register_module("neocortex", "neocortex_module")
    
    # 测试记忆传输
    test_data = torch.randn(10, 128)
    try:
        success = interface.transfer_memory(
            source_module="hippocampus",
            target_module="neocortex",
            memory_data=test_data,
            transfer_type="full"
        )
        print(f"传输测试结果: {success}")
        
        # 获取状态
        status = interface.get_interface_status()
        print(f"接口状态: {status}")
        
    except Exception as e:
        print(f"测试失败: {e}")
