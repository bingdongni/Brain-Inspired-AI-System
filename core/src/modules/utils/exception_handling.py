"""
脑启发AI模块异常处理基类
提供统一的异常处理机制
"""

import logging
from typing import Optional, Dict, Any

# 设置日志记录器
logger = logging.getLogger(__name__)

class BrainAIError(Exception):
    """脑启发AI模块异常基类"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.details = details or {}
        super().__init__(self.message)
        
        # 记录错误日志
        logger.error(f"错误代码: {self.error_code}, 消息: {self.message}, 详情: {self.details}")

class MemoryEncodingError(BrainAIError):
    """记忆编码异常"""
    
    def __init__(self, message: str, memory_type: Optional[str] = None, **kwargs):
        super().__init__(message, "MEMORY_ENCODING_ERROR", kwargs)
        self.memory_type = memory_type

class RoutingError(BrainAIError):
    """路由异常"""
    
    def __init__(self, message: str, route_info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, "ROUTING_ERROR", kwargs)
        self.route_info = route_info or {}

class ModelConfigurationError(BrainAIError):
    """模型配置异常"""
    
    def __init__(self, message: str, config_param: Optional[str] = None, **kwargs):
        super().__init__(message, "CONFIG_ERROR", kwargs)
        self.config_param = config_param

class InputValidationError(BrainAIError):
    """输入验证异常"""
    
    def __init__(self, message: str, input_type: Optional[str] = None, **kwargs):
        super().__init__(message, "INPUT_VALIDATION_ERROR", kwargs)
        self.input_type = input_type

def validate_input_tensor(tensor: Any, tensor_name: str = "input", allow_none: bool = False) -> None:
    """验证输入张量"""
    
    if tensor is None and not allow_none:
        raise InputValidationError(f"{tensor_name} cannot be None", input_type="None")
    
    if tensor is not None and not isinstance(tensor, __import__('torch').Tensor):
        raise InputValidationError(
            f"{tensor_name} must be torch.Tensor, got {type(tensor)}",
            input_type=type(tensor).__name__
        )

def validate_config(config: Dict[str, Any], required_keys: list, config_name: str = "config") -> None:
    """验证配置字典"""
    
    if not isinstance(config, dict):
        raise InputValidationError(f"{config_name} must be a dictionary", input_type=type(config).__name__)
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ModelConfigurationError(
            f"{config_name} missing required keys: {missing_keys}",
            config_param=", ".join(missing_keys)
        )

def safe_execute(func, *args, fallback_return: Any = None, **kwargs):
    """安全执行函数，提供错误处理和回退机制"""
    
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"函数 {func.__name__} 执行失败: {e}")
        if fallback_return is not None:
            logger.warning(f"返回默认回退值: {fallback_return}")
            return fallback_return
        raise

# 装饰器：自动添加异常处理
def brain_ai_exception_handler(func):
    """异常处理装饰器"""
    
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except BrainAIError:
            # 重新抛出已有的BrainAIError
            raise
        except Exception as e:
            # 包装其他异常为BrainAIError
            raise BrainAIError(
                f"函数 {func.__name__} 执行时发生未知错误: {str(e)}",
                details={"original_error": str(e), "args": str(args), "kwargs": str(kwargs)}
            ) from e
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

# 装饰器：性能监控
def performance_monitor(func):
    """性能监控装饰器"""
    
    import time
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.debug(f"{func.__name__} 执行时间: {end_time - start_time:.3f}秒")
            return result
        except Exception as e:
            end_time = time.time()
            logger.error(f"{func.__name__} 执行失败，耗时: {end_time - start_time:.3f}秒，错误: {e}")
            raise
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper
