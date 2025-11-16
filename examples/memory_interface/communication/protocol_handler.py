"""
通信协议处理器
实现海马体与新皮层之间的通信协议，处理消息的发送、接收和路由
"""

import asyncio
import threading
import time
import queue
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import hashlib

from .message_types import (
    Message, MemoryMessage, QueryMessage, ControlMessage, 
    SyncMessage, LearningMessage, MessageType, Priority,
    MessageBuilder, MessageValidator, MessageSerializer
)


@dataclass
class NetworkConfig:
    """网络配置"""
    max_message_size: int = 1024 * 1024  # 1MB
    max_queue_size: int = 10000
    message_timeout: float = 30.0
    retry_attempts: int = 3
    heartbeat_interval: float = 10.0
    compression_enabled: bool = True
    encryption_enabled: bool = False


@dataclass
class ConnectionState:
    """连接状态"""
    is_connected: bool = False
    last_heartbeat: float = 0.0
    message_count: int = 0
    error_count: int = 0
    latency_avg: float = 0.0
    bandwidth_usage: float = 0.0


class CommunicationProtocol:
    """
    通信协议处理器
    管理海马体与新皮层之间的通信
    """
    
    def __init__(self, config: NetworkConfig = None):
        self.config = config or NetworkConfig()
        
        # 消息队列
        self.outgoing_queue = queue.PriorityQueue(maxsize=self.config.max_queue_size)
        self.incoming_queue = queue.PriorityQueue(maxsize=self.config.max_queue_size)
        
        # 消息处理器
        self.message_handlers = {
            MessageType.MEMORY_STORE: self._handle_memory_store,
            MessageType.MEMORY_RETRIEVE: self._handle_memory_retrieve,
            MessageType.MEMORY_UPDATE: self._handle_memory_update,
            MessageType.MEMORY_DELETE: self._handle_memory_delete,
            MessageType.QUERY_PROCESS: self._handle_query_process,
            MessageType.CONTROL_COMMAND: self._handle_control_command,
            MessageType.SYNC_REQUEST: self._handle_sync_request,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.STATUS_UPDATE: self._handle_status_update,
            MessageType.ERROR_REPORT: self._handle_error_report,
            MessageType.LEARNING_UPDATE: self._handle_learning_update
        }
        
        # 连接管理
        self.connections = {
            'hippocampus': ConnectionState(),
            'neocortex': ConnectionState()
        }
        
        # 路由表
        self.routing_table = {
            'hippocampus': ['neocortex'],
            'neocortex': ['hippocortex']  # 注意：这里假设有多个海马体
        }
        
        # 性能统计
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'messages_failed': 0,
            'total_latency': 0.0,
            'average_latency': 0.0,
            'bandwidth_usage': 0.0,
            'queue_overflows': 0
        }
        
        # 回调函数
        self.callbacks = {
            'on_message_sent': [],
            'on_message_received': [],
            'on_connection_change': [],
            'on_error': []
        }
        
        # 运行状态
        self.is_running = False
        self.sender_thread = None
        self.receiver_thread = None
        
        # 消息缓存
        self.message_cache = {}
        self.pending_responses = {}
        
    async def start(self):
        """启动通信协议"""
        self.is_running = True
        
        # 启动发送和接收线程
        self.sender_thread = threading.Thread(target=self._message_sender, daemon=True)
        self.receiver_thread = threading.Thread(target=self._message_receiver, daemon=True)
        
        self.sender_thread.start()
        self.receiver_thread.start()
        
        # 启动心跳
        asyncio.create_task(self._heartbeat_sender())
        
        print("通信协议已启动")
    
    async def stop(self):
        """停止通信协议"""
        self.is_running = False
        
        # 等待线程结束
        if self.sender_thread:
            self.sender_thread.join(timeout=5)
        if self.receiver_thread:
            self.receiver_thread.join(timeout=5)
        
        print("通信协议已停止")
    
    def send_message(self, message: Message) -> bool:
        """
        发送消息
        
        Args:
            message: 要发送的消息
            
        Returns:
            是否成功加入队列
        """
        try:
            # 验证消息
            if not MessageValidator.validate_message(message):
                self._trigger_error(f"消息验证失败: {message.message_id}")
                return False
            
            # 检查队列容量
            if self.outgoing_queue.full():
                self.stats['queue_overflows'] += 1
                self._trigger_error("发送队列已满")
                return False
            
            # 添加到队列（优先级：URGENT=0, HIGH=1, NORMAL=2, LOW=3）
            priority_value = message.priority.value - 1
            self.outgoing_queue.put((priority_value, time.time(), message))
            
            # 记录发送统计
            self.stats['messages_sent'] += 1
            
            self._trigger_callback('on_message_sent', message)
            
            return True
            
        except Exception as e:
            self._trigger_error(f"发送消息失败: {e}")
            return False
    
    def receive_message(self, timeout: float = None) -> Optional[Message]:
        """
        接收消息
        
        Args:
            timeout: 超时时间
            
        Returns:
            接收到的消息或None
        """
        try:
            if timeout is None:
                _, _, message = self.incoming_queue.get()
            else:
                _, _, message = self.incoming_queue.get(timeout=timeout)
            
            self.stats['messages_received'] += 1
            self._trigger_callback('on_message_received', message)
            
            return message
            
        except queue.Empty:
            return None
        except Exception as e:
            self._trigger_error(f"接收消息失败: {e}")
            return None
    
    def send_memory_store(self, content, memory_type=None, importance=0.5, **kwargs) -> bool:
        """发送记忆存储消息"""
        message = MessageBuilder.create_memory_store(
            content=content,
            memory_type=memory_type,
            importance=importance,
            **kwargs
        )
        return self.send_message(message)
    
    def send_memory_retrieve(self, query_vector, memory_type=None, **kwargs) -> bool:
        """发送记忆检索消息"""
        message = MessageBuilder.create_memory_retrieve(
            query_vector=query_vector,
            memory_type=memory_type,
            **kwargs
        )
        return self.send_message(message)
    
    def send_query(self, query_vector, **kwargs) -> bool:
        """发送查询消息"""
        message = MessageBuilder.create_query_message(
            query_vector=query_vector,
            **kwargs
        )
        return self.send_message(message)
    
    def send_control_command(self, command, target_component="", **kwargs) -> bool:
        """发送控制命令"""
        message = MessageBuilder.create_control_message(
            command=command,
            target_component=target_component,
            **kwargs
        )
        return self.send_message(message)
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """注册消息处理器"""
        self.message_handlers[message_type] = handler
    
    def register_callback(self, event_type: str, callback: Callable):
        """注册回调函数"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def get_connection_status(self) -> Dict[str, ConnectionState]:
        """获取连接状态"""
        return self.connections.copy()
    
    def get_statistics(self) -> Dict:
        """获取通信统计信息"""
        return self.stats.copy()
    
    def _message_sender(self):
        """消息发送线程"""
        while self.is_running:
            try:
                # 从队列获取消息
                priority, enqueue_time, message = self.outgoing_queue.get(timeout=1.0)
                
                # 检查消息是否过期
                if message.is_expired():
                    continue
                
                # 检查目标连接
                if not self._check_connection(message.receiver):
                    # 记录错误但不丢弃消息
                    self.connections[message.receiver].error_count += 1
                
                # 序列化消息
                serialized_message = MessageSerializer.serialize_message(message)
                
                # 模拟网络传输延迟
                latency = self._simulate_network_latency()
                time.sleep(latency)
                
                # 发送到目标
                self._transmit_message(serialized_message, message.receiver)
                
                # 更新统计
                current_time = time.time()
                message_latency = current_time - enqueue_time
                self.stats['total_latency'] += message_latency
                self.stats['average_latency'] = (
                    self.stats['total_latency'] / self.stats['messages_sent']
                )
                
                # 触发发送回调
                self._trigger_callback('on_message_sent', message)
                
                self.connections[message.receiver].message_count += 1
                
            except queue.Empty:
                continue
            except Exception as e:
                self._trigger_error(f"发送线程错误: {e}")
                self.stats['messages_failed'] += 1
    
    def _message_receiver(self):
        """消息接收线程"""
        while self.is_running:
            try:
                # 模拟接收消息（实际应用中这里应该是网络接收）
                received_message = self._simulate_message_reception()
                
                if received_message:
                    # 反序列化消息
                    message = MessageSerializer.deserialize_message(received_message)
                    
                    # 验证消息
                    if not MessageValidator.validate_message(message):
                        continue
                    
                    # 处理消息
                    response_message = self._process_message(message)
                    
                    # 如果需要响应，将响应消息加入发送队列
                    if response_message and message.requires_response:
                        self.send_message(response_message)
                    
                    # 将消息放入接收队列
                    self.incoming_queue.put((message.priority.value - 1, time.time(), message))
                
                time.sleep(0.001)  # 防止CPU占用过高
                
            except Exception as e:
                self._trigger_error(f"接收线程错误: {e}")
    
    def _process_message(self, message: Message) -> Optional[Message]:
        """处理接收到的消息"""
        try:
            # 获取处理器
            handler = self.message_handlers.get(message.message_type)
            
            if handler:
                # 调用处理器
                response = handler(message)
                return response
            else:
                # 没有处理器，发送错误响应
                return self._create_error_response(message, f"未找到消息处理器: {message.message_type}")
                
        except Exception as e:
            self._trigger_error(f"消息处理错误: {e}")
            return self._create_error_response(message, str(e))
    
    def _handle_memory_store(self, message: MemoryMessage) -> Optional[Message]:
        """处理记忆存储消息"""
        # 验证消息
        if not MessageValidator.validate_memory_message(message):
            return self._create_error_response(message, "记忆存储消息验证失败")
        
        # 这里应该调用实际的记忆存储逻辑
        # 暂时返回成功响应
        response = MessageBuilder.create_control_message(
            command="memory_store_response",
            parameters={
                "success": True,
                "memory_id": f"mem_{int(time.time())}",
                "timestamp": time.time()
            },
            sender=message.receiver,
            receiver=message.sender
        )
        
        return response
    
    def _handle_memory_retrieve(self, message: MemoryMessage) -> Optional[Message]:
        """处理记忆检索消息"""
        # 验证消息
        if not MessageValidator.validate_memory_message(message):
            return self._create_error_response(message, "记忆检索消息验证失败")
        
        # 实际应用中这里会执行检索逻辑
        # 暂时返回空结果
        response = MessageBuilder.create_memory_store(
            content=torch.zeros(512),  # 空的检索结果
            memory_type=message.memory_type,
            importance=0.0,
            sender=message.receiver,
            receiver=message.sender,
            results=[],
            success=True
        )
        
        return response
    
    def _handle_memory_update(self, message: MemoryMessage) -> Optional[Message]:
        """处理记忆更新消息"""
        return self._create_success_response(message, "memory_update_success")
    
    def _handle_memory_delete(self, message: MemoryMessage) -> Optional[Message]:
        """处理记忆删除消息"""
        return self._create_success_response(message, "memory_delete_success")
    
    def _handle_query_process(self, message: QueryMessage) -> Optional[Message]:
        """处理查询消息"""
        if not MessageValidator.validate_query_message(message):
            return self._create_error_response(message, "查询消息验证失败")
        
        response = MessageBuilder.create_query_message(
            query_vector=message.query_vector,
            sender=message.receiver,
            receiver=message.sender,
            results=[],
            execution_time=0.1,
            max_results=0
        )
        
        return response
    
    def _handle_control_command(self, message: ControlMessage) -> Optional[Message]:
        """处理控制命令"""
        response = MessageBuilder.create_control_message(
            command=f"{message.command}_response",
            parameters={"status": "executed", "timestamp": time.time()},
            sender=message.receiver,
            receiver=message.sender
        )
        
        return response
    
    def _handle_sync_request(self, message: SyncMessage) -> Optional[Message]:
        """处理同步请求"""
        response = MessageBuilder.create_sync_message(
            sync_data={"status": "synced"},
            sender=message.receiver,
            receiver=message.sender,
            sync_status="completed"
        )
        
        return response
    
    def _handle_heartbeat(self, message: Message) -> Optional[Message]:
        """处理心跳消息"""
        # 更新心跳时间
        if message.sender in self.connections:
            self.connections[message.sender].last_heartbeat = time.time()
            self.connections[message.sender].is_connected = True
        
        # 发送心跳响应
        response = Message(
            message_type=MessageType.HEARTBEAT,
            sender=message.receiver,
            receiver=message.sender,
            requires_response=False
        )
        
        return response
    
    def _handle_status_update(self, message: Message) -> Optional[Message]:
        """处理状态更新消息"""
        # 记录状态信息
        self.connections[message.sender].last_heartbeat = time.time()
        
        return None  # 不需要响应
    
    def _handle_error_report(self, message: Message) -> Optional[Message]:
        """处理错误报告"""
        self._trigger_error(f"接收到错误报告: {message.metadata}")
        return None
    
    def _handle_learning_update(self, message: LearningMessage) -> Optional[Message]:
        """处理学习更新消息"""
        response = MessageBuilder.create_learning_message(
            learning_data={"status": "processed"},
            sender=message.receiver,
            receiver=message.sender,
            confidence_score=0.8
        )
        
        return response
    
    async def _heartbeat_sender(self):
        """心跳发送任务"""
        while self.is_running:
            try:
                for component in self.connections.keys():
                    heartbeat_msg = Message(
                        message_type=MessageType.HEARTBEAT,
                        sender="protocol",
                        receiver=component,
                        requires_response=False
                    )
                    
                    self.send_message(heartbeat_msg)
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                self._trigger_error(f"心跳发送错误: {e}")
                await asyncio.sleep(1)
    
    def _check_connection(self, receiver: str) -> bool:
        """检查连接状态"""
        if receiver not in self.connections:
            return False
        
        connection = self.connections[receiver]
        
        # 检查最后心跳时间
        time_since_heartbeat = time.time() - connection.last_heartbeat
        if time_since_heartbeat > self.config.heartbeat_interval * 3:
            connection.is_connected = False
            return False
        
        return connection.is_connected
    
    def _simulate_network_latency(self) -> float:
        """模拟网络延迟"""
        import random
        return random.uniform(0.001, 0.01)  # 1-10ms延迟
    
    def _transmit_message(self, serialized_message: Dict, receiver: str):
        """传输消息到目标组件"""
        # 在实际应用中，这里会实现真正的网络传输
        # 目前只是模拟，将消息添加到接收队列
        message = MessageSerializer.deserialize_message(serialized_message)
        
        # 检查接收队列容量
        if not self.incoming_queue.full():
            self.incoming_queue.put((message.priority.value - 1, time.time(), message))
        else:
            self.stats['queue_overflows'] += 1
            self._trigger_error("接收队列已满，丢弃消息")
    
    def _simulate_message_reception(self) -> Optional[Dict]:
        """模拟消息接收"""
        # 在实际应用中，这里会从网络接收消息
        # 目前返回None，实际使用时需要实现真正的网络接收
        return None
    
    def _create_success_response(self, original_message: Message, status: str) -> Message:
        """创建成功响应"""
        response = Message(
            message_type=MessageType.STATUS_UPDATE,
            sender=original_message.receiver,
            receiver=original_message.sender,
            requires_response=False,
            metadata={"status": status, "original_message_id": original_message.message_id}
        )
        return response
    
    def _create_error_response(self, original_message: Message, error_msg: str) -> Message:
        """创建错误响应"""
        response = Message(
            message_type=MessageType.ERROR_REPORT,
            sender=original_message.receiver,
            receiver=original_message.sender,
            requires_response=False,
            metadata={
                "error": error_msg,
                "original_message_id": original_message.message_id,
                "timestamp": time.time()
            }
        )
        return response
    
    def _trigger_callback(self, event_type: str, *args):
        """触发回调函数"""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(*args)
            except Exception as e:
                print(f"回调函数错误 ({event_type}): {e}")
    
    def _trigger_error(self, error_msg: str):
        """触发错误回调"""
        self._trigger_callback('on_error', error_msg)
        self.stats['messages_failed'] += 1


if __name__ == "__main__":
    import torch
    
    # 创建协议实例
    protocol = CommunicationProtocol()
    
    # 启动协议
    asyncio.run(protocol.start())
    
    # 发送测试消息
    content = torch.randn(512)
    success = protocol.send_memory_store(
        content=content,
        memory_type="episodic",
        importance=0.8,
        sender="hippocampus",
        receiver="neocortex"
    )
    
    print(f"消息发送成功: {success}")
    
    # 接收消息
    received_msg = protocol.receive_message(timeout=5.0)
    if received_msg:
        print(f"接收到消息: {received_msg.message_type}")
    
    # 获取统计信息
    stats = protocol.get_statistics()
    print(f"通信统计: {stats}")
    
    # 停止协议
    asyncio.run(protocol.stop())