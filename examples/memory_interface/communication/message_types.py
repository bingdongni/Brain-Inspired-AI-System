"""
消息类型定义
定义海马体与新皮层通信中使用的各种消息类型
"""

import torch
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass, field
import time
import uuid


class MessageType(Enum):
    """消息类型枚举"""
    # 记忆操作消息
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_CONSOLIDATE = "memory_consolidate"
    MEMORY_COMPRESS = "memory_compress"
    
    # 搜索查询消息
    QUERY_PROCESS = "query_process"
    SEARCH_REQUEST = "search_request"
    SEARCH_RESPONSE = "search_response"
    
    # 同步消息
    SYNC_REQUEST = "sync_request"
    SYNC_RESPONSE = "sync_response"
    HEARTBEAT = "heartbeat"
    
    # 控制消息
    CONTROL_COMMAND = "control_command"
    STATUS_UPDATE = "status_update"
    ERROR_REPORT = "error_report"
    
    # 学习消息
    LEARNING_UPDATE = "learning_update"
    PATTERN_DISCOVERY = "pattern_discovery"
    KNOWLEDGE_TRANSFER = "knowledge_transfer"


class MemoryType(Enum):
    """记忆类型"""
    EPISODIC = "episodic"  # 情景记忆
    SEMANTIC = "semantic"   # 语义记忆
    PROCEDURAL = "procedural"  # 程序性记忆
    DECLARATIVE = "declarative"  # 陈述性记忆
    WORKING = "working"  # 工作记忆


class Priority(Enum):
    """消息优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class Message:
    """基础消息类"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.MEMORY_STORE
    sender: str = ""
    receiver: str = ""
    timestamp: float = field(default_factory=time.time)
    priority: Priority = Priority.NORMAL
    ttl: float = 300.0  # 生存时间（秒）
    requires_response: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """检查消息是否过期"""
        return time.time() - self.timestamp > self.ttl
    
    def get_age(self) -> float:
        """获取消息年龄"""
        return time.time() - self.timestamp


@dataclass
class MemoryMessage(Message):
    """记忆相关消息"""
    memory_type: MemoryType = MemoryType.EPISODIC
    content: Optional[torch.Tensor] = None
    importance: float = 0.5
    context: Optional[torch.Tensor] = None
    memory_id: Optional[str] = None
    query_vector: Optional[torch.Tensor] = None
    top_k: int = 10
    threshold: float = 0.5
    consolidation_ratio: Optional[float] = None
    compression_ratio: Optional[float] = None
    results: Optional[List[Dict]] = None
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class QueryMessage(Message):
    """查询相关消息"""
    query_vector: torch.Tensor = None
    search_space: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    result_format: str = "detailed"
    max_results: int = 50
    results: Optional[List[Dict]] = None
    execution_time: float = 0.0
    relevance_scores: Optional[torch.Tensor] = None


@dataclass
class ControlMessage(Message):
    """控制消息"""
    command: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    target_component: str = ""
    status: str = "pending"
    progress: float = 0.0
    result: Optional[Any] = None
    error_details: Optional[str] = None


@dataclass
class SyncMessage(Message):
    """同步消息"""
    sync_type: str = "full"  # full, partial, incremental
    data_version: str = "1.0"
    sync_data: Dict[str, Any] = field(default_factory=dict)
    sequence_number: int = 0
    checksum: Optional[str] = None
    ack_required: bool = True
    sync_status: str = "pending"


@dataclass
class LearningMessage(Message):
    """学习相关消息"""
    learning_type: str = "consolidation"  # consolidation, transfer, adaptation
    learning_data: Dict[str, Any] = field(default_factory=dict)
    pattern_results: Optional[List[Dict]] = None
    transfer_results: Optional[Dict] = None
    confidence_score: float = 0.0
    adaptation_rate: float = 0.1


class MessageBuilder:
    """消息构建器"""
    
    @staticmethod
    def create_memory_store(
        content: torch.Tensor,
        memory_type: MemoryType = MemoryType.EPISODIC,
        importance: float = 0.5,
        sender: str = "",
        receiver: str = "",
        context: Optional[torch.Tensor] = None,
        **kwargs
    ) -> MemoryMessage:
        """创建记忆存储消息"""
        message = MemoryMessage(
            message_type=MessageType.MEMORY_STORE,
            sender=sender,
            receiver=receiver,
            memory_type=memory_type,
            content=content,
            importance=importance,
            context=context,
            **kwargs
        )
        return message
    
    @staticmethod
    def create_memory_retrieve(
        query_vector: torch.Tensor,
        memory_type: Optional[MemoryType] = None,
        top_k: int = 10,
        threshold: float = 0.5,
        sender: str = "",
        receiver: str = "",
        **kwargs
    ) -> MemoryMessage:
        """创建记忆检索消息"""
        message = MemoryMessage(
            message_type=MessageType.MEMORY_RETRIEVE,
            sender=sender,
            receiver=receiver,
            query_vector=query_vector,
            top_k=top_k,
            threshold=threshold,
            memory_type=memory_type or MemoryType.EPISODIC,
            **kwargs
        )
        return message
    
    @staticmethod
    def create_query_message(
        query_vector: torch.Tensor,
        search_space: List[str] = None,
        filters: Dict[str, Any] = None,
        sender: str = "",
        receiver: str = "",
        **kwargs
    ) -> QueryMessage:
        """创建查询消息"""
        message = QueryMessage(
            message_type=MessageType.QUERY_PROCESS,
            sender=sender,
            receiver=receiver,
            query_vector=query_vector,
            search_space=search_space or [],
            filters=filters or {},
            **kwargs
        )
        return message
    
    @staticmethod
    def create_control_message(
        command: str,
        target_component: str = "",
        parameters: Dict[str, Any] = None,
        sender: str = "",
        receiver: str = "",
        **kwargs
    ) -> ControlMessage:
        """创建控制消息"""
        message = ControlMessage(
            message_type=MessageType.CONTROL_COMMAND,
            sender=sender,
            receiver=receiver,
            command=command,
            target_component=target_component,
            parameters=parameters or {},
            **kwargs
        )
        return message
    
    @staticmethod
    def create_sync_message(
        sync_type: str = "full",
        sync_data: Dict[str, Any] = None,
        sender: str = "",
        receiver: str = "",
        **kwargs
    ) -> SyncMessage:
        """创建同步消息"""
        message = SyncMessage(
            message_type=MessageType.SYNC_REQUEST,
            sender=sender,
            receiver=receiver,
            sync_type=sync_type,
            sync_data=sync_data or {},
            **kwargs
        )
        return message
    
    @staticmethod
    def create_learning_message(
        learning_type: str = "consolidation",
        learning_data: Dict[str, Any] = None,
        sender: str = "",
        receiver: str = "",
        **kwargs
    ) -> LearningMessage:
        """创建学习消息"""
        message = LearningMessage(
            message_type=MessageType.LEARNING_UPDATE,
            sender=sender,
            receiver=receiver,
            learning_type=learning_type,
            learning_data=learning_data or {},
            **kwargs
        )
        return message


class MessageValidator:
    """消息验证器"""
    
    @staticmethod
    def validate_memory_message(message: MemoryMessage) -> bool:
        """验证记忆消息"""
        if not isinstance(message.content, torch.Tensor):
            return False
        
        if message.importance < 0.0 or message.importance > 1.0:
            return False
        
        if message.content.dim() == 0:
            return False
        
        return True
    
    @staticmethod
    def validate_query_message(message: QueryMessage) -> bool:
        """验证查询消息"""
        if not isinstance(message.query_vector, torch.Tensor):
            return False
        
        if message.query_vector.dim() == 0:
            return False
        
        if message.max_results <= 0:
            return False
        
        return True
    
    @staticmethod
    def validate_control_message(message: ControlMessage) -> bool:
        """验证控制消息"""
        if not message.command:
            return False
        
        if not isinstance(message.parameters, dict):
            return False
        
        return True
    
    @staticmethod
    def validate_message(message: Message) -> bool:
        """验证基础消息"""
        if not message.sender or not message.receiver:
            return False
        
        if message.timestamp <= 0:
            return False
        
        if message.ttl <= 0:
            return False
        
        return True


class MessageSerializer:
    """消息序列化器"""
    
    @staticmethod
    def serialize_message(message: Message) -> Dict:
        """序列化消息为字典"""
        data = {
            'message_id': message.message_id,
            'message_type': message.message_type.value,
            'sender': message.sender,
            'receiver': message.receiver,
            'timestamp': message.timestamp,
            'priority': message.priority.value,
            'ttl': message.ttl,
            'requires_response': message.requires_response,
            'metadata': message.metadata
        }
        
        # 处理特定类型消息的额外字段
        if isinstance(message, MemoryMessage):
            data.update({
                'memory_type': message.memory_type.value,
                'importance': message.importance,
                'memory_id': message.memory_id,
                'top_k': message.top_k,
                'threshold': message.threshold,
                'success': message.success,
                'error_message': message.error_message
            })
            
            if message.content is not None:
                data['content'] = message.content.tolist()
            
            if message.context is not None:
                data['context'] = message.context.tolist()
            
            if message.query_vector is not None:
                data['query_vector'] = message.query_vector.tolist()
            
            if message.results is not None:
                data['results'] = message.results
        
        elif isinstance(message, QueryMessage):
            data.update({
                'search_space': message.search_space,
                'filters': message.filters,
                'result_format': message.result_format,
                'max_results': message.max_results,
                'execution_time': message.execution_time
            })
            
            if message.query_vector is not None:
                data['query_vector'] = message.query_vector.tolist()
        
        # ... 其他消息类型的序列化 ...
        
        return data
    
    @staticmethod
    def deserialize_message(data: Dict) -> Message:
        """从字典反序列化消息"""
        message_type = MessageType(data['message_type'])
        
        if message_type in [MessageType.MEMORY_STORE, MessageType.MEMORY_RETRIEVE, 
                           MessageType.MEMORY_UPDATE, MessageType.MEMORY_DELETE]:
            message = MemoryMessage(
                message_id=data['message_id'],
                message_type=message_type,
                sender=data['sender'],
                receiver=data['receiver'],
                timestamp=data['timestamp'],
                priority=Priority(data['priority']),
                ttl=data['ttl'],
                requires_response=data['requires_response'],
                metadata=data['metadata'],
                memory_type=MemoryType(data.get('memory_type', MemoryType.EPISODIC.value)),
                importance=data.get('importance', 0.5),
                memory_id=data.get('memory_id'),
                top_k=data.get('top_k', 10),
                threshold=data.get('threshold', 0.5),
                success=data.get('success', False),
                error_message=data.get('error_message')
            )
            
            # 恢复张量数据
            if 'content' in data and data['content']:
                message.content = torch.tensor(data['content'])
            if 'context' in data and data['context']:
                message.context = torch.tensor(data['context'])
            if 'query_vector' in data and data['query_vector']:
                message.query_vector = torch.tensor(data['query_vector'])
            if 'results' in data:
                message.results = data['results']
        
        else:
            # 基础消息
            message = Message(
                message_id=data['message_id'],
                message_type=message_type,
                sender=data['sender'],
                receiver=data['receiver'],
                timestamp=data['timestamp'],
                priority=Priority(data['priority']),
                ttl=data['ttl'],
                requires_response=data['requires_response'],
                metadata=data['metadata']
            )
        
        return message


# 使用示例和测试
if __name__ == "__main__":
    # 创建测试消息
    content = torch.randn(512)
    query = torch.randn(512)
    
    # 记忆存储消息
    store_msg = MessageBuilder.create_memory_store(
        content=content,
        memory_type=MemoryType.EPISODIC,
        importance=0.8,
        sender="hippocampus",
        receiver="neocortex"
    )
    
    # 记忆检索消息
    retrieve_msg = MessageBuilder.create_memory_retrieve(
        query_vector=query,
        top_k=5,
        threshold=0.6,
        sender="neocortex",
        receiver="hippocampus"
    )
    
    # 验证消息
    print(f"存储消息验证: {MessageValidator.validate_memory_message(store_msg)}")
    print(f"检索消息验证: {MessageValidator.validate_memory_message(retrieve_msg)}")
    
    # 序列化测试
    serialized = MessageSerializer.serialize_message(store_msg)
    deserialized = MessageSerializer.deserialize_message(serialized)
    
    print(f"原始消息ID: {store_msg.message_id}")
    print(f"反序列化消息ID: {deserialized.message_id}")
    print(f"消息内容匹配: {torch.allclose(store_msg.content, deserialized.content)}")