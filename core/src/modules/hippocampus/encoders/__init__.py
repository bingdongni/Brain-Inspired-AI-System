"""
记忆编码器模块

实现基于Transformer的海马体记忆编码机制
"""

from .transformer_encoder import (
    TransformerMemoryEncoder,
    EpisodicMemoryEncoder,
    MultiSynapticEngram,
    PositionalEncoding,
    AttentionMechanism,
    create_memory_encoder
)

__all__ = [
    "TransformerMemoryEncoder",
    "EpisodicMemoryEncoder", 
    "MultiSynapticEngram",
    "PositionalEncoding",
    "AttentionMechanism",
    "create_memory_encoder"
]