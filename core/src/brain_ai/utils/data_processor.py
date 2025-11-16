#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据处理工具
===========

提供统一的数据处理功能，包括:
- 数据加载和预处理
- 数据增强
- 数据分割
- 数据验证
- 批处理
"""

import os
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd

class BaseDataProcessor(ABC):
    """数据处理器基类"""
    
    @abstractmethod
    def process(self, data: Any) -> Any:
        """处理数据"""
        pass
    
    @abstractmethod
    def inverse_transform(self, data: Any) -> Any:
        """逆变换"""
        pass

class ImageDataProcessor(BaseDataProcessor):
    """图像数据处理器"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True, 
                 augment: bool = False):
        """
        初始化图像数据处理器
        
        Args:
            target_size: 目标尺寸
            normalize: 是否归一化
            augment: 是否数据增强
        """
        self.target_size = target_size
        self.normalize = normalize
        self.augment = augment
        self.scaler = None
        
    def process(self, images: List[np.ndarray]) -> np.ndarray:
        """处理图像数据"""
        processed_images = []
        
        for image in images:
            # 调整尺寸
            if image.shape[:2] != self.target_size:
                image = self._resize_image(image)
            
            # 数据增强
            if self.augment:
                image = self._augment_image(image)
            
            # 归一化
            if self.normalize:
                image = image.astype(np.float32) / 255.0
            
            processed_images.append(image)
        
        return np.array(processed_images)
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """调整图像尺寸"""
        from PIL import Image
        pil_image = Image.fromarray(image)
        resized = pil_image.resize(self.target_size, Image.Resampling.LANCZOS)
        return np.array(resized)
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """数据增强（简单实现）"""
        # 这里可以实现更复杂的增强策略
        # 暂时只实现基本的随机翻转
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        return image
    
    def inverse_transform(self, images: np.ndarray) -> np.ndarray:
        """逆归一化"""
        if self.normalize:
            images = (images * 255).astype(np.uint8)
        return images

class TextDataProcessor(BaseDataProcessor):
    """文本数据处理器"""
    
    def __init__(self, max_length: int = 128, 
                 vocab_size: int = 10000,
                 oov_token: str = "<OOV>"):
        """
        初始化文本数据处理器
        
        Args:
            max_length: 最大长度
            vocab_size: 词汇表大小
            oov_token: 未知词标记
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.oov_token = oov_token
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_built = False
        
    def build_vocabulary(self, texts: List[str]):
        """构建词汇表"""
        word_counts = {}
        
        # 统计词频
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # 按频率排序并构建词汇表
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 添加特殊标记
        special_tokens = [self.oov_token, "<PAD>", "<START>", "<END>"]
        
        self.word_to_idx = {token: idx for idx, token in enumerate(special_tokens)}
        self.idx_to_word = {idx: token for token, idx in self.word_to_idx.items()}
        
        # 添加常用词
        for word, _ in sorted_words[:self.vocab_size - len(special_tokens)]:
            if word not in self.word_to_idx:
                idx = len(self.word_to_idx)
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
        
        self.vocab_built = True
        print(f"词汇表构建完成，大小: {len(self.word_to_idx)}")
    
    def process(self, texts: List[str]) -> np.ndarray:
        """处理文本数据"""
        if not self.vocab_built:
            raise ValueError("请先构建词汇表")
        
        processed_texts = []
        
        for text in texts:
            words = text.lower().split()[:self.max_length]
            
            # 转换为索引
            indices = []
            for word in words:
                idx = self.word_to_idx.get(word, self.word_to_idx[self.oov_token])
                indices.append(idx)
            
            # 填充到固定长度
            if len(indices) < self.max_length:
                pad_idx = self.word_to_idx["<PAD>"]
                indices.extend([pad_idx] * (self.max_length - len(indices)))
            
            processed_texts.append(indices)
        
        return np.array(processed_texts)
    
    def inverse_transform(self, indices: np.ndarray) -> List[str]:
        """索引转换为文本"""
        texts = []
        
        for seq in indices:
            words = []
            for idx in seq:
                if idx != self.word_to_idx["<PAD>"]:
                    word = self.idx_to_word.get(idx, self.oov_token)
                    words.append(word)
            texts.append(" ".join(words))
        
        return texts

class TabularDataProcessor(BaseDataProcessor):
    """表格数据处理器"""
    
    def __init__(self, target_column: Optional[str] = None,
                 feature_columns: Optional[List[str]] = None,
                 scaler_type: str = 'standard'):
        """
        初始化表格数据处理器
        
        Args:
            target_column: 目标列名
            feature_columns: 特征列名
            scaler_type: 缩放器类型 ('standard', 'minmax', 'none')
        """
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.scaler_type = scaler_type
        self.scaler = None
        self.label_encoder = None
        self.column_names = []
        
    def process(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """处理表格数据"""
        if isinstance(data, pd.DataFrame):
            return self._process_dataframe(data)
        else:
            return self._process_array(data)
    
    def _process_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """处理DataFrame"""
        # 确定特征列
        if self.feature_columns:
            features = df[self.feature_columns].copy()
        else:
            features = df.drop(columns=[self.target_column] if self.target_column else [])
        
        self.column_names = features.columns.tolist()
        
        # 处理数值特征
        numeric_features = features.select_dtypes(include=[np.number])
        
        # 缩放
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            numeric_features = self.scaler.fit_transform(numeric_features)
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            numeric_features = self.scaler.fit_transform(numeric_features)
        
        # 处理分类特征
        categorical_features = features.select_dtypes(include=['object'])
        encoded_features = []
        
        for col in categorical_features.columns:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                encoded_col = self.label_encoder.fit_transform(categorical_features[col].astype(str))
            else:
                encoded_col = self.label_encoder.transform(categorical_features[col].astype(str))
            encoded_features.append(encoded_col.reshape(-1, 1))
        
        # 合并特征
        all_features = [numeric_features] + encoded_features
        X = np.hstack(all_features) if all_features else numeric_features
        
        # 处理目标变量
        if self.target_column and self.target_column in df.columns:
            y = df[self.target_column].values
            
            # 如果是分类标签，编码
            if y.dtype == 'object' or not np.issubdtype(y.dtype, np.number):
                if self.label_encoder is None:
                    self.label_encoder = LabelEncoder()
                    y = self.label_encoder.fit_transform(y.astype(str))
        else:
            y = np.zeros(len(df))
        
        return X, y
    
    def _process_array(self, array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """处理numpy数组"""
        # 简单处理：假设最后一列是目标
        X = array[:, :-1]
        y = array[:, -1]
        
        # 缩放
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            X = self.scaler.fit_transform(X)
        
        return X, y
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """逆变换"""
        if self.scaler:
            return self.scaler.inverse_transform(X)
        return X

class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_path: str, processor: BaseDataProcessor,
                 batch_size: int = 32, shuffle: bool = True):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据路径
            processor: 数据处理器
            batch_size: 批次大小
            shuffle: 是否打乱
        """
        self.data_path = Path(data_path)
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = None
        self.labels = None
        
    def load_data(self) -> bool:
        """加载数据"""
        try:
            if self.data_path.suffix == '.csv':
                df = pd.read_csv(self.data_path)
                self.data, self.labels = self.processor.process(df)
            elif self.data_path.suffix == '.npy':
                data_array = np.load(self.data_path)
                self.data, self.labels = self.processor.process(data_array)
            elif self.data_path.suffix == '.pkl':
                with open(self.data_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                if isinstance(loaded_data, dict):
                    self.data = loaded_data['data']
                    self.labels = loaded_data.get('labels')
                else:
                    self.data, self.labels = self.processor.process(loaded_data)
            else:
                raise ValueError(f"不支持的文件格式: {self.data_path.suffix}")
            
            print(f"数据加载成功，形状: {self.data.shape}")
            return True
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return False
    
    def get_batches(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """获取批次数据"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(indices)
        
        batches = []
        for i in range(0, len(self.data), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_data = self.data[batch_indices]
            batch_labels = self.labels[batch_indices] if self.labels is not None else None
            
            batches.append((batch_data, batch_labels))
        
        return batches
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, 
                   random_state: int = 42) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """分割数据"""
        if self.data is None:
            raise ValueError("请先加载数据")
        
        indices = np.arange(len(self.data))
        
        # 训练集和测试集
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        
        # 训练集和验证集
        train_indices, val_indices = train_test_split(
            train_indices, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        splits = {}
        for name, idxs in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
            splits[name] = (self.data[idxs], self.labels[idxs] if self.labels is not None else None)
        
        return splits

def create_torch_dataset(data: np.ndarray, labels: Optional[np.ndarray] = None, 
                        transform: Optional[Callable] = None) -> Dataset:
    """创建PyTorch数据集"""
    if labels is not None:
        dataset = TensorDataset(
            torch.FloatTensor(data), 
            torch.LongTensor(labels) if labels.dtype in [np.int32, np.int64] else torch.FloatTensor(labels)
        )
    else:
        dataset = TensorDataset(torch.FloatTensor(data))
    
    if transform:
        # 自定义transform应用
        class TransformDataset(Dataset):
            def __init__(self, base_dataset, transform):
                self.base_dataset = base_dataset
                self.transform = transform
            
            def __len__(self):
                return len(self.base_dataset)
            
            def __getitem__(self, idx):
                item = self.base_dataset[idx]
                if self.transform:
                    item = self.transform(item)
                return item
        
        dataset = TransformDataset(dataset, transform)
    
    return dataset

def save_data(data: Any, filepath: str, format: str = 'pickle'):
    """保存数据"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    elif format == 'numpy':
        np.save(filepath, data)
    elif format == 'csv' and isinstance(data, pd.DataFrame):
        data.to_csv(filepath, index=False)
    else:
        raise ValueError(f"不支持的格式: {format}")

def load_data(filepath: str, format: str = 'auto'):
    """加载数据"""
    filepath = Path(filepath)
    
    if format == 'auto':
        if filepath.suffix == '.csv':
            format = 'csv'
        elif filepath.suffix == '.npy':
            format = 'numpy'
        elif filepath.suffix == '.pkl':
            format = 'pickle'
        else:
            format = 'pickle'
    
    if format == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    elif format == 'numpy':
        return np.load(filepath)
    elif format == 'csv':
        return pd.read_csv(filepath)
    else:
        raise ValueError(f"不支持的格式: {format}")

class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_image_data(images: np.ndarray) -> Dict[str, Any]:
        """验证图像数据"""
        report = {
            'shape': images.shape,
            'dtype': images.dtype,
            'min_value': float(np.min(images)),
            'max_value': float(np.max(images)),
            'mean': float(np.mean(images)),
            'std': float(np.std(images)),
            'has_nan': bool(np.isnan(images).any()),
            'has_inf': bool(np.isinf(images).any())
        }
        
        # 检查异常值
        if report['has_nan']:
            report['warnings'] = report.get('warnings', []) + ['数据包含NaN值']
        if report['has_inf']:
            report['warnings'] = report.get('warnings', []) + ['数据包含无穷值']
        
        return report
    
    @staticmethod
    def validate_text_data(texts: List[str]) -> Dict[str, Any]:
        """验证文本数据"""
        lengths = [len(text.split()) for text in texts]
        
        report = {
            'num_samples': len(texts),
            'avg_length': float(np.mean(lengths)),
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'unique_texts': len(set(texts)),
            'empty_texts': sum(1 for text in texts if not text.strip())
        }
        
        return report
    
    @staticmethod
    def validate_tabular_data(data: Union[np.ndarray, pd.DataFrame]) -> Dict[str, Any]:
        """验证表格数据"""
        if isinstance(data, pd.DataFrame):
            array = data.values
            columns = data.columns.tolist()
        else:
            array = data
            columns = None
        
        report = {
            'shape': array.shape,
            'dtype': array.dtype,
            'missing_values': int(np.isnan(array).sum()),
            'numeric_columns': int(array.shape[1]) if np.issubdtype(array.dtype, np.number) else 0
        }
        
        if columns:
            report['columns'] = columns
        
        return report