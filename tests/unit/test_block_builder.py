"""KVBlock 单元测试."""

import unittest
import redis
import numpy as np
from typing import Tuple
import sys


class KVBlockBuilder:
    """KVBlock 类，用于处理 KV 缓存块操作."""
    
    client = None  # 类变量，所有实例共享同一个连接

    @classmethod
    def init_connection(cls, host='localhost', port=9221):
        """初始化 Redis 连接"""
        if cls.client is None:
            try:
                cls.client = redis.Redis(host=host, port=port, decode_responses=False)
                cls.client.ping()
                print(f"已成功连接到 PikiwiDB: {host}:{port}")
            except redis.ConnectionError:
                print(f"无法连接到 PikiwiDB: {host}:{port}")
                sys.exit(1)

    def __init__(self):
        """初始化 KVBlockBuilder 实例"""
        if self.client is None:
            self.init_connection()

    def build_key(self, ns: str, layer_id: int, block_id: int, 
                  k_type: int) -> str:
        """构建 KV 缓存块的 Redis 键"""
        return f"kvblock:{ns}:{layer_id}:{block_id}:{k_type}"

    def validate_params(self, ns: str, layer_id: int, block_id: int,
                       k_type: int, dtype: int, block_size: int) -> None:
        """验证块参数"""
        if not ns:
            raise ValueError("命名空间不能为空")
        if block_size <= 0:
            raise ValueError("无效的块大小")
        if dtype not in [1, 2]:  # FP16 或 FP32
            raise ValueError("无效的数据类型")
        if k_type not in [0, 1]:  # K 或 V
            raise ValueError("无效的块类型")

    def serialize_tensor(self, tensor: np.ndarray) -> bytes:
        """将 numpy 张量序列化为字节"""
        return tensor.tobytes()

    def deserialize_tensor(self, data: bytes, block_size: int, 
                         dtype: int) -> np.ndarray:
        """将字节反序列化为 numpy 张量"""
        np_dtype = np.float16 if dtype == 1 else np.float32
        return np.frombuffer(data, dtype=np_dtype).reshape(-1, block_size)

    def validate_tensor_shape(self, tensor: np.ndarray, 
                            block_size: int) -> None:
        """验证张量形状"""
        if tensor.shape[1] != block_size:
            raise ValueError(f"无效的张量形状: {tensor.shape}, "
                           f"期望块大小: {block_size}")


class TestKVBlockBuilder(unittest.TestCase):
    """KVBlockBuilder 类的测试用例"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化，确保 Redis 连接"""
        KVBlockBuilder.init_connection()

    def setUp(self):
        """设置测试环境"""
        self.builder = KVBlockBuilder()
        self.test_params = {
            'ns': 'test_ns',
            'layer_id': 0,
            'block_id': 1,
            'k_type': 0,
            'dtype': 1,  # FP16
            'block_size': 128
        }

    def create_test_tensor(self) -> Tuple[np.ndarray, bytes]:
        """创建测试张量及其二进制表示"""
        data = np.random.randn(
            1,  # 单个块
            self.test_params['block_size']
        ).astype(np.float16)
        return data, data.tobytes()

    def test_build_key(self):
        """测试键构建功能"""
        expected = "kvblock:test_ns:0:1:0"
        key = self.builder.build_key(
            self.test_params['ns'],
            self.test_params['layer_id'],
            self.test_params['block_id'],
            self.test_params['k_type']
        )
        self.assertEqual(key, expected)

    def test_validate_params(self):
        """测试参数验证功能"""
        # 测试有效参数
        try:
            self.builder.validate_params(**self.test_params)
        except ValueError:
            self.fail("validate_params 意外抛出 ValueError")

        # 测试无效命名空间
        invalid_params = dict(self.test_params)
        invalid_params['ns'] = ''
        with self.assertRaises(ValueError):
            self.builder.validate_params(**invalid_params)

        # 测试无效块大小
        invalid_params = dict(self.test_params)
        invalid_params['block_size'] = 0
        with self.assertRaises(ValueError):
            self.builder.validate_params(**invalid_params)

        # 测试无效块类型
        invalid_params = dict(self.test_params)
        invalid_params['k_type'] = 2
        with self.assertRaises(ValueError):
            self.builder.validate_params(**invalid_params)

    def test_serialize_deserialize(self):
        """测试序列化和反序列化功能"""
        tensor, binary = self.create_test_tensor()
        
        # 测试序列化
        serialized = self.builder.serialize_tensor(tensor)
        self.assertEqual(serialized, binary)

        # 测试反序列化
        deserialized = self.builder.deserialize_tensor(
            binary,
            self.test_params['block_size'],
            self.test_params['dtype']
        )
        np.testing.assert_array_equal(deserialized, tensor)

        # 测试与 Redis 服务器的交互
        key = self.builder.build_key(
            self.test_params['ns'],
            self.test_params['layer_id'],
            self.test_params['block_id'],
            self.test_params['k_type']
        )

        # 在 Redis 中存储张量
        result = self.builder.client.execute_command(
            'KVBLOCKSET',
            self.test_params['ns'],
            self.test_params['layer_id'],
            self.test_params['block_id'],
            self.test_params['k_type'],
            binary
        )
        self.assertEqual(result, b'OK')

        # 获取并验证
        retrieved = self.builder.client.execute_command(
            'KVBLOCKGET',
            self.test_params['ns'],
            self.test_params['layer_id'],
            self.test_params['block_id'],
            self.test_params['k_type']
        )
        retrieved_tensor = self.builder.deserialize_tensor(
            retrieved,
            self.test_params['block_size'],
            self.test_params['dtype']
        )
        np.testing.assert_array_equal(retrieved_tensor, tensor)

        # 清理
        self.builder.client.delete(key)

    def test_validate_tensor_shape(self):
        """测试张量形状验证功能"""
        tensor = np.random.randn(
            1,  # 单个块
            self.test_params['block_size']
        ).astype(np.float16)

        # 测试有效形状
        try:
            self.builder.validate_tensor_shape(
                tensor,
                self.test_params['block_size']
            )
        except ValueError:
            self.fail("validate_tensor_shape 意外抛出 ValueError")

        # 测试无效形状
        invalid_tensor = np.random.randn(1, 64).astype(np.float16)
        with self.assertRaises(ValueError):
            self.builder.validate_tensor_shape(
                invalid_tensor,
                self.test_params['block_size']
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)