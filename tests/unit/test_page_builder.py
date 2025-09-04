"""Page Builder 单元测试."""

import unittest
import redis
import numpy as np
from typing import Tuple
import sys


class PageBuilder:
    """Page Builder 类，用于处理 KV 缓存页面操作."""
    
    client = None  # 类变量，所有实例共享同一个连接

    @classmethod
    def init_connection(cls, host='localhost', port=9221):
        """初始化 Redis 连接"""
        if cls.client is None:
            try:
                cls.client = redis.Redis(host=host, port=port, decode_responses=False)
                cls.client.ping()
                print(f"\n✅ 成功连接到 PikiwiDB: {host}:{port}")
            except redis.ConnectionError:
                print(f"❌ 无法连接到 PikiwiDB: {host}:{port}")
                sys.exit(1)

    def __init__(self):
        """初始化 PageBuilder 实例"""
        if self.client is None:
            self.init_connection()

    def build_key(self, req_id: str, layer_idx: int, head_idx: int, 
                  page_id: int, kv_type: int) -> str:
        """构建 KV 缓存页面的 Redis 键"""
        return f"kv:{req_id}:{layer_idx}:{head_idx}:{page_id}:{kv_type}"

    def validate_params(self, req_id: str, layer_idx: int, head_idx: int,
                       page_id: int, kv_type: int, dtype: int, page_size: int,
                       head_dim: int, ttl: int) -> None:
        """验证页面参数"""
        if not req_id:
            raise ValueError("请求 ID 不能为空")
        if page_size <= 0 or head_dim <= 0:
            raise ValueError("无效的维度")
        if dtype not in [1, 2]:  # FP16 或 FP32
            raise ValueError("无效的数据类型")

    def serialize_tensor(self, tensor: np.ndarray) -> bytes:
        """将 numpy 张量序列化为字节"""
        return tensor.tobytes()

    def deserialize_tensor(self, data: bytes, page_size: int, 
                         head_dim: int, dtype: int) -> np.ndarray:
        """将字节反序列化为 numpy 张量"""
        np_dtype = np.float16 if dtype == 1 else np.float32
        return np.frombuffer(data, dtype=np_dtype).reshape(page_size, head_dim)

    def validate_tensor_shape(self, tensor: np.ndarray, 
                            page_size: int, head_dim: int) -> None:
        """验证张量形状"""
        if tensor.shape != (page_size, head_dim):
            raise ValueError(f"无效的张量形状: {tensor.shape}, "
                           f"期望形状: ({page_size}, {head_dim})")


class TestPageBuilder(unittest.TestCase):
    """PageBuilder 类的测试用例"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化，确保 Redis 连接"""
        PageBuilder.init_connection()

    def setUp(self):
        """设置测试环境"""
        self.builder = PageBuilder()
        self.test_params = {
            'req_id': 'test_req',
            'layer_idx': 0,
            'head_idx': 0,
            'page_id': 1,
            'kv_type': 0,
            'dtype': 1,  # FP16
            'page_size': 128,
            'head_dim': 64,
            'ttl': 3600
        }

    def create_test_tensor(self) -> Tuple[np.ndarray, bytes]:
        """创建测试张量及其二进制表示"""
        data = np.random.randn(
            self.test_params['page_size'], 
            self.test_params['head_dim']
        ).astype(np.float16)
        return data, data.tobytes()

    def test_build_key(self):
        """测试键构建功能"""
        expected = "kv:test_req:0:0:1:0"
        key = self.builder.build_key(
            self.test_params['req_id'],
            self.test_params['layer_idx'],
            self.test_params['head_idx'], 
            self.test_params['page_id'],
            self.test_params['kv_type']
        )
        self.assertEqual(key, expected)

    def test_validate_params(self):
        """测试参数验证功能"""
        # 测试有效参数
        try:
            self.builder.validate_params(**self.test_params)
        except ValueError:
            self.fail("validate_params 意外抛出 ValueError")

        # 测试无效请求 ID
        invalid_params = dict(self.test_params)
        invalid_params['req_id'] = ''
        with self.assertRaises(ValueError):
            self.builder.validate_params(**invalid_params)

        # 测试无效维度
        invalid_params = dict(self.test_params)
        invalid_params['page_size'] = 0
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
            self.test_params['page_size'],
            self.test_params['head_dim'],
            self.test_params['dtype']
        )
        np.testing.assert_array_equal(deserialized, tensor)

        # 测试与 Redis 服务器的交互
        key = self.builder.build_key(
            self.test_params['req_id'],
            self.test_params['layer_idx'],
            self.test_params['head_idx'],
            self.test_params['page_id'],
            self.test_params['kv_type']
        )

        # 在 Redis 中存储张量
        result = self.builder.client.execute_command(
            'KVPAGESET',
            self.test_params['req_id'],
            self.test_params['layer_idx'],
            self.test_params['head_idx'],
            self.test_params['page_id'],
            self.test_params['kv_type'],
            self.test_params['dtype'],
            self.test_params['page_size'],
            self.test_params['head_dim'],
            self.test_params['ttl'],
            binary
        )
        self.assertEqual(result, b'OK')

        # 获取并验证
        retrieved = self.builder.client.execute_command(
            'KVPAGEGET',
            self.test_params['req_id'],
            self.test_params['layer_idx'],
            self.test_params['head_idx'],
            self.test_params['page_id'],
            self.test_params['kv_type']
        )
        retrieved_tensor = self.builder.deserialize_tensor(
            retrieved,
            self.test_params['page_size'],
            self.test_params['head_dim'],
            self.test_params['dtype']
        )
        np.testing.assert_array_equal(retrieved_tensor, tensor)

        # 清理
        self.builder.client.delete(key)

    def test_validate_tensor_shape(self):
        """测试张量形状验证功能"""
        tensor = np.random.randn(
            self.test_params['page_size'],
            self.test_params['head_dim']
        ).astype(np.float16)

        # 测试有效形状
        try:
            self.builder.validate_tensor_shape(
                tensor,
                self.test_params['page_size'],
                self.test_params['head_dim']
            )
        except ValueError:
            self.fail("validate_tensor_shape 意外抛出 ValueError")

        # 测试无效形状
        invalid_tensor = np.random.randn(64, 32).astype(np.float16)
        with self.assertRaises(ValueError):
            self.builder.validate_tensor_shape(
                invalid_tensor,
                self.test_params['page_size'],
                self.test_params['head_dim']
            )


if __name__ == '__main__':
    unittest.main(verbosity=2)