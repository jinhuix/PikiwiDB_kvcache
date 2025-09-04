#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KVBlock 命令测试
测试 KVBLOCKSET/KVBLOCKGET/KVBLOCKMSET/KVBLOCKMGET/KVBLOCKEXISTS 命令
"""

import unittest
import redis
import struct
import time
import numpy as np
import sys
from typing import List, Tuple, Optional

class TestKVBlockCommands(unittest.TestCase):
    """KVBlock 命令测试用例"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，建立Redis连接"""
        try:
            cls.client = redis.Redis(host='localhost', port=9221, decode_responses=False)
            cls.client.ping()
            print("已成功连接到 PikiwiDB: localhost:9221")
        except redis.ConnectionError:
            print("无法连接到 PikiwiDB: localhost:9221")
            sys.exit(1)

    @classmethod
    def tearDownClass(cls):
        """测试类清理，关闭Redis连接"""
        if hasattr(cls, 'client'):
            cls.client.close()

    def create_test_tensor_data(self, block_size: int = 128, dtype: int = 1) -> bytes:
        """创建测试用的 tensor 数据"""
        if dtype == 1:  # FP16
            np_dtype = np.float16
        elif dtype == 2:  # FP32
            np_dtype = np.float32
        else:
            raise ValueError(f"不支持的数据类型: {dtype}")
        
        data = np.random.randn(1, block_size).astype(np_dtype)
        return data.tobytes()

    def test_kvblockset_get(self):
        """测试单个块的存取"""
        ns = "test_ns_001"
        layer_id = 5
        block_id = 42
        k_type = 0
        block_size = 128
        dtype = 1
        
        tensor_data = self.create_test_tensor_data(block_size, dtype)
        
        # 测试 KVBLOCKSET
        result = self.client.execute_command(
            'KVBLOCKSET',
            ns, layer_id, block_id, k_type,
            tensor_data
        )
        self.assertEqual(result, b'OK', "KVBLOCKSET 失败")
        
        # 测试 KVBLOCKGET
        result = self.client.execute_command(
            'KVBLOCKGET',
            ns, layer_id, block_id, k_type
        )
        self.assertIsNotNone(result, "KVBLOCKGET 返回空值")
        self.assertEqual(len(result), len(tensor_data), "数据长度不匹配")
        
        # 验证数据一致性
        original_array = np.frombuffer(tensor_data, dtype=np.float16).reshape(1, block_size)
        retrieved_array = np.frombuffer(result, dtype=np.float16).reshape(1, block_size)
        np.testing.assert_array_equal(original_array, retrieved_array, "数据内容不匹配")
        
        # 清理
        key = f"kvblock:{ns}:{layer_id}:{block_id}:{k_type}"
        self.client.delete(key)

    def test_kvblockmset_mget(self):
        """测试批量块存取"""
        blocks = []
        ns = "test_batch_ns"
        block_size = 128
        dtype = 1
        
        # 创建测试数据
        for layer_id in [0, 1]:
            for k_type in [0, 1]:
                block_id = layer_id * 10 + k_type
                tensor_data = self.create_test_tensor_data(block_size, dtype)
                blocks.append({
                    'ns': ns,
                    'layer_id': layer_id,
                    'block_id': block_id,
                    'k_type': k_type,
                    'tensor_data': tensor_data
                })
        
        # 构建并执行 KVBLOCKMSET
        cmd_args = ['KVBLOCKMSET', str(len(blocks))]
        for block in blocks:
            cmd_args.extend([
                block['ns'], str(block['layer_id']),
                str(block['block_id']), str(block['k_type']),
                block['tensor_data']
            ])
        
        result = self.client.execute_command(*cmd_args)
        self.assertEqual(result, b'OK', "KVBLOCKMSET 失败")
        
        # 构建并执行 KVBLOCKMGET
        cmd_args = ['KVBLOCKMGET', str(len(blocks))]
        for block in blocks:
            cmd_args.extend([
                block['ns'], str(block['layer_id']),
                str(block['block_id']), str(block['k_type'])
            ])
        
        results = self.client.execute_command(*cmd_args)
        self.assertIsInstance(results, list, "KVBLOCKMGET 应返回列表")
        self.assertEqual(len(results), len(blocks), "结果数量不匹配")
        
        # 验证数据
        for i, (block, result) in enumerate(zip(blocks, results)):
            self.assertIsNotNone(result, f"块 {i} 返回空值")
            self.assertEqual(len(result), len(block['tensor_data']), f"块 {i} 长度不匹配")
            
            original_array = np.frombuffer(block['tensor_data'], dtype=np.float16).reshape(1, block_size)
            retrieved_array = np.frombuffer(result, dtype=np.float16).reshape(1, block_size)
            np.testing.assert_array_equal(original_array, retrieved_array, f"块 {i} 内容不匹配")
        
        # 清理
        for block in blocks:
            key = f"kvblock:{block['ns']}:{block['layer_id']}:{block['block_id']}:{block['k_type']}"
            self.client.delete(key)

    def test_kvblockexists(self):
        """测试 KVBLOCKEXISTS 命令"""
        ns = "exists_test"
        block_size = 128
        
        # 测试不存在的块
        result = self.client.execute_command('KVBLOCKEXISTS', ns, 0, 0, 0)
        self.assertEqual(result, 0, "不存在的块检查失败")
        
        # 创建并测试存在的块
        tensor_data = self.create_test_tensor_data(block_size, 1)
        self.client.execute_command(
            'KVBLOCKSET',
            ns, 0, 0, 0,
            tensor_data
        )
        
        result = self.client.execute_command('KVBLOCKEXISTS', ns, 0, 0, 0)
        self.assertEqual(result, 1, "存在的块检查失败")
        
        # 清理
        self.client.delete(f"kvblock:{ns}:0:0:0")

    def test_performance(self):
        """性能测试"""
        num_blocks = 10
        block_size = 128
        dtype = 1
        
        # 创建测试数据
        blocks = []
        for i in range(num_blocks):
            tensor_data = self.create_test_tensor_data(block_size, dtype)
            blocks.append({
                'ns': f'perf_test_{i}',
                'layer_id': 0,
                'block_id': i,
                'k_type': 0,
                'tensor_data': tensor_data
            })
        
        # 测试批量写入
        start_time = time.time()
        cmd_args = ['KVBLOCKMSET', str(len(blocks))]
        for block in blocks:
            cmd_args.extend([
                block['ns'], str(block['layer_id']),
                str(block['block_id']), str(block['k_type']),
                block['tensor_data']
            ])
        
        result = self.client.execute_command(*cmd_args)
        self.assertEqual(result, b'OK', "批量写入失败")
        write_time = time.time() - start_time
        
        # 测试批量读取
        start_time = time.time()
        cmd_args = ['KVBLOCKMGET', str(len(blocks))]
        for block in blocks:
            cmd_args.extend([
                block['ns'], str(block['layer_id']),
                str(block['block_id']), str(block['k_type'])
            ])
        
        results = self.client.execute_command(*cmd_args)
        read_time = time.time() - start_time
        
        self.assertEqual(len(results), num_blocks, "批量读取结果数量不匹配")
        
        # 清理
        for block in blocks:
            key = f"kvblock:{block['ns']}:{block['layer_id']}:{block['block_id']}:{block['k_type']}"
            self.client.delete(key)

if __name__ == '__main__':
    unittest.main(verbosity=2)