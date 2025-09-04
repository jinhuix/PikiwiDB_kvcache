#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Page-oriented KV Cache Test for vLLM PagedAttention
Tests the new KVPAGESET/KVPAGEGET/KVPAGEMSET/KVPAGEMGET/KVPAGEEXISTS commands.
"""

import unittest
import redis
import struct
import time
import numpy as np
import sys
from typing import List, Tuple, Optional

class TestKVPageCommands(unittest.TestCase):
    """Page-oriented KV Cache 测试用例"""
    
    @classmethod
    def setUpClass(cls):
        """测试类初始化，建立Redis连接"""
        try:
            cls.client = redis.Redis(host='localhost', port=9221, decode_responses=False)
            cls.client.ping()
            print(f"✅ 成功连接到 PikiwiDB: localhost:9221")
        except redis.ConnectionError:
            print(f"❌ 无法连接到 PikiwiDB: localhost:9221")
            sys.exit(1)

    @classmethod
    def tearDownClass(cls):
        """测试类清理，关闭Redis连接"""
        if hasattr(cls, 'client'):
            cls.client.close()

    def create_test_tensor_data(self, page_size: int = 128, head_dim: int = 64, 
                              dtype: int = 1) -> bytes:
        """创建测试用的 tensor 数据"""
        if dtype == 1:  # FP16
            np_dtype = np.float16
        elif dtype == 2:  # FP32
            np_dtype = np.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        
        data = np.random.randn(page_size, head_dim).astype(np_dtype)
        return data.tobytes()

    def test_kvpageset_get(self):
        """测试单个页面的存取"""
        req_id = "test_req_001"
        layer_idx = 5
        head_idx = 8
        page_id = 42
        kv_type = 0
        dtype = 1
        page_size = 128
        head_dim = 64
        ttl = 3600
        
        tensor_data = self.create_test_tensor_data(page_size, head_dim, dtype)
        
        # 测试 KVPAGESET
        result = self.client.execute_command(
            'KVPAGESET',
            req_id, layer_idx, head_idx, page_id, kv_type,
            dtype, page_size, head_dim, ttl,
            tensor_data
        )
        self.assertEqual(result, b'OK', "KVPAGESET failed")
        
        # 测试 KVPAGEGET
        result = self.client.execute_command(
            'KVPAGEGET',
            req_id, layer_idx, head_idx, page_id, kv_type
        )
        self.assertIsNotNone(result, "KVPAGEGET returned None")
        self.assertEqual(len(result), len(tensor_data), "Data length mismatch")
        
        # 验证数据一致性
        original_array = np.frombuffer(tensor_data, dtype=np.float16).reshape(page_size, head_dim)
        retrieved_array = np.frombuffer(result, dtype=np.float16).reshape(page_size, head_dim)
        np.testing.assert_array_equal(original_array, retrieved_array, "Data content mismatch")
        
        # 清理
        key = f"kv:{req_id}:{layer_idx}:{head_idx}:{page_id}:{kv_type}"
        self.client.delete(key)

    def test_kvpagemset_mget(self):
        """测试批量页面存取"""
        pages = []
        req_id = "test_batch_req"
        page_size = 128
        head_dim = 64
        dtype = 1
        ttl = 3600
        
        # 创建测试数据
        for layer_idx in [0, 1]:
            for head_idx in [0, 1]:
                for kv_type in [0, 1]:
                    page_id = layer_idx * 100 + head_idx * 10 + kv_type
                    tensor_data = self.create_test_tensor_data(page_size, head_dim, dtype)
                    pages.append({
                        'req_id': req_id,
                        'layer_idx': layer_idx,
                        'head_idx': head_idx,
                        'page_id': page_id,
                        'kv_type': kv_type,
                        'dtype': dtype,
                        'page_size': page_size,
                        'head_dim': head_dim,
                        'ttl': ttl,
                        'tensor_data': tensor_data
                    })
        
        # 构建并执行 KVPAGEMSET
        cmd_args = ['KVPAGEMSET', str(len(pages))]
        for page in pages:
            cmd_args.extend([
                page['req_id'], str(page['layer_idx']), str(page['head_idx']),
                str(page['page_id']), str(page['kv_type']), str(page['dtype']),
                str(page['page_size']), str(page['head_dim']), str(page['ttl']),
                page['tensor_data']
            ])
        
        result = self.client.execute_command(*cmd_args)
        self.assertEqual(result, b'OK', "KVPAGEMSET failed")
        
        # 构建并执行 KVPAGEMGET
        cmd_args = ['KVPAGEMGET', str(len(pages))]
        for page in pages:
            cmd_args.extend([
                page['req_id'], str(page['layer_idx']), str(page['head_idx']),
                str(page['page_id']), str(page['kv_type'])
            ])
        
        results = self.client.execute_command(*cmd_args)
        self.assertIsInstance(results, list, "KVPAGEMGET should return a list")
        self.assertEqual(len(results), len(pages), "Results count mismatch")
        
        # 验证数据
        for i, (page, result) in enumerate(zip(pages, results)):
            self.assertIsNotNone(result, f"Page {i} returned None")
            self.assertEqual(len(result), len(page['tensor_data']), f"Page {i} length mismatch")
            
            original_array = np.frombuffer(page['tensor_data'], dtype=np.float16)
            retrieved_array = np.frombuffer(result, dtype=np.float16)
            np.testing.assert_array_equal(original_array, retrieved_array, f"Page {i} content mismatch")
        
        # 清理
        for page in pages:
            key = f"kv:{page['req_id']}:{page['layer_idx']}:{page['head_idx']}:{page['page_id']}:{page['kv_type']}"
            self.client.delete(key)

    def test_kvpageexists(self):
        """测试 KVPAGEEXISTS 命令"""
        req_id = "exists_test"
        page_size = 128
        head_dim = 64
        
        # 测试不存在的页面
        result = self.client.execute_command('KVPAGEEXISTS', req_id, 0, 0, 0, 0)
        self.assertEqual(result, 0, "Non-existent page check failed")
        
        # 创建并测试存在的页面
        tensor_data = self.create_test_tensor_data(page_size, head_dim, 1)
        self.client.execute_command(
            'KVPAGESET',
            req_id, 0, 0, 0, 0,
            1, page_size, head_dim, 3600,
            tensor_data
        )
        
        result = self.client.execute_command('KVPAGEEXISTS', req_id, 0, 0, 0, 0)
        self.assertEqual(result, 1, "Existing page check failed")
        
        # 清理
        self.client.delete(f"kv:{req_id}:0:0:0:0")

    def test_performance(self):
        """性能测试"""
        num_pages = 10
        page_size = 128
        head_dim = 128
        dtype = 1
        
        # 创建测试数据
        pages = []
        for i in range(num_pages):
            tensor_data = self.create_test_tensor_data(page_size, head_dim, dtype)
            pages.append({
                'req_id': f'perf_test_{i}',
                'layer_idx': 0,
                'head_idx': 0,
                'page_id': i,
                'kv_type': 0,
                'dtype': dtype,
                'page_size': page_size,
                'head_dim': head_dim,
                'ttl': 3600,
                'tensor_data': tensor_data
            })
        
        # 测试批量写入
        start_time = time.time()
        cmd_args = ['KVPAGEMSET', str(len(pages))]
        for page in pages:
            cmd_args.extend([
                page['req_id'], str(page['layer_idx']), str(page['head_idx']),
                str(page['page_id']), str(page['kv_type']), str(page['dtype']),
                str(page['page_size']), str(page['head_dim']), str(page['ttl']),
                page['tensor_data']
            ])
        
        result = self.client.execute_command(*cmd_args)
        self.assertEqual(result, b'OK', "Batch write failed")
        write_time = time.time() - start_time
        
        # 测试批量读取
        start_time = time.time()
        cmd_args = ['KVPAGEMGET', str(len(pages))]
        for page in pages:
            cmd_args.extend([
                page['req_id'], str(page['layer_idx']), str(page['head_idx']),
                str(page['page_id']), str(page['kv_type'])
            ])
        
        results = self.client.execute_command(*cmd_args)
        read_time = time.time() - start_time
        
        self.assertEqual(len(results), num_pages, "Batch read results count mismatch")
        
        # 清理
        for page in pages:
            key = f"kv:{page['req_id']}:{page['layer_idx']}:{page['head_idx']}:{page['page_id']}:{page['kv_type']}"
            self.client.delete(key)

if __name__ == '__main__':
    unittest.main(verbosity=2)