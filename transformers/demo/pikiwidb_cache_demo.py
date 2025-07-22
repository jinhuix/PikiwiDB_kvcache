#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PikiwidbCache 功能演示脚本

本脚本演示了如何使用PikiwidbCache进行文本生成，并验证其核心功能，
包括：批处理、左填充、连续生成、数值正确性和内存优化。

使用方法:
1. 确保PikiwiDB/Pika服务器正在运行。
   默认连接地址: localhost:9221
   可通过环境变量 PIKIWIDB_HOST 和 PIKIWIDB_PORT 修改。
2. 运行脚本:
   python pikiwidb_cache_demo.py
"""

import os
import uuid
import logging
import torch
import redis
import sys

# 将src目录添加到python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers.cache_utils import PikiwidbCache, DynamicCache

# 日志配置
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pikiwidb_cache_demo.log"),
        logging.StreamHandler()
    ]
)

# PikiwiDB服务器配置
PIKIWIDB_HOST = os.environ.get("PIKIWIDB_HOST", "localhost")
PIKIWIDB_PORT = int(os.environ.get("PIKIWIDB_PORT", 9221))


class PikiwidbCacheDemo:
    """
    PikiwidbCache演示类
    """
    def __init__(self, model_name="facebook/opt-125m"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map=self.device, torch_dtype=torch.float16
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        self.redis_client = None
        self.cache_prefix = None

    def setup_redis(self):
        """连接到Redis并为演示创建唯一前缀"""
        try:
            self.redis_client = redis.Redis(host=PIKIWIDB_HOST, port=PIKIWIDB_PORT, db=0, decode_responses=True)
            self.redis_client.ping()
            logging.info(f"成功连接到PikiwiDB at {PIKIWIDB_HOST}:{PIKIWIDB_PORT}")
        except redis.exceptions.ConnectionError as e:
            logging.error(f"无法连接到PikiwiDB at {PIKIWIDB_HOST}:{PIKIWIDB_PORT}. 请确保它正在运行。")
            raise e
        self.cache_prefix = f"cache:{uuid.uuid4()}"
        logging.info(f"使用缓存前缀: {self.cache_prefix}")

    def teardown_redis(self):
        """清理数据库并关闭连接"""
        if self.redis_client and self.cache_prefix:
            keys = self.redis_client.keys(f"{self.cache_prefix}:*")
            if keys:
                self.redis_client.delete(*keys)
                logging.info(f"已清理缓存键: {len(keys)} 个")
            self.redis_client.close()
            logging.info("已关闭PikiwiDB连接")

    def _get_pikiwidb_cache(self):
        """创建PikiwidbCache实例"""
        return PikiwidbCache(
            host=PIKIWIDB_HOST,
            port=PIKIWIDB_PORT,
            prefix=self.cache_prefix,
        )

    def run_all_demos(self):
        """按顺序运行所有演示"""
        try:
            self.setup_redis()
            self.demo_batched_generation()
            self.demo_left_padding()
            self.demo_continue_generation()
            self.demo_correctness()
            self.demo_memory_usage()
        finally:
            self.teardown_redis()
    
    def demo_batched_generation(self):
        """演示批处理生成"""
        logging.info("\n--- 1. 批处理生成演示 ---")
        prompts = ["A sequence: 1, 2, 3, 4, 5", "A sequence: A, B, C"]
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        
        cache = self._get_pikiwidb_cache()
        gen_out = self.model.generate(
            **inputs, do_sample=False, max_new_tokens=10, past_key_values=cache
        )
        decoded = self.tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        
        logging.info("批处理输入:")
        for p in prompts:
            logging.info(f"'{p}'")
        logging.info("批处理生成结果:")
        for d in decoded:
            logging.info(f"'{d}'")

    def demo_left_padding(self):
        """演示左填充处理"""
        logging.info("\n--- 2. 左填充处理演示 ---")
        prompt = "The cat is cute"
        inputs = self.tokenizer([prompt], padding=True, return_tensors="pt").to(self.device)
        
        cache1 = self._get_pikiwidb_cache()
        gen_out = self.model.generate(**inputs, max_new_tokens=5, past_key_values=cache1)
        decoded = self.tokenizer.decode(gen_out[0], skip_special_tokens=True)
        logging.info(f"无额外填充: '{decoded}'")

        self.cache_prefix = f"cache:{uuid.uuid4()}" # 新前缀
        inputs_padded = self.tokenizer([prompt], padding=True, pad_to_multiple_of=16, return_tensors="pt").to(self.device)
        
        cache2 = self._get_pikiwidb_cache()
        gen_out_padded = self.model.generate(**inputs_padded, max_new_tokens=5, past_key_values=cache2)
        decoded_padded = self.tokenizer.decode(gen_out_padded[0], skip_special_tokens=True)
        logging.info(f"有额外填充: '{decoded_padded}'")
        
        if decoded == decoded_padded:
            logging.info("结果一致，左填充处理正确。")
        else:
            logging.warning("结果不一致，左填充处理可能存在问题。")

    def demo_continue_generation(self):
        """演示从缓存继续生成"""
        logging.info("\n--- 3. 从缓存继续生成演示 ---")
        inputs = self.tokenizer("Once upon a time", return_tensors="pt").to(self.device)
        cache = self._get_pikiwidb_cache()

        output1 = self.model.generate(**inputs, max_new_tokens=5, past_key_values=cache, return_dict_in_generate=True)
        logging.info(f"第一步生成: '{self.tokenizer.decode(output1.sequences[0], skip_special_tokens=True)}'")
        logging.debug(f"output1.past_key_values: {output1.past_key_values[0][0].shape}")

        output2 = self.model.generate(output1.sequences, max_new_tokens=5, past_key_values=output1.past_key_values, return_dict_in_generate=True)
        logging.info(f"第二步生成: '{self.tokenizer.decode(output2.sequences[0], skip_special_tokens=True)}'")
        logging.debug(f"output2.past_key_values: {output2.past_key_values[0][0].shape}")
        
        self.cache_prefix = f"cache:{uuid.uuid4()}"
        cache_full = self._get_pikiwidb_cache()
        output_full = self.model.generate(**inputs, max_new_tokens=10, past_key_values=cache_full, return_dict_in_generate=True)
        logging.info(f"一次性生成: '{self.tokenizer.decode(output_full.sequences[0], skip_special_tokens=True)}'")
        logging.debug(f"output_full.past_key_values: {output_full.past_key_values[0][0].shape}")

        if torch.allclose(output2.sequences, output_full.sequences):
            logging.info("结果一致，连续生成功能正常。")
        else:
            logging.warning("结果不一致，连续生成功能可能存在问题。")

    def demo_correctness(self):
        """演示数值正确性（与DynamicCache对比）"""
        logging.info("\n--- 4. 数值正确性演示 ---")
        model_name = "Qwen/Qwen2-0.5B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device, torch_dtype=torch.float16)
        inputs = tokenizer(["Here's everything I know about cats. Cats"], return_tensors="pt").to(model.device)

        set_seed(0)
        gen_out_dynamic = model.generate(**inputs, do_sample=True, max_new_tokens=32)
        decoded_dynamic = tokenizer.decode(gen_out_dynamic[0], skip_special_tokens=True)

        set_seed(0)
        self.cache_prefix = f"cache:{uuid.uuid4()}"
        cache_pikiwi = self._get_pikiwidb_cache()
        gen_out_pikiwi = model.generate(**inputs, do_sample=True, max_new_tokens=32, past_key_values=cache_pikiwi)
        decoded_pikiwi = tokenizer.decode(gen_out_pikiwi[0], skip_special_tokens=True)

        logging.info(f"DynamicCache 生成: '{decoded_dynamic}'")
        logging.info(f"PikiwidbCache 生成: '{decoded_pikiwi}'")
        
        if decoded_pikiwi == decoded_dynamic:
            logging.info("生成结果完全一致，数值正确性通过。")
        else:
            logging.warning("生成结果不一致，数值正确性需要检查。")

    def demo_memory_usage(self):
        """演示内存使用情况对比"""
        if self.device.type != 'cuda':
            logging.warning("内存使用演示需要CUDA设备，已跳过。")
            return
            
        logging.info("\n--- 5. 内存使用演示 ---")
        model_name = "microsoft/Phi-3-mini-4k-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map=self.device, torch_dtype=torch.float16)
        
        input_text = "Fun fact:" * 100
        inputs = tokenizer(input_text, return_tensors="pt").to(self.device)

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        _ = model.generate(**inputs, max_new_tokens=128) # DynamicCache
        dynamic_peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        logging.info(f"DynamicCache 峰值内存: {dynamic_peak_memory:.2f} MB")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        self.cache_prefix = f"cache:{uuid.uuid4()}"
        cache_pikiwi = self._get_pikiwidb_cache()
        _ = model.generate(**inputs, max_new_tokens=128, past_key_values=cache_pikiwi)
        pikiwi_peak_memory = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        logging.info(f"PikiwidbCache 峰值内存: {pikiwi_peak_memory:.2f} MB")

        if pikiwi_peak_memory < dynamic_peak_memory:
            logging.info("PikiwidbCache 使用的GPU内存明显更少，内存卸载功能验证通过。")
        else:
            logging.warning("PikiwidbCache 使用的GPU内存未减少，内存卸载功能可能未生效。")


if __name__ == "__main__":
    demo = PikiwidbCacheDemo()
    demo.run_all_demos() 