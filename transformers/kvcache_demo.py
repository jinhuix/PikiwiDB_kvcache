#!/usr/bin/env python3

import torch
import time
import psutil
import gc
import os
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    DynamicCache,
    OffloadedCache,
    StaticCache,
    QuantizedCache,
    QuantizedCacheConfig
)


def demonstrate_opt_125m():
    """
    加载OPT-125m模型并演示KV Cache生成流程
    """
    
    # 直接加载模型（transformers会自动使用缓存）
    model_name = "facebook/opt-125m"    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda:0")
    device = torch.device("cuda:0")
        
    # 输出模型基本信息
    print(f"\n\n模型信息:")
    print(f"  参数量: {model.num_parameters():,}")
    print(f"  层数: {model.config.num_hidden_layers}")
    print(f"  注意力头数: {model.config.num_attention_heads}")
    print(f"  隐藏维度: {model.config.hidden_size}")
    print(f"  头维度: {model.config.hidden_size // model.config.num_attention_heads}")
    
    # 准备输入
    prompt = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to(device)
    print(f"\n输入文本: {prompt}")
    print(f"输入token数: {input_ids.shape[1]}")
    print(f"输入token IDs: {input_ids.tolist()}")
    
    # 第一阶段：不使用缓存
    print(f"\n[阶段1] 不使用缓存")
    print("-" * 40)
    outputs_no_cache = model(input_ids, use_cache=False)
    print(f"输出logits形状: {outputs_no_cache.logits.shape}")
    print(f"返回past_key_values: {outputs_no_cache.past_key_values is not None}")
    
    # 第二阶段：使用缓存
    print(f"\n[阶段2] 使用缓存")
    print("-" * 40)
    outputs_with_cache = model(input_ids, use_cache=True)    
    print(f"输出logits形状: {outputs_with_cache.logits.shape}")
    print(f"返回past_key_values: {outputs_with_cache.past_key_values is not None}")
    
    # 分析KV Cache结构
    if outputs_with_cache.past_key_values is not None:
        past_kv = outputs_with_cache.past_key_values
        print(f"\nKV Cache 结构分析:")
        print(f"  缓存类型: {type(past_kv).__name__}")
        print(f"  缓存层数: {len(past_kv)}")
        
        # 分析前3层的缓存
        for layer_idx in range(min(3, len(past_kv))):
            if hasattr(past_kv, 'key_cache'):
                # 新版本的Cache对象
                key_cache = past_kv.key_cache[layer_idx]
                value_cache = past_kv.value_cache[layer_idx]
            else:
                # 传统的tuple格式
                key_cache, value_cache = past_kv[layer_idx]
            
            memory_usage = key_cache.numel() * key_cache.element_size() + value_cache.numel() * value_cache.element_size()
            print(f"  层{layer_idx}: Key{key_cache.shape}, Value{value_cache.shape}, 内存{memory_usage}bytes")
    
    # 第三阶段：增量生成演示（核心流程）
    print(f"\n[阶段3] 增量生成流程演示")
    print("-" * 40)
    
    # 创建DynamicCache
    cache = DynamicCache()
    current_input = input_ids.clone()
    
    print(f"开始增量生成，每步详细流程:")
    generated_tokens = []
    
    for step in range(5):  # 生成5个token
        print(f"\n步骤 {step + 1}:")
        
        # 确定输入：第一次是完整序列，后续只是新token
        if step == 0:
            step_input = current_input
            print(f"  输入模式: 完整序列")
        else:
            step_input = torch.tensor([[next_token]], device=current_input.device)
            print(f"  输入模式: 新token")
        
        print(f"  输入形状: {step_input.shape}")
        print(f"  输入内容: '{tokenizer.decode(step_input[0])}'")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(
                step_input,
                past_key_values=cache,
                use_cache=True
            )
        
        # 更新缓存
        cache = outputs.past_key_values
        
        # 选择下一个token (贪心搜索)
        next_token = torch.argmax(outputs.logits[0, -1, :]).item()
        generated_tokens.append(next_token)
        
        print(f"  生成token: {next_token} -> '{tokenizer.decode([next_token])}'")
        print(f"  缓存序列长度: {cache.get_seq_length()}")
        print(f"  缓存层数: {len(cache)}")
        
        # 更新当前序列
        current_input = torch.cat([current_input, torch.tensor([[next_token]], device=current_input.device)], dim=1)
    
    # 显示最终结果
    final_text = tokenizer.decode(current_input[0], skip_special_tokens=True)
    print(f"\n生成结果:")
    print(f"  完整文本: {final_text}")
    print(f"  生成的tokens: {generated_tokens}")
    
    
    # 清理内存
    del cache
    torch.cuda.empty_cache()


def demonstrate_opt_2_7b_offloadcache():
    """
    函数3: 加载OPT-2.7b模型并使用offloadcache
    """
    model_name = "facebook/opt-2.7b"    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda:0")
    device = torch.device("cuda:0")
    print(f"使用设备: {device}")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU总内存: {gpu_memory:.1f} GB")

    # 输出模型基本信息
    print(f"\nOPT-2.7b 模型信息:")
    print(f"  - 参数量: {model.num_parameters():,}")
    print(f"  - 层数: {model.config.num_hidden_layers}")
    print(f"  - 注意力头数: {model.config.num_attention_heads}")
    print(f"  - 隐藏维度: {model.config.hidden_size}")
    print(f"  - 头维度: {model.config.hidden_size // model.config.num_attention_heads}")
    
    # 检查当前GPU内存使用
    print(f"\n当前GPU内存使用:")
    print(f"  - 已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  - 已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    # 准备输入 - 使用更长的序列来展示OffloadedCache的优势
    prompt = "The advancement of large language models has revolutionized natural language processing. These models can understand and generate human-like text, enabling applications such as"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"\n输入文本: {prompt}")
    print(f"输入token数: {input_ids.shape[1]}")
    
    # 第一阶段：使用DynamicCache（标准缓存）
    print(f"\n[阶段1] 使用 DynamicCache")
    print("-" * 40)
    try:
        dynamic_cache = DynamicCache()
        
        # 记录开始内存状态
        start_memory = torch.cuda.memory_allocated() / 1024**3
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
                past_key_values=dynamic_cache,
                pad_token_id=tokenizer.pad_token_id
            )
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() / 1024**3
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"生成时间: {end_time - start_time:.2f}s")
        print(f"内存变化: {start_memory:.2f} GB -> {end_memory:.2f} GB")
        print(f"生成结果: {generated_text}")
        
        # 清理内存
        del dynamic_cache, outputs
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"DynamicCache 测试失败: {e}")
        print("可能是GPU内存不足")
    
    # 第二阶段：使用OffloadedCache
    print(f"\n[阶段2] 使用 OffloadedCache")
    print("-" * 40)
    
    try:
        offloaded_cache = OffloadedCache()
        
        print(f"OffloadedCache 配置:")
        print(f"  - 预取流: {offloaded_cache.prefetch_stream}")
        print(f"  - 原始设备列表: {offloaded_cache.original_device}")
        print(f"  - Beam索引: {offloaded_cache.beam_idx}")
        
        # 记录开始内存状态
        start_memory = torch.cuda.memory_allocated() / 1024**3
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
                past_key_values=offloaded_cache,
                pad_token_id=tokenizer.pad_token_id
            )
        
        end_time = time.time()
        end_memory = torch.cuda.memory_allocated() / 1024**3
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"生成时间: {end_time - start_time:.2f}s")
        print(f"内存变化: {start_memory:.2f} GB -> {end_memory:.2f} GB")
        print(f"生成结果: {generated_text}")
        
        # 分析OffloadedCache的状态
        print(f"\nOffloadedCache 详细分析:")
        print(f"  - 缓存层数: {len(offloaded_cache)}")
        print(f"  - 原始设备记录: {offloaded_cache.original_device}")
        
        # 检查缓存设备分布
        gpu_layers = 0
        cpu_layers = 0
        
        for i in range(len(offloaded_cache)):
            if offloaded_cache.key_cache[i].device.type == 'cuda':
                gpu_layers += 1
            else:
                cpu_layers += 1
        
        print(f"  - GPU上的层数: {gpu_layers}")
        print(f"  - CPU上的层数: {cpu_layers}")
        
        # 演示缓存的动态调度
        print(f"\n演示缓存动态调度:")
        for layer_idx in range(min(3, len(offloaded_cache))):
            key_device = offloaded_cache.key_cache[layer_idx].device
            value_device = offloaded_cache.value_cache[layer_idx].device
            print(f"  - 层 {layer_idx}: Key在{key_device}, Value在{value_device}")
        
        del offloaded_cache, outputs
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"OffloadedCache 测试失败: {e}")
        print("可能是GPU内存不足或CUDA版本不兼容")


def main():
    # demonstrate_opt_125m()
    demonstrate_opt_2_7b_offloadcache()


if __name__ == "__main__":
    main()
