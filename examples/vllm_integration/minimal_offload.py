#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import argparse
from typing import List

# 让哈希/分词更可控（可选）
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("VLLM_LOG_LEVEL", "INFO")

import redis
from transformers import AutoTokenizer

# vLLM
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

import logging
logging.getLogger("vllm.distributed.kv_transfer.kv_connector.v1.pika_connector").setLevel(logging.INFO)

# ---------- Redis 工具 ----------
def get_redis(host: str, port: int, db: int = 0) -> redis.Redis:
    r = redis.Redis(host=host, port=port, db=db, decode_responses=False)
    r.ping()
    return r

def scan_keys(r: redis.Redis, pattern: str, count: int = 1000) -> List[bytes]:
    cursor = 0
    out = []
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=count)
        out.extend(keys)
        if cursor == 0:
            break
    return out

def del_pattern(r: redis.Redis, pattern: str) -> int:
    total = 0
    cursor = 0
    pipe = r.pipeline()
    batch = 0
    while True:
        cursor, keys = r.scan(cursor=cursor, match=pattern, count=1000)
        for k in keys:
            pipe.delete(k)
            batch += 1
            if batch >= 500:
                total += sum(pipe.execute()); batch = 0
        if cursor == 0:
            break
    if batch > 0:
        total += sum(pipe.execute())
    return total

# ---------- Prompt 构造 ----------
def long_enough_prompt(tokenizer, min_tokens: int) -> str:
    base = ("In the field of artificial intelligence, recent advances in natural language processing "
            "and large language models have demonstrated remarkable capabilities across a wide range of tasks. ")
    s = base
    while len(tokenizer.encode(s)) < min_tokens:
        s += base
    return s

# ---------- LLM 构建（同一进程内可重复调用） ----------
def build_llm(model: str, host: str, port: int, max_len: int, engine_id: str):
    kv_cfg = KVTransferConfig(
        kv_connector="PikaConnector",
        kv_role="kv_both",
        engine_id=engine_id,  # 固定 engine_id（可选，但便于排查/命名空间稳定）
        kv_connector_extra_config={"host": host, "port": port, "db": 0},
        kv_buffer_device="cuda",
        kv_buffer_size=1_000_000_000,       # int，避免 float
    )
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_len,
        enable_prefix_caching=True,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=0.85,
    )
    return llm

# ---------- 主流程（一个进程内顺序执行两轮） ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9221)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--engine-id", default="kvtest-opt125m-nhd-fp16")  # 固定命名空间/引擎标识
    ap.add_argument("--min-prefix-tokens", type=int, default=256)      # 确保至少 1~2 个块
    args = ap.parse_args()

    r = get_redis(args.host, args.port)
    removed = del_pattern(r, "kvblock:*")
    print(f"[INIT] Cleared old keys: {removed}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    common = long_enough_prompt(tok, min_tokens=args.min_prefix_tokens)
    suffixes = [
        " This experiment focuses on efficiency.",
        " This experiment focuses on reliability.",
        " This experiment focuses on latency.",
        " This experiment focuses on throughput.",
    ]
    prompts = [common + s for s in suffixes]
    sp = SamplingParams(temperature=args.temperature, max_tokens=32)

    # ===== 第一轮（写入） =====
    llm = build_llm(args.model, args.host, args.port, args.max_len, args.engine_id)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    dt = time.time() - t0
    print(f"[STORE] First run time: {dt:.3f}s")
    print("[STORE] Example output:", outs[0].outputs[0].text.strip()[:120].replace("\n", " "))

    keys1 = scan_keys(r, "kvblock:*")
    print(f"[STORE] kvblock keys count: {len(keys1)}")
    if keys1[:5]:
        print("[STORE] sample keys:")
        for k in keys1[:5]:
            print("  -", (k.decode("utf-8") if isinstance(k, bytes) else str(k)))

    # 释放第一轮引擎（同一进程里“重建引擎”的效果）
    del llm
    time.sleep(0.5)

    # ===== 第二轮（读取/命中） =====
    llm = build_llm(args.model, args.host, args.port, args.max_len, args.engine_id)
    t0 = time.time()
    outs2 = llm.generate(prompts, sp)
    dt2 = time.time() - t0
    print(f"[LOAD ] Second run time: {dt2:.3f}s")
    print("[LOAD ] Example output:", outs2[0].outputs[0].text.strip()[:120].replace("\n", " "))

    keys2 = scan_keys(r, "kvblock:*")
    print(f"[LOAD ] kvblock keys count: {len(keys2)}")

    if len(keys2) == len(keys1):
        print("[OK   ] Key count unchanged. Likely hit & reuse succeeded.")
    else:
        print(f"[WARN ] Key count changed: {len(keys1)} -> {len(keys2)} (可能未命中或又写入了新块)")

if __name__ == "__main__":
    main()
