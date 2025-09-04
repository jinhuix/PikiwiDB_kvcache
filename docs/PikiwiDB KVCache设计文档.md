# PikiwiDB KVCache设计文档

## 1. KVPage 数据结构

### 1.1 设计特点

这是一个为 PikiwiDB 实现的专门用于管理 KV cache 的数据结构，专门设计用于与 vLLM 的 PagedAttention 机制高效配合。

### 设计特点

**1. 兼容 vLLM 内存布局**

- K cache 布局: `[num_kv_heads, head_size/x, block_size, x]`
- V cache 布局: `[num_kv_heads, head_size, block_size]`
- KVCache 的存储单位采用 **Page（页面）**，与 vLLM 的 PagedAttention 内存管理保持一致。每个 Page 对应固定大小的矩阵切片（例如 [page_size, head_dim]），便于在 GPU/CPU/存储之间高效迁移。

**2. 高效的 Key 命名规范**

```
kv:{req_id}:{layer_idx}:{head_idx}:{page_id}:{kv_type}
```

- req_id：请求 ID（区分不同推理会话）
- layer_idx：Transformer 层索引
- head_idx：注意力头索引
- page_id：页号（在同一 head 内的偏移）
- kv_type：K=0 / V=1

**示例**

```
kv:test_req_001:5:8:42:0
```

表示请求 `test_req_001` 的第 5 层、第 8 个头、第 42 页、Key 缓存。这种设计保证了 vLLM 能够快速定位对应的 Page。

**3. 紧凑的 Blob 格式**
 每个 KVCache blob 存储为 **二进制数据**，减少 Redis 协议和内存开销。Blob 由 Header + Body 组成：

- Header：元信息（dtype、page_size、head_dim、ttl 等）
- Body：tensor 原始二进制数据（fp16/fp32）

**4. 索引支持**
 Page 的 Key 本身就是索引，因此无需额外索引表。未来可在请求维度增加 `kv:{req_id}:index`，记录该请求所有 Page 的 key，方便批量清理。



### 1.2 命令接口

新增 **Page-oriented 命令族**：

1. `KVPAGESET`
    写入单个页面。

   ```
   KVPAGESET req_id layer_idx head_idx page_id kv_type dtype page_size head_dim ttl blob
   ```

2. `KVPAGEGET`
    获取单个页面。

   ```
   KVPAGEGET req_id layer_idx head_idx page_id kv_type
   ```

3. `KVPAGEMSET`
    批量写入多个页面。

   ```
   KVPAGEMSET N (req_id layer_idx head_idx page_id kv_type dtype page_size head_dim ttl blob) * N
   ```

4. `KVPAGEMGET`
    批量获取多个页面。

   ```
   KVPAGEMGET N (req_id layer_idx head_idx page_id kv_type) * N
   ```

5. `KVPAGEEXISTS`
    判断页面是否存在。

   ```
   KVPAGEEXISTS req_id layer_idx head_idx page_id kv_type
   ```

**使用示例**

```bash
redis-cli -p 9221 \
  KVPAGESET test_req_001 5 8 42 0 1 128 64 3600 "BLOB_DATA"

redis-cli -p 9221 \
  KVPAGEGET test_req_001 5 8 42 0
```



### 1.3 编译和测试

测试文件位于 `tests/` 目录：

- **test_page_builder.py**
   验证 Key 构建和 Blob 格式编码解码是否符合设计。
- **test_page_commands.py**
   验证命令族的正确性，包括：
  - 单页写入/读取（KVPAGESET/KVPAGEGET）
  - 批量写入/读取（KVPAGEMSET/KVPAGEMGET）
  - 存在性检查（KVPAGEEXISTS）
  - 性能压力测试（批量写/读耗时）

**编译 PikiwiDB**

```bash
make
./output/pika -c ./conf/pika.conf
```

**运行单元测试**

```bash
python3 -m unittest test_page_builder.py -v
python3 -m unittest test_page_commands.py -v
```

