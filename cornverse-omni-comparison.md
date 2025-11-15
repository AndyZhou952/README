## CORNSERVE Data Plane vs vllm-omni 执行流程深度对比

### 一、核心架构对比

**CORNSERVE Data Plane**：分布式微服务架构，Router/Engine分离，跨节点通信  
**vllm-omni**：单机多进程流水线，直接进程间通信，专注性能

---

### 二、代码位置与模型实现

#### CORNSERVE Data Plane（以 Eric 编码器为例）

**1. Router层（HTTP入口）**
```python
# python/cornserve/task_executors/eric/router/app.py:46-90
@router.post("/embeddings")
async def embeddings(request: EmbeddingRequest, ...) -> EmbeddingResponse:
    # 1. 预处理多模态数据（异步线程池）
    processor: Processor = raw_request.app.state.processor
    processed = await processor.process(request.data)
    
    # 2. 通过Engine Client发送到Engine进程
    engine_client: EngineClient = raw_request.app.state.engine_client
    response = await engine_client.embed(uuid.uuid4().hex, processed)
    return response
```

**2. Engine Client层（异步ZMQ通信）**
```python
# python/cornserve/task_executors/eric/engine/client.py:114-133
async def embed(self, request_id: str, processed: list[ProcessedEmbeddingData]):
    # 创建Future等待响应
    fut: Future[EmbeddingResponse] = self.loop.create_future()
    self.responses[request_id] = fut
    
    # 通过ZMQ发送到Engine进程
    await self.request_sock.send_multipart(
        (EngineOpcode.ENQUEUE.value, msg_bytes),
        copy=False,
    )
    return await fut  # 等待Engine响应
```

**3. Engine层（同步调度）**
```python
# python/cornserve/task_executors/eric/engine/core.py:181-233
def run(self):
    while True:
        # 1. 从ZMQ接收请求，放入队列
        req = self.request_queue.get(timeout=3.0)
        self._handle_client_request(*req)
        
        # 2. 调度批处理
        batch = self.scheduler.schedule()
        
        # 3. 执行模型推理
        batch_result = self.executor.execute_model(batch.to_worker_batch())
        
        # 4. 发送响应回Router
        self.response_queue.put_nowait(responses)
```

**4. Executor层（共享内存广播）**
```python
# python/cornserve/task_executors/eric/executor/executor.py:159-174
def execute_model(self, batch: WorkerBatch) -> BatchResult:
    # 通过共享内存MessageQueue广播到所有Workers
    self.run_workers("execute_model", kwargs={"batch": batch})
    # Workers并行执行，结果通过Sidecar发送
    return BatchResult(...)
```

**5. Worker层（GPU执行）**
```python
# python/cornserve/task_executors/eric/executor/worker.py:59-100
class Worker:
    def __init__(self, ...):
        # 1. 初始化分布式环境（Tensor Parallelism）
        init_distributed(world_size=tp_size, rank=tp_rank)
        
        # 2. 加载模型
        self.model = load_model(model_name_or_path=model_id, ...)
        
        # 3. 初始化Sidecar客户端（用于发送张量）
        self.sender_sidecar_client = Sidecar(...)
    
    def execute_model(self, batch: WorkerBatch):
        # 执行推理
        embeddings = self.model(batch.data)
        # 通过Sidecar发送到目标executor
        self.sender_sidecar_client.send(embeddings, ...)
```

**6. Sidecar（张量传输）**
```python
# python/cornserve/services/sidecar/sender.py:39-100
class SidecarSender:
    def __init__(self, config: SidecarSenderConfig):
        # 使用gRPC + 共享内存进行高效张量传输
        self.shm_manager = SharedMemoryManager(...)
        # 支持跨节点传输（UCX）
```

#### vllm-omni 执行流程

**1. Orchestrator层（主进程）**
```python
# vllm-omni/vllm_omni/entrypoints/omni_llm.py:147-309
class OmniLLM:
    def generate(self, prompts, sampling_params_list):
        # 1. 初始化所有Stage进程
        for stage_id, stage in enumerate(self.stage_list):
            stage.init_stage_worker(model, ...)
        
        # 2. 将请求提交到Stage 0
        for req_id, prompt in enumerate(request_prompts):
            self.stage_list[0].submit({
                "request_id": req_id,
                "engine_inputs": prompt,
                "sampling_params": sp0,
            })
        
        # 3. 轮询各Stage输出，流水线转发
        while completed_requests < total_requests:
            for stage_id, stage in enumerate(self.stage_list):
                result = stage.try_collect()  # 非阻塞获取
                if result:
                    # 转发到下一Stage
                    next_inputs = next_stage.process_engine_inputs(...)
                    self.stage_list[next_stage_id].submit(ipc_payload)
```

**2. Stage层（进程管理）**
```python
# vllm-omni/vllm_omni/entrypoints/omni_stage.py:87-113
def init_stage_worker(self, model: str, ...):
    # 创建进程，运行_stage_worker函数
    self._proc = ctx.Process(
        target=_stage_worker,
        args=(model, stage_payload, self._in_q, self._out_q, ...),
    )
    self._proc.start()
```

**3. Stage Worker（工作进程）**
```python
# vllm-omni/vllm_omni/entrypoints/omni_stage.py:177-396
def _stage_worker(model, stage_payload, in_q, out_q, ...):
    # 1. 设置GPU设备
    set_stage_gpu_devices(stage_id, runtime_cfg.get("devices"))
    
    # 2. 初始化vLLM引擎
    stage_engine = OmniStageLLM(model=model, **engine_args)
    
    # 3. 批处理循环
    while True:
        task = in_q.get()  # 从队列获取任务
        if task is None:
            break
        
        # 4. 批处理（支持max_batch_size > 1）
        batch_tasks = [task]
        while len(batch_tasks) < max_batch_size:
            if not in_q.empty():
                batch_tasks.append(in_q.get(timeout=batch_timeout))
        
        # 5. 解码IPC负载（可能来自共享内存）
        ein, _rx_metrics = maybe_load_from_ipc_with_metrics(
            t, obj_key="engine_inputs", shm_key="engine_inputs_shm"
        )
        
        # 6. 执行推理
        gen_outputs = []
        for ro in stage_engine.generate(batch_engine_inputs, sampling_params):
            gen_outputs.append(ro)
        
        # 7. 编码输出（可能使用共享内存）
        use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
        out_q.put({"request_id": rid, "engine_outputs": payload, ...})
```

**4. 阶段间数据转换**
```python
# vllm-omni/vllm_omni/model_executor/stage_input_processors/qwen2_5_omni.py:10-58
def thinker2talker(stage_list, engine_input_source, prompt):
    # 从上游Stage的engine_outputs提取隐藏状态
    thinker_outputs = stage_list[source_stage_id].engine_outputs
    thinker_hidden_states = output.multimodal_output["latent"].clone().detach().cuda()
    
    # 构造Talker输入（包含特殊token和隐藏状态）
    talker_inputs.append(OmniTokensPrompt(
        prompt_token_ids=[TALKER_CODEC_START_TOKEN_ID] + ...,
        additional_information={
            "thinker_result": thinker_hidden_states[...],
            "prompt_embeds": thinker_hidden_states[:prompt_token_ids_len],
        }
    ))
    return talker_inputs
```

### 三、执行流程对比

#### CORNSERVE 流程（Eric编码器为例）

```
请求流程：
┌─────────────────────────────────────────────────────────┐
│ 1. HTTP请求 → Router (FastAPI)                          │
│    python/cornserve/task_executors/eric/router/app.py    │
│    - 异步预处理多模态数据                                │
│    - 调用Engine Client                                   │
└──────────────────┬──────────────────────────────────────┘
                   │ async embed()
┌──────────────────▼──────────────────────────────────────┐
│ 2. Engine Client (异步ZMQ)                              │
│    python/cornserve/task_executors/eric/engine/client.py│
│    - ZMQ PUSH发送到Engine进程                           │
│    - Future等待响应                                      │
└──────────────────┬──────────────────────────────────────┘
                   │ ZMQ IPC
┌──────────────────▼──────────────────────────────────────┐
│ 3. Engine (同步进程)                                     │
│    python/cornserve/task_executors/eric/engine/core.py   │
│    - 从ZMQ接收请求                                       │
│    - 调度批处理                                          │
│    - 调用Executor                                        │
└──────────────────┬──────────────────────────────────────┘
                   │ execute_model()
┌──────────────────▼──────────────────────────────────────┐
│ 4. Executor (共享内存广播)                               │
│    python/cornserve/task_executors/eric/executor/executor.py│
│    - MessageQueue广播到所有Workers                      │
│    - 等待Workers完成                                     │
└──────────────────┬──────────────────────────────────────┘
                   │ 共享内存
┌──────────────────▼──────────────────────────────────────┐
│ 5. Workers (多进程，Tensor Parallelism)                  │
│    python/cornserve/task_executors/eric/executor/worker.py│
│    - 执行模型推理                                        │
│    - 通过Sidecar发送结果                                 │
└──────────────────┬──────────────────────────────────────┘
                   │ Sidecar (gRPC + 共享内存)
┌──────────────────▼──────────────────────────────────────┐
│ 6. Sidecar (跨节点张量传输)                              │
│    python/cornserve/services/sidecar/sender.py          │
│    - gRPC通信                                           │
│    - 共享内存/UCX传输                                    │
└─────────────────────────────────────────────────────────┘
```

#### vllm-omni 流程

```
请求流程：
┌─────────────────────────────────────────────────────────┐
│ 1. OmniLLM.generate() (主进程)                         │
│    vllm-omni/vllm_omni/entrypoints/omni_llm.py         │
│    - 提交请求到Stage 0队列                              │
│    - 轮询各Stage输出                                    │
│    - 流水线转发到下一Stage                              │
└──────────────────┬──────────────────────────────────────┘
                   │ submit() → Queue
┌──────────────────▼──────────────────────────────────────┐
│ 2. Stage 0 Worker Process (Thinker)                     │
│    vllm-omni/vllm_omni/entrypoints/omni_stage.py       │
│    - 从in_q获取任务                                      │
│    - 批处理（max_batch_size）                           │
│    - 解码IPC负载（可能来自共享内存）                     │
│    - OmniStageLLM.generate()执行推理                    │
│    - 编码输出到out_q（可能使用共享内存）                 │
└──────────────────┬──────────────────────────────────────┘
                   │ try_collect() → process_engine_inputs()
┌──────────────────▼──────────────────────────────────────┐
│ 3. 数据转换 (主进程)                                     │
│    thinker2talker()函数                                  │
│    - 提取隐藏状态                                        │
│    - 构造Talker输入                                      │
└──────────────────┬──────────────────────────────────────┘
                   │ submit() → Queue
┌──────────────────▼──────────────────────────────────────┐
│ 4. Stage 1 Worker Process (Talker)                      │
│    类似Stage 0，但处理Talker模型                         │
└──────────────────┬──────────────────────────────────────┘
                   │ try_collect()
┌──────────────────▼──────────────────────────────────────┐
│ 5. Stage 2 Worker Process (Code2Wav)                    │
│    类似Stage 1，但处理Codec模型                         │
└─────────────────────────────────────────────────────────┘
```

### 四、服务拉起方式对比

#### CORNSERVE

**1. Kubernetes部署**
```bash
# Task Manager创建Pod
# python/cornserve/services/task_manager/manager.py:319-442
pod = kclient.V1Pod(
    containers=[kclient.V1Container(
        image=self.descriptor.get_container_image(),  # Docker镜像
        args=container_args,  # 启动参数
        resources={"nvidia.com/gpu": len(gpus)},
    )],
)
await self.core_client.create_namespaced_pod(...)
```

**2. Eric启动流程**
```python
# python/cornserve/task_executors/eric/router/app.py:100-105
def create_app(config: EricConfig) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    init_app_state(app, config)  # 初始化Processor和EngineClient
    return app

# Engine Client自动启动Engine进程
# python/cornserve/task_executors/eric/engine/client.py:64-68
self.engine_proc = Engine.spawn_engine(
    config=config,
    request_sock_path=self.request_sock_path,
    response_sock_path=self.response_sock_path,
)
```

**3. Executor启动Workers**
```python
# python/cornserve/task_executors/eric/executor/executor.py:70-88
for tp_rank in range(tp_size):
    worker = Worker.spawn_worker(
        model_id=model_id,
        tp_rank=tp_rank,
        tp_size=tp_size,
        input_mq_handle=input_mq_handle,
        sender_sidecar_ranks=sender_sidecar_ranks,
    )
    self.workers.append(worker)
```

#### vllm-omni

**1. Python直接启动**
```python
# vllm-omni/examples/offline_inference/qwen_2_5_omni/end2end.py:172-180
omni_llm = OmniLLM(
    model="Qwen/Qwen2.5-Omni-7B",
    init_sleep_seconds=20,
    batch_timeout=5,
)

# 自动加载Stage配置
# vllm-omni/vllm_omni/entrypoints/omni_llm.py:67-80
if stage_configs is None:
    self.stage_configs = load_stage_configs_from_model(model)  # 从YAML加载

self._initialize_stages(model, ...)  # 启动所有Stage进程
```

**2. Stage进程启动**
```python
# vllm-omni/vllm_omni/entrypoints/omni_llm.py:109-127
def _start_stage_processes(self, model: str):
    for stage_id, stage in enumerate(self.stage_list):
        in_q = self._ctx.Queue(maxsize=0)
        out_q = self._ctx.Queue(maxsize=0)
        stage.attach_queues(in_q, out_q)
        stage.init_stage_worker(model, ...)  # 启动进程
        time.sleep(self._init_sleep_seconds)  # 等待初始化
```

**3. Worker进程初始化**
```python
# vllm-omni/vllm_omni/entrypoints/omni_stage.py:229-243
def _stage_worker(...):
    # 设置GPU设备
    set_stage_gpu_devices(stage_id, runtime_cfg.get("devices"))
    
    # 初始化vLLM引擎
    stage_engine = OmniStageLLM(model=model, **engine_args)
    
    # 发送就绪信号
    out_q.put({"type": "stage_ready", "stage_id": stage_id})
    
    # 进入批处理循环
    while True:
        task = in_q.get()
        # ... 处理任务
```

### 五、性能深度对比

#### 通信开销对比

**CORNSERVE：**
```
HTTP请求 → ZMQ序列化 → 共享内存广播 → Sidecar gRPC → 共享内存
开销：HTTP解析 + ZMQ序列化 + 共享内存拷贝 + gRPC序列化 + 网络传输
延迟：~5-10ms（本地）到~50-100ms（跨节点）
```

**vllm-omni：**
```
Queue.put() → 共享内存（>65KB）或直接序列化 → Queue.get()
开销：进程间队列 + 共享内存拷贝（大对象）或pickle序列化（小对象）
延迟：~0.5-2ms（同机进程间）
```

#### 流水线并行效率

**CORNSERVE（无流水线）：**
```
请求1: [Thinker Embedding] → [Talker Vocoder] = 450ms
请求2:                    [Thinker Embedding] → [Talker Vocoder] = 450ms
总时间：900ms（串行）
```

**vllm-omni（有流水线）：**
```
时间轴：
0ms:    [Req1 Thinker开始]
200ms:  [Req1 Thinker完成] → [Req1 Talker开始]
        [Req2 Thinker开始]（并发！）
350ms:  [Req1 Talker完成] → [Req1 Codec开始]
        [Req2 Thinker完成] → [Req2 Talker开始]
        [Req3 Thinker开始]（并发！）
总时间：~550ms（流水线并行，44%更快）
```

#### 批处理能力

**CORNSERVE：**
```python
# python/cornserve/task_executors/eric/engine/scheduler.py
# 支持批处理，但受限于Scheduler配置
max_batch_size = config.server.max_batch_size  # 通常较小（1-4）
```

**vllm-omni：**
```python
# vllm-omni/vllm_omni/entrypoints/omni_stage.py:253-264
max_batch_size = int(runtime_cfg.get("max_batch_size", 1) or 1)
# 可配置，Talker stage支持max_batch_size=3
# 但当前实现：window=-1，阶段内串行
```

#### 性能测试场景 - vllm-omni更优（得益于流水线并行）

### 六、协作空间

**1. vllm-omni可借鉴CORNSERVE的技术**

**Sidecar张量传输优化：**
```python
# CORNSERVE的Sidecar使用gRPC + 共享内存 + UCX
# vllm-omni当前只有基础共享内存
# 可借鉴：UCX跨节点传输、gRPC流式传输
```

**批处理调度策略：**
```python
# CORNSERVE的Scheduler有更成熟的批处理逻辑
# vllm-omni当前批处理较简单
# 可借鉴：动态批处理、优先级调度
```

**2. CORNSERVE可借鉴vllm-omni的技术**

**流水线并行：**
```python
# vllm-omni的流水线并行可提升吞吐
# CORNSERVE当前是串行执行
# 可借鉴：Stage间流水线，提升多阶段模型性能
```

**阶段间数据转换模式：**
```python
# vllm-omni的custom_process_input_func设计清晰
# CORNSERVE当前通过DataForward，可借鉴模块化转换函数
```

**3. 可能的集成方案**

**方案A：vllm-omni作为CORNSERVE的Executor**
```python
# CORNSERVE可以支持vllm-omni作为Task Executor
# 优势：结合CORNSERVE的分布式能力 + vllm-omni的性能优势
# 实现：创建VLLMOmniDescriptor，类似VLLMDescriptor
```

**方案B：共享Sidecar机制**
```python
# vllm-omni可以集成CORNSERVE的Sidecar
# 优势：跨节点传输能力
# 实现：vllm-omni的Stage间通信使用Sidecar（可选）
```

**方案C：统一配置接口**
```python
# 两者都支持YAML配置
# 可以定义统一的Stage配置格式
# 优势：用户可以在两个框架间迁移
```

### 七、总结

| 维度 | CORNSERVE Data Plane | vllm-omni |
|------|---------------------|-----------|
| **架构** | Router/Engine分离，ZMQ通信 | 直接进程间Queue |
| **通信** | HTTP → ZMQ → 共享内存 → Sidecar | Queue → 共享内存 |
| **开销** | 高（多层序列化） | 低（直接进程通信） |
| **流水线** | 无（串行执行） | 有（阶段并发） |
| **批处理** | 支持（Scheduler） | 支持（max_batch_size） |
| **性能** | 良好（分布式） | 优秀（单机流水线） |
| **适用场景** | 生产多应用 | 研究/批量推理 |

**结论：**
- 单机批量推理：vllm-omni性能更优（流水线并行，低开销）
- 生产多应用：CORNSERVE更合适（分布式，高可用）
- 协作空间：vllm-omni可借鉴Sidecar优化，CORNSERVE可借鉴流水线并行

两者互补：vllm-omni专注性能优化，CORNSERVE专注生产能力。



# 三个协作方案的详细设计计划：

## 总结对比 （细节见下）

| 方案 | 复杂度 | 价值 | 实施时间 | 维护成本 |
|------|--------|------|----------|----------|
| **方案A** | 高 | 高 | 4-5周 | 中 |
| **方案B** | 中 | 中 | 2-3周 | 低 |

### 推荐优先级

1. 方案B（共享Sidecar）：实施快，价值明确，风险低
2. 方案A（Executor集成）：价值高但复杂度高，可作为长期目标

## 方案A：vllm-omni作为CORNSERVE的Executor

### 一、规划

目标：让vllm-omni可以作为CORNSERVE的一个Task Executor，在K8s中部署，享受CORNSERVE的分布式能力，同时保留流水线并行性能。

价值：
- CORNSERVE：获得高性能多阶段模型执行能力
- vllm-omni：获得分布式部署和资源管理能力

### 二、设计

#### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│         CORNSERVE Control Plane                         │
│  Task Manager → 创建Pod → vllm-omni Executor Pod      │
└─────────────────────────────────────────────────────────┘
                    ↓ HTTP API
┌─────────────────────────────────────────────────────────┐
│      vllm-omni Executor Pod (K8s Container)             │
│  ┌──────────────────────────────────────────────┐      │
│  │  HTTP Server (FastAPI)                       │      │
│  │  - /v1/chat/completions                       │      │
│  │  - /health                                    │      │
│  └──────────────┬───────────────────────────────┘      │
│                 │                                       │
│  ┌──────────────▼───────────────────────────────┐      │
│  │  OmniLLM Wrapper                             │      │
│  │  - 包装vllm-omni的OmniLLM                    │      │
│  │  - 处理HTTP请求 → OmniLLM.generate()         │      │
│  │  - 转换响应格式                               │      │
│  └──────────────────────────────────────────────┘      │
│                 │                                       │
│  ┌──────────────▼───────────────────────────────┐      │
│  │  vllm-omni Core                               │      │
│  │  - Stage 0 (Thinker) Process                 │      │
│  │  - Stage 1 (Talker) Process                  │      │
│  │  - Stage 2 (Codec) Process                   │      │
│  └──────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────┘
```

#### 代码结构设计

在vllm-omni仓库中新增：

```
vllm-omni/
├── vllm_omni/
│   ├── entrypoints/
│   │   ├── omni_llm.py          # 现有
│   │   └── cornserve_wrapper.py # 新增：CORNSERVE包装器
│   └── server/
│       └── cornserve_server.py  # 新增：HTTP服务器
└── examples/
    └── cornserve/
        └── Dockerfile           # 新增：CORNSERVE镜像
```

### 三、实现方式：在vllm-omni仓库集成

原因：
- vllm-omni需要提供HTTP接口
- 需要适配CORNSERVE的请求/响应格式
- 保持vllm-omni的独立性，不依赖CORNSERVE代码

#### 步骤1：创建HTTP服务器包装器

```python
# vllm-omni/vllm_omni/server/cornserve_server.py
"""
HTTP server wrapper for vllm-omni to work as CORNSERVE executor.
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import json

from vllm_omni.entrypoints.omni_llm import OmniLLM
from vllm_omni.entrypoints.utils import load_stage_configs_from_model
from vllm.sampling_params import SamplingParams

app = FastAPI()

# 全局OmniLLM实例
omni_llm: Optional[OmniLLM] = None

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[dict]
    stream: bool = True
    temperature: float = 0.7
    max_tokens: int = 2048
    # CORNSERVE特定字段
    cornserve_hidden_states_recv_id: Optional[str] = None
    cornserve_return_audio: bool = False

@app.on_event("startup")
async def startup():
    """Initialize OmniLLM on startup."""
    global omni_llm
    model = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Omni-7B")
    omni_llm = OmniLLM(
        model=model,
        init_sleep_seconds=int(os.getenv("INIT_SLEEP_SECONDS", "20")),
        batch_timeout=int(os.getenv("BATCH_TIMEOUT", "10")),
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if omni_llm is None:
        raise HTTPException(status_code=503, detail="Not initialized")
    return {"status": "healthy"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    if omni_llm is None:
        raise HTTPException(status_code=503, detail="Not initialized")
    
    # 转换请求格式
    prompts = [convert_messages_to_prompt(request.messages)]
    
    # 创建采样参数（每个Stage一个）
    sampling_params_list = [
        SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    ] * len(omni_llm.stage_list)
    
    # 调用vllm-omni
    outputs = omni_llm.generate(prompts, sampling_params_list)
    
    # 转换响应格式
    if request.stream:
        return StreamingResponse(
            generate_stream_response(outputs),
            media_type="text/event-stream"
        )
    else:
        return convert_outputs_to_response(outputs)

def convert_messages_to_prompt(messages: List[dict]) -> dict:
    """Convert OpenAI messages to vllm-omni prompt format."""
    # 实现消息格式转换
    pass

def generate_stream_response(outputs):
    """Generate streaming response."""
    for output in outputs:
        if output.final_output_type == "audio":
            # 处理音频输出
            chunk = {
                "choices": [{
                    "delta": {
                        "audio": {
                            "data": base64_encode_audio(output.request_output)
                        }
                    }
                }]
            }
        else:
            # 处理文本输出
            chunk = {
                "choices": [{
                    "delta": {"content": output.request_output.outputs[0].text}
                }]
            }
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"
```

#### 步骤2：创建CORNSERVE Descriptor（在CORNSERVE仓库）

```python
# tasklib/cornserve_tasklib/task_executors/descriptor/vllm_omni.py
"""
Task execution descriptor for vllm-omni executor.
"""
from typing import Any
import aiohttp
from cornserve import constants
from cornserve.services.resource import GPU
from cornserve.task.base import Stream
from cornserve.task_executors.descriptor.base import TaskExecutionDescriptor

from cornserve_tasklib.task.unit.llm import (
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionRequest,
)

class VLLMOmniDescriptor(
    TaskExecutionDescriptor[
        LLMBaseUnitTask,  # 或其他合适的Task类型
        OpenAIChatCompletionRequest,
        Stream[OpenAIChatCompletionChunk]
    ]
):
    """Descriptor for vllm-omni executor."""
    
    def create_executor_name(self) -> str:
        return f"vllm-omni-{self.task.model_id.split('/')[-1]}"
    
    def get_container_image(self) -> str:
        return constants.CONTAINER_IMAGE_VLLM_OMNI  # 需要定义
    
    def get_container_args(self, gpus: list[GPU], port: int) -> list[str]:
        return [
            "--model", self.task.model_id,
            "--port", str(port),
            "--host", "0.0.0.0",
            # GPU配置通过环境变量传递
        ]
    
    def get_api_url(self, base: str) -> str:
        return f"{base}/v1/chat/completions"
    
    def to_request(
        self,
        task_input: OpenAIChatCompletionRequest,
        task_output: Stream[OpenAIChatCompletionChunk],
    ) -> dict[str, Any]:
        """Convert to vllm-omni request format."""
        return {
            "model": self.task.model_id,
            "messages": [msg.model_dump() for msg in task_input.messages],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 2048,
        }
    
    async def from_response(
        self,
        task_output: Stream[OpenAIChatCompletionChunk],
        response: aiohttp.ClientResponse,
    ) -> Stream[OpenAIChatCompletionChunk]:
        """Parse streaming response."""
        # 复用现有的parse_stream_to_completion_chunks
        from cornserve_tasklib.task_executors.descriptor.llm import (
            parse_stream_to_completion_chunks
        )
        return Stream[OpenAIChatCompletionChunk](
            async_iterator=parse_stream_to_completion_chunks(response),
            response=response,
        )
```

#### 步骤3：创建Docker镜像

```dockerfile
# vllm-omni/examples/cornserve/Dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 安装Python和依赖
RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip install uv
RUN uv pip install vllm==0.11.0
RUN uv pip install fastapi uvicorn

# 复制vllm-omni代码
COPY vllm_omni/ /app/vllm_omni/
COPY pyproject.toml /app/

# 安装vllm-omni
WORKDIR /app
RUN uv pip install -e .

# 启动HTTP服务器
CMD ["python", "-m", "vllm_omni.server.cornserve_server"]
```

### 四、集成步骤

#### Phase 1：vllm-omni侧

1. HTTP服务器
   - 实现`cornserve_server.py`
   - 支持OpenAI兼容API
   - 实现健康检查
   - 单元测试

2. 请求/响应转换
   - 实现消息格式转换
   - 实现流式响应生成
   - 处理多阶段输出
   - 集成测试

3. Docker镜像和文档
   - 创建Dockerfile
   - 编写使用文档
   - 示例代码

#### Phase 2：CORNSERVE侧

1. Descriptor实现
   - 实现`VLLMOmniDescriptor`
   - 注册到Descriptor Registry
   - 单元测试

2. 集成测试
   - 端到端测试
   - 性能对比测试
   - 文档更新

### 五、价值和协作方式

   - 对CORNSERVE：高性能多阶段模型执行，流水线并行提升吞吐
   - 对vllm-omni：获得分布式部署能力，扩大使用场景

   - vllm-omni提供HTTP接口，保持独立性
   - CORNSERVE通过Descriptor集成，无需修改核心代码
   - 向后兼容，不影响现有功能

   - vllm-omni先实现HTTP服务器
   - CORNSERVE实现Descriptor
   - 联合测试和文档

   - vllm-omni维护HTTP服务器和Docker镜像
   - CORNSERVE维护Descriptor代码
   - 共同维护集成文档


## 方案B：共享Sidecar机制

### 一、规划

目标：让vllm-omni可以使用CORNSERVE的Sidecar进行跨节点张量传输，同时保持vllm-omni的轻量级架构。

价值：
- vllm-omni：获得跨节点传输能力
- CORNSERVE：Sidecar被更多项目使用，提升影响力

### 二、设计

#### 架构设计

```
┌─────────────────────────────────────────────────────────┐
│  vllm-omni Stage Process                                │
│  ┌──────────────────────────────────────────────┐      │
│  │  Stage Worker                                │      │
│  │  - 执行推理                                  │      │
│  │  - 生成张量                                  │      │
│  └──────────────┬───────────────────────────────┘      │
│                 │                                       │
│  ┌──────────────▼───────────────────────────────┐      │
│  │  Sidecar Client (可选)                       │      │
│  │  - 如果配置了sidecar_ranks，使用Sidecar      │      │
│  │  - 否则使用Queue/IPC（默认）                 │      │
│  └──────────────┬───────────────────────────────┘      │
│                 │                                       │
│         ┌───────┴────────┐                            │
│         │                │                            │
│    Sidecar          Queue/IPC                          │
│  (跨节点)         (同机进程)                           │
└─────────────────────────────────────────────────────────┘
```

#### 代码结构设计

在vllm-omni仓库中新增：

```
vllm-omni/
├── vllm_omni/
│   ├── sidecar/                    # 新增：Sidecar客户端
│   │   ├── __init__.py
│   │   ├── client.py              # Sidecar客户端实现
│   │   └── config.py              # Sidecar配置
│   └── entrypoints/
│       └── omni_stage.py          # 修改：支持Sidecar
```

### 三、实现方式：在vllm-omni仓库集成CORNSERVE Sidecar客户端

原因：
- vllm-omni需要调用Sidecar，但不依赖CORNSERVE完整代码
- 通过gRPC调用Sidecar服务，解耦
- Sidecar是独立服务，可单独使用

#### 步骤1：创建Sidecar客户端（在vllm-omni）

```python
# vllm-omni/vllm_omni/sidecar/client.py
"""
Sidecar client for vllm-omni to use CORNSERVE Sidecar for tensor transfer.
This is a lightweight client that only depends on gRPC and protobuf.
"""
import grpc
import torch
from typing import Optional, List
from vllm_omni.sidecar.proto import sidecar_pb2, sidecar_pb2_grpc

class SidecarClient:
    """Client for CORNSERVE Sidecar service."""
    
    def __init__(
        self,
        sidecar_rank: int,
        group: List[int],
        grpc_url: Optional[str] = None,
    ):
        self.sidecar_rank = sidecar_rank
        self.group = group
        
        # 构建gRPC URL（默认使用CORNSERVE约定）
        if grpc_url is None:
            from vllm_omni.sidecar.config import grpc_url_from_rank
            grpc_url = grpc_url_from_rank(sidecar_rank)
        
        self.channel = grpc.insecure_channel(grpc_url)
        self.stub = sidecar_pb2_grpc.SidecarStub(self.channel)
        
        # 注册到Sidecar
        self._register()
    
    def _register(self):
        """Register with Sidecar server."""
        request = sidecar_pb2.RegisterRequest(
            rank=self.sidecar_rank,
            group=self.group,
            dtype="float32",  # 根据实际需求调整
            send_slot_numel=1024 * 1024,  # 根据实际需求调整
            recv_slot_numel=1024 * 1024,
            concurrent_copy=True,
        )
        response = self.stub.Register(request)
        if response.status != 0:  # STATUS_OK
            raise RuntimeError(f"Failed to register with Sidecar: {response}")
    
    def send_tensor(
        self,
        tensor: torch.Tensor,
        dst_ranks: List[int],
        data_id: str,
        chunk_id: int = 0,
        num_chunks: int = 1,
    ):
        """Send tensor to destination ranks via Sidecar."""
        # 序列化张量
        tensor_bytes = tensor.cpu().numpy().tobytes()
        
        # 构建请求
        request = sidecar_pb2.SendRequest(
            id=data_id,
            dst_ranks=[sidecar_pb2.RankGroup(ranks=dst_ranks)],
            shard_rank=0,  # TP rank
            data=tensor_bytes,
            chunk_id=chunk_id,
            num_chunks=num_chunks,
        )
        
        response = self.stub.Send(request)
        if response.status != 0:
            raise RuntimeError(f"Failed to send tensor: {response}")
    
    def receive_tensor(
        self,
        data_id: str,
        chunk_id: int = 0,
    ) -> torch.Tensor:
        """Receive tensor from Sidecar."""
        request = sidecar_pb2.ReceiveRequest(
            id=data_id,
            chunk_id=chunk_id,
        )
        response = self.stub.Receive(request)
        if response.status != 0:
            raise RuntimeError(f"Failed to receive tensor: {response}")
        
        # 反序列化张量（需要知道shape和dtype）
        # 这里简化，实际需要从metadata获取
        tensor = torch.frombuffer(response.data, dtype=torch.float32)
        return tensor
```

#### 步骤2：修改Stage Worker支持Sidecar（可选）

```python
# vllm-omni/vllm_omni/entrypoints/omni_stage.py
# 修改_stage_worker函数

def _stage_worker(...):
    # ... 现有代码 ...
    
    # 可选：初始化Sidecar客户端
    sidecar_client = None
    sidecar_ranks = runtime_cfg.get("sidecar_ranks")
    if sidecar_ranks:
        from vllm_omni.sidecar.client import SidecarClient
        sidecar_client = SidecarClient(
            sidecar_rank=sidecar_ranks[stage_id],
            group=sidecar_ranks,
        )
    
    # 在发送输出时，如果配置了Sidecar，使用Sidecar
    for rid in batch_request_ids:
        r_outputs = req_to_outputs.get(rid, [])
        
        # 提取张量（隐藏状态）
        if hasattr(r_outputs[0], 'multimodal_output'):
            tensor = r_outputs[0].multimodal_output.get("latent")
            
            if sidecar_client and tensor is not None:
                # 使用Sidecar发送（跨节点场景）
                sidecar_client.send_tensor(
                    tensor=tensor,
                    dst_ranks=[next_stage_sidecar_rank],
                    data_id=f"stage_{stage_id}_req_{rid}",
                )
                # 在payload中标记使用Sidecar
                payload = {"sidecar_id": f"stage_{stage_id}_req_{rid}"}
            else:
                # 使用Queue/IPC（同机场景）
                use_shm, payload = maybe_dump_to_shm(r_outputs, shm_threshold_bytes)
        
        out_q.put({"request_id": rid, "engine_outputs": payload, ...})
```

#### 步骤3：添加Sidecar配置支持

```python
# vllm-omni/vllm_omni/model_executor/stage_configs/qwen2_5_omni.yaml
stage_args:
  - stage_id: 0
    runtime:
      process: true
      devices: "0"
      max_batch_size: 1
      # 新增：Sidecar配置（可选）
      sidecar_ranks: [0, 1, 2]  # 如果配置，使用Sidecar
      # 如果不配置，使用Queue/IPC（默认）
    # ... 其他配置
```

### 四、集成步骤

#### Phase 1：Sidecar客户端实现

1. 客户端基础
   - 实现gRPC客户端
   - 实现Register/Send/Receive
   - 单元测试

2. 集成和测试
   - 集成到Stage Worker
   - 端到端测试
   - 性能对比

#### Phase 2：文档和示例（1周）

1. 编写使用文档
2. 提供配置示例
3. 性能测试报告

### 五、协作价值

1. 价值主张
   - 对CORNSERVE：Sidecar被更多项目使用，提升影响力
   - 对vllm-omni：获得跨节点传输能力，扩展使用场景

2. 技术方案
   - vllm-omni通过gRPC调用Sidecar，不依赖CORNSERVE代码
   - Sidecar是独立服务，可单独使用
   - 向后兼容，默认使用Queue/IPC

3. 实施计划
   - vllm-omni实现Sidecar客户端
   - 联合测试
   - 文档更新

4. 维护责任
   - CORNSERVE维护Sidecar服务
   - vllm-omni维护客户端代码
   - 共同维护协议文档




