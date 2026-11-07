# Inference Engine: Architecture

A small, single-GPU LLM inference engine in the spirit of vLLM. It is meant to be read, profiled and pulled apart, and it gives custom GPU kernels a real workload to show their worth.

- **Scope:** single node, single GPU (tensor parallelism is a stretch goal), decoder-only Llama/Qwen2-family models, offline batch generation plus an interactive CLI. There is no HTTP server, no multi-modal support and no LoRA.
- **Stack:** Python + PyTorch for orchestration, Triton for most custom kernels, and CUDA C++ (via a `torch.utils.cpp_extension` module) for one or two kernels where hand-tuning pays off.
- **Reference models:** `Qwen2.5-0.5B-Instruct` for development (ungated, fast to iterate on), and `Qwen2.5-1.5B-Instruct` / `Llama-3.2-1B-Instruct` for benchmarks.

---

## 0. Framing: vLLM vs SGLang, and where kernels fit

A few corrections to the starting assumptions:

- **vLLM is not the "small-scale" engine and SGLang is not the "large-scale" one.** Both are production serving engines that run on multi-GPU, multi-node clusters. They differ in emphasis, not scale:
  - **vLLM** is known for PagedAttention (block-based KV-cache memory management), continuous batching, and wide model and hardware coverage. Its v1 engine uses a unified token-budget scheduler and hash-based prefix caching.
  - **SGLang** is known for RadixAttention (a radix tree of cached KV prefixes, very strong for multi-turn and agentic workloads that share prefixes), a frontend language for structured generation, a low-overhead overlapped scheduler, and large-scale deployment features such as prefill/decode disaggregation and expert parallelism.
- **Neither engine is "the kernel one."** Both lean mostly on shared kernel libraries (FlashAttention, FlashInfer, Triton, CUTLASS) and add their own custom kernels on top. A from-scratch engine lets you write every hot kernel yourself, and either design would let you do that.
- **Building a vLLM-shaped engine is still the right call.** Its core ideas (paged KV cache, continuous batching, CUDA graphs) form a clean, well-documented skeleton with obvious spots for custom kernels. Prefix caching is included below in vLLM's hash-based form, which is simpler than SGLang's radix tree. The closest open-source relatives are `nano-vllm` and `mini-sglang`.
- **Kernels alone won't move end-to-end numbers on small models.** For a 0.5B–1.5B model, PyTorch eager decode is **CPU/launch-overhead bound**, not GPU bound (see §1). Your kernel wins only show up after CUDA graphs and a lean scheduler remove that overhead. That is why the plan builds the engine skeleton first and then optimizes kernels against a measured roofline.
- **Hardware note:** you are on macOS. Triton and CUDA kernels need an NVIDIA GPU, so plan on a cloud GPU (an L4, A10G, A100, H100 or RTX 4090 are all fine). Phases 1–3 of the plan (the pure-PyTorch reference engine) can be written and tested on a Mac with CPU or MPS.

---

## 1. Performance model: what actually costs time

Every design decision below is justified against these numbers. Example: Qwen2.5-0.5B in bf16 on an A100 (≈2.0 TB/s HBM, ≈312 TFLOP/s bf16).

| Quantity | Value | Consequence |
|---|---|---|
| Weights | ≈1.0 GB | Every decode step must stream all weights at least once, so the floor is ≈0.5 ms/step at any batch size. |
| KV cache per token | 2 (K,V) × 24 layers × 2 KV heads × 64 dim × 2 B = **12 KB** | 40 GB of free HBM holds ≈3.4M tokens. Memory is not the bottleneck for small models, but fragmentation still is (see §5). |
| Decode GEMM arithmetic intensity | ≈ B FLOPs/byte (B = batch tokens) | The A100 ridge point is ≈156 FLOPs/byte, so decode is **memory-bound until B ≈ 150**. Batching is nearly free throughput. |
| Decode attention intensity | ≈ GQA group size (14/2 = 7) | Always memory-bound, so the kernel must read each KV byte exactly once and use it for every query head in the group. |
| Kernel launches per step (eager) | 24 layers × ~10–20 ops | ≈300–500 launches × 5–20 µs of Python/launch overhead ≈ **3–10 ms**, which is 6–20× the GPU floor. CUDA graphs are mandatory. |
| Prefill | Compute-bound (thousands of tokens per GEMM) | Optimize for FLOP/s: large tiles, tensor cores, no padding waste. |

Two rules follow:
1. **Decode:** minimize bytes moved and launches. Fuse elementwise ops, read the KV cache once, capture with CUDA graphs, and batch aggressively.
2. **Prefill:** minimize wasted FLOPs. Use ragged batches with no padding, compute logits only for the last token, and run flash-style attention.

Every kernel benchmark reports **achieved bandwidth or FLOP/s as a percentage of the hardware peak**, not just "x faster than PyTorch". This is how you demonstrate kernel skill credibly.

---

## 2. Component map

```
                ┌──────────────────────────────────────────────────────────┐
  CLI ────────► │ LLMEngine                                                │
 (generate /    │  ┌───────────┐  ┌───────────┐  ┌──────────────────────┐   │
  chat / bench) │  │ Tokenizer │  │ Scheduler │◄►│ BlockManager         │   │
                │  │ +Detok    │  │           │  │ (paged KV + prefix $)│   │
                │  └───────────┘  └─────┬─────┘  └──────────────────────┘   │
                │                       │ SchedulerOutput                   │
                │                ┌──────▼───────┐                           │
                │                │ ModelRunner  │── CUDA graphs (decode)    │
                │                │ (input prep, │── KV cache tensors        │
                │                │  forward,    │                           │
                │                │  sampling)   │                           │
                │                └──────┬───────┘                           │
                │          ┌────────────▼─────────────┐                     │
                │          │ Model (Qwen2/Llama)      │                     │
                │          │  layers → ops registry   │──► kernels/ (Triton,│
                │          └──────────────────────────┘     CUDA, torch ref)│
                └──────────────────────────────────────────────────────────┘
```

Proposed layout:

```
engine/
  cli.py                  # generate | chat | bench
  config.py               # EngineConfig, ModelConfig, SamplingParams
  llm_engine.py           # top-level step loop
  sequence.py             # Sequence / request state machine
  scheduler.py            # continuous batching + chunked prefill + preemption
  block_manager.py        # paged KV allocator, ref counts, prefix-cache hashing
  model_runner.py         # input prep, KV alloc, CUDA graphs, forward, sample
  sampler.py
  tokenizer.py            # tokenizer wrapper + incremental detokenizer
  models/
    qwen2.py, llama.py    # model definitions built from layers/
    loader.py             # safetensors loading + weight fusion
  layers/                 # Linear, RMSNorm, RotaryEmbedding, Attention, MLP, Embedding/LMHead
  ops/                    # op registry: each op has ref/triton/cuda backends
  kernels/
    triton/               # rmsnorm, rope, kv_store, silu_mul, paged_decode, prefill_attn, sampling
    cuda/                 # hand-written CUDA (paged decode v2, quant GEMM)
  metrics.py              # TTFT / TPOT / throughput / MBU accounting
tests/                    # unit (per op), integration (vs HF), e2e
bench/                    # per-kernel microbenchmarks + end-to-end benchmarks
docs/
```

---

## 3. Components

### 3.1 CLI / entry points (`cli.py`)

**What it does.** This is the only user interface. It has three subcommands:
- `generate --model M --prompts prompts.jsonl --max-tokens N [--temperature ...]` runs offline batch generation and writes JSONL output.
- `chat --model M` runs an interactive REPL with streamed tokens and a chat template.
- `bench --model M --num-prompts 512 --input-len 512 --output-len 128 [--backend ref|triton|cuda]` reports throughput, TTFT and TPOT.

The Python API (`LLM(model).generate(prompts, sampling_params)`) is what the CLI calls, so tests and notebooks can use the engine directly.

**Why it's efficient.** `generate` submits *all* prompts up front so the scheduler can batch them. It does not loop over prompts one at a time.

### 3.2 Config (`config.py`)

`ModelConfig` is parsed from HF `config.json`: layers, hidden size, heads, KV heads, head dim, rope theta, vocab, tied embeddings and QKV bias.

`EngineConfig` holds `block_size=16`, `gpu_memory_utilization=0.9`, `max_num_seqs=256`, `max_num_batched_tokens=8192`, `enable_prefix_caching`, `enforce_eager`, `kernel_backend`, and `cuda_graph_batch_sizes`.

`SamplingParams` holds `temperature`, `top_k`, `top_p`, `max_tokens`, `stop_token_ids`, `ignore_eos` and `seed`.

### 3.3 Weight loader (`models/loader.py`)

**What it does.** Reads `*.safetensors` from a local HF snapshot (downloaded with `huggingface_hub`) and maps HF parameter names onto engine modules.

**Efficiency design.**
- **Weight fusion at load time:** `q_proj`/`k_proj`/`v_proj` are concatenated into one `qkv_proj`, and `gate_proj`/`up_proj` into one `gate_up_proj`. That turns 5 GEMM launches per layer into 2 and gives each GEMM a wider N, which improves tensor-core utilization.
- Tensors are loaded with `safetensors.safe_open` (memory-mapped) straight to the GPU in the target dtype, which avoids a full CPU copy of the model.
- Tied embeddings (`lm_head` shares `embed_tokens`) are handled without duplicating memory.

### 3.4 Tokenizer and detokenizer (`tokenizer.py`)

**What it does.** Wraps HF `tokenizers`/`AutoTokenizer`, applies chat templates, and supplies EOS/stop IDs.

**Efficiency design.**
- **Incremental detokenization:** each sequence keeps `(prefix_offset, read_offset)` and decodes only the newest few tokens per step. Text is emitted only when it no longer ends in an incomplete UTF-8 sequence. This is O(1) per token rather than re-decoding the whole output (which is O(n²) over a generation).
- Tokenization of a batch of prompts uses the fast Rust tokenizer's batch API.

### 3.5 Sequence / request state (`sequence.py`)

**What it does.** A `Sequence` holds `seq_id`, `token_ids` (prompt + generated), `num_prompt_tokens`, `num_computed_tokens` (how many tokens already have KV in cache), `block_table` (a list of physical block IDs), `status ∈ {WAITING, RUNNING, PREEMPTED, FINISHED}`, its sampling params and detok state.

**Key idea.** `num_computed_tokens` is the single source of truth that unifies prefill, chunked prefill, decode, prefix-cache hits and recompute after preemption. Each step, the scheduler decides how many *new* tokens to compute for each sequence (`num_new_tokens`). A full prefill, a prefill chunk and a decode step are all the same operation with different sizes. This is vLLM v1's model, and it removes a whole class of special cases.

### 3.6 Block manager / KV-cache manager (`block_manager.py`)

**What it does.** Manages a fixed pool of `num_blocks` physical KV blocks, each holding `block_size` (16) tokens for every layer. It allocates blocks as sequences grow, frees them when sequences finish or are preempted, and implements prefix caching.

**Efficiency design.**
- **Paging (the PagedAttention idea):** KV memory is allocated on demand in 16-token blocks instead of reserving `max_seq_len` per request. Waste is bounded by fewer than 16 tokens per sequence instead of up to `max_len − actual_len`, which is what allows hundreds of concurrent sequences.
- **O(1) allocation:** a free list (a deque) of block IDs. There is no Python-side compaction and nothing ever moves on the GPU.
- **Prefix caching (hash-based, vLLM v1 style):**
  - Each *full* block gets `hash = H(parent_block_hash, tuple(tokens_in_block))`. The chained hash means a block hash identifies the whole prefix up to that block.
  - `hash → block_id` lives in a dict. On admission, the manager walks the prompt's full blocks and reuses cached blocks (incrementing `ref_count`) until the first miss, then advances `num_computed_tokens` past the hit.
  - When `ref_count` drops to 0, the block goes to the free list **but keeps its hash and contents**. It is evicted lazily only when reallocated. Because the free list is FIFO, eviction approximates LRU.
  - Shared system prompts and multi-turn chat get prefill almost for free.
- **Capacity sizing:** at startup the model runner runs a dummy forward with `max_num_batched_tokens` to measure peak activation memory. It then sets `num_blocks = (gpu_memory_utilization × total − weights − peak_activations − cuda_graph_pool) / bytes_per_block`.

**KV tensor layout.** One tensor per layer for K and one for V, each shaped `[num_blocks, num_kv_heads, block_size, head_dim]` ("HND").
- In HND, one (block, head) tile is `16 × 64 × 2 B = 2 KB` contiguous, so the decode kernel's loads are fully coalesced and vectorizable.
- The cost is that the per-token write in `store_kvcache` is strided. That is a single small scatter per token, so it's an acceptable trade.
- FlashInfer supports HND, so you can use it as a speed and correctness reference.
- Keep the layout behind one helper so you can benchmark NHD against HND later.

### 3.7 Scheduler (`scheduler.py`)

**What it does.** Each engine step, it decides which sequences run and how many tokens each computes. It runs **continuous batching** at the iteration level: sequences join and leave the batch every step, so the GPU never waits for the slowest request in a batch.

**Algorithm (per step):**
```
budget = max_num_batched_tokens
1. For each RUNNING seq (oldest first):
     need = 1 (decode) or min(remaining_prompt, budget) (unfinished chunked prefill)
     ensure blocks for num_computed_tokens + need
       └─ if no free block: preempt the NEWEST running seq
          (free its blocks, mark PREEMPTED, push to front of waiting) and retry
     schedule (seq, need); budget -= need
2. While budget > 0 and len(running) < max_num_seqs and waiting not empty:
     seq = waiting[0]; apply prefix-cache hits
     chunk = min(seq.remaining_prompt_tokens, budget)       # chunked prefill
     if can't allocate blocks for chunk: break              # FCFS, no head-of-line skipping
     schedule (seq, chunk); move to running; budget -= chunk
3. Emit SchedulerOutput: ordered list [decode seqs..., prefill seqs...] with num_new_tokens each
```

**Efficiency design.**
- **A token budget instead of separate prefill and decode phases.** Decodes are scheduled first, so a long incoming prompt never stalls running generations (good TPOT). Prefill chunks fill the remaining budget, so the GPU stays saturated. A budget of 2–8K tokens keeps each step's latency bounded.
- **Chunked prefill** splits a 32K-token prompt across steps. Without it, one long prompt freezes every decode for the length of its prefill.
- **Preemption by recompute** (not swap): on a small model, recomputing a prefill is cheaper than PCIe-swapping KV, and the code is far simpler. The newest sequence is victimized because it has the least sunk work.
- **Ordering decodes before prefills** in the batch lets attention use two specialized kernels on contiguous slices (§3.10), with no per-token branching.
- **Low CPU cost:** the scheduler touches only Python ints and lists, with no tensor ops. Its overhead per step should be well under 100 µs at 256 sequences; measure this.

### 3.8 Model runner (`model_runner.py`)

**What it does.** Turns a `SchedulerOutput` into GPU tensors, runs the forward pass, samples, and returns token IDs. It owns the KV-cache tensors and the CUDA graphs.

**Input preparation (ragged and flattened, no padding):**
- `input_ids[T]` and `positions[T]`: all scheduled tokens from all sequences, concatenated. T is the total token count.
- `slot_mapping[T]`: the physical KV slot for each new token, `block_table[pos // B] * B + pos % B`.
- Attention metadata:
  - `cu_seqlens_q[N+1]`: cumulative query lengths.
  - `context_lens[N]`: total KV length after this step.
  - `block_tables[N, max_blocks]`, padded.
  - `num_decode_seqs` and `num_decode_tokens`, used to split the batch.
- `logits_indices[N]`: the index of the last token of each sequence that needs a sample. Non-final prefill chunks need none.

**Efficiency design.**
- **No padding anywhere.** Ragged tensors plus `cu_seqlens` mean the GEMMs process exactly T rows. Padding a batch to its max length can waste over 50% of prefill FLOPs.
- **Logits only for sampled positions:** `hidden[logits_indices]` is gathered *before* `lm_head`. With a 151K vocab, a 4K-token prefill would otherwise produce a 1.2 GB fp32 logits tensor and a 4K×896×151K GEMM that is thrown away.
- **Cheap host-to-device transfers:** metadata is built with NumPy into **pinned** CPU buffers and copied with `non_blocking=True`. The only sync point per step is reading back the sampled token IDs, which is unavoidable without async scheduling (stretch goal).
- **CUDA graphs for decode:** graphs are captured at startup for decode batch sizes `[1, 2, 4, 8, 16, 24, 32, ..., max_num_seqs]`. Each step pads to the next captured size, copies inputs into the graph's *static* buffers, and calls `graph.replay()`. All graphs share one memory pool. Prefill and mixed batches run eagerly, because their shapes vary too much and they are GPU-bound anyway. This one feature typically gives a 3–10× decode speedup on small models.
- **Memory profiling** at startup sizes the KV cache (§3.6).

### 3.9 Model and layers (`models/`, `layers/`)

**What it does.** A Qwen2/Llama decoder. Each layer runs:
```
residual, h = fused_add_rmsnorm(h, residual)
qkv        = qkv_proj(h)                      # one GEMM (+bias for Qwen2)
q, k       = rope(q, k, positions)            # in-place
store_kvcache(k, v, k_cache, v_cache, slot_mapping)
o          = attention(q, k_cache, v_cache, attn_metadata)
h          = o_proj(o)
residual, h = fused_add_rmsnorm(h, residual)
h          = down_proj(silu_and_mul(gate_up_proj(h)))
```

**Efficiency design.**
- Layers call ops through the **op registry** (`ops/`), never `torch.*` directly for hot ops. That makes swapping backends a config flag, which gives free A/B benchmarks and correctness tests against the reference.
- **Fused residual add + RMSNorm:** one pass over memory instead of three (add, norm, weight multiply).
- **RoPE fused with the KV-cache write** (a later-phase kernel): one read of K is used both to rotate and to scatter into the cache.
- The GEMMs stay on cuBLAS (`torch.nn.functional.linear`) at first. For bf16 at these shapes cuBLAS is very hard to beat, so custom GEMMs are reserved for quantized weights (§3.10.8), where a custom kernel is genuinely needed.

### 3.10 Kernels (`kernels/`, `ops/`): the showcase

Every op has a `ref` backend (plain PyTorch, used as the correctness oracle), a `triton` backend, and optionally a `cuda` backend. Each kernel ships with a unit test (comparing to `ref` across dtypes and edge shapes), a microbenchmark that reports µs and % of peak bandwidth or FLOP/s, and a short write-up in `docs/kernels/<name>.md`.

| # | Kernel | Bound by | Key design points |
|---|---|---|---|
| 1 | **Fused add + RMSNorm** | Memory | One program per row. fp32 accumulation for the sum of squares. Writes both the new residual and the normalized output in one pass. Target ≥80% of HBM bandwidth. |
| 2 | **SiLU-and-mul** | Memory | Reads `[T, 2I]`, writes `[T, I]`. Vectorized loads and fp32 math. Later fused into the down-proj prologue or the gate_up epilogue. |
| 3 | **RoPE (+ KV store)** | Memory | Precomputed cos/sin cache indexed by `positions`. In-place on q/k. The fused variant also scatters k/v via `slot_mapping` into the HND cache. Must handle Qwen/Llama's neox-style (half-rotation) layout. |
| 4 | **Paged decode attention** | Memory | The centerpiece; see below. |
| 5 | **Prefill / chunked-prefill attention** | Compute | FlashAttention-2 style: tile Q (e.g. 64–128 rows), stream K/V tiles, online softmax, causal mask with tile skipping. Reads *new* tokens' KV plus *cached prefix* KV through the block table, which chunked prefill and prefix caching require. Varlen via `cu_seqlens_q`. |
| 6 | **Sampling** | Memory | Greedy = argmax. Temperature-only uses the exponential-race / Gumbel trick, `argmax(p / Exp(1))`, which avoids any sort. Top-k/top-p use a sorting-free kernel (threshold search or rejection sampling, as in FlashInfer) instead of `torch.sort` over a 151K vocab per row. |
| 7 | **Embedding / LM-head gather** | Memory | Minor. Mostly about doing the gather before the LM head (§3.8). |
| 8 | *Stretch:* **W8A16 / W4A16 GEMM**, **FP8 KV cache** | Memory (decode) | Weight-only quantization halves or quarters the bytes per decode step, which raises the decode floor directly. Dequantize in registers inside the GEMM mainloop. FP8 KV halves attention bytes, so the decode kernel dequantizes on load. |

**Paged decode attention: design.**
- *Work decomposition:* the grid is `(num_seqs, num_kv_heads, num_splits)`.
  - Each program loads the **whole GQA group** of query heads for its KV head (`[G, D]`, e.g. 7×64), so each K/V tile is loaded from HBM once and reused G times. This is the most important decode-attention optimization for GQA models.
  - Loop over the program's share of blocks using `block_tables[seq]`. Load the K tile `[block_size, D]` (2 KB contiguous in HND) and compute `q·Kᵀ → [G, block_size]`. Keep online softmax state `(m, l, acc[G, D])` in registers and accumulate `P·V`.
- *Split-K (flash-decoding):* with batch=1 and 2 KV heads, a split-free grid has only 2 programs on a 108-SM A100, which leaves the GPU about 98% idle. Splitting the context into `num_splits` chunks fills the machine. A second tiny kernel merges the partial `(out, lse)` results. `num_splits` is chosen from `(num_seqs × num_kv_heads)` against SM count and context length. This is a nice tunable to show off with a heatmap.
- *Pipelining:* multiple blocks are processed per loop iteration (e.g. 4×16 = 64 tokens). In Triton this uses `num_stages`, and in CUDA it uses `cp.async` double-buffering, to hide HBM latency.
- *Target:* ≥70–80% of peak HBM bandwidth at long contexts. Report it against the bytes-moved model `(context_len × 2 × kv_heads × D × 2 B)` per sequence.
- *CUDA C++ version (v2):* a hand-written kernel with warp-level reductions (`__shfl_xor_sync`), 128-bit vectorized loads and a shared-memory K/V staging buffer, compared against the Triton version and FlashInfer. This is where the "I can write CUDA, not just Triton" story lives.

**Attention dispatch inside one forward:** tokens `[0, num_decode_tokens)` go to the paged decode kernel and the rest go to the prefill kernel. Both write into the same output tensor, and both are captured inside the decode CUDA graph when the batch is decode-only.

### 3.11 Sampler (`sampler.py`)

**What it does.** Logits `[N, V]` → token IDs `[N]`, with per-sequence temperature, top-k and top-p (each packed into a GPU tensor once per step).

**Efficiency design.**
- Everything stays on the GPU, and one `.tolist()` at the end is the only sync.
- A fast path applies when the whole batch is greedy (argmax only).
- Kernel 6 replaces sort-based top-p.
- Per-request seeds use a counter-based RNG (Philox offset = `seed, step`), so results are reproducible regardless of batch composition.

### 3.12 Engine loop (`llm_engine.py`)

```
add_request(prompt, params) → tokenize → Sequence(WAITING) → scheduler.waiting
step():
    sched_out = scheduler.schedule()
    token_ids = model_runner.run(sched_out)            # GPU
    scheduler.postprocess(sched_out, token_ids)        # append tokens, advance num_computed,
                                                       # check stop, free finished blocks,
                                                       # register newly-full blocks in prefix cache
    return detokenized deltas for streaming
generate(prompts): add all; while unfinished: step()
```
*Stretch:* **async scheduling / CPU–GPU overlap.** Schedule step *n+1* on the CPU while step *n* runs on the GPU. This is SGLang's "zero-overhead" scheduler idea, and it matters once the GPU step is under 1 ms.

### 3.13 Metrics and profiling (`metrics.py`, `bench/`)

- **Per request:**
  - TTFT (time to first token).
  - TPOT/ITL (time per output token / inter-token latency), with p50/p99.
  - End-to-end latency.
- **Per engine:**
  - Output tokens/s and total tokens/s.
  - Steps/s.
  - Batch occupancy.
  - KV-cache utilization.
  - Prefix-cache hit rate.
  - Preemption count.
- **Efficiency:**
  - **MBU (model bandwidth utilization)** = `(weight_bytes + kv_bytes_read) / step_time / peak_BW` for decode steps.
  - **MFU** for prefill.
  - These say how close the whole engine runs to the hardware limit, which is the headline number for a kernel-focused project.
- **Profiling:** `torch.profiler` traces for a step breakdown, and Nsight Systems for launch gaps and CPU overhead. Nsight Compute is for individual kernels: achieved bandwidth, occupancy and stall reasons.
- **Baselines:**
  - The same `bench` workload run against real vLLM on the same GPU.
  - HF `transformers.generate` as a sanity baseline.

### 3.14 Testing strategy

- **Op unit tests:** each backend is compared to `ref` (bf16 tolerances around `atol=1e-2`; the softmax/attention output is checked in fp32), including edge shapes: context length 1, not a multiple of the block size, max blocks, a GQA group of 1.
- **Model correctness:** the per-layer hidden states and final logits for a fixed prompt are compared with HF `transformers` (fp32 on CPU for the strict check). **Greedy output must match HF token-for-token** for a set of prompts.
- **Engine invariants:**
  - Batching invariance: greedy output for prompt P is identical whether it runs alone or in a batch of 256, and with or without prefix caching, chunked prefill, preemption or CUDA graphs.
  - Block accounting: free + used = total, and nothing leaks after all requests finish.
- **Scheduler property tests** (pure Python, run on a Mac): random arrivals and lengths under tight KV memory → no deadlock, no starvation, every request finishes.

---

## 4. Non-goals and stretch goals

**Non-goals:** an HTTP/OpenAI API (a thin FastAPI wrapper can be added later), beam search, LoRA, multi-modal, MoE, pipeline parallelism.

**Stretch goals, in rough order of payoff for a kernel portfolio:**
1. Weight-only INT4/INT8 GEMM kernel (W4A16), which raises the decode floor directly.
2. FP8 KV cache with dequant fused into the decode attention.
3. Speculative decoding, either n-gram/prompt-lookup (no draft model) or a small draft model. This needs a "verify k tokens" attention path, which is just chunked prefill with k query tokens.
4. Async scheduling (CPU/GPU overlap).
5. Tensor parallelism (2 GPUs, NCCL all-reduce after `o_proj` and `down_proj`), plus a custom one-shot all-reduce kernel.
6. Radix-tree prefix cache (SGLang style) as an alternative to hash-based caching, with a comparison on a multi-turn workload.
