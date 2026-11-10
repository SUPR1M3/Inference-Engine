# Inference Engine: Implementation Plan

This is the companion to [`ARCHITECTURE.md`](./ARCHITECTURE.md), and section references (§) point there.

The plan is ordered so that **you always have a working, measurable engine**:
- Correctness comes first, against Hugging Face.
- The engine skeleton comes second (paging, batching, CUDA graphs). That removes CPU overhead.
- Kernels come third, where each one moves a measured number.

Every phase ends with tests passing and a benchmark recorded in `bench/results/` with the GPU, commit and numbers. Those results become your write-up.

**Hardware:**
- Phases 0–3 can be developed on a Mac (CPU or MPS, fp32/bf16).
- Phase 4 onward needs an NVIDIA GPU. Rent one (L4/A10G is cheap for iteration; A100/H100 for headline numbers), and pin a single GPU model for all reported numbers.

**Effort estimates** assume part-time work (about 10–15 h/week) and are rough.

---

## Phase 0: Project setup (≈2–3 days)

**Tasks**
- Project skeleton per the layout in §2: `pyproject.toml` (`uv` or `pip`), with `torch`, `transformers` (reference only), `safetensors`, `tokenizers`, `huggingface_hub`, `numpy`, `pytest`, and `triton` (Linux only, as an optional extra).
- Tooling: `ruff`, `pytest`, `pre-commit`, a `Makefile` or `justfile` with `test`, `bench` and `lint` targets.
- `scripts/download_model.py` to fetch `Qwen2.5-0.5B-Instruct` into `~/.cache`.
- `tests/reference/`: a script that runs HF `transformers` on about 10 fixed prompts (greedy, 64 tokens) and saves the token IDs plus last-layer logits as fixtures.
- A GPU box bootstrap script (`scripts/setup_gpu.sh`): driver check, `nvidia-smi`, install, run the tests.

**Exit criteria:** `make test` runs (trivially) on the Mac and on the GPU box, and the HF reference fixtures exist.

---

## Phase 1: Naive, correct engine (≈1 week)

*Goal: a pure-PyTorch model that matches HF exactly. No paging or batching yet.*

**Tasks**
- `config.py`: `ModelConfig.from_hf(path)`.
- `models/loader.py`: safetensors loading, QKV and gate/up **fusion**, tied embeddings (§3.3).
- `layers/`: `RMSNorm`, `RotaryEmbedding` (neox style, cos/sin cache), `QKVLinear`, `MLP`, `Attention` using `F.scaled_dot_product_attention` on a **contiguous** per-request KV cache `[layers, 2, max_len, kv_heads, D]`.
- `models/qwen2.py` (with QKV bias) and `models/llama.py` (no bias), sharing most code.
- `ops/` registry with only `ref` backends: `rms_norm`, `fused_add_rms_norm`, `rope`, `silu_and_mul`, `attention`.
- `tokenizer.py` with a chat template and an **incremental detokenizer** (§3.4).
- A minimal `generate(prompt)` loop for a single request with greedy decoding, plus `cli.py generate` and `cli.py chat`.

**Tests**
- Per-layer hidden states vs HF (hook-based) at fp32: max abs diff < 1e-4.
- Greedy token-for-token match vs fixtures for all 10 prompts (fp32 strict; also check bf16 on the GPU and note any divergences).
- The detokenizer round-trips multi-byte and emoji text.

**Exit criteria:** `python -m engine.cli chat --model qwen2.5-0.5b` works, and the greedy outputs match HF.

---

## Phase 2: Paged KV cache and block manager (≈1 week)

*Goal: replace the contiguous cache with paged memory, still in pure PyTorch.*

**Tasks**
- `sequence.py`: `Sequence` with `num_computed_tokens`, `block_table` and status (§3.5).
- `block_manager.py`: free-list allocator, `allocate`/`append_slot`/`free`, ref counts. Leave prefix-cache hooks stubbed (§3.6).
- KV tensors in **HND** layout, `[num_blocks, kv_heads, block_size, D]` per layer, behind one layout helper.
- `ref` ops:
  - `store_kvcache(k, v, k_cache, v_cache, slot_mapping)`, an index scatter.
  - `paged_attention_ref`, which gathers K/V through the block table into contiguous tensors and runs SDPA. It's slow but obviously correct, and it is the oracle for every later attention kernel.
- KV sizing via memory profiling (§3.6). On a Mac, use a fixed `num_blocks`.

**Tests**
- Block manager unit tests: allocate/free cycles, no leaks, OOM detection.
- Greedy output is still identical to the fixtures.
- Odd lengths (1, 15, 16, 17, 1000 tokens) are correct.

**Exit criteria:** single-request generation runs on the paged cache and matches HF.

---

## Phase 3: Continuous batching, scheduler, sampler, benchmarks (≈1.5–2 weeks)

*Goal: a real (but slow) serving engine. After this phase, everything is about speed.*

**Tasks**
- `scheduler.py`: the full algorithm from §3.7:
  - Token budget, decode-first ordering.
  - Admission from waiting (FCFS).
  - **Preemption by recompute.**
  - `max_num_seqs`.
  - Hold off on chunked prefill: admit whole prompts first, then add chunking at the end of this phase once batching is stable.
- `model_runner.py`: **ragged, flattened input prep** (`input_ids`, `positions`, `slot_mapping`, `cu_seqlens_q`, `context_lens`, `block_tables`, `logits_indices`) built with NumPy into pinned buffers (§3.8).
  - Gather the hidden states at `logits_indices` *before* the LM head.
  - Use the ref attention with decode/prefill split dispatch.
- `sampler.py`: greedy, temperature, top-k, top-p (sort-based ref is fine for now), and per-request seeds (§3.11).
- `llm_engine.py`: `add_request` / `step` / `generate` with streaming deltas (§3.12).
- `metrics.py` and `cli.py bench`:
  - Synthetic workloads: fixed input/output lengths, and random lengths with a ShareGPT-like distribution.
  - Report throughput, TTFT/TPOT p50/p99, steps/s and KV utilization.
- `bench/run_vllm_baseline.py`: the same workload against real vLLM, the target to chase.

**Tests**
- **Batching invariance:** each prompt's greedy output alone equals its output in a batch of 64.
- **Scheduler property test** (CPU, no model; fake the model runner): random arrivals and tiny KV pools → every request finishes, there are no leaks, and preemption is exercised.
- Chunked prefill invariance: the output is identical with `max_num_batched_tokens` set to 16, 64 or 8192.

**Exit criteria:** `bench` runs end to end on the GPU box. Record **Baseline v0** (expect it to be far slower than vLLM; that is the point). Capture a `torch.profiler` trace and an Nsight Systems timeline of a decode step showing the launch-bound gaps. That screenshot motivates Phases 4–5.

---

## Phase 4: CUDA graphs and CPU overhead (≈1 week)

*Goal: make decode GPU-bound before writing kernels, so kernel wins become visible end to end.*

**Tasks**
- Persistent static input buffers sized to `max_num_seqs` (decode) and `max_num_batched_tokens`.
- Capture decode graphs for batch sizes `[1, 2, 4, 8, 16, 24, 32, ..., max_num_seqs]` with a shared memory pool. Pad to the next size at runtime. `--enforce-eager` disables graphs (§3.8).
- Make sure the ref attention path is graph-capturable. That means no `.item()`, no data-dependent shapes and no CPU syncs inside the forward, and it forces you to make metadata fixed-shape (padded `block_tables`, a `max_context_len` bucket).
- Profile and trim the scheduler and input-prep CPU time: vectorize the NumPy work, avoid per-token Python loops, and reuse buffers.

**Benchmarks**
- Decode-step latency at batch sizes 1, 8, 64 and 256, eager vs graphs.
- Per-step time breakdown, CPU side vs GPU side.

**Exit criteria:** batch-1 decode step time is within about 2× of the weight-read floor (≈0.5 ms for 0.5B on an A100; compute your own GPU's floor), and greedy outputs are unchanged. Record **v1**.

---

## Phase 5: Kernels round 1, the memory-bound elementwise ops (≈1–1.5 weeks)

*Goal: build the kernel workflow (test, microbench, roofline, write-up) on simple kernels first.*

**Tasks**
- `bench/kernels/harness.py`: CUDA-event timing with warmup, L2 flush between iterations, and output of µs, GB/s and % of peak.
- Triton kernels, each with tests vs `ref`, a microbench and a `docs/kernels/<name>.md`:
  1. `fused_add_rms_norm`, plus plain `rms_norm` (§3.10 #1)
  2. `silu_and_mul` (#2)
  3. `rope` in place, then `rope_and_store_kvcache` fused (#3)
  4. `store_kvcache` standalone, for comparison
- A `--kernel-backend triton` flag on the engine, with per-op overrides for A/B testing.
- Optional comparison point: `torch.compile` on the ref ops.

**Exit criteria:** each kernel reaches ≥80% of peak bandwidth at realistic shapes (T from 1 to 8192, hidden 896 to 2048). End-to-end greedy output is still identical (or within documented bf16 drift), and **v2** is recorded.

---

## Phase 6: Paged decode attention kernel, the centerpiece (≈2–3 weeks)

*Goal: the most impressive kernel in the project. Build it in stages and benchmark each one.*

**Stages** (§3.10, "Paged decode attention: design")
1. **v0, the straightforward Triton version:** one program per (seq, q_head), looping over blocks, with online softmax. This establishes correctness against `paged_attention_ref`.
2. **v1, GQA-aware:** one program per (seq, kv_head) that processes all G query heads together, so each KV tile is loaded once. Expect a large win at G=7 (Qwen 0.5B).
3. **v2, split-K / flash-decoding:** `num_splits` partitions of the context, partial `(out, lse)` outputs and a reduce kernel. A heuristic picks `num_splits` from the batch, KV heads, SM count and context length.
4. **v3, tuning:** blocks per iteration, `num_warps`, `num_stages`, `@triton.autotune` keyed on (G, D, block_size, context bucket). Keep it capturable in a CUDA graph (autotuning must happen before capture).
5. **v4, CUDA C++ version** (`kernels/cuda/paged_decode.cu`, built with `torch.utils.cpp_extension`): warp-per-token-chunk layout, `__shfl_xor_sync` reductions, 128-bit vectorized loads, `cp.async` double-buffered K/V staging. Compare against Triton v3 and **FlashInfer**'s decode kernel.

**Benchmarks**
- A heatmap of µs and % peak bandwidth over batch ∈ {1, 8, 32, 128, 256} × context ∈ {128, 512, 2K, 8K, 32K}, for every stage and for FlashInfer.
- An Nsight Compute report for the best version: achieved bandwidth, occupancy, stall reasons.

**Exit criteria:** ≥70% of peak HBM bandwidth at long contexts, within about 10–20% of FlashInfer (beating it on some shapes is a great talking point), and end-to-end **v3** recorded. Expect the biggest gains at long contexts and large batches.

---

## Phase 7: Prefill attention, chunked prefill and prefix caching (≈2 weeks)

*Goal: make prefill fast, and support prefixes already in the cache.*

**Tasks**
- **Prefill attention kernel** (Triton, FlashAttention-2 style; §3.10 #5):
  - Varlen queries via `cu_seqlens_q`, with a causal mask and tile skipping.
  - K/V read through the **block table**, so it covers the cached prefix plus the new chunk. That is what makes chunked prefill and prefix-cache hits work with no special cases.
  - Start with K/V from the paged cache only (store first, then attend). Optimizing the "new tokens" path to read K/V from registers or SMEM is an optional refinement.
- Turn on chunked prefill in the scheduler by default (if not done in Phase 3) and tune `max_num_batched_tokens` for the TTFT/TPOT trade-off.
- **Prefix caching** in `block_manager.py` (§3.6):
  - Chained block hashes and the `hash → block` map.
  - Hit lookup at admission.
  - Registering blocks when they become full.
  - Lazy eviction from the free list.
  - `--enable-prefix-caching`.
- Workload: `bench --shared-prefix-len 1024` (many requests sharing a system prompt) and a multi-turn chat replay.

**Tests**
- Prefill kernel vs ref across varlen, causal and prefix-offset cases.
- Prefix-cache invariance: outputs are identical with caching on and off. A hit-rate assertion on the shared-prefix workload.
- Eviction stress: a tiny KV pool with many distinct prefixes → correct outputs and no leaks.

**Exit criteria:** prefill MFU recorded vs `F.scaled_dot_product_attention` / FlashAttention; TTFT on the shared-prefix workload drops by roughly the prefix fraction; **v4** recorded.

---

## Phase 8: Sampling kernel and remaining hot spots (≈1 week)

**Tasks**
- A sorting-free top-k/top-p sampling kernel (§3.10 #6), using a threshold or pivot search over probabilities instead of a full sort. Also the Gumbel / exponential-race path for temperature-only sampling.
- Re-profile the full engine. Whatever is now the largest non-GEMM slice is the next kernel target. It is often the LM-head plus sampling, or the remaining small elementwise ops.
- Optional: fuse `silu_and_mul` into a Triton GEMM epilogue to test whether it beats cuBLAS + kernel at decode shapes.

**Exit criteria:** sampling with top-p costs < 5% of decode step time at batch 256; **v5** recorded.

---

## Phase 9: Write-up and comparison (≈1 week, overlapping with earlier phases)

- `docs/RESULTS.md`:
  - A table showing each version (v0 to vN) against vLLM on the same GPU and workload: throughput, TTFT, TPOT, MBU.
  - A chart of "what each optimization bought".
- One page per kernel in `docs/kernels/` covering the design, roofline, results, what didn't work and why.
- A README with a quick start, architecture diagram, headline numbers and links.

---

## Stretch phases (pick by interest)

| Phase | What | Why it's worth it | Rough effort |
|---|---|---|---|
| S1 | **W4A16 / W8A16 weight-only GEMM** (Triton, then CUDA), loading AWQ/GPTQ checkpoints | Directly lowers the decode bandwidth floor, and it is the most "real-world kernel" item | 2–3 wks |
| S2 | **FP8 KV cache** with dequant fused into decode/prefill attention | Halves attention bytes and touches all your attention kernels | 1 wk |
| S3 | **Speculative decoding** (n-gram prompt-lookup first, then a draft model) | Reuses chunked-prefill attention for verification, and needs rejection sampling | 2 wks |
| S4 | **Async scheduling** (CPU/GPU overlap) | Removes the remaining CPU gap; SGLang-style | 1–2 wks |
| S5 | **Tensor parallel (TP=2)** + custom one-shot all-reduce kernel | Covers multi-GPU and communication kernels | 2–3 wks |
| S6 | **Radix-tree prefix cache** vs hash-based, on a multi-turn workload | A direct vLLM-vs-SGLang design comparison | 1–2 wks |
| S7 | Thin **OpenAI-compatible HTTP server** (FastAPI + SSE streaming) | Makes the engine demoable; little to do with kernels | 3–4 days |

---

## Working agreements (apply in every phase)

1. **Correctness gate:** no optimization merges unless the HF greedy fixtures and the batching-invariance tests still pass.
2. **Measure before and after:** every performance PR includes a microbench result and an end-to-end `bench` result, both on the pinned GPU.
3. **Reference backend is permanent:** `ref` ops are never deleted. They are the oracle and they keep the Mac dev loop alive.
4. **One variable at a time:** introduce a kernel behind a backend flag, A/B test it, then make it the default.
5. **Log the failures:** record approaches that didn't pay off (e.g. "NHD vs HND layout: HND +12% decode BW") in the kernel docs. They're as valuable to readers as the wins.

## Milestone summary

| Version | After phase | Headline capability |
|---|---|---|
| v0 | 3 | Paged KV + continuous batching, pure PyTorch, matches HF |
| v1 | 4 | CUDA graphs; decode near the launch-free floor at small batch |
| v2 | 5 | Triton fused elementwise kernels |
| v3 | 6 | Custom paged decode attention (Triton + CUDA), compared with FlashInfer |
| v4 | 7 | Custom prefill attention, chunked prefill, prefix caching |
| v5 | 8 | Sorting-free sampling; full engine benchmarked against vLLM |
