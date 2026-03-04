# KV Cache Compression Controls in vLLM and Where to Find Real SnapKV/H2O/PyramidKV Implementations

## What “KV budget control” means in long-context inference

“KV cache compression” is an umbrella term for multiple, technically distinct ways of reducing the *memory footprint* (and sometimes *compute/latency*) of the key/value cache used during autoregressive decoding. Recent systems papers commonly group approaches into at least four families: **token dropping / eviction** (remove some tokens’ K/V entries), **quantization** (store K/V at lower precision), **merging** (combine multiple tokens’ K/V into fewer representatives), and **prompt compression** (reduce the prompt itself before prefill). citeturn21view1

In most research codebases, “budget” is expressed in one of two forms:

- **A token budget**: keep only *K* tokens’ KV entries per layer (or per head), e.g., “keep 1024 prompt KV caches” in SnapKV-style reporting. citeturn10search23turn10search30  
- **A compression ratio**: keep only *r* fraction of tokens (or equivalently drop 1−r), often specified as a `compression_ratio` in tooling frameworks. citeturn10search18

This distinction matters because serving engines typically need explicit support for whichever budget mechanism you want: a system may support KV quantization but not token-dropping eviction, or support a fixed-size sliding window but not attention-score-based sparsification. citeturn15view1turn21view1

## What vLLM supports today that affects KV cache size

In entity["organization","vLLM","llm inference engine"], most knobs that influence KV memory are **engine configuration** (startup flags), not **per-request** controls.

### Engine-level KV memory sizing and dtype
The official engine arguments include:

- `--gpu-memory-utilization` to cap how much GPU memory the engine can use overall (and therefore how much is left for KV blocks). citeturn15view1  
- `--kv-cache-memory-bytes` for explicitly sizing KV cache memory (overriding the heuristic derived from GPU utilization). citeturn15view0  
- `--kv-cache-dtype` to select the KV cache storage dtype, including FP8 variants on supported hardware. citeturn15view0turn15view1  
- `--kv-offloading-size` / `--kv-offloading-backend` for CPU offloading of KV cache via supported backends. citeturn15view0  

These features are real KV-memory optimizations, but they **do not implement SnapKV/H2O/PyramidKV token selection**; they change memory format, pool size, or placement. citeturn15view0turn21view1

### Sliding window and “sink” attention patterns
vLLM supports **sliding window attention** when the model architecture/config provides it and exposes a server flag to disable it (`--disable-sliding-window`). citeturn14view0turn15view1  

Separately, vLLM has explicit support for **static sink tokens** via a `StaticSinkAttention` implementation and tracks “Sink” support as an attention-backend capability (described as “Attention sink support (for StreamingLLM)”). citeturn12search4turn8view0  

For hybrid models (mixtures of full + sliding-window layers), vLLM’s “Hybrid KV Cache Manager” design documentation emphasizes per-attention-type allocation rules (full layers keep all tokens; sliding-window layers retain only the most recent window) and corresponding prefix-caching semantics. citeturn8view1  

This is important context: vLLM already has sophisticated block/page-based KV allocation and attention-pattern support, but **that is not the same as implementing token-dropping policies like SnapKV or H2O**. citeturn8view1turn21view1

## Why a `kv_budget` field in your OpenAI request body is not doing anything in vLLM

The OpenAI-compatible server documentation in vLLM enumerates supported request parameters (including “extra parameters”). There is **no mention of `kv_budget`** anywhere on that page. citeturn13view0  

The same page *does* show vLLM-specific extensions such as `truncate_prompt_tokens` (a prompt truncation facility), reinforcing that vLLM only acts on fields it explicitly understands. citeturn13view2  

In practice, vLLM’s server-side validation and parsing behavior has historically varied by endpoint/model/schema:

- Some extra or unknown fields are **ignored with a warning**, e.g., logs like “The following fields were present in the request but ignored: {…}”. citeturn17view0turn19search4turn19search5  
- Other fields can trigger strict schema validation failures (“Extra inputs are not permitted”), including cases where users attempted to pass “extra” fields inside messages. citeturn11view0  
- There are also documented periods where even a *documented* extra parameter (e.g., `truncate_prompt_tokens`) was rejected depending on server version/endpoint plumbing. citeturn11view1  

Given (a) `kv_budget` is not a documented server parameter, and (b) vLLM frequently either ignores or rejects unknown fields depending on validation context, a top-level `kv_budget` added to your JSON payload is not a reliable control knob—and in deployments where it is accepted, it is consistent with being silently ignored. citeturn13view0turn17view0turn11view0

## Evidence that SnapKV/H2O/PyramidKV aren’t exposed in vLLM’s OpenAI server (and why vLLM is still mentioned in research)

### Mainline vLLM: active discussion, not a stable “KV budget” API
Within vLLM’s own issue tracker, token-dropping / compaction has repeatedly appeared as a requested feature or RFC rather than a completed, user-facing capability:

- A 2024 RFC explicitly proposes a “sparse KV cache framework” and even sketches a hypothetical `--sparse-kv-cache-type` flag and compression ratio workflow—indicating this was design exploration rather than an existing API. citeturn16search0  
- A feature request specifically asking for H2O-style eviction was closed “as not planned.” citeturn16search1  
- A KV cache compaction RFC describes needs like exposing attention weights from kernels, enabling `free_and_reallocate`, and handling non-uniform layouts—then was also closed “as not planned.” citeturn20view1  
- A KVPress integration request was similarly closed “as not planned.” citeturn20view0  
- More recent RFC threads propose a common sparse KV framework (including external storage hooks), which again suggests the feature is being architected rather than already shipped as a stable knob. citeturn20view2  

Taken together with the absence of any `kv_budget`/SnapKV/H2O parameter in the OpenAI server docs, this supports the conclusion that **mainline vLLM does not provide a per-request SnapKV/H2O/PyramidKV “budget” control in its OpenAI endpoint today**. citeturn13view0turn16search1turn20view1turn20view2  

### Why vLLM still shows up in papers and project prompts
A key nuance: **research prototypes frequently extend vLLM** because it is a high-performance baseline, not because upstream vLLM already has all methods implemented.

Concrete examples:

- vLLM issue #10942 describes “KV-Compress” experiments performed on a **vLLM integration fork**, including modifications to flash attention, paged attention, and block manager logic—explicitly acknowledging the work happened out-of-tree and traded off compatibility with newer vLLM features. citeturn21view0  
- The KV-Compress paper itself frames vLLM’s paged KV cache organization (block-based) as a core constraint and designs compression/eviction around it. citeturn3view2  
- The EvicPress system paper reports implementing joint compression+eviction by **extending vLLM and LMCache** with ~3K lines of code and integrating with vLLM’s paged memory manager—again illustrating “vLLM as substrate,” not “vLLM as turnkey SnapKV API.” citeturn21view1  

So, you are not “missing” a hidden request field. The more precise interpretation of your professor’s phrasing (and the broader ecosystem reality) is: **vLLM can be used as a base engine if you adopt an existing fork/patchset or implement the method yourself**, but **stock vLLM’s OpenAI server does not expose SnapKV-like per-request KV budget controls.** citeturn21view0turn3view2turn20view2  

## Where to run “true” KV compression methods with adjustable budgets today

If the goal is to evaluate multiple token-dropping policies (SnapKV / H2O / PyramidKV / StreamingLLM) with budgets like 10/20/50%, the most direct path is to use frameworks that already implement them (typically on top of PyTorch + entity["company","Hugging Face","transformers platform"] Transformers and FlashAttention), where “budget” is a first-class parameter.

### KVCache-Factory (unified evaluation & LongBench scripts)
The KVCache-Factory repository explicitly states support for **PyramidKV, SnapKV, H2O, and StreamingLLM**, and exposes a `max_capacity_prompts` parameter described as “Selected KV Size in each layer,” used to control how many KV entries are kept. citeturn5view0  

It provides LongBench scripts where the method and budget are provided as CLI parameters (`--method`, `--max_capacity_prompts`, attention backend choice, etc.). citeturn5view0  

Independent papers also cite KVCache-Factory as a benchmarking framework supporting these methods, reinforcing that this is a commonly used evaluation testbed. citeturn4search26  

### SnapKV: reference implementation (Transformers monkeypatch)
The SnapKV reference repo provides a “monkeypatch” integration approach for Transformers models (Llama family / Mistral / Mixtral) and describes the algorithm implementation location. citeturn5view2  

The SnapKV paper reports comparisons at different “prompt KV cache” capacities (e.g., 1024), showing how “budget” is naturally expressed as a target retained KV length. citeturn10search23turn10search30  

### H2O: reference implementation with real KV dropping
The H2O repo explicitly states it includes an HF-based implementation and that it provides **both simulation and “real KV dropping”** codepaths. citeturn18search0  

Its HF instructions expose “budget” as ratios of prompt length (e.g., `recent_ratio` and `heavy_ratio`), which directly determines how many KV entries are kept in the cache. citeturn18search3  

The H2O paper provides the underlying motivation and reported throughput/latency gains for eviction under a fixed heavy-hitter percentage. citeturn18search10turn18search18  

### StreamingLLM: attention sinks + fixed window (bounded KV memory)
StreamingLLM’s core mechanism is explicit: keep a small number of “sink” tokens plus a sliding window of recent tokens, enabling bounded KV memory while maintaining stability over very long sequences. citeturn12search7  

The official repo is widely used as the reference artifact for the method. citeturn18search2  

### KVPress (NVIDIA): a research framework with compression ratios as a first-class API
The entity["company","NVIDIA","gpu manufacturer"] KVPress framework is designed specifically to “implement multiple KV cache compression methods and benchmarks” on top of Transformers, and uses “presses” applied during prefill with an explicit `compression_ratio` parameter. citeturn10search0turn10search18  

Importantly for evaluating “budget curves,” KVPress models compression as a parameterized object interface (presses) rather than an inference-engine flag, which often makes it easier to sweep 0.1/0.2/0.5 ratios. citeturn10search18  

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["SnapKV KV cache compression diagram","StreamingLLM attention sink sliding window diagram","vLLM PagedAttention KV cache blocks diagram","PyramidKV KV cache pyramidal allocation diagram"],"num_per_query":1}

## Reconciling your observed invariance with the “vLLM should work” expectation

Your empirical observation—accuracy/latency curves not changing across “kv_budget” sweeps—matches what you would expect if the serving engine is not applying any token-dropping/eviction policy keyed off that field. vLLM’s OpenAI server documentation contains no `kv_budget` parameter, and vLLM’s own logs/issues show a pattern of either ignoring unknown request fields (with warnings) or rejecting them when strict validation is enabled. citeturn13view0turn17view0turn11view0  

At the same time, your professor’s mention of vLLM is still consistent with the literature and tooling landscape: many KV cache compression systems *do* build on vLLM, but they do so via forks/patches (e.g., KV-Compress) or by extending vLLM’s cache manager and paging system (e.g., EvicPress). Those projects demonstrate feasibility—just not a ready-made, per-request `kv_budget` switch in stock vLLM’s OpenAI endpoint. citeturn21view0turn3view2turn21view1