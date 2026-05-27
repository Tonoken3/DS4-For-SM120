# Sub-Graph Refactoring Specification

## Goal

Fix CUDA Graph capture to produce **correct** MoE output by splitting the single
full-model capture into **per-layer sub-graphs** (43 graphs). The host-side
compact MoE logic runs **between** sub-graph launches, filling the compact
scratch with the correct selected-expert weights for each layer.

## Background

Current state (commit `c7d9c17`):
- Compact MoE (6/256 experts, 40 MB/layer) works correctly at **5.69 t/s**
- Graph capture infrastructure (`cudaStreamBeginCapture/EndCapture/Instantiate/Launch`) works
- **Bug**: The single captured graph contains all 43 layers. Each layer's MoE
  kernels read from `g_moe_compact_gate/up/down` (the same device buffer).
  During capture, the scratch was filled with layer 42's weights (from warmup).
  On replay, ALL 43 layers read layer 42's weights → **wrong MoE output**.

## Solution Overview

1. Split into 43 per-layer sub-graphs (`cudaGraphExec_t sub_graph[43]`)
2. The router (`ds4_gpu_router_select_tensor`) runs **outside** each sub-graph
3. Between router and MoE, host fills compact scratch with correct weights
4. Sub-graph contains: attention ops + MoE kernels (gate/up/down/sum) + shared FFN + HC

## Detailed Changes

### File: `ds4.c` — `metal_graph_encode_decode_layer()`

Add `bool skip_router` as a new parameter (after `int token`).

**Function signature change** (line ~9973):

```c
static bool metal_graph_encode_decode_layer(
        ds4_gpu_graph  *g,
        const ds4_model        *model,
        const ds4_layer_weights *layer,
        uint32_t                il,
        uint32_t                pos,
        ds4_gpu_tensor       *raw_cache,
        uint32_t                raw_cap,
        uint32_t                raw_row,
        uint32_t                n_raw,
        int                     token,
        bool                    skip_router);   // <-- NEW
```

**Router section guard** (lines ~10624–10646):

Wrap the router matmul + select + debug dumps in `if (!skip_router)`:

```c
    const uint64_t down_expert_bytes = routed_out_dim * down_row_bytes;
    if (!skip_router) {                           // <-- ADD
    if (ok) ok = metal_graph_matmul_plain_tensor(g->router_logits, ...);
    if (ok) ok = ds4_gpu_router_select_tensor(g->router_selected, ...);
    DS4_METAL_PROFILE_DECODE_STAGE("router");
    if (ok) {
        metal_graph_debug_dump_tensor(...);
        metal_graph_debug_dump_tensor(...);
        metal_graph_debug_dump_i32_tensor(...);
        metal_graph_debug_dump_tensor(...);
    }
    } /* !skip_router */                          // <-- ADD
    if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out, ...);
```

**All callers**: Add `, false` (not skipping router — existing behavior) as the
last argument before `)`.

Callers (8 locations numbered by line, approximate):
1. `ds4.c:~11365` — prefill call `metal_graph_encode_decode_layer(&g, model, layer, ..., token)` → `, false)`
2. `ds4.c:~11531` — decode batch `(&g, model, &weights->layer[il], il, 0, ..., token)` → `, false)`
3. `ds4.c:~11575` — decode single `(&g, model, &weights->layer[il], il, 0, ..., token)` → `, false)`
4. `ds4.c:~11671` — spec decode `(g, model, layer, il, pos, ..., token)` → `, false)`
5. `ds4.c:~13851` — MTP draft `(g, model, layer, il, pos, ..., token)` → `, false)`
6. `ds4.c:~14750` — spec verifier `(g, model, layer, il, pos, ..., token)` → `, false)`
7. `ds4.c:~14767` — spec verifier prefix1 `(g, model, layer, il, pos, ..., token)` → `, false)`

### File: `ds4.c` — `metal_graph_eval_token_raw_swa()`

Replace the single capture/replay with per-layer sub-graph management.

**Add per-layer sub-graph handles** (in `ds4_gpu_graph` struct):

```c
cudaGraphExec_t *sub_graph_exec;  /* [43] allocated in metal_graph_alloc_raw_cap */
```

Or in `ds4_cuda.cu` as a global array:

```c
static cudaGraphExec_t g_sub_graph_exec[43];
static int g_sub_graph_exec_ready[43];  // 1 if captured
```

**Modified eval function flow:**

```c
static bool metal_graph_eval_token_raw_swa(g, model, weights, token, pos, logits) {
    // ... fill host params ...
    const bool graph_mode = getenv("DS4_CUDA_DECODE_GRAPH") != NULL;
    const bool can_capture = graph_mode && g->cuda_params_host &&
                             ds4_gpu_decode_graph_can_capture();

    bool ok = ds4_gpu_begin_commands() != 0;

    if (can_capture && sub_graphs_ready()) {
        // === REPLAY PHASE (pos >= capture_token_pos + 1) ===
        const bool need_logits = (logits != NULL);
        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const ds4_layer_weights *layer = &weights->layer[il];

            // a) Run router outside capture
            ok = ok && metal_graph_matmul_plain_tensor(g->router_logits,
                         model, layer->ffn_gate_inp, DS4_N_EMBD, DS4_N_EXPERT,
                         g->ffn_norm, 1);
            ok = ok && ds4_gpu_router_select_tensor(
                         g->router_selected, g->router_weights, ...);

            // b) Flush router to ensure selected is on device
            ok = ok && ds4_gpu_flush_commands();

            // c) Host: fill compact scratch with layer il's selected experts
            ok = ok && ds4_gpu_moe_compact_fill(model, layer, g->router_selected,
                                                  !ds4_gpu_need_logits);

            // d) Launch sub-graph (attention + MoE + shared FFN, no router)
            ok = ok && ds4_gpu_decode_subgraph_launch(il);

            // e) After last layer + need_logits, copy logits
            if (il == DS4_N_LAYER - 1 && need_logits) {
                ok = ok && ds4_gpu_tensor_read(g->logits, 0, logits, ...);
            }
        }
    } else if (can_capture && !sub_graphs_ready()) {
        // === CAPTURE PHASE (first decode token after warmup) ===
        // Warmup first: run normal encode to fill all weight caches
        ok = ok && metal_graph_encode_token_raw_swa(g, model, weights,
                                                     token, pos, logits != NULL, true);

        for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
            const ds4_layer_weights *layer = &weights->layer[il];

            // a) Run router outside capture
            ok = ok && metal_graph_matmul_plain_tensor(...);
            ok = ok && ds4_gpu_router_select_tensor(...);
            ok = ok && ds4_gpu_flush_commands();

            // b) Fill compact scratch
            ok = ok && ds4_gpu_moe_compact_fill(...);

            // c) Begin capture
            ok = ok && ds4_gpu_decode_graph_capture();

            // d) Encode layer with skip_router=true
            ok = ok && metal_graph_encode_decode_layer(g, model, layer, il,
                         pos, raw_cache, raw_cap, raw_row, n_raw, token,
                         /*skip_router=*/true);

            // e) End capture, instantiate, store handle for layer il
            ok = ok && ds4_gpu_decode_graph_capture_end_store(il);
        }
    } else {
        // === NORMAL ENCODE (no graph mode, or warmup) ===
        ok = ok && metal_graph_encode_token_raw_swa(g, model, weights,
                                                     token, pos, logits != NULL, true);
    }

    if (ok) ok = ds4_gpu_end_commands() != 0;
    // ... read logits if normal path ...
}
```

### File: `ds4_cuda.cu` — New functions

**1. `ds4_gpu_moe_compact_fill()`** — Host-side compact logic (extracted from `routed_moe_launch`):

```c
extern "C" int ds4_gpu_moe_compact_fill(
        const ds4_model *model,
        const ds4_layer_weights *layer,
        const ds4_gpu_tensor *selected,  // router output (device tensor)
        int need_logits);  // placeholder
```

Implementation:
1. Read selected indices from device (cudaMemcpy D2H, 6 int32s)
2. For each selected expert: cudaMemcpyAsync H2D to g_moe_compact_gate/up/down
3. Remap indices (0-5) and cudaMemcpyAsync to g_moe_compact_selected_dev

**2. `ds4_gpu_decode_subgraph_launch(int il)`** — Launch layer il's sub-graph:

```c
extern "C" int ds4_gpu_decode_subgraph_launch(int il) {
    if (il < 0 || il >= 43 || !g_sub_graph_exec[il]) return 0;
    return cuda_ok(cudaGraphLaunch(g_sub_graph_exec[il],
                                    g_cuda_decode_stream), "subgraph launch");
}
```

**3. `ds4_gpu_decode_graph_capture_end_store(int il)`** — End capture + store at index:

```c
extern "C" int ds4_gpu_decode_graph_capture_end_store(int il) {
    if (il < 0 || il >= 43) return 0;
    cudaGraph_t graph = NULL;
    cudaError_t ce = cudaStreamEndCapture(g_cuda_decode_stream, &graph);
    if (ce != cudaSuccess || !graph) return 0;
    ce = cudaGraphInstantiate(&g_sub_graph_exec[il], graph, NULL, NULL, 0);
    (void)cudaGraphDestroy(graph);
    if (ce != cudaSuccess) return 0;
    g_sub_graph_exec_ready[il] = 1;
    return 1;
}
```

**4. `ds4_gpu_decode_subgraphs_ready()`** — Check if all sub-graphs are ready:

```c
extern "C" int ds4_gpu_decode_subgraphs_ready(void) {
    for (int i = 0; i < 43; i++)
        if (!g_sub_graph_exec_ready[i]) return 0;
    return 1;
}
```

### File: `ds4_gpu.h` — New declarations

```c
int ds4_gpu_moe_compact_fill(const void *model_map, uint64_t model_size,
                               uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                               uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
                               uint64_t down_expert_bytes, uint64_t down_row_bytes,
                               uint32_t n_total_expert, uint32_t n_expert,
                               const ds4_gpu_tensor *selected);
int ds4_gpu_decode_subgraph_launch(int il);
int ds4_gpu_decode_graph_capture_end_store(int il);
int ds4_gpu_decode_subgraphs_ready(void);
```

### File: `ds4_cuda.cu` — Cleanup additions

Add to `ds4_gpu_cleanup()`:
```c
for (int i = 0; i < 43; i++) {
    if (g_sub_graph_exec[i]) {
        cudaGraphExecDestroy(g_sub_graph_exec[i]);
        g_sub_graph_exec[i] = NULL;
    }
    g_sub_graph_exec_ready[i] = 0;
}
```

## Test Procedure

```bash
# Build
make cuda CUDA_ARCH=sm_120

# Test: graph capture + replay
DS4_CUDA_DECODE_GRAPH=1 DS4_CUDA_DECODE_GRAPH_PROFILE=1 \
DS4_CUDA_Q8_F16_CACHE_MB=0 DS4_CUDA_WEIGHT_CACHE_LIMIT_GB=12 \
./ds4 --backend cuda --model ds4flash.gguf --temp 0 -p "Hello" -n 32

# Verify:
# 1. Output text is coherent (not garbage like 了的/unravelability)
# 2. "graph token pos=11" shows "captured and instantiated"  
# 3. "graph token pos=12+" shows replay with encode <= 1ms
# 4. Generation t/s > 10
```

## Risks

1. **Template/SED errors**: The function signature change affects 8 call sites.
   Use a script (Python/AWK) to do all replacements atomically.

2. **Compact scratch fill ordering**: The host-side fill uses `cudaMemcpyAsync`
   on the graph stream. The stream ordering ensures fills complete before
   sub-graph kernels start. Verify with cuda-memcheck.

3. **First-token weight cache**: The warmup encode (normal) must complete before
   the capture encode. Verify weight caches are warm (no staging messages).

4. **Router-resident params**: The router uses `token` as an argument. During
   replay, this is fine because the router runs OUTSIDE the sub-graph (host
   passes the current token). No device-params needed for the router.
