# Pipeline Parallelism (PP=7) Design

## Motivation

Compact MoE (6/256 experts) reduces H2D from 73GB/token to 1.7GB/token — but
still consumes ~53ms/token in PCIe transfers. To reach 30 t/s (33ms/token),
H2D must be **eliminated entirely**.

The solution: split 43 layers across 7 GPUs. Each GPU caches ALL weights for
its assigned layers. During decode, each GPU processes its layers with ZERO
H2D weight transfer. Activations (16KB) flow between GPUs via P2P.

## VRAM Budget per GPU (16GB)

```
GPU i (layers: start_i .. end_i, typically 6 layers):

Attention weights:     6 layers × 150 MB =    0.9 GB
MoE weights (256 exp):  6 layers × 1.7 GB  = 10.2 GB
KV cache (6 layers):    ~0.2 GB
cuBLAS workspace:        32 MiB
Scratch / temp:          ~0.5 GB
-------------------------------------------
Total:                  ~11.8 GB  → fits in 16 GB ✓
```

## Layer Assignment (PP=7)

```
GPU 0: layers  0-5   (6 layers)
GPU 1: layers  6-11  (6 layers)
GPU 2: layers 12-17  (6 layers)
GPU 3: layers 18-23  (6 layers)
GPU 4: layers 24-29  (6 layers)
GPU 5: layers 30-35  (6 layers)
GPU 6: layers 36-42  (7 layers)
```

## Architecture

```
Token flow (per decode step):
  ┌──────────┐   act   ┌──────────┐   act   ┌──────────┐
  │  GPU 0   │────────▶│  GPU 1   │────────▶│  GPU 2   │─ ...
  │ L0-L5    │  16KB   │ L6-L11   │  16KB   │ L12-L17  │
  └──────────┘         └──────────┘         └──────────┘
       ↑                                         │
  token emb                                    logits
  (GPU 0)                                    (GPU 6)
```

Per GPU per token: run `metal_graph_encode_decode_layer` for each assigned
layer. All weights already cached on this GPU.

## Implementation Plan

### Phase A: P2P Infrastructure

1. `ds4_gpu_enable_peer_access_all()`: iterate GPU pairs, enable P2P
2. `pp_layer_range[7][2]`: start/end layer indices per GPU
3. `pp_activation_buf[7]`: device buffers for inter-GPU activation transfer

### Phase B: Weight Caching

1. `pp_cache_weights(int gpu)`: for GPU `gpu`, cache all attention + MoE
   weights for assigned layer range using `cuda_model_range_ptr` + arena
2. Called during startup for each GPU

### Phase C: Decode Pipeline (Naive, Sequential)

```c
for (int g = 0; g < 7; g++) {
    cudaSetDevice(g);
    for (int il = pp_layer_range[g][0]; il <= pp_layer_range[g][1]; il++) {
        metal_graph_encode_decode_layer(gpu_graph[g], model, layer[il],
                                         il, pos, ..., token, /*skip_router=*/false);
    }
    // Transfer activation to next GPU
    if (g < 6)
        cudaMemcpyPeer(pp_activation_buf[g+1], g+1,
                       pp_activation_buf[g], g,
                       16*1024);  // 16KB HC residual
}
cudaSetDevice(0);
```

### Phase D: Pipeline Overlap (Advanced)

Overlap GPU execution with activation transfer using CUDA streams.
Each GPU runs async, activation copy overlaps with next GPU's first layer.

### Phase E: Per-GPU Sub-Graphs

Each GPU captures its layer range as sub-graphs. Replay via
`cudaGraphLaunch` per GPU → no per-layer kernel launch overhead.

## New Functions

```c
// ds4_cuda.cu
int ds4_gpu_pp_init(int ngpu);                    // enable P2P, alloc bufs
int ds4_gpu_pp_cache_weights(int gpu,             // cache weights for this GPU
                              const void *model_map, uint64_t model_size,
                              const ds4_weights *weights);
int ds4_gpu_pp_decode_token(int token, uint32_t pos, float *logits);
void ds4_gpu_pp_cleanup(void);

// ds4.c
// pp_layer_range: extern array [7][2] with layer start/end
// pp_activation_buf: extern device pointer array [7]
```

## Test Procedure

```bash
# 1-gpu baseline
./ds4 --backend cuda --model ds4flash.gguf -p "Hello" -n 32 --temp 0

# PP=7
DS4_CUDA_PP=1 ./ds4 --backend cuda --model ds4flash.gguf -p "Hello" -n 32 --temp 0

# Verify: generation t/s > 20, token text coherent
```

## Milestones

| Phase | Expected t/s | Description |
|---|---|---|
| Compact MoE (current) | 5.7 | Host-streamed 6-expert |
| + Sub-graph capture | ~10 | Encode overhead eliminated |
| + PP=7 naive | ~15-20 | Zero H2D, sequential GPU |
| + PP=7 pipelined | ~25 | Overlapping compute + transfer |
| + PP=7 sub-graphs | ~30 | Per-GPU captured graphs |
