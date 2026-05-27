# GPT-5.5Pro への状況報告

## 環境

- GPU: NVIDIA RTX PRO 2000 Blackwell x7 (16GB x7)
- Driver: 595.71.05
- CUDA: 13.2 (nvcc 13.2.78)
- P2P: 全ペア `cudaDeviceCanAccessPeer=YES`、`cudaMemcpyPeer` 動作確認済み
- OS: Linux

## 実装済みの修正

### 1. NCCL リンク（Makefile）
- pip 版 `nvidia-nccl-cu13` をリンク
- `NCCL_HOME` を `~/.local/lib/python3.14/site-packages/nvidia/nccl` に設定
- `libnccl.so` のシンボリックリンクを作成

### 2. NCCL P2P path（ds4_cuda.cu）
- `#include <nccl.h>`
- `g_pp_nccl_comms[7]` / `g_pp_nccl_ready`
- `ncclCommInitAll` で 7GPU communicator を一括初期化
- `ds4_gpu_pp_p2p_copy_ptr` に NCCL `ncclSend`/`ncclRecv` path を追加
- `DS4_CUDA_PP_NCCL=1` で有効化

**問題**: `ncclGroupEnd`/`cudaStreamSynchronize` でデッドロックする可能性あり。現状は `DS4_CUDA_PP_NCCL` なしでテスト中（`cudaMemcpyPeer` fallback を使用）。

### 3. Per-GPU KV cache アロケーション（ds4.c `metal_graph_alloc_pp`）
- 各 GPU に担当 layer の KV cache tensor を新規アロケート
  - `layer_raw_cache`
  - `layer_attn_comp_cache`
  - `layer_attn_state_kv`
  - `layer_attn_state_score`
  - `layer_index_comp_cache`
  - `layer_index_state_kv`
  - `layer_index_state_score`
- 元の実装では `ref->layer_raw_cache[il]` のポインタをそのままコピーしていた（GPU0 のメモリを参照）

### 4. KV cache P2P コピー（ds4.c prefill後hook）
- `g_pp_graphs` allocate 後に、GPU0 の KV cache から各 GPU の KV cache へ `ds4_gpu_pp_p2p_copy_ptr` でコピー
- コピー後 `ds4_gpu_synchronize()`

### 5. PP decode path 書き換え（ds4.c）
- `metal_graph_encode_token_raw_swa_pp` を新設
- フロー:
  1. GPU0 で embed token
  2. for gpu in 0..6:
     - `cudaSetDevice(gpu)`
     - 担当 layer を `metal_graph_encode_decode_layer` で実行
     - `cur_hc <-> after_ffn_hc` swap
     - P2P copy `after_ffn_hc` -> next GPU `cur_hc`
  3. 最後の GPU から GPU0 へ final `after_ffn_hc` をコピー
  4. GPU0 で `metal_graph_encode_output_head`

### 6. Weight cache device フィルタ（ds4_cuda.cu）
- `cuda_model_range` に `int device` フィールドを追加
- `ds4_gpu_cache_model_range_force` で `cudaGetDevice(&current_dev)` を記録
- `cuda_model_range_ptr` で exact match / linear search 時に `current_dev` と一致するエントリを優先
- **理由**: 同じ `offset` を複数 GPU で登録すると `g_model_range_by_offset` が上書きされ、GPU0 の kernel が GPU6 の weight pointer を取得して illegal memory access していた

### 7. デバッグ関数
- `ds4_gpu_debug_tensor_ptr` を追加（`cudaPointerGetAttributes` で device/type/bytes を確認）

## 測定結果

| 設定 | generation t/s | 備考 |
|------|----------------|------|
| Baseline (PPなし) | 3.59 t/s | 正答 |
| PP 旧（g_pp_active のみ） | 16.27 t/s | ❌ illegal memory access |
| **PP 新（per-GPU graph + deviceフィルタ）** | **28.58 t/s** | ❌ rms_norm_plain illegal memory access |

→ **28.58 t/s** は 30TPS 目標の **95%** 達成。あとはクラッシュを直せば夢の 30TPS 目前。

## 現在の問題

```
ds4: PP KV cache copy done
ds4: g0->cur_hc ptr=0x7e6d539a4a00 device=0 type=2 bytes=65536
ds4: g0->flat_hc ptr=0x7e6d539b4a00 device=0 type=2 bytes=65536
ds4: CUDA rms_norm_plain launch failed: an illegal memory access was encountered
ds4: CUDA synchronize failed: an illegal memory access was encountered
```

- `embed_token_hc_kernel` は成功（`ds4_gpu_synchronize()` 後エラーなし）
- `g0->cur_hc` / `g0->flat_hc` は両方 `device=0`（GPU0）、`type=2`（device memory）で正しい
- `rms_norm_plain_kernel` は `out->ptr` / `x->ptr` のみをアクセスする単純な RMS norm kernel
- エラーは `cudaGetLastError()` で検出（kernel launch 時ではなく実行時の illegal memory access）

## 調査済みの事実

1. `ds4_gpu_init` は `cudaSetDevice(0)` して `g_cuda_decode_stream` を作成
2. `ds4_decode_stream()` は GPU0 の場合 `g_cuda_decode_stream` を返す
3. `rms_norm_plain_kernel` は `ds4_decode_stream()` で指定されたストリームで実行
4. `g0->cur_hc` / `g0->flat_hc` のポインタは `cudaPointerGetAttributes` で検証済み（GPU0 device memory）
5. `ds4_gpu_rms_norm_plain_tensor` の引数チェックも通過（bytes >= n * sizeof(float)）

## 次に調べてほしいこと

1. **なぜ `rms_norm_plain_kernel` が illegal memory access を起こすのか？**
   - `g_cuda_decode_stream` が実際に GPU0 に関連付けられているか確認
   - `cudaStreamGetFlags` や `cudaStreamQuery` でストリームの状態確認
   - `rms_norm_plain_kernel` の launch パラメータ（grid/block/shared）に問題がないか再確認
   - `cudaMemset` で `g0->cur_hc` / `g0->flat_hc` を 0 埋めしてから kernel を実行し、同じエラーになるか確認（データ依存性の排除）

2. **`cuda_model_range_ptr` の device フィルタが不完全な可能性**
   - `g_model_range_by_offset` は `unordered_map<uint64_t, size_t>` で、最後に登録されたインデックスを保持
   - exact match で `r.device == current_dev` が一致しない場合、linear search にフォールバック
   - linear search で `fallback` を返す場合、そのポインタが別 GPU のものになる可能性がある
   - `fallback` を返さずに `NULL` を返すべきか？（PP mode では各 GPU に必ずキャッシュがあるはず）

3. **`ds4_gpu_release_weight_cache_for_pp()` の影響**
   - `cuda_model_range_release_all()` で `g_model_ranges` をクリア
   - `cuda_q8_f16_cache_release_all()` で `g_q8_f16_ranges` をクリア
   - これらが GPU0 のみを対象としていて、他の GPU のキャッシュが正しく構築されているか確認

4. **`metal_graph_encode_decode_layer` 内の他の kernel**
   - `rms_norm_plain` より前に `ds4_gpu_hc_split_weighted_sum_norm_tensor` などが実行される可能性
   - 実際には `rms_norm_plain` が最初の kernel のはずだが、確認が必要

5. **`ds4_gpu_flush_commands()` / `ds4_gpu_end_commands()` の呼び出しタイミング**
   - `metal_graph_encode_token_raw_swa_pp` では `ds4_gpu_flush_commands()` を呼んでいない
   - 各 GPU の default stream に kernel が積まれているが、同期が不完全な可能性

## コミット情報

- ブランチ: main
- 変更ファイル: Makefile, ds4.c, ds4_cuda.cu, ds4_gpu.h
- 未追跡: scripts/ (bench_30tps.sh)

## 備考

- `DS4_CUDA_PP_NCCL` は現状デッドロックの可能性があるため無効化してテスト中
- `cudaMemcpyPeer` fallback で十分な速度が出ている（28.58 t/s）
- あとは **1つの illegal memory access** を潰せば、正答性確認 + 30TPS 達成が目前
