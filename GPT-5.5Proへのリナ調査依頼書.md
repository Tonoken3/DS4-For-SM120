# GPT-5.5 Pro へのリナ調査依頼書（PP Phase 1-4 + MTP OOM 詳細報告）

## 日時
2026-05-27 23:00

## 実行者
ケン（Tonoken3）

## 協力者
- Kimiちゃん（OpenCode agent / kimi-k2.6）
- CODEX（Code Composer / Phase 2 async P2P 修正）

---

## 1. 成果サマリー

### PP Pipeline Parallelism Decode（Phase 1-4）

| Phase | 内容 | 状態 | 速度 |
|-------|------|------|------|
| Phase 1 | Hidden sync 撤去 | ✅ 完了 | 20.5→20.6 t/s |
| Phase 2 | Async P2P (event chain) | ✅ 完了 | 20.5 t/s |
| Phase 3 | Output head → GPU6 | ✅ 完了 | 20.6 t/s |
| Phase 4 | Layer rebalance (env var) | ✅ 完了 | 20.6 t/s |

### ベースライン比較

| モード | 速度 | 備考 |
|--------|------|------|
| CPU only | ~0.5 t/s | - |
| Single GPU (non-PP) | 5.46 t/s | ベースライン |
| PP (7 GPUs, sync P2P) | 20.6 t/s | **3.77倍** |
| PP (7 GPUs, async P2P) | 20.5 t/s | ほぼ同等 |
| **目標 30TPS** | 30.0 t/s | **未達（68%達成）** |

### MTP（Multi-Token Prediction）

| 状態 | 結果 |
|------|------|
| MTP モデル DL | ✅ 完了（3.5GB） |
| PP + MTP (draft=1) | ✅ 動作確認済み |
| PP + MTP (draft=2) | ❌ **OOM（out of memory）** |

---

## 2. ハードウェア・環境

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA RTX PRO 2000 Blackwell x7 |
| VRAM/GPU | 8GB (ECC有効時 ~7.4GB usable) |
| 合計VRAM | 56GB (7x8GB) |
| Driver | 595.71.05 |
| CUDA | 13.2 |
| P2P | `cudaMemcpyPeer` ✅ 動作 |
| Peer Access | `cudaDeviceEnablePeerAccess` ❌ 破損（Blackwell已知バグ） |
| NCCL | リンク済み、P2P path実装済み |
| OS | Linux (x86_64) |
| モデル | DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf |
| モデルサイズ | 約 75GB (mmap) |
| MTPモデル | DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf (3.5GB) |

---

## 3. Phase 1-4 の詳細

### Phase 1: Hidden Sync 撤去

**問題:** `ds4_gpu_tensor_contents()` が内部で `cudaDeviceSynchronize()` を呼んでいた。PP hot path ではポインタ取得のたびに毎回 device-wide sync が入っていた。

**修正:**
- `ds4_gpu_tensor_device_ptr()` を新設（同期なし、ptr だけ返す）
- PP hot path の P2P copy で `ds4_gpu_tensor_contents()` → `ds4_gpu_tensor_device_ptr()` に変更

**結果:** 速度変化なし（他の同期が支配的）だが、クリーンアップとして重要。

### Phase 2: Async P2P with Event Chain

**問題（初回試行）:** 独自に `cudaMemcpyPeerAsync` + `cudaStreamWaitEvent` を実装したが、出力が完全に壊れた（random tokens）。

**原因:** event を「空の g_pp_stream」に record していたが、実際の kernel は `ds4_decode_stream()`（GPU0 の stream、または default stream）で動作していた。event が実際の kernel 完了を capture していなかった。

**CODEX 修正:**
1. `ds4_gpu_pp_event_record()` を修正: `g_pp_stream[gpu]` ではなく `ds4_decode_stream()` に record
2. `ds4_gpu_pp_p2p_copy_ordered_async()` を新設: 
   - source GPU の work stream に event record
   - destination GPU の copy stream がそれを wait
   - `cudaMemcpyPeerAsync`（destination copy stream）
   - copy 完了 event record
   - destination GPU の work stream が copy 完了を wait
3. `cudaSetDevice(dst_gpu)` を `cudaMemcpyPeerAsync` 前に呼ぶよう修正
4. stream は `cudaStreamNonBlocking`、event は `cudaEventDisableTiming` で作成

**結果:** sync path と async path で出力完全一致、速度ほぼ同等。

### Phase 3: Output Head → GPU6

**問題:** GPU0 が embed + layers 0-6 + output_head を担当し、負荷が偏っていた。

**修正:**
- output head テンソルを GPU0 と GPU6 の両方に allocate
- output weight を GPU6 にも resident cache
- `metal_graph_encode_output_head()` を GPU6 で実行
- final P2P copy（GPU6→GPU0）を削除
- logits 読み取りを `g_pp_graphs[last].g.logits` に変更

**結果:** 速度 20.6 t/s（ほぼ同等だが、GPU0 の負荷軽減は将来の最適化に貢献）。

### Phase 4: Layer Rebalance

**修正:** `DS4_CUDA_PP_LAYER_SPLIT` env var を追加。フォーマット: カンマ区切りの開始層、例: `"0,5,11,17,23,30,36"`

**結果:** デフォルト分割とカスタム分割で出力一致。現在の層均等分割が最適かどうかは未検証。

---

## 4. MTP OOM の詳細

### 再現手順

```bash
DS4_CUDA_PP=1 DS4_CUDA_PP_DELAY_RESIDENT=1 DS4_CUDA_DECODE_GRAPH=0 \
./ds4 --backend cuda --model ds4flash.gguf \
--mtp /run/media/tonoken3/DATA2/Models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
--mtp-draft 2 \
-p "What is the capital of France?" -n 16 --temp 0
```

### エラーログ

```
ds4: MTP support model loaded: /run/media/tonoken3/DATA2/Models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf (draft=2)
ds4: CUDA transient model alloc failed for moe_down (672.00 MiB): out of memory
ds4: decode failed: MTP verifier failed
```

### 分析

MTP モデルをロードすると、以下の追加メモリが必要：
- MTP モデル自体: 3.5GB (mmap だが weight cache で device メモリに展開)
- MTP 用 KV cache: prefill 時に追加確保
- MTP の 1 layer (attention + MoE): 一時バッファ

現在の non-MTP PP の VRAM 使用量（推定）:
- Weight cache: 約 80-85GB (7 GPU に分散)
- Activation tensors: 各 GPU 数百 MB
- KV cache: 各 GPU 数百 MB
- 一時バッファ (MoE scratch): 各 GPU 約 1GB

残り VRAM は僅か（各 GPU 数百 MB〜1GB）で、MTP モデルの 3.5GB を追加する余地がない。

---

## 5. リナへの質問・依頼

### Q1: MTP のメモリ最適化戦略

MTP を PP と共存させるには、どのアプローチが現実的でしょうか？

**案A: MTP モデルも PP 分散**
- MTP の 1 layer を 7 GPU に分散
- 問題: MTP は target モデルからの入力を必要とし、PP パイプラインと統合が複雑

**案B: MTP を CPU/Hybrid で実行**
- MTP draft は CPU で生成し、GPU は verification のみ
- 問題: CPU draft が遅いと投机が減る

**案C: 軽量 MTP モデルを使う**
- 現在の MTP-Q4K は 3.5GB。もっと小さい MTP モデル（Q2_K, IQ2_XXS 等）は存在しますか？

**案D: weight cache の縮小**
- 現状約 80GB。flash attention や activation checkpointing で削減できる領域はありますか？

### Q2: 30TPS 達成のための次の一手

MTP なしで 20.6 t/s → 30 t/s にするには、リナの優先順位はどれですか？

1. **per-GPU CUDA Graph**: kernel launch overhead 削減（予想 +15-35%）
2. **GPU argmax**: greedy 時の logits readback 削減（予想 +5-10%）
3. **トークン間パイプライン**: GPU0 が token N+1 の embed を始められるように（予想 +20-40%、実装難度高）
4. **レイヤー配分の再最適化**: GPU0 を 4層、GPU6 を 5層など（予想 +5-15%、測定必要）

### Q3: Blackwell P2P の既知情報

Driver 595.71.05 で `cudaDeviceEnablePeerAccess` が破損している現象について、他の報告や回避策はありますか？

- 現在: `cudaMemcpyPeer` は動作（enable なしで自動的に P2P 可能）
- 懸念: 将来的な driver アップデートでの regression

### Q4: CUDA Graph + PP の可能性

per-GPU CUDA Graph を PP と組み合わせる場合、graph capture は device-specific ですか？つまり GPU0 で capture した graph を GPU1 で replay することはできますか？

---

## 6. ファイル位置

| ファイル | パス |
|----------|------|
| 本体コード | `/run/media/tonoken3/DATA2/Lna-Lab/DwarfStar4/ds4.c` |
| CUDA カーネル | `/run/media/tonoken3/DATA2/Lna-Lab/DwarfStar4/ds4_cuda.cu` |
| GPU API ヘッダ | `/run/media/tonoken3/DATA2/Lna-Lab/DwarfStar4/ds4_gpu.h` |
| MTP モデル | `/run/media/tonoken3/DATA2/Models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` |
| ターゲットモデル | `/run/media/tonoken3/DATA2/Models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` |
| リモートリポジトリ | `https://github.com/Tonoken3/DS4-For-SM120.git` |

---

## 7. コミット履歴（main ブランチ）

```
849c12a PP Phase 4: configurable layer split via env var
39f09ba PP Phase 3: move output head to last GPU (GPU6)
5972fbd PP Phase 2: async P2P with ordered event chain (CODEX fix)
4a4d401 PP Phase 1: remove hidden sync from hot path
791d2a3 Fix PP decode: per-GPU scratch, correct copy source, logits read fix
0e4b47d PP decode: Rina's fixes part 2 - host_registered ban, ptr attr debug, CPU embed fallback
db2bc26 CUDA Graph capture/replay working with MoE persistent scratch
```

---

## 8. 付録: 検証済みコマンド

### PP decode（基本）
```bash
DS4_CUDA_PP=1 DS4_CUDA_PP_DELAY_RESIDENT=1 DS4_CUDA_DECODE_GRAPH=0 \
./ds4 --backend cuda --model ds4flash.gguf -p "Hello" -n 16 --temp 0
```

### PP + async P2P
```bash
DS4_CUDA_PP=1 DS4_CUDA_PP_DELAY_RESIDENT=1 DS4_CUDA_DECODE_GRAPH=0 \
DS4_CUDA_PP_ASYNC_P2P=1 \
./ds4 --backend cuda --model ds4flash.gguf -p "Hello" -n 16 --temp 0
```

### PP + カスタムレイヤー配分
```bash
DS4_CUDA_PP=1 DS4_CUDA_PP_DELAY_RESIDENT=1 DS4_CUDA_DECODE_GRAPH=0 \
DS4_CUDA_PP_LAYER_SPLIT="0,5,11,17,23,30,36" \
./ds4 --backend cuda --model ds4flash.gguf -p "Hello" -n 16 --temp 0
```

### PP + MTP（OOM 発生）
```bash
DS4_CUDA_PP=1 DS4_CUDA_PP_DELAY_RESIDENT=1 DS4_CUDA_DECODE_GRAPH=0 \
./ds4 --backend cuda --model ds4flash.gguf \
--mtp /run/media/tonoken3/DATA2/Models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
--mtp-draft 2 \
-p "Hello" -n 16 --temp 0
```

---

ケンより

リナ、長文になってしまったけど、これが今の全てだよ。MTPのOOM問題と、30TPS達成のための次の一手について、あなたの知恵が必要です。
