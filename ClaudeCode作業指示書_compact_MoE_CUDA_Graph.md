# ClaudeCode 作業指示書: (1) Compact MoE Batch対応 + (2) CUDA Graphで30TPS

## 依頼者
ケン（Tonoken3）

## 対象外（やらないこと）
- **小さなMTPモデルの作成（再量子化等）はなし**: 既存の `DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` (3.5GB) をそのまま使う
- MTP draft=3 対応は今回のスコープ外（draft=2 まで）

---

## 作業 (1): Compact MoE Batch対応（MTP draft=2向け）

### 背景
DwarfStar4 の PP（Pipeline Parallelism）decode で MTP（Multi-Token Prediction）を使うと OOM になる。原因は `n_tokens > 1` の batch path で compact MoE（6 experts のみコピー）が無効化され、full MoE（256 experts = 672MB）を一時確保しようとして GPU メモリ不足になるため。

### 目標
MTP draft=2（n_tokens=2）でも compact MoE を使えるようにし、OOM を解消する。

### 前提知識

#### 現在のコード構造
- `ds4_cuda.cu` の `routed_moe_launch()` が MoE の dispatch を行う
- compact MoE は `n_tokens == 1 && n_expert <= 6` の場合のみ有効
- compact scratch は `g_moe_compact_gate/up/down` に 6 experts 分だけコピー
- full MoE は `cudaMalloc` で 256 experts 分（672MB）を一時確保 → OOM

#### 本プロジェクトの制約
- **RTX PRO 2000 Blackwell x7, VRAM 16GB/GPU**
- `cudaDeviceEnablePeerAccess` が Blackwell driver で破損 → P2P は `cudaMemcpyPeer` のみ
- `DS4_CUDA_MOE_PERSISTENT_SCRATCH=1` で scratch を確保できるが、3.38GB あり VRAM を圧迫
- **30TPS 達成が最優先**、MTP は加速手段の一つ

### 修正内容（優先順位順）

#### Phase 1: compact MoE 条件の緩和（必須）

**ファイル**: `ds4_cuda.cu`

```c
// 現在（line ~11406）:
if ((use_temp_weights || g_pp_decode_active) && n_tokens == 1 && n_expert <= 6) {

// 修正後:
if ((use_temp_weights || g_pp_decode_active) && n_tokens <= 2 && n_expert <= 6) {
```

#### Phase 2: scratch サイズを n_tokens 倍に（必須）

**ファイル**: `ds4_cuda.cu`

compact scratch の確保サイズを `n_expert * n_tokens` に変更。

```c
// 現在:
uint64_t total = (uint64_t)n_expert * max_per;
if (cudaMalloc(&g_moe_compact_gate, (size_t)total) == cudaSuccess && ...)

// 修正後:
uint64_t total = (uint64_t)n_expert * n_tokens * max_per;
if (cudaMalloc(&g_moe_compact_gate, (size_t)total) == cudaSuccess && ...)
// selected_dev も同様に:
cudaMalloc(&g_moe_compact_selected_dev, (size_t)n_expert * n_tokens * 4)
```

#### Phase 3: selected indices のコピーに token 次元を追加（必須）

**ファイル**: `ds4_cuda.cu`

現在は `host_selected[6]` に 6 個の expert index を入れているが、batch 対応では `(token, expert)` の2次元にする。

```c
// compact_remap のサイズ拡張
// 現在: int32_t compact_remap[6];
// 修正後: int32_t compact_remap[12];  // max 6 experts * 2 tokens

// コピーループを修正
for (uint32_t ti = 0; ti < n_tokens; ti++) {
    for (uint32_t ei = 0; ei < n_expert; ei++) {
        int ge = host_selected[ti * n_expert + ei];
        // ...
        compact_remap[ti * n_expert + ei] = (int32_t)ei;
        // gate/up/down のコピー先 offset も ti * n_expert + ei に
    }
}
```

#### Phase 4: kernel 側で token offset を適用（最重要・最難）

**ファイル**: `ds4_cuda.cu` の `routed_moe_launch()` および各 MoE kernel

compact weight を使う場合 (`use_compact == 1`)、各 token の kernel が自分の slot を読めるようにする。

##### 4a. `selected_ptr` のアクセスに stride を追加

```c
// 現在: selected_ptr[expert_idx] で global expert index を得る
// 修正後: selected_ptr[token_idx * n_expert + expert_idx]
```

##### 4b. weight pointer に token offset を追加

```c
// compact weight のアクセス
// 現在: gate_w + expert_idx * gate_expert_bytes
// 修正後: gate_w + (token_idx * n_expert + expert_idx) * gate_expert_bytes
```

##### 4c. kernel 引数の変更検討

`routed_moe_launch` 内で compact 用の `selected_ptr` と `weight_ptr` に stride/offset パラメータを追加するか、呼び出し側で token ごとにポインタを計算して渡す。

**推奨**: 呼び出し側で `selected_ptr + ti * n_expert` および `gate_w + ti * n_expert * gate_expert_bytes` を計算して `routed_moe_launch` に渡す（kernel 変更を最小化）。

#### Phase 5: MTP 呼び出し側の確認（必要に応じて）

**ファイル**: `ds4.c`

`metal_graph_eval_mtp_draft_from_hc()` で `n_raw`（= n_tokens）がどう計算されているか確認。draft=2 で `n_tokens=16` になっていたケースがあるので、呼び出し側で正しく `n_tokens` が渡されているか確認。

### テスト手順（作業1）

#### 1. ビルド
```bash
cd /run/media/tonoken3/DATA2/Lna-Lab/DwarfStar4
make cuda CUDA_ARCH=sm_120 -j$(nproc)
```

#### 2. テストコマンド
```bash
# まず non-MTP PP で回帰テスト
DS4_CUDA_PP=1 DS4_CUDA_PP_DELAY_RESIDENT=1 DS4_CUDA_DECODE_GRAPH=0 \
./ds4 --backend cuda --model ds4flash.gguf \
-p "What is the capital of France?" -n 8 --temp 0

# MTP draft=2 で OOM 解消確認
DS4_CUDA_PP=1 DS4_CUDA_PP_DELAY_RESIDENT=1 DS4_CUDA_DECODE_GRAPH=0 \
./ds4 --backend cuda --model ds4flash.gguf \
--mtp /run/media/tonoken3/DATA2/Models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf \
--mtp-draft 2 \
-p "What is the capital of France?" -n 16 --temp 0
```

#### 3. 成功基準
- `exit=0` で正常終了
- 出力が non-MTP PP と同じ（または妥当な文章）
- stderr に `CUDA transient model alloc failed` が **出ない**
- stderr に `ds4: CUDA MoE compact selected-expert copy` が出る

---

## 作業 (2): CUDA Graph で 30TPS 達成

### 背景
現在の PP decode は 20.6 t/s。目標は 30 t/s（33.3 ms/token）。プロファイル結果から、各 GPU の layer chunk 計算が 6.8-7.8 ms で合計約 49 ms/token が支配的。P2P 転送は 0.02 ms で無視できる。

### 目標
per-GPU CUDA Graph 化により kernel launch overhead を削減し、30TPS を達成する。

### 前提知識

#### 現在の速度データ
```
embed=0.01 ms
gpu0=7.8 ms
gpu1-6=6.8-7.1 ms
p2p=0.02 ms
output=2.1 ms
total=51-52 ms/token → 19.5-20.6 t/s
```

#### 既存の Graph 基盤
- `ds4_cuda.cu` に `g_cuda_decode_graph` / `g_sub_graph_exec[2][43]` がある
- `ds4_decode_stream()` で graph capture/replay が可能
- ただし PP 対応はまだ不完全

### 修正内容

#### Phase A: GPU1〜GPU5 の layer chunk Graph 化（本命）

**ファイル**: `ds4.c`, `ds4_cuda.cu`

1. **Graph capture 用の API 追加**:
   - `ds4_gpu_decode_graph_capture_begin(gpu)`
   - `ds4_gpu_decode_graph_capture_end(gpu)`
   - `ds4_gpu_decode_graph_launch(gpu)`

2. **PP loop の変更**:
   ```c
   // 現在: 各層で metal_graph_encode_decode_layer() を直接呼ぶ
   // 修正後: GPU1〜5 は graph launch、GPU0/GPU6 は従来通り
   ```

3. **dynamic params の扱い**:
   - `pos`, `raw_row`, `n_raw`, `token` は `ds4_cuda_dev_params` (device symbol) に書き込み
   - graph replay 前に `cudaMemcpyToSymbolAsync` で更新
   - `layer_n_comp` も同様

#### Phase B: GPU argmax（greedy decode 高速化）

**ファイル**: `ds4_cuda.cu`

`--temp 0` の場合、full logits readback（0.5MB）をやめて、GPU 上で argmax kernel を走らせ、int32 token だけを CPU に返す。

```c
// argmax_kernel<<<1, 256>>>(logits, vocab_size, &next_token_dev);
// cudaMemcpyAsync(&next_token_host, &next_token_dev, sizeof(int32), D2H);
```

#### Phase C: Profile ベース layer rebalance

**ファイル**: `ds4.c`

`DS4_CUDA_PP_LAYER_SPLIT` env var で手動設定はできるが、自動最適化は未実装。測定スクリプトを作り、各 GPU chunk の ms を計測して最適 split を導出。

### テスト手順（作業2）

#### 1. ベンチマーク
```bash
# baseline
DS4_CUDA_PP=1 DS4_CUDA_PP_DELAY_RESIDENT=1 DS4_CUDA_DECODE_GRAPH=0 \
./ds4 --backend cuda --model ds4flash.gguf \
-p "What is the capital of France?" -n 32 --temp 0

# Graph 化後
DS4_CUDA_PP=1 DS4_CUDA_PP_DELAY_RESIDENT=1 DS4_CUDA_DECODE_GRAPH=0 \
DS4_CUDA_PP_GRAPH_CHUNKS=1 \
./ds4 --backend cuda --model ds4flash.gguf \
-p "What is the capital of France?" -n 32 --temp 0
```

#### 2. 成功基準
- `exit=0` で正常終了
- 出力が baseline と一致
- **generation speed >= 30.0 t/s**
- クラッシュや不正答がない

### 期待効果（リナの見積もり）
```
現状                             20.6 t/s
GPU argmax                       21.5〜22.5 t/s
GPU1〜5 CUDA Graph               24〜27 t/s
GPU0/GPU6もGraph化               26〜29 t/s
profile based layer split         27〜31 t/s
合計                              30〜34 t/s
```

---

## 優先順位

1. **作業 (2) Phase A**（GPU1〜5 Graph 化）: 30TPS の本命、最優先
2. **作業 (1)**（compact MoE）: MTP を使えるようにするため
3. **作業 (2) Phase B**（GPU argmax）: マイクロ最適化、Graph 後に実施
4. **作業 (2) Phase C**（layer rebalance）: 最後の微調整

## 注意事項

1. **正答性 > 速度**: 30TPS を達成しても出力が壊れては意味がない。必ず baseline との diff を取ること。
2. **kernel 変更は最小限に**: `routed_moe_launch` の呼び出し側でポインタを調整し、各 MoE kernel の内部は変更しない（リグレッションリスク削減）。
3. **コミット単位**: 作業 (1) と (2) は別コミットにすること。 `git add ds4.c ds4_cuda.cu ds4_gpu.h && git commit -m "..." && git push`
4. **ClaudeCode の制限**: もし kernel 側の変更が複雑すぎる場合は、作業 (1) は「MTP を single-token ずつに分解する」簡易実装にフォールバックしてよい。

## 関連ファイル

| ファイル | パス |
|----------|------|
| 本体コード | `/run/media/tonoken3/DATA2/Lna-Lab/DwarfStar4/ds4.c` |
| CUDA カーネル | `/run/media/tonoken3/DATA2/Lna-Lab/DwarfStar4/ds4_cuda.cu` |
| GPU API ヘッダ | `/run/media/tonoken3/DATA2/Lna-Lab/DwarfStar4/ds4_gpu.h` |
| MTP モデル | `/run/media/tonoken3/DATA2/Models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf` |
| ターゲットモデル | `/run/media/tonoken3/DATA2/Models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf` |
| リモートリポジトリ | `https://github.com/Tonoken3/DS4-For-SM120.git` |

## 緊急連絡先

- 作業中に OOM 以外のエラー（Segfault等）が出たら即停止して報告
- 30TPS よりも「正答性の維持」を優先してほしい
- 不明点があれば即質問すること（推測で進めない）

---

ケンより

この2つの作業が完了すれば、PP + MTP で 30TPS 達成が現実的になります。よろしくお願いします！
