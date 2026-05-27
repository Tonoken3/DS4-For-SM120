# GPT-5.5 Pro への追加報告（2026-05-27 20:30）

## リナのパッチ適用結果

### 適用済みパッチ

1. **embed_token_hc_kernel に n_vocab + clamp を追加** ✅
2. **wrapper 側でも token >= n_vocab で clamp** ✅
3. **PP decode 開始前に全 GPU で ds4_gpu_decode_params_deactivate()** ✅
4. **cuda_model_range_ptr の PP 中 fallback 禁止** ✅
5. **DS4_CUDA_PP_DEFAULT_STREAM=1 で non-blocking stream を封印** ✅
6. **ds4_debug_decode_symbol_token で device symbol の値を確認** ✅

### 測定結果

```
ds4: pp_embed_before dev=0 active=0 sym_token=0 host_token=19923 n_vocab=129280
ds4: PP memset test OK - memory is accessible
ds4: embed token wptr first_val=-0.107422
ds4: CUDA synchronize failed: an illegal memory access was encountered
```

- **device symbol**: `active=0` → 無効。`sym_token=0` → 関係ない。✅
- **memset test**: `g0->cur_hc` のメモリは kernel なしでアクセス可能。✅
- **wptr readback**: `cudaMemcpy` で先頭データ `-0.107422` を正常に読み出し。✅
- **しかし**: `embed_token_hc_kernel` の後に `illegal memory access`。❌

### 結論

**device symbol も boundary check も stream も無関係。**
**`wptr` は CPU から読めるが、kernel から読めない。**

これは以下のいずれか：

1. **Blackwell + CUDA 13.2 の謎バグ**
   - `cudaMalloc` したメモリに `cudaMemcpy` は成功するが、kernel からのアクセスだけが失敗する
   - または `cudaMemcpyHostToDevice` でコピーされたデータが、kernel から見ると別のメモリ領域になっている

2. **カーネルの引数渡しに問題**
   - `embed_token_hc_kernel` は非常にシンプルだが、何らかの理由で引数が壊れている可能性
   - `token=19923` は正しいが、カーネル内では別の値になっている？
   - `ds4_cuda_params_active` は `active=0` だが、カーネル内では別の値を読んでいる？

3. **メモリ保護/セキュリティ機能**
   - Blackwell の Confidential Computing や他のセキュリティ機能が、kernel アクセスをブロックしている可能性
   - `cudaPointerGetAttributes` や `cudaMemcpy` は bypass されるが、kernel はブロックされる

### 次に試すべきこと

1. **compute-sanitizer --tool memcheck**
   - これが最も効果的な次のステップ
   - `cudaMemset` は成功するが kernel は失敗する、という状況を詳細に診断できる

2. **embed_token_hc_kernel の引数を全部プリント**
   - カーネル内で `token`, `n_vocab`, `n_embd`, `n_hc`, `w`, `out` の値を `printf` してみる
   - 実際にカーネルがどの値を見ているか確認

3. **embed_token_hc_kernel を最小化**
   - `w` と `token` を使わず、単に `out[i] = 1.0f;` だけにしてみる
   - これで成功するなら `w` アクセスが原因
   - 失敗するなら `out` アクセスが原因（でも memset は成功しているので謎）

4. **embed_token_hc_kernel を CPU 実装にフォールバック**
   - `ds4_gpu_embed_token_hc_tensor` で `cudaMemcpy` を使って CPU で計算し、結果を `cudaMemcpy` で GPU に戻す
   - これで PP decode が全体として動くか確認

5. **別カーネルで同じ wptr を読んでみる**
   - `wptr` の先頭 1 element を読むだけのダミーカーネルを作って試す
   - ダミーカーネルでも失敗するなら `wptr` 自体の問題

6. **g_model_device_owned / g_model_registered をクリア**
   - `ds4_gpu_release_weight_cache_for_pp()` でこれらをクリアする
   - 万が一 `cuda_model_range_ptr` がフォールバックして `cuda_model_ptr` を返していても、PP 中は禁止しているはず

### 最終手段

- **compute-sanitizer --tool memcheck** を実行して、正確なアクセス違反の場所を特定する
- これが出せば、あとはその 1 行を直すだけ

---

ケンより

夢の30TPS、もう少しだと思うけど、ここが最後の壁。
