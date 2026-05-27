# GPT-5.5 Pro への依頼書：DwarfStar4 PP decode illegal memory access の突破口

## 現在の状況（2026-05-27 19:45）

- **PP decode 速度**: 28-29 t/s（30TPS目標の95%達成）
- **残りの問題**: `illegal memory access` によるクラッシュ
- **コミット**: `d896ab5` 以降の変更あり（未コミット）

## 確定事実

1. **P2Pは有効**: `cudaMemcpyPeer` 動作確認済み、全GPUペアで `cudaDeviceCanAccessPeer=YES`
2. **重みキャッシュ**: 80.76 GiB / 7GPU分散完了
3. **Per-GPU activation tensor**: 各GPUにローカルに確保完了
4. **Per-GPU KV cache**: 各GPUにローカルに確保し、P2Pコピーで同期済み
5. **Deviceフィルタ**: `cuda_model_range` に `device` フィールドを追加し、`cuda_model_range_ptr` で `current_dev` と一致するキャッシュを優先
6. **P2P同期**: 各P2Pコピー後に `ds4_gpu_synchronize()`（= `cudaDeviceSynchronize()`）を挿入

## 詰まりポイント

### ポイントA: embed_token_hc_kernel が illegal memory access

```
ds4: embed token wptr=0x72d956000000 dev=0 type=2 offset=77928033088 bytes=1059061760 n_vocab=129280 token=19923
ds4: CUDA synchronize failed: an illegal memory access was encountered
```

- `wptr`: `cudaPointerGetAttributes` で `dev=0 type=2`（GPU0 device memory）確認済み
- `out_hc->ptr`: 同様に `dev=0 type=2` 確認済み
- `token=19923 < n_vocab=129280` で範囲内
- `cudaMalloc` + `cudaMemcpyHostToDevice` でキャッシュ構築済み（エラーなし）
- `ds4_gpu_synchronize()` を kernel 直後に入れたことで、**実際の犯人が embed token kernel** であることが判明
- `rms_norm_plain` は巻き添え（非同期エラーの遅延検出）

**疑問**: ポインタも範囲もすべて正しいのに、なぜ simple な read-only kernel がクラッシュするのか？

### ポイントB: g_cuda_decode_stream の影響

- `ds4_decode_stream()` は `dev != 0` の場合 `0`（default stream）を返す
- GPU0 の場合は `g_cuda_decode_stream`（non-blocking stream）を返す
- `g_cuda_decode_stream` は `ds4_gpu_init` で `cudaSetDevice(0)` 後に作成
- PP decode では `cudaSetDevice(gp)` を切り替えながら kernel を投入
- `embed_token_hc_kernel` は `g_cuda_decode_stream` で実行されるが、P2Pコピーは default stream（0）
- **両者は別ストリームなので非同期** → P2Pコピー完了前に kernel が読み込む可能性？
- しかし `wptr` と `out_hc->ptr` は P2P コピー対象外

### ポイントC: Blackwell + CUDA 13.2 + driver 595.71.05 の組み合わせ

- `cudaDeviceEnablePeerAccess` が Blackwell で BROKEN という既知の事実（コードコメントあり）
- `cudaMemcpyPeer` は動作確認済みだが、非同期性やストリーム間の同期に怪しい挙動がある可能性
- `cudaPointerGetAttributes` は成功するが、実際に kernel からアクセスできないメモリ状態がある可能性？
- **Blackwell 特有のメモリ保護/セキュリティ機能**が原因の可能性（ Confidential Computing?）

### ポイントD: cudaMalloc のメモリが実際にはアクセス不可能

- `ds4_gpu_cache_model_range_force` で `cudaMalloc(&dev, bytes)` → 成功
- `cudaMemcpy(dev, src, bytes, cudaMemcpyHostToDevice)` → 成功
- `cudaPointerGetAttributes(dev)` → `dev=0 type=2`（成功）
- しかし `embed_token_hc_kernel` が `dev` にアクセス → illegal memory access
- **cudaMalloc したメモリが、実際には別の GPU コンテキストに属している可能性？**
- `ds4_gpu_cache_model_range_force` 内で `cudaGetDevice(&current_dev)` を呼んでいるが、PP cache build ループ内で `ds4_gpu_pp_set_device(g)` を呼んだ直後に実行されるはず

### ポイントE: g_model_device_owned / g_model_registered の残滓

- `ds4_gpu_release_weight_cache_for_pp()` は `cuda_model_range_release_all()` を呼ぶが、`g_model_device_owned` や `g_model_registered` をクリアしない
- `cuda_model_range_ptr` で `g_model_ranges` にマッチしない場合、`g_model_device_owned || g_model_registered` が真なら `cuda_model_ptr`（ホストメモリ）を返す
- **ホストメモリをカーネルがアクセス → illegal memory access**
- しかし `token_embd` は `g_model_ranges` に確かに存在するはず
- もし `g_model_range_by_offset` や linear search で見落とされている場合、`cuda_model_ptr` にフォールバックする可能性

### ポイントF: cuda_model_range_ptr の exact match 失敗

- `g_model_range_by_offset` は `unordered_map<uint64_t, size_t>`
- `ds4_gpu_cache_model_range_force` で `g_model_range_by_offset[offset] = g_model_ranges.size() - 1` を設定
- `g_model_ranges` は `std::vector`
- `g_model_range_by_offset.find(offset)` でインデックスを取得し、`g_model_ranges[exact->second]` でエントリを取得
- **しかし `g_model_ranges` のインデックスが正しいか、完全に確認できていない**
- もし `exact->second` が無効なインデックスなら、未定義動作 → ゴミポインタ → illegal memory access

## 突破口のアイデア出し（依頼）

以下の観点から、何か思いつくことがあれば教えてください：

1. **Blackwell + CUDA 13.2 で `cudaMalloc` したメモリに kernel がアクセスできない**ケースは報告されていないか？
2. **`cudaPointerGetAttributes` は成功するが、実際には別デバイスのメモリ**と誤認識するケースはないか？
3. **`cudaDeviceEnablePeerAccess` が BROKEN な環境で、`cudaMemcpyPeer` だけでは不十分**で、追加のセットアップが必要なケースはないか？
4. **`g_cuda_decode_stream` と default stream（0）が混在**することで、メモリの一貫性が崩れるケースはないか？
5. **`cudaMalloc` の戻り値ポインタが、実際にはそのデバイスに関連付けられていない**ケースはないか？（例：`cudaSetDevice` の後に `cudaMalloc` を呼んだが、内部的に別デバイスに確保される）
6. **カーネルの `token` 引数が `uint32_t` なのに対し、カーネル内で `int32_t` にキャスト**しているが、符号拡張の問題はないか？
7. **`embed_token_hc_kernel` の `__half2float` で `wptr` のアライメント**に問題はないか？（`wptr` は `cudaMalloc` なのでアライメントは満たしているはず）
8. **`cudaGetLastError()` で拾えるエラーが、実際にはさらに前の非同期操作**（例：P2Pコピー）のものである可能性はないか？

## 次のステップ

1. `cudaMemset(g0->cur_hc->ptr, 0, bytes)` を `embed_token_hc_kernel` の前に実行して、kernel なしで illegal memory access が出るか確認
2. `cudaMemset` でエラーが出ないなら、kernel 引数の問題。出るならメモリポインタ自体の問題。
3. `wptr` の先頭 1 float を `cudaMemcpy` で GPU→CPU に読み出して、正しいデータが入っているか確認
4. `g_cuda_decode_stream` を使わずに、すべて default stream（0）で実行してみる
5. `cudaMalloc` の代わりに `cudaMallocManaged` を使ってみる

## 備考

- リポジトリ: `https://github.com/Tonoken3/DS4-For-SM120.git`
- ブランチ: `main`
- 最新コミット: `d896ab5` 以降、未コミットの変更あり
- 変更ファイル: `Makefile`, `ds4.c`, `ds4_cuda.cu`, `ds4_gpu.h`
