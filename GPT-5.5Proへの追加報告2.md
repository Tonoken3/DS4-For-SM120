# GPT-5.5 Pro への追加報告2（2026-05-27 20:55）

## 重大発見

### compute-sanitizer + CUDA_LAUNCH_BLOCKING=1 で PP decode が動いた

```bash
CUDA_LAUNCH_BLOCKING=1 \
DS4_CUDA_PP=1 DS4_CUDA_PP_DELAY_RESIDENT=1 DS4_CUDA_PP_DEFAULT_STREAM=1 DS4_CUDA_DECODE_GRAPH=0 \
compute-sanitizer --tool memcheck --show-backtrace yes --error-exitcode 99 \
./ds4 --backend cuda --model ds4flash.gguf -p "Hello" -n 1 --temp 0
```

**結果**: `Hello` が正しく出力された！`exit=0` で正常終了。

```
ds4: PP decode enabled
...
ds4: PP GPU 6 graph allocated (layers 37-42)
ds4: PP KV cache copy done
Hello
ds4: prefill: 1.32 t/s, generation: 14907.13 t/s
```

### しかし -n=8 では失敗

同じ条件で `-n 8` を実行すると `exit=99`（compute-sanitizer のエラーコード）で失敗。

エラーが変化した：

```
ds4: CUDA matmul_q8_0 quantize launch failed: an illegal memory access was encountered
```

以前は `rms_norm_plain` で失敗していたが、今回は `matmul_q8_0` で失敗。**weight pointer の問題ではなく、非同期性の累積**が示唆される。

### 検証済み事実

1. **device symbol**: `active=0` → 無関係 ✅
2. **boundary check**: `token=19923 < n_vocab=129280` → 範囲内 ✅
3. **wptr 属性**: `current_dev=0 attr.device=0 attr.type=2` → GPU0 device memory ✅
4. **wptr 実アクセス行 readback**: `row_off=163209216 first=no error/-0.143555 last=no error/0.0727539` ✅
5. **memset test**: `g0->cur_hc` に `fill_f32` → 成功 ✅
6. **`-n=1` で PP decode 成功** → 1 token 目は動く ✅
7. **`-n=8` で失敗** → 2 token 目以降で問題発生 ❌

### 結論

**非同期性が累積して、2 token 目以降でメモリ一貫性が崩れている。**

`CUDA_LAUNCH_BLOCKING=1` は kernel を同期的に実行するが、P2Pコピー（`cudaMemcpyPeer`）は非同期的のまま。1 token 目では P2Pコピーが少ないので成功するが、2 token 目以降で P2Pコピーが kernel 完了前に実行され、未完了の転送領域に kernel がアクセスする可能性がある。

### 次のステップ

1. **P2Pコピーを同期的に実行する**
   - `ds4_gpu_pp_p2p_copy_ptr` の後に `cudaStreamSynchronize(cudaStreamDefault)` を追加
   - あるいは `cudaMemcpyPeer` の代わりに `cudaMemcpyPeerAsync` + `cudaStreamSynchronize` を使う

2. **各 GPU の kernel 完了を明示的に待つ**
   - `metal_graph_encode_decode_layer` の後に `cudaDeviceSynchronize()` を追加
   - あるいは `cudaStreamSynchronize(ds4_decode_stream())` を追加

3. **`-n=1` の成功を `-n=8` に拡張する**
   - 上記の同期を追加して、`-n=8` で成功するか確認

4. **compute-sanitizer は Blackwell をサポートしていない**
   - `Device not supported` と出ている
   - ただしツールのロードによる副作用で `-n=1` が成功した可能性がある
   - `compute-sanitizer` なしで `CUDA_LAUNCH_BLOCKING=1` だけで `-n=1` が成功するか確認

---

ケンより

30TPSの扉が見えてきた。1 token 目は動く。あとは非同期性を制御するだけ。
