#!/usr/bin/env bash
set -euo pipefail

PROMPT="${1:-Hello}"
TOKENS="${TOKENS:-32}"
CTX="${CTX:-256}"
MODEL="${MODEL:-ds4flash.gguf}"

./ds4 --backend cuda --model "$MODEL" --temp 0 --nothink -c "$CTX" -n "$TOKENS" -p "$PROMPT" \
  > /tmp/ds4_base.out 2> /tmp/ds4_base.err

DS4_CUDA_DECODE_GRAPH=1 \
./ds4 --backend cuda --model "$MODEL" --temp 0 --nothink -c "$CTX" -n "$TOKENS" -p "$PROMPT" \
  > /tmp/ds4_graph_fallback.out 2> /tmp/ds4_graph_fallback.err

diff -u /tmp/ds4_base.out /tmp/ds4_graph_fallback.out

echo "baseline:"
grep -E "generation:|prefill:" /tmp/ds4_base.err || true

echo "graph fallback:"
grep -E "generation:|prefill:|sub-graph|graph token" /tmp/ds4_graph_fallback.err || true
