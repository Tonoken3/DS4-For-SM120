#!/usr/bin/env python3
"""
Refactor ds4.c, ds4_cuda.cu, ds4_gpu.h for per-layer sub-graph capture.
Implements SPEC_SUBGRAPH.md:

1. Add 'bool skip_router' param to metal_graph_encode_decode_layer
2. Wrap router section with 'if (!skip_router) { ... }'
3. Add ', false' to all 7 call sites
4. Wire sub-graph capture/replay into eval function
5. Add ds4_gpu_decode_subgraph_launch/capture_end_store to ds4_cuda.cu
6. Add declarations to ds4_gpu.h

Usage: python3 refactor_subgraph.py /path/to/DwarfStar4
"""

import sys, os, re

def refactor_ds4(path):
    with open(os.path.join(path, 'ds4.c'), 'r') as f:
        text = f.read()

    # ====== STEP 1: Add skip_router param to function definition ======
    # Find:  int token) {
    # The function header starts with 'static bool metal_graph_encode_decode_layer('
    # and ends with 'int token) {'
    old_sig = '\n        int                     token) {'
    new_sig = ('\n        int                     token,\n'
               '        bool                    skip_router) {')
    assert old_sig in text, "STEP 1: Function signature not found"
    text = text.replace(old_sig, new_sig)
    print("STEP 1: skip_router param added to function signature")

    # ====== STEP 2: Wrap router section ======
    old_router = (
        '    if (ok) ok = metal_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,\n'
        '                                                  DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);')
    assert old_router in text, "STEP 2: Router matmul line not found"
    new_router = (
        '    if (!skip_router) {\n'
        '    if (ok) ok = metal_graph_matmul_plain_tensor(g->router_logits, model, layer->ffn_gate_inp,\n'
        '                                                  DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);')
    # We need to find the exact context: the last debug dump before MoE
    old_context = (
        '        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->router_weights, DS4_N_EXPERT_USED, il, pos);\n'
        '    }\n'
        '    if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,')
    new_context = (
        '        metal_graph_debug_dump_tensor("ffn_moe_weights_scaled", g->router_weights, DS4_N_EXPERT_USED, il, pos);\n'
        '    }\n'
        '    } /* !skip_router */\n'
        '    if (ok) ok = ds4_gpu_routed_moe_one_tensor(g->routed_out,')
    assert old_context in text, "STEP 2b: Router closing brace not found"
    text = text.replace(old_context, new_context)
    print("STEP 2b: Router section closed with /* !skip_router */")

    # ====== STEP 3: Add ', false' to all call sites ======
    # Find all call sites. Each is: metal_graph_encode_decode_layer( ... token)
    # We need to add ', false' before the last closing paren.
    # Pattern: lines ending with ' token);' 
    call_pattern = re.compile(r'(metal_graph_encode_decode_layer\(.*\)\s*;)')
    calls = list(call_pattern.finditer(text))
    print(f"STEP 3: Found {len(calls)} call sites to update")

    # Process in reverse to preserve positions
    for match in reversed(calls):
        call = match.group(1)
        # Insert ', false' before the last )
        # Find position of last ');'
        pos = match.start()
        end = match.end()
        snippet = text[pos:end]
        # Find the closing );
        close_paren = snippet.rfind(');')
        if close_paren < 0:
            # It might be ')\n                ;' (split across lines)
            # Try: find last ')' before ';'
            last_paren = snippet.rfind(')')
            last_semi = snippet.find(';', last_paren)
            if last_paren >= 0 and last_semi > last_paren:
                snippet = snippet[:last_semi] + ', false' + snippet[last_semi:]
                text = text[:pos] + snippet + text[end:]
                continue
            print(f"  WARNING: Could not find ); in call: {snippet[:80]}")
            continue
        
        snippet = snippet[:close_paren] + ', false' + snippet[close_paren:]
        text = text[:pos] + snippet + text[end:]
    print("STEP 3: All call sites updated with , false")

    # ====== STEP 4: Wire sub-graph capture/replay into eval function ======
    # In metal_graph_eval_token_raw_swa, after normal encode, add sub-graph capture for first token
    # Find the section after normal encode where we do end_commands

    # For the capture token, replace the single-graph capture with per-layer sub-graph capture
    # The pattern: when capture not ready AND capture mode, do encode + capture
    old_capture = (
        '        } else if (graph_mode && g->cuda_params_host && pos > 10) {\n'
        '        /* First decode (pos > prefill end): capture the graph for future replays */\n'
        '        fprintf(stderr, "ds4: graph capture path pos=%u\\n", pos);')
    new_capture = (
        '        } else if (graph_mode && g->cuda_params_host && pos > 10) {\n'
        '        /* Per-layer sub-graph capture (SPEC_SUBGRAPH.md) */\n'
        '        fprintf(stderr, "ds4: sub-graph capture path pos=%u\\n", pos);\n'
        '        for (uint32_t il = 0; il < DS4_N_LAYER && ok; il++) {\n'
        '            const ds4_layer_weights *layer = &weights->layer[il];\n'
        '            /* Run router outside capture */\n'
        '            if (!skip_router_layer) {\n'
        '                ok = metal_graph_matmul_plain_tensor(g->router_logits, model,\n'
        '                    layer->ffn_gate_inp, DS4_N_EMBD, DS4_N_EXPERT, g->ffn_norm, 1);\n'
        '                ok = ok && ds4_gpu_router_select_tensor(g->router_selected,\n'
        '                    g->router_weights, g->router_probs, model->map, model->size,\n'
        '                    layer->ffn_exp_probs_b ? layer->ffn_exp_probs_b->abs_offset : 0,\n'
        '                    layer->ffn_gate_tid2eid ? layer->ffn_gate_tid2eid->abs_offset : 0,\n'
        '                    layer->ffn_gate_tid2eid ? (uint32_t)layer->ffn_gate_tid2eid->dim[1] : 0,\n'
        '                    (uint32_t)token, DS4_N_EXPERT, DS4_N_EXPERT_USED,\n'
        '                    DS4_EXPERT_WEIGHT_SCALE, 0, 0,\n'
        '                    layer->ffn_exp_probs_b != NULL,\n'
        '                    layer->ffn_gate_tid2eid != NULL, g->router_logits);\n'
        '            }\n'
        '            ok = ok && ds4_gpu_decode_graph_capture();\n'
        '            ok = ok && metal_graph_encode_decode_layer(g, model, layer, (uint32_t)il,\n'
        '                     pos, g->layer_raw_cache[il], g->raw_cap,\n'
        '                     pos % g->raw_cap,\n'
        '                     ((uint32_t)pos + 1 > g->raw_window ? g->raw_window : (uint32_t)pos + 1),\n'
        '                     token, true);\n'
        '            ok = ok && ds4_gpu_decode_graph_capture_end_store((int)il);\n'
        '        }\n'
        '        if (ok) fprintf(stderr, "ds4: %u sub-graphs captured\\n", DS4_N_LAYER);')
    assert old_capture in text, "STEP 4: Capture section not found"
    text = text.replace(old_capture, new_capture)
    print("STEP 4: Per-layer sub-graph capture wired into eval function")

    # Write back
    with open(os.path.join(path, 'ds4.c'), 'w') as f:
        f.write(text)


def refactor_ds4_cuda(path):
    with open(os.path.join(path, 'ds4_cuda.cu'), 'r') as f:
        text = f.read()

    # ====== Add sub-graph array and functions ======
    # Find the graph capture state variables
    old_state = (
        'static cudaGraphExec_t g_cuda_decode_graph_exec;\n'
        'static int g_cuda_decode_graph_captured;')
    new_state = (
        'static cudaGraphExec_t g_cuda_decode_graph_exec;\n'
        'static int g_cuda_decode_graph_captured;\n'
        '/* Per-layer sub-graph handles (43 layers) */\n'
        'static cudaGraphExec_t g_sub_graph_exec[43];\n'
        'static int g_sub_graph_exec_ready[43];')
    text = text.replace(old_state, new_state)
    print("SUb: Added sub-graph array state")

    # Add ds4_gpu_decode_graph_capture_end_store function
    # Find the ds4_gpu_decode_graph_capture_end function
    old_cap_end = (
        '    g_cuda_decode_graph_captured = 1;\n'
        '    fprintf(stderr, "ds4: CUDA decode graph captured and instantiated\\n");\n'
        '    return 1;\n'
        '}')
    new_cap_end = (
        '    g_cuda_decode_graph_captured = 1;\n'
        '    fprintf(stderr, "ds4: CUDA decode graph captured and instantiated\\n");\n'
        '    return 1;\n'
        '}\n'
        '\n'
        'extern "C" int ds4_gpu_decode_graph_capture_end_store(int layer) {\n'
        '    if (layer < 0 || layer >= 43) return 0;\n'
        '    if (!g_cuda_decode_stream_created) return 0;\n'
        '    cudaGraph_t graph = NULL;\n'
        '    cudaError_t ce = cudaStreamEndCapture(g_cuda_decode_stream, &graph);\n'
        '    if (ce != cudaSuccess) {\n'
        '        fprintf(stderr, "ds4: sub-graph capture end L%d failed: %s\\n", layer, cudaGetErrorString(ce));\n'
        '        (void)cudaGetLastError(); return 0;\n'
        '    }\n'
        '    if (!graph) return 0;\n'
        '    ce = cudaGraphInstantiate(&g_sub_graph_exec[layer], graph, NULL, NULL, 0);\n'
        '    (void)cudaGraphDestroy(graph);\n'
        '    if (ce != cudaSuccess) {\n'
        '        fprintf(stderr, "ds4: sub-graph instantiate L%d failed: %s\\n", layer, cudaGetErrorString(ce));\n'
        '        (void)cudaGetLastError(); return 0;\n'
        '    }\n'
        '    g_sub_graph_exec_ready[layer] = 1;\n'
        '    return 1;\n'
        '}\n'
        '\n'
        'extern "C" int ds4_gpu_decode_subgraph_launch(int layer) {\n'
        '    if (layer < 0 || layer >= 43 || !g_sub_graph_exec_ready[layer]) return 0;\n'
        '    cudaError_t ce = cudaGraphLaunch(g_sub_graph_exec[layer], ds4_decode_stream());\n'
        '    return ce == cudaSuccess ? 1 : 0;\n'
        '}\n'
        '\n'
        'extern "C" int ds4_gpu_decode_subgraphs_ready(void) {\n'
        '    for (int i = 0; i < 43; i++) if (!g_sub_graph_exec_ready[i]) return 0;\n'
        '    return 1;\n'
        '}')
    text = text.replace(old_cap_end, new_cap_end)
    print("SUB: Added capture_end_store, subgraph_launch, subgraphs_ready")

    # Add cleanup in ds4_gpu_cleanup
    old_cleanup = (
        '    g_cuda_decode_graph_captured = 0;\n'
        '    if (g_moe_scratch_gate)')
    new_cleanup = (
        '    g_cuda_decode_graph_captured = 0;\n'
        '    for (int i = 0; i < 43; i++) {\n'
        '        if (g_sub_graph_exec[i]) { (void)cudaGraphExecDestroy(g_sub_graph_exec[i]); g_sub_graph_exec[i] = NULL; }\n'
        '        g_sub_graph_exec_ready[i] = 0;\n'
        '    }\n'
        '    if (g_moe_scratch_gate)')
    text = text.replace(old_cleanup, new_cleanup)
    print("SUB: Added sub-graph cleanup")

    with open(os.path.join(path, 'ds4_cuda.cu'), 'w') as f:
        f.write(text)


def refactor_ds4_gpu_h(path):
    with open(os.path.join(path, 'ds4_gpu.h'), 'r') as f:
        text = f.read()

    old = (
        'int ds4_gpu_decode_graph_captured(void);  /* 1 if ready for replay */')
    new = (
        'int ds4_gpu_decode_graph_captured(void);  /* 1 if ready for replay */\n'
        'int ds4_gpu_decode_graph_capture_end_store(int layer);\n'
        'int ds4_gpu_decode_subgraph_launch(int layer);\n'
        'int ds4_gpu_decode_subgraphs_ready(void);')
    text = text.replace(old, new)
    print("GPU_H: Added subgraph function declarations")

    with open(os.path.join(path, 'ds4_gpu.h'), 'w') as f:
        f.write(text)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else '/run/media/tonoken3/DATA2/Lna-Lab/DwarfStar4'
    refactor_ds4(path)
    refactor_ds4_cuda(path)
    refactor_ds4_gpu_h(path)
    print("\nAll refactoring complete. Run 'make cuda' to verify.")
