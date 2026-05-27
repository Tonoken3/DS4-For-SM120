#!/usr/bin/env python3
"""Phase C helper: analyze split point in metal_graph_encode_decode_layer.
Prints the line range for pre-router and post-router sections."""

with open('/run/media/tonoken3/DATA2/Lna-Lab/DwarfStar4/ds4.c', 'r') as f:
    lines = f.readlines()

# Find function start
start = None
for i, line in enumerate(lines):
    if 'static bool metal_graph_encode_decode_layer(' in line:
        start = i
        break

if start is None:
    print("ERROR: function not found")
    exit(1)

# Find the router section
router_start = None
router_end = None
for i in range(start, len(lines)):
    if 'if (!skip_router) {' in lines[i]:
        router_start = i
    if router_start and '} /* !skip_router */' in lines[i]:
        router_end = i
        break

if router_start is None or router_end is None:
    print("ERROR: router section not found")
    exit(1)

# Find function end
func_end = None
depth = 0
for i in range(start, len(lines)):
    depth += lines[i].count('{') - lines[i].count('}')
    if depth == 0 and i > start:
        func_end = i
        break

print(f"Function: lines {start+1}-{func_end+1}")
print(f"Router section: lines {router_start+1}-{router_end+1}")
print(f"Pre-router: lines {start+1}-{router_start} ({router_start - start} lines)")
print(f"Post-router: lines {router_end+2}-{func_end+1} ({func_end - router_end - 1} lines)")

# Show first/last few lines of each section
print("\n--- PRE-ROUTER first 3 lines ---")
for l in lines[start:start+3]:
    print(f"  {l.rstrip()}")
print("--- PRE-ROUTER last 3 lines before router ---")
for l in lines[router_start-3:router_start]:
    print(f"  {l.rstrip()}")

print("\n--- POST-ROUTER first 3 lines ---")
for l in lines[router_end+1:router_end+4]:
    print(f"  {l.rstrip()}")
print("--- POST-ROUTER last 3 lines ---")
for l in lines[func_end-3:func_end+1]:
    print(f"  {l.rstrip()}")
