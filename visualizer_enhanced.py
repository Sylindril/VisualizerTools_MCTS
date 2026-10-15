#!/usr/bin/env python3
"""Enhanced MCTS Visualizer (v31) - Streamlined for VWA MCTS data.

Generates a fast, interactive HTML visualization with:
- Left panel: Screenshot canvas with small red dot coordinate markers
- Right panel: vis-network tree with distinct think/action node colors
- Bottom info panel: Node details, thought text, rollout info

Usage:
    python visualizer_enhanced.py input.jsonl
    python visualizer_enhanced.py input.jsonl --output_filename my_vis --max_depth 5
"""

import argparse
import json
import html
import re
import os
from pathlib import Path


def _extract_coords(text):
    """Extract all (x, y) and [x, y] coordinates from a string."""
    if not isinstance(text, str):
        return []
    pattern = r'[\(\[]\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*[\)\]]'
    matches = re.findall(pattern, text)
    coords = []
    for m in matches:
        try:
            x, y = float(m[0]), float(m[1])
            if 0 <= x <= 10000 and 0 <= y <= 10000:
                coords.append([x, y])
        except (ValueError, IndexError):
            continue
    return coords


def get_tree_statistics(tree_data):
    """Compute tree statistics."""
    stats = {
        'total_nodes': 0, 'max_depth': 0, 'terminal_nodes': 0,
        'total_rollouts': 0, 'successful_rollouts_0': 0,
        'successful_rollouts_05': 0, 'successful_rollouts_1': 0,
        'best_reward': -1.0, 'think_nodes': 0, 'action_nodes': 0,
    }
    root = tree_data.get('tree')
    if not root:
        return stats

    def _walk(node, depth):
        if not node:
            return
        stats['total_nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)
        nt = node.get('node_type', '')
        if nt == 'think':
            stats['think_nodes'] += 1
        elif nt == 'action':
            stats['action_nodes'] += 1
        if node.get('is_terminal'):
            stats['terminal_nodes'] += 1
        for r in node.get('rollouts', []):
            stats['total_rollouts'] += 1
            reward = r.get('reward', 0)
            if reward > 0:
                stats['successful_rollouts_0'] += 1
            if reward >= 0.5:
                stats['successful_rollouts_05'] += 1
            if reward == 1.0:
                stats['successful_rollouts_1'] += 1
            if reward > stats['best_reward']:
                stats['best_reward'] = reward
        for child in node.get('children', []):
            _walk(child, depth + 1)

    _walk(root, 0)
    return stats


def build_graph_data(tree_data, max_depth_vis):
    """Build vis-network nodes/edges and a node data map.

    Key behavior:
    - Think nodes and action nodes get distinct colors/shapes
    - Each node stores its display_image (for think nodes, this is the
      parent action node's image)
    - Coordinates are extracted from used_coords (action) or thought_text (think)
    """
    nodes = []
    edges = []
    node_data_map = {}
    counter = [0]

    root = tree_data.get('tree')
    if not root:
        return {"nodes": [], "edges": [], "node_data": {}}

    def _build(node, parent_id, depth, parent_image):
        nid = f"n{counter[0]}"
        counter[0] += 1

        if max_depth_vis is not None and depth > max_depth_vis:
            return

        node_type = node.get('node_type', 'unknown')
        thought = node.get('thought_text', '')
        value = node.get('value', 0.0)
        visit_count = node.get('visit_count', 0)
        is_terminal = node.get('is_terminal', False)
        raw_pred = node.get('raw_prediction', '')
        node_image = node.get('image', '')

        # Determine display image: think nodes use parent action's image
        if node_type == 'think':
            display_image = parent_image or ''
        else:
            display_image = node_image or parent_image or ''

        # Extract coordinates
        coords = []
        if node_type == 'action':
            # Use used_coords field
            for cs in node.get('used_coords', []):
                parsed = _extract_coords(cs)
                coords.extend(parsed)
        elif node_type == 'think':
            # Extract from thought_text
            coords = _extract_coords(thought)

        # Build label
        short_text = thought[:80] + '...' if len(thought) > 80 else thought
        short_text = short_text.replace('\n', ' ')
        label = f"{html.escape(short_text)}\nv={value:.2f} n={visit_count}"

        # Color/shape by node type
        if node_type == 'think':
            color = '#7CB9E8'   # Soft blue for think
            shape = 'box'
            border_color = '#4A90D9'
        elif node_type == 'action':
            color = '#F4A460'   # Sandy brown for action
            shape = 'box'
            border_color = '#D2691E'
        else:
            color = '#97C2FC'
            shape = 'box'
            border_color = '#666'

        if is_terminal:
            border_color = '#CC0000'

        if depth == 0:
            color = '#4CAF50'   # Green for root
            border_color = '#2E7D32'

        nodes.append({
            "id": nid,
            "label": label,
            "shape": shape,
            "color": {"background": color, "border": border_color,
                       "highlight": {"background": color, "border": "#000"}},
            "font": {"size": 11, "color": "#333", "multi": "html"},
            "margin": 8,
            "borderWidth": 2 if is_terminal else 1,
            "level": depth,
        })

        if parent_id is not None:
            edges.append({"from": parent_id, "to": nid, "arrows": "to"})

        # Build rollout summary
        rollout_info = []
        for i, r in enumerate(node.get('rollouts', [])):
            rollout_info.append({
                "final_answer": r.get('final_answer', ''),
                "reward": r.get('reward', 0),
            })

        node_data_map[nid] = {
            "node_type": node_type,
            "value": value,
            "visit_count": visit_count,
            "depth": depth,
            "is_terminal": is_terminal,
            "thought_text": thought,
            "raw_prediction": raw_pred,
            "browser_url": node.get('browser_url', ''),
            "action_success": node.get('action_success', True),
            "display_image": display_image,
            "coords": coords,
            "rollouts": rollout_info,
            "accumulated_thoughts": node.get('accumulated_thoughts', []),
        }

        # The image that children (think nodes) should inherit
        next_parent_image = node_image if node_image else parent_image

        for child in node.get('children', []):
            _build(child, nid, depth + 1, next_parent_image)

    _build(root, None, 0, tree_data.get('image', ''))
    return {"nodes": nodes, "edges": edges, "node_data": node_data_map}


def generate_html(tree_data, graph_data, stats, output_path):
    """Generate a self-contained HTML visualization.

    Layout: left half = full-height image canvas.
    Right half = top area with switchable tabs (Tree, Node Output, Thoughts, Stats),
    bottom strip = compact node info bar.
    """

    question = html.escape(tree_data.get('question', 'N/A'))
    best_score = tree_data.get('best_score', stats['best_reward'])
    total_nodes = stats['total_nodes']
    search_time = tree_data.get('global_search_time', 0)
    if isinstance(search_time, (int, float)):
        time_str = f"{search_time:.0f}s"
    else:
        time_str = "N/A"

    # Separate images from node metadata to avoid dumping base64 to terminal
    node_data_for_json = {}
    image_map = {}
    for nid, nd in graph_data['node_data'].items():
        nd_copy = {k: v for k, v in nd.items() if k != 'display_image'}
        node_data_for_json[nid] = nd_copy
        if nd.get('display_image'):
            image_map[nid] = nd['display_image']

    nodes_json = json.dumps(graph_data['nodes'])
    edges_json = json.dumps(graph_data['edges'])
    node_data_json = json.dumps(node_data_for_json)

    image_map_js_parts = []
    for nid, img in image_map.items():
        image_map_js_parts.append(f'imageMap["{nid}"] = "{img}";')
    image_map_js = "\n".join(image_map_js_parts)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MCTS Viewer - {question[:60]}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html, body {{ height:100%; overflow:hidden; }}
body {{ font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#1a1a2e; color:#eee; display:flex; flex-direction:column; }}

/* Header */
.header {{ background:#16213e; padding:6px 20px; border-bottom:1px solid #0f3460; display:flex; justify-content:space-between; align-items:center; flex-shrink:0; }}
.header h1 {{ font-size:15px; color:#e94560; white-space:nowrap; }}
.header .question {{ font-size:12px; color:#ccc; max-width:50%; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.header .meta {{ font-size:11px; color:#aaa; white-space:nowrap; }}

/* Main 2-column layout */
.main {{ display:flex; flex:1; min-height:0; }}

/* LEFT: full-height image */
.left {{ width:50%; display:flex; align-items:center; justify-content:center; background:#0f0f23; border-right:1px solid #0f3460; position:relative; overflow:hidden; }}
.left canvas {{ max-width:100%; max-height:100%; display:block; }}
.no-image {{ color:#555; font-size:13px; position:absolute; }}

/* RIGHT: tabs on top, info bar on bottom */
.right {{ width:50%; display:flex; flex-direction:column; min-height:0; }}

/* Tab bar */
.tab-bar {{ display:flex; background:#16213e; border-bottom:1px solid #0f3460; flex-shrink:0; }}
.tab-btn {{ padding:7px 16px; cursor:pointer; font-size:12px; color:#888; border-bottom:2px solid transparent; transition:color 0.15s; white-space:nowrap; }}
.tab-btn:hover {{ color:#ccc; }}
.tab-btn.active {{ color:#e94560; border-bottom-color:#e94560; }}

/* Tab panes - fill remaining space */
.tab-panes {{ flex:1; position:relative; min-height:0; }}
.tab-pane {{ position:absolute; top:0; left:0; right:0; bottom:0; overflow:hidden; display:none; }}
.tab-pane.active {{ display:flex; flex-direction:column; }}

/* Tree pane */
.tree-pane {{ position:relative; flex:1; }}
.tree-pane #network {{ position:absolute; top:0; left:0; right:0; bottom:0; }}
.tree-controls {{ position:absolute; top:6px; left:6px; z-index:10; display:flex; gap:5px; }}
.tree-controls button {{ background:rgba(108,117,125,0.85); color:#fff; border:none; padding:4px 10px; border-radius:3px; cursor:pointer; font-size:11px; }}
.tree-controls button:hover {{ background:rgba(90,98,104,0.95); }}
.tree-controls button.active {{ background:rgba(40,167,69,0.9); }}

/* Text panes */
.text-pane {{ flex:1; overflow-y:auto; padding:12px 15px; }}
.text-pane pre {{ background:#0f0f23; padding:12px; border-radius:5px; font-family:'Menlo','Monaco','Courier New',monospace; font-size:12px; white-space:pre-wrap; word-break:break-word; color:#ccc; line-height:1.5; }}
.text-pane .empty {{ color:#555; font-style:italic; }}
.text-pane .section-title {{ color:#e94560; font-weight:600; font-size:12px; margin:12px 0 6px 0; }}
.text-pane .section-title:first-child {{ margin-top:0; }}

/* Info bar at bottom of right panel */
.info-bar {{ background:#16213e; border-top:1px solid #0f3460; padding:8px 14px; flex-shrink:0; font-size:11px; overflow-y:auto; max-height:180px; }}
.info-row {{ display:flex; gap:10px; flex-wrap:wrap; margin-bottom:3px; }}
.info-row .label {{ color:#e94560; font-weight:600; }}
.info-row .val {{ color:#ddd; }}
.thought-preview {{ background:#0f0f23; padding:6px 8px; border-radius:3px; margin-top:5px; font-family:monospace; font-size:11px; color:#aaa; max-height:50px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; cursor:pointer; }}
.thought-preview:hover {{ color:#ccc; white-space:pre-wrap; max-height:200px; overflow-y:auto; }}
.rollout-item {{ background:#0f0f23; padding:3px 8px; border-radius:3px; margin-top:2px; font-family:monospace; font-size:10px; display:flex; justify-content:space-between; gap:8px; }}
.reward-good {{ color:#4CAF50; font-weight:600; }}
.reward-mid {{ color:#FFC107; }}
.reward-low {{ color:#F44336; }}

/* Footer legend */
.footer {{ background:#10132a; padding:3px 15px; font-size:10px; color:#666; display:flex; gap:16px; align-items:center; flex-shrink:0; border-top:1px solid #0f3460; }}
.ldot {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:3px; vertical-align:middle; }}
</style>
</head>
<body>

<div class="header">
    <h1>MCTS Tree Viewer</h1>
    <div class="question" title="{question}">{question}</div>
    <div class="meta">Score: {best_score:.3f} | Nodes: {total_nodes} | Time: {time_str}</div>
</div>

<div class="main">
    <!-- LEFT: full-height screenshot -->
    <div class="left">
        <canvas id="canvas"></canvas>
        <div id="no-image" class="no-image">Click a node to view its screenshot</div>
    </div>

    <!-- RIGHT: tabs + info bar -->
    <div class="right">
        <div class="tab-bar">
            <div class="tab-btn active" data-tab="tab-tree">Tree</div>
            <div class="tab-btn" data-tab="tab-output">Node Output</div>
            <div class="tab-btn" data-tab="tab-thoughts">Thoughts</div>
            <div class="tab-btn" data-tab="tab-stats">Stats</div>
        </div>

        <div class="tab-panes">
            <!-- Tree tab -->
            <div id="tab-tree" class="tab-pane active">
                <div class="tree-pane">
                    <div class="tree-controls">
                        <button id="btn-best-path" onclick="toggleBestPath()">Best Path</button>
                        <button id="btn-value-colors" onclick="toggleValueColors()">Value Colors</button>
                        <button onclick="network.fit()">Fit</button>
                    </div>
                    <div id="network"></div>
                </div>
            </div>

            <!-- Node Output tab -->
            <div id="tab-output" class="tab-pane">
                <div class="text-pane" id="pane-output">
                    <div class="empty">Click a node to see its full output.</div>
                </div>
            </div>

            <!-- Accumulated Thoughts tab -->
            <div id="tab-thoughts" class="tab-pane">
                <div class="text-pane" id="pane-thoughts">
                    <div class="empty">Click a node to see accumulated thoughts.</div>
                </div>
            </div>

            <!-- Stats tab -->
            <div id="tab-stats" class="tab-pane">
                <div class="text-pane">
                    <pre>Nodes: {total_nodes} (Think: {stats['think_nodes']}, Action: {stats['action_nodes']})
Max Depth: {stats['max_depth']}
Terminal Nodes: {stats['terminal_nodes']}
Total Rollouts: {stats['total_rollouts']}
Best Reward: {stats['best_reward']:.3f}
Perfect Rollouts (R=1): {stats['successful_rollouts_1']}
Good Rollouts (R>=0.5): {stats['successful_rollouts_05']}
Positive Rollouts (R>0): {stats['successful_rollouts_0']}
Search Time: {time_str}</pre>
                </div>
            </div>
        </div>

        <!-- Compact info bar pinned at bottom -->
        <div class="info-bar" id="info-bar">
            <div class="info-row">
                <span><span class="label">Node:</span> <span class="val" id="info-type">-</span></span>
                <span><span class="label">Value:</span> <span class="val" id="info-value">-</span></span>
                <span><span class="label">Visits:</span> <span class="val" id="info-visits">-</span></span>
                <span><span class="label">Depth:</span> <span class="val" id="info-depth">-</span></span>
            </div>
            <div class="info-row">
                <span><span class="label">URL:</span> <span class="val" id="info-url">-</span></span>
            </div>
            <div id="info-thought-preview" class="thought-preview" title="Hover to expand, click to jump to Node Output tab" style="display:none;"></div>
            <div id="info-rollouts"></div>
        </div>
    </div>
</div>

<div class="footer">
    <span><span class="ldot" style="background:#4CAF50;"></span>Root</span>
    <span><span class="ldot" style="background:#7CB9E8;"></span>Think</span>
    <span><span class="ldot" style="background:#F4A460;"></span>Action</span>
    <span><span class="ldot" style="background:#CC0000;"></span>Terminal</span>
    <span style="margin-left:auto;">Think: {stats['think_nodes']} | Action: {stats['action_nodes']} | Best: {stats['best_reward']:.3f}</span>
</div>

<script>
// === Data ===
const nodesArr = {nodes_json};
const edgesArr = {edges_json};
const nodeData = {node_data_json};
const imageMap = {{}};
{image_map_js}

// === Tab switching ===
document.querySelectorAll('.tab-btn').forEach(btn => {{
    btn.addEventListener('click', () => {{
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(btn.dataset.tab).classList.add('active');
        // Redraw network when switching to tree tab (vis-network needs visible container)
        if (btn.dataset.tab === 'tab-tree') {{
            setTimeout(() => network.redraw(), 50);
        }}
    }});
}});

// === Network ===
const nodesDS = new vis.DataSet(nodesArr);
const edgesDS = new vis.DataSet(edgesArr);
const network = new vis.Network(document.getElementById('network'),
    {{ nodes: nodesDS, edges: edgesDS }},
    {{
        layout: {{ hierarchical: {{ direction:'UD', sortMethod:'directed', levelSeparation:100, nodeSpacing:120 }} }},
        physics: false,
        interaction: {{ hover:true, selectConnectedEdges:false }},
        edges: {{ color:{{ color:'#555', highlight:'#e94560' }}, width:1.5 }}
    }}
);
network.once('afterDrawing', () => network.fit());

// === Canvas for screenshots ===
const cvs = document.getElementById('canvas');
const ctx = cvs.getContext('2d');
const noImageEl = document.getElementById('no-image');
let imgCache = {{}};

function drawScreenshot(nodeId) {{
    const src = imageMap[nodeId];
    const nd = nodeData[nodeId];
    if (!src) {{
        cvs.style.display = 'none';
        noImageEl.style.display = 'block';
        return;
    }}
    cvs.style.display = 'block';
    noImageEl.style.display = 'none';

    function render(img) {{
        cvs.width = img.width;
        cvs.height = img.height;
        ctx.drawImage(img, 0, 0);
        if (nd && nd.coords && nd.coords.length > 0) {{
            nd.coords.forEach(c => {{
                let px = c[0], py = c[1];
                if (px <= 1 && py <= 1) {{ px *= img.width; py *= img.height; }}
                ctx.beginPath();
                ctx.arc(px, py, 5, 0, 2 * Math.PI);
                ctx.fillStyle = 'rgba(255,0,0,0.9)';
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }});
        }}
    }}

    if (imgCache[nodeId]) {{ render(imgCache[nodeId]); }}
    else {{
        const img = new Image();
        img.onload = () => {{ imgCache[nodeId] = img; render(img); }};
        img.src = src;
    }}
}}

// === Info bar (bottom of right panel) ===
function updateInfo(nodeId) {{
    const nd = nodeData[nodeId];
    if (!nd) return;
    document.getElementById('info-type').textContent = nd.node_type + (nd.is_terminal ? ' [TERM]' : '') + ' (' + nodeId + ')';
    document.getElementById('info-value').textContent = nd.value.toFixed(3);
    document.getElementById('info-visits').textContent = nd.visit_count;
    document.getElementById('info-depth').textContent = nd.depth;
    document.getElementById('info-url').textContent = nd.browser_url || '-';

    // Thought preview (single line, expand on hover)
    const preview = document.getElementById('info-thought-preview');
    const text = nd.raw_prediction || nd.thought_text || '';
    if (text) {{
        preview.textContent = text;
        preview.style.display = 'block';
    }} else {{
        preview.style.display = 'none';
    }}

    // Rollouts
    const rolloutsEl = document.getElementById('info-rollouts');
    if (nd.rollouts && nd.rollouts.length > 0) {{
        let h = '';
        nd.rollouts.forEach(r => {{
            const rc = r.reward >= 0.75 ? 'reward-good' : (r.reward >= 0.3 ? 'reward-mid' : 'reward-low');
            h += '<div class="rollout-item"><span>' + (r.final_answer||'').substring(0,120) + '</span><span class="' + rc + '">R=' + r.reward.toFixed(3) + '</span></div>';
        }});
        rolloutsEl.innerHTML = h;
    }} else {{ rolloutsEl.innerHTML = ''; }}
}}

// === Detail panes (Node Output + Accumulated Thoughts) ===
function updateDetail(nodeId) {{
    const nd = nodeData[nodeId];
    if (!nd) return;

    // Node Output
    const out = document.getElementById('pane-output');
    let h = '<div class="section-title">Node: ' + nd.node_type + ' (depth=' + nd.depth + ', value=' + nd.value.toFixed(3) + ', visits=' + nd.visit_count + ')</div>';
    if (nd.browser_url) h += '<div style="font-size:11px;color:#888;margin-bottom:8px;">URL: ' + nd.browser_url + '</div>';
    h += '<div class="section-title">Thought Text</div><pre>' + (nd.thought_text || '(none)') + '</pre>';
    if (nd.raw_prediction) {{
        h += '<div class="section-title">Raw Prediction</div><pre>' + nd.raw_prediction + '</pre>';
    }}
    if (nd.rollouts && nd.rollouts.length > 0) {{
        h += '<div class="section-title">Rollouts (' + nd.rollouts.length + ')</div>';
        nd.rollouts.forEach((r, i) => {{
            const rc = r.reward >= 0.75 ? 'reward-good' : (r.reward >= 0.3 ? 'reward-mid' : 'reward-low');
            h += '<div class="rollout-item" style="margin-bottom:4px;"><span>Rollout ' + (i+1) + ': ' + (r.final_answer||'') + '</span><span class="' + rc + '">R=' + r.reward.toFixed(3) + '</span></div>';
        }});
    }}
    out.innerHTML = h;

    // Accumulated Thoughts
    const th = document.getElementById('pane-thoughts');
    if (nd.accumulated_thoughts && nd.accumulated_thoughts.length > 0) {{
        let ht = '';
        nd.accumulated_thoughts.forEach((t, i) => {{
            ht += '<div class="section-title">Thought ' + (i+1) + '</div><pre>' + t + '</pre>';
        }});
        th.innerHTML = ht;
    }} else {{
        th.innerHTML = '<div class="empty">No accumulated thoughts for this node.</div>';
    }}
}}

// === Click thought preview to jump to Node Output tab ===
document.getElementById('info-thought-preview').addEventListener('click', () => {{
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    document.querySelector('[data-tab="tab-output"]').classList.add('active');
    document.getElementById('tab-output').classList.add('active');
}});

// === Highlight selected node ===
function highlightNode(nodeId) {{
    nodesDS.forEach(n => {{
        const orig = nodesArr.find(o => o.id === n.id);
        if (orig) nodesDS.update({{id:n.id, color:orig.color, borderWidth:orig.borderWidth||1}});
    }});
    const sel = nodesDS.get(nodeId);
    if (sel) {{
        nodesDS.update({{id:nodeId, borderWidth:4, color:{{border:'#e94560', background:sel.color.background, highlight:{{border:'#e94560',background:sel.color.background}}}}}});
    }}
}}

// === Click: update everything ===
network.on('click', function(params) {{
    if (params.nodes.length > 0) {{
        const nid = params.nodes[0];
        drawScreenshot(nid);
        updateInfo(nid);
        updateDetail(nid);
        highlightNode(nid);
    }}
}});

// === Double-click: update + switch to Node Output tab ===
network.on('doubleClick', function(params) {{
    if (params.nodes.length > 0) {{
        const nid = params.nodes[0];
        drawScreenshot(nid);
        updateInfo(nid);
        updateDetail(nid);
        highlightNode(nid);
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        document.querySelector('[data-tab="tab-output"]').classList.add('active');
        document.getElementById('tab-output').classList.add('active');
        document.getElementById('pane-output').scrollTop = 0;
    }}
}});

// === Best path ===
let bestPathOn = false;
function toggleBestPath() {{
    bestPathOn = !bestPathOn;
    const btn = document.getElementById('btn-best-path');
    if (bestPathOn) {{
        btn.classList.add('active');
        const byDepth = {{}};
        Object.entries(nodeData).forEach(([id, nd]) => {{
            const d = nd.depth;
            if (!byDepth[d] || nd.value > byDepth[d].value) byDepth[d] = {{id, value:nd.value}};
        }});
        const pathIds = Object.values(byDepth).map(x => x.id);
        edgesDS.forEach(e => {{
            const onPath = pathIds.includes(e.from) && pathIds.includes(e.to);
            edgesDS.update({{id:e.id, color:onPath?'#4CAF50':'#555', width:onPath?3:1.5}});
        }});
        pathIds.forEach(pid => nodesDS.update({{id:pid, shadow:{{enabled:true, color:'rgba(40,167,69,0.5)', size:12, x:0, y:0}}}}));
    }} else {{
        btn.classList.remove('active');
        edgesDS.forEach(e => edgesDS.update({{id:e.id, color:'#555', width:1.5}}));
        nodesDS.forEach(n => nodesDS.update({{id:n.id, shadow:{{enabled:false}}}}));
    }}
}}

// === Value colors ===
let valueColorsOn = false;
let origColors = {{}};
function toggleValueColors() {{
    valueColorsOn = !valueColorsOn;
    const btn = document.getElementById('btn-value-colors');
    if (valueColorsOn) {{
        btn.classList.add('active');
        nodesDS.forEach(n => {{
            const nd = nodeData[n.id];
            if (nd) {{
                origColors[n.id] = n.color;
                const hue = Math.round(nd.value * 120);
                const bg = 'hsl(' + hue + ',70%,' + (50+Math.round((1-nd.value)*15)) + '%)';
                nodesDS.update({{id:n.id, color:{{background:bg, border:n.color.border, highlight:{{background:bg,border:'#000'}}}}}});
            }}
        }});
    }} else {{
        btn.classList.remove('active');
        nodesDS.forEach(n => {{ if (origColors[n.id]) nodesDS.update({{id:n.id, color:origColors[n.id]}}); }});
    }}
}}

// === Init ===
setTimeout(() => {{
    const rootId = nodesArr.length > 0 ? nodesArr[0].id : null;
    if (rootId) {{
        drawScreenshot(rootId);
        updateInfo(rootId);
        updateDetail(rootId);
        network.selectNodes([rootId]);
        network.fit();
    }}
}}, 300);
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Generated: {output_path}")
    print(f"  Nodes: {len(graph_data['nodes'])}, Edges: {len(graph_data['edges'])}")
    print(f"  Question: {tree_data.get('question', 'N/A')[:80]}")


def main():
    parser = argparse.ArgumentParser(description='MCTS Tree Visualizer - generates interactive HTML from JSONL data.')
    parser.add_argument('input_file', type=str, help='Path to MCTS JSONL file.')
    parser.add_argument('--output_filename', type=str, default='mcts_visualization_enhanced',
                        help='Output filename base (without .html extension).')
    parser.add_argument('--max_depth', type=int, default=None, help='Max tree depth to visualize.')
    args = parser.parse_args()

    input_path = Path(args.input_file)
    print(f"Loading: {input_path}")

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line:
                print("Error: File is empty.")
                return
            data = json.loads(first_line)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error: {e}")
        return

    stats = get_tree_statistics(data)
    graph_data = build_graph_data(data, args.max_depth)

    output_path = f"{args.output_filename}.html"
    generate_html(data, graph_data, stats, output_path)
    print(f"Done: {os.path.abspath(output_path)}")


if __name__ == '__main__':
    main()
