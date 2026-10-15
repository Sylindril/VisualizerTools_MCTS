#!/usr/bin/env python3
"""MCTS Tree Growth Simulator - Interactive parameter exploration.

Generates a self-contained HTML file that simulates MCTS tree growth
with tunable parameters. All simulation logic runs client-side in
JavaScript. No data, no browser, no LLM calls needed.

Goal: understand how many simulations you need for a given branching
factor to get a "sufficiently deep" tree.

Usage:
    python visualizers/mcts_growth_simulator.py [-o output.html]
"""

import argparse
import os
import webbrowser


def generate_html() -> str:
    """Generate self-contained HTML with embedded MCTS growth simulator."""

    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MCTS Tree Growth Simulator</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body { height:100%; overflow:hidden; }
body { font-family: -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; background:#1a1a2e; color:#eee; display:flex; flex-direction:column; }

.header { background:#16213e; padding:8px 20px; border-bottom:1px solid #0f3460; display:flex; justify-content:space-between; align-items:center; flex-shrink:0; }
.header h1 { font-size:16px; color:#e94560; }
.header .subtitle { font-size:12px; color:#aaa; }

.main { display:flex; flex:1; min-height:0; }

/* Sidebar */
.sidebar { width:280px; background:#16213e; border-right:1px solid #0f3460; display:flex; flex-direction:column; flex-shrink:0; overflow-y:auto; }
.sidebar-section { padding:12px 14px; border-bottom:1px solid #0f3460; }
.sidebar-section h3 { font-size:12px; color:#e94560; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:8px; }

.param-row { margin-bottom:8px; }
.param-label { display:flex; justify-content:space-between; align-items:center; font-size:11px; color:#ccc; margin-bottom:3px; }
.param-value { color:#7CB9E8; font-weight:600; min-width:30px; text-align:right; }
.param-row input[type="range"] { width:100%; height:4px; -webkit-appearance:none; background:#0f3460; border-radius:2px; outline:none; }
.param-row input[type="range"]::-webkit-slider-thumb { -webkit-appearance:none; width:14px; height:14px; background:#e94560; border-radius:50%; cursor:pointer; }
.param-row input[type="range"]::-moz-range-thumb { width:14px; height:14px; background:#e94560; border-radius:50%; cursor:pointer; border:none; }

.toggle-row { display:flex; justify-content:space-between; align-items:center; margin-bottom:6px; font-size:11px; color:#ccc; }
.toggle-switch { position:relative; width:36px; height:18px; cursor:pointer; }
.toggle-switch input { opacity:0; width:0; height:0; }
.toggle-slider { position:absolute; top:0; left:0; right:0; bottom:0; background:#0f3460; border-radius:9px; transition:0.2s; }
.toggle-slider:before { content:""; position:absolute; height:14px; width:14px; left:2px; bottom:2px; background:#aaa; border-radius:50%; transition:0.2s; }
.toggle-switch input:checked + .toggle-slider { background:#e94560; }
.toggle-switch input:checked + .toggle-slider:before { transform:translateX(18px); background:#fff; }

.controls { padding:12px 14px; }
.btn-row { display:flex; gap:6px; margin-bottom:8px; }
.btn { padding:6px 12px; border:none; border-radius:4px; cursor:pointer; font-size:12px; font-weight:600; transition:background 0.15s; }
.btn-step { background:#2196F3; color:#fff; }
.btn-step:hover { background:#1976D2; }
.btn-play { background:#4CAF50; color:#fff; }
.btn-play:hover { background:#388E3C; }
.btn-play.playing { background:#FF9800; }
.btn-play.playing:hover { background:#F57C00; }
.btn-reset { background:#F44336; color:#fff; }
.btn-reset:hover { background:#D32F2F; }
.btn-run-all { background:#9C27B0; color:#fff; }
.btn-run-all:hover { background:#7B1FA2; }

/* Right panel */
.right-panel { flex:1; display:flex; flex-direction:column; min-width:0; }

.tree-container { flex:1; background:#0f0f23; position:relative; min-height:0; }
#network { position:absolute; top:0; left:0; right:0; bottom:0; }

.stats-bar { background:#16213e; border-top:1px solid #0f3460; padding:8px 14px; flex-shrink:0; font-size:11px; display:flex; flex-wrap:wrap; gap:8px 16px; }
.stat-item { display:flex; gap:4px; }
.stat-label { color:#e94560; font-weight:600; }
.stat-value { color:#ddd; }

.legend { background:#16213e; border-top:1px solid #0f3460; padding:6px 14px; flex-shrink:0; font-size:11px; display:flex; gap:16px; justify-content:center; }
.legend-item { display:flex; align-items:center; gap:5px; }
.legend-color { width:14px; height:14px; border-radius:3px; }
.legend-ellipse { width:16px; height:12px; border-radius:50%; }

/* Node tooltip */
.node-tooltip { position:fixed; background:#16213e; border:1px solid #0f3460; border-radius:6px; padding:10px 14px; font-size:11px; z-index:1000; pointer-events:none; display:none; max-width:280px; box-shadow:0 4px 12px rgba(0,0,0,0.5); }
.node-tooltip .tt-row { margin-bottom:3px; display:flex; gap:6px; }
.node-tooltip .tt-label { color:#e94560; font-weight:600; min-width:60px; }
.node-tooltip .tt-value { color:#ddd; }

/* Sim counter */
.sim-counter { font-size:13px; color:#7CB9E8; font-weight:600; margin-bottom:6px; text-align:center; }
</style>
</head>
<body>
<div class="header">
    <div>
        <h1>MCTS Tree Growth Simulator</h1>
        <span class="subtitle">Tune parameters and watch the tree grow — no data, no LLM calls</span>
    </div>
</div>

<div class="main">
    <div class="sidebar">
        <div class="sidebar-section">
            <h3>MCTS Parameters</h3>

            <div class="param-row">
                <div class="param-label"><span>n_simulations</span><span class="param-value" id="val_n_simulations">10</span></div>
                <input type="range" id="n_simulations" min="1" max="100" value="10" step="1">
            </div>
            <div class="param-row">
                <div class="param-label"><span>branching_factor</span><span class="param-value" id="val_branching_factor">5</span></div>
                <input type="range" id="branching_factor" min="1" max="15" value="5" step="1">
            </div>
            <div class="param-row">
                <div class="param-label"><span>think_children</span><span class="param-value" id="val_think_children">3</span></div>
                <input type="range" id="think_children" min="1" max="10" value="3" step="1">
            </div>
            <div class="param-row">
                <div class="param-label"><span>think_maximum</span><span class="param-value" id="val_think_maximum">3</span></div>
                <input type="range" id="think_maximum" min="0" max="10" value="3" step="1">
            </div>
            <div class="param-row">
                <div class="param-label"><span>c_puct</span><span class="param-value" id="val_c_puct">1.4</span></div>
                <input type="range" id="c_puct" min="0.1" max="5.0" value="1.4" step="0.1">
            </div>
            <div class="param-row">
                <div class="param-label"><span>max_depth</span><span class="param-value" id="val_max_depth">10</span></div>
                <input type="range" id="max_depth" min="1" max="30" value="10" step="1">
            </div>
            <div class="param-row">
                <div class="param-label"><span>max_actions</span><span class="param-value" id="val_max_actions">10</span></div>
                <input type="range" id="max_actions" min="1" max="20" value="10" step="1">
            </div>
        </div>

        <div class="sidebar-section">
            <h3>Mode</h3>
            <div class="toggle-row">
                <span>Progressive Deepening</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="progressive_deepening" checked>
                    <span class="toggle-slider"></span>
                </label>
            </div>
            <div class="toggle-row">
                <span>Lazy Expansion</span>
                <label class="toggle-switch">
                    <input type="checkbox" id="lazy_expansion">
                    <span class="toggle-slider"></span>
                </label>
            </div>
            <div class="param-row" id="lazy_max_row" style="opacity:0.4;">
                <div class="param-label"><span>lazy_expansion_max</span><span class="param-value" id="val_lazy_expansion_max">8</span></div>
                <input type="range" id="lazy_expansion_max" min="2" max="30" value="8" step="1" disabled>
            </div>
        </div>

        <div class="sidebar-section controls">
            <div class="sim-counter">Simulation: <span id="simCounter">0</span> / <span id="simTotal">10</span></div>
            <div class="btn-row">
                <button class="btn btn-step" id="btnStep" title="Run one simulation">Step</button>
                <button class="btn btn-play" id="btnPlay" title="Auto-play simulations">Play</button>
                <button class="btn btn-reset" id="btnReset" title="Reset tree">Reset</button>
            </div>
            <div class="btn-row">
                <button class="btn btn-run-all" id="btnRunAll" title="Run all simulations instantly">Run All</button>
            </div>
            <div class="param-row">
                <div class="param-label"><span>Play speed (ms)</span><span class="param-value" id="val_speed">500</span></div>
                <input type="range" id="speed" min="50" max="2000" value="500" step="50">
            </div>
        </div>
    </div>

    <div class="right-panel">
        <div class="tree-container">
            <div id="network"></div>
        </div>
        <div class="stats-bar" id="statsBar">
            <div class="stat-item"><span class="stat-label">Total:</span><span class="stat-value" id="statTotal">1</span></div>
            <div class="stat-item"><span class="stat-label">Think:</span><span class="stat-value" id="statThink">0</span></div>
            <div class="stat-item"><span class="stat-label">Action:</span><span class="stat-value" id="statAction">0</span></div>
            <div class="stat-item"><span class="stat-label">Max depth:</span><span class="stat-value" id="statMaxDepth">0</span></div>
            <div class="stat-item"><span class="stat-label">Terminals:</span><span class="stat-value" id="statTerminals">0</span></div>
            <div class="stat-item"><span class="stat-label">Nodes/level:</span><span class="stat-value" id="statPerLevel">1</span></div>
        </div>
        <div class="legend">
            <div class="legend-item"><div class="legend-ellipse" style="background:#4CAF50;"></div><span>Think node</span></div>
            <div class="legend-item"><div class="legend-color" style="background:#2196F3;"></div><span>Action node</span></div>
            <div class="legend-item"><div class="legend-color" style="background:transparent; border:2px solid #F44336;"></div><span>Terminal</span></div>
            <div class="legend-item"><div class="legend-color" style="background:#FF9800;"></div><span>Root</span></div>
        </div>
    </div>
</div>

<div class="node-tooltip" id="nodeTooltip">
    <div class="tt-row"><span class="tt-label">ID:</span><span class="tt-value" id="ttId">-</span></div>
    <div class="tt-row"><span class="tt-label">Type:</span><span class="tt-value" id="ttType">-</span></div>
    <div class="tt-row"><span class="tt-label">Depth:</span><span class="tt-value" id="ttDepth">-</span></div>
    <div class="tt-row"><span class="tt-label">Value:</span><span class="tt-value" id="ttValue">-</span></div>
    <div class="tt-row"><span class="tt-label">Visits:</span><span class="tt-value" id="ttVisits">-</span></div>
    <div class="tt-row"><span class="tt-label">UCB:</span><span class="tt-value" id="ttUCB">-</span></div>
    <div class="tt-row"><span class="tt-label">Children:</span><span class="tt-value" id="ttChildren">-</span></div>
    <div class="tt-row"><span class="tt-label">Actions:</span><span class="tt-value" id="ttActions">-</span></div>
</div>

<script>
// =============================================
// MCTS Growth Simulator — Client-side engine
// =============================================

// --- Simulation Node ---
class SimNode {
    constructor(id, nodeType, parent, depth) {
        this.id = id;
        this.nodeType = nodeType; // "think" | "action" | "root"
        this.parent = parent;
        this.children = [];
        this.depth = depth;
        this.value = 0.0;
        this.visitCount = 0;
        this.isTerminal = false;
        this.actionCount = 0; // action nodes from root to here
        this.maxChildren = 0; // for lazy expansion
    }
}

// --- Globals ---
let nodeIdCounter = 0;
let root = null;
let allNodes = {};
let simCount = 0;
let playInterval = null;

// vis-network
let visNodes, visEdges, network;

// --- Parameter getters ---
function getParam(id) {
    const el = document.getElementById(id);
    if (el.type === 'checkbox') return el.checked;
    if (el.step && el.step.includes('.')) return parseFloat(el.value);
    return parseInt(el.value);
}

function params() {
    return {
        n_simulations: getParam('n_simulations'),
        branching_factor: getParam('branching_factor'),
        think_children: getParam('think_children'),
        think_maximum: getParam('think_maximum'),
        c_puct: getParam('c_puct'),
        max_depth: getParam('max_depth'),
        max_actions: getParam('max_actions'),
        progressive_deepening: getParam('progressive_deepening'),
        lazy_expansion: getParam('lazy_expansion'),
        lazy_expansion_max: getParam('lazy_expansion_max'),
    };
}

// --- Slider value display ---
const sliderIds = ['n_simulations','branching_factor','think_children','think_maximum','c_puct','max_depth','max_actions','lazy_expansion_max','speed'];
sliderIds.forEach(id => {
    const el = document.getElementById(id);
    const valEl = document.getElementById('val_' + id);
    if (el && valEl) {
        el.addEventListener('input', () => {
            valEl.textContent = el.value;
            if (id === 'n_simulations') {
                document.getElementById('simTotal').textContent = el.value;
            }
        });
    }
});

// Lazy expansion toggle
document.getElementById('lazy_expansion').addEventListener('change', function() {
    const row = document.getElementById('lazy_max_row');
    const slider = document.getElementById('lazy_expansion_max');
    if (this.checked) {
        row.style.opacity = '1';
        slider.disabled = false;
    } else {
        row.style.opacity = '0.4';
        slider.disabled = true;
    }
});

// --- Tree helpers ---
function createNode(nodeType, parent, depth) {
    const id = nodeIdCounter++;
    const node = new SimNode(id, nodeType, parent, depth);
    if (parent) {
        // Inherit action count
        node.actionCount = parent.actionCount + (nodeType === 'action' ? 1 : 0);
        parent.children.push(node);
    }
    allNodes[id] = node;
    return node;
}

function countConsecutiveThinks(node) {
    let count = 0;
    let current = node;
    while (current !== null) {
        if (current.nodeType === 'think') {
            count++;
        } else if (current.nodeType === 'action' || current.nodeType === 'root') {
            break;
        }
        current = current.parent;
    }
    return count;
}

// --- UCB1 ---
function ucbScore(parent, child, c_puct) {
    const exploitation = child.value;
    const exploration = c_puct * Math.sqrt(
        Math.log(parent.visitCount + 1) / (child.visitCount + 1)
    );
    return exploitation + exploration;
}

// --- Selection (faithful to mcts_search.py lines 488-551) ---
function select(root, p) {
    const path = [];
    let current = root;

    while (true) {
        path.push(current);

        if (current.isTerminal) return path;
        if (current.children.length === 0) return path;

        // Filter out terminal children
        const liveChildren = current.children.filter(c => !c.isTerminal);

        // Lazy expansion: can this node accept more children?
        const canExpand = p.lazy_expansion
            && current.maxChildren > 0
            && current.children.length < current.maxChildren;

        if (liveChildren.length === 0) {
            if (canExpand) return path;
            return path;
        }

        // Prefer unvisited live children
        const unvisited = liveChildren.filter(c => c.visitCount === 0);
        if (unvisited.length > 0) {
            const chosen = unvisited[Math.floor(Math.random() * unvisited.length)];
            path.push(chosen);
            return path;
        }

        if (canExpand) {
            // Virtual child UCB competes with best existing child
            const expandScore = p.c_puct * Math.sqrt(
                Math.log(current.visitCount + 1) / 1.0
            );
            let bestChild = liveChildren[0];
            let bestScore = -Infinity;
            for (const c of liveChildren) {
                const s = ucbScore(current, c, p.c_puct);
                if (s > bestScore) { bestScore = s; bestChild = c; }
            }
            if (expandScore > bestScore) {
                return path; // expand here
            }
            current = bestChild;
            continue;
        }

        // All visited, pick best UCB
        let bestChild = liveChildren[0];
        let bestScore = -Infinity;
        for (const c of liveChildren) {
            const s = ucbScore(current, c, p.c_puct);
            if (s > bestScore) { bestScore = s; bestChild = c; }
        }
        current = bestChild;
    }
}

// --- Expansion (faithful to mcts_search.py lines 567-605) ---
function expand(node, p) {
    if (node.depth >= p.max_depth) return [];
    if (node.isTerminal) return [];

    const thinkCount = countConsecutiveThinks(node);
    const children = [];

    if (thinkCount < p.think_maximum) {
        // Generate think children
        for (let i = 0; i < p.think_children; i++) {
            const child = createNode('think', node, node.depth + 1);
            // Terminal check: random 5% chance
            if (Math.random() < 0.05) child.isTerminal = true;
            children.push(child);
        }
    } else {
        // Generate action children
        const numActions = p.branching_factor;
        for (let i = 0; i < numActions; i++) {
            const child = createNode('action', node, node.depth + 1);
            // Terminal if max_actions reached or random 5%
            if (child.actionCount >= p.max_actions || Math.random() < 0.05) {
                child.isTerminal = true;
            }
            children.push(child);
        }
    }

    // Lazy expansion: set max_children on first expansion
    if (p.lazy_expansion && node.maxChildren === 0) {
        node.maxChildren = p.lazy_expansion_max;
    }

    return children;
}

// --- Rollout (structural only: random score) ---
function rollout() {
    return Math.random();
}

// --- Backprop (faithful to mcts_search.py lines 1041-1062) ---
function backprop(path, reward) {
    if (isNaN(reward)) {
        for (const node of path) node.visitCount++;
        return;
    }
    for (const node of path) {
        node.visitCount++;
        node.value += (reward - node.value) / node.visitCount;
    }
}

// --- Run one simulation ---
function runSimulation(p) {
    // 1. Select
    const path = select(root, p);
    const leaf = path[path.length - 1];

    // Already terminal or fully expanded with all terminal children
    if (leaf.isTerminal) {
        backprop(path, 0.0);
        return;
    }

    if (p.progressive_deepening) {
        // Progressive deepening: expand multiple layers
        let expandNode = leaf;
        const deepeningPath = []; // [{siblings, chosen}]

        while (true) {
            if (expandNode.depth >= p.max_depth) break;
            if (expandNode.actionCount >= p.max_actions) break;
            if (expandNode.isTerminal) break;

            const newChildren = expand(expandNode, p);
            if (newChildren.length === 0) break;

            const actionChildren = newChildren.filter(c => c.nodeType === 'action');
            const thinkChildren = newChildren.filter(c => c.nodeType === 'think');

            if (actionChildren.length > 0) {
                const chosen = actionChildren[Math.floor(Math.random() * actionChildren.length)];
                deepeningPath.push({ siblings: newChildren, chosen });
                if (chosen.isTerminal) break;
                expandNode = chosen;
            } else if (thinkChildren.length > 0) {
                const chosen = thinkChildren[Math.floor(Math.random() * thinkChildren.length)];
                deepeningPath.push({ siblings: newChildren, chosen });
                expandNode = chosen;
            } else {
                break;
            }
        }

        if (deepeningPath.length === 0) {
            backprop(path, 0.0);
            return;
        }

        // Rollout all action siblings at every level, backprop each
        for (const { siblings, chosen } of deepeningPath) {
            for (const child of siblings) {
                if (child.visitCount === 0 && child.nodeType !== 'think') {
                    const reward = rollout();
                    // Build path from root through deepening path to this child
                    const childPath = [...path];
                    for (const dp of deepeningPath) {
                        if (dp.siblings.includes(child)) {
                            childPath.push(child);
                            break;
                        }
                        if (!childPath.includes(dp.chosen)) {
                            childPath.push(dp.chosen);
                        }
                    }
                    backprop(childPath, reward);
                }
            }
        }

    } else {
        // Regular MCTS: expand one level, rollout each child
        const newChildren = expand(leaf, p);
        if (newChildren.length === 0) {
            backprop(path, 0.0);
            return;
        }
        for (const child of newChildren) {
            const reward = rollout();
            backprop([...path, child], reward);
        }
    }
}

// --- vis-network rendering ---
function buildVisData() {
    const nodes = [];
    const edges = [];

    for (const id in allNodes) {
        const node = allNodes[id];
        let color, shape, borderColor, borderWidth;

        if (node.nodeType === 'root') {
            color = '#FF9800';
            shape = 'diamond';
            borderColor = '#E65100';
            borderWidth = 2;
        } else if (node.nodeType === 'think') {
            color = '#4CAF50';
            shape = 'ellipse';
            borderColor = node.isTerminal ? '#F44336' : '#2E7D32';
            borderWidth = node.isTerminal ? 3 : 1;
        } else {
            color = '#2196F3';
            shape = 'box';
            borderColor = node.isTerminal ? '#F44336' : '#1565C0';
            borderWidth = node.isTerminal ? 3 : 1;
        }

        const label = `v=${node.value.toFixed(2)} n=${node.visitCount}`;

        nodes.push({
            id: node.id,
            label: label,
            shape: shape,
            color: {
                background: color,
                border: borderColor,
                highlight: { background: color, border: '#fff' }
            },
            borderWidth: borderWidth,
            font: { size: 10, color: '#fff' },
            level: node.depth,
            margin: 5,
        });

        if (node.parent !== null) {
            edges.push({
                from: node.parent.id,
                to: node.id,
                arrows: 'to',
                color: { color: '#555', highlight: '#aaa' },
                width: 1,
            });
        }
    }

    return { nodes, edges };
}

function updateVis() {
    const data = buildVisData();
    visNodes.clear();
    visEdges.clear();
    visNodes.add(data.nodes);
    visEdges.add(data.edges);
}

function updateStats() {
    let total = 0, thinkCount = 0, actionCount = 0, maxDepth = 0, terminals = 0;
    const perLevel = {};

    for (const id in allNodes) {
        const node = allNodes[id];
        total++;
        if (node.nodeType === 'think') thinkCount++;
        if (node.nodeType === 'action') actionCount++;
        if (node.depth > maxDepth) maxDepth = node.depth;
        if (node.isTerminal) terminals++;
        perLevel[node.depth] = (perLevel[node.depth] || 0) + 1;
    }

    document.getElementById('statTotal').textContent = `${total} (T:${thinkCount}, A:${actionCount})`;
    document.getElementById('statThink').textContent = thinkCount;
    document.getElementById('statAction').textContent = actionCount;
    document.getElementById('statMaxDepth').textContent = maxDepth;
    document.getElementById('statTerminals').textContent = terminals;

    const levels = Object.keys(perLevel).map(Number).sort((a, b) => a - b);
    const perLevelStr = levels.map(l => perLevel[l]).join(',');
    document.getElementById('statPerLevel').textContent = perLevelStr;

    document.getElementById('simCounter').textContent = simCount;
}

// --- Reset ---
function resetTree() {
    stopPlay();
    nodeIdCounter = 0;
    allNodes = {};
    simCount = 0;
    root = createNode('root', null, 0);
    root.actionCount = 0;
    updateVis();
    updateStats();
    network.fit();
}

// --- Step ---
function stepSim() {
    const p = params();
    if (simCount >= p.n_simulations) return;
    simCount++;
    runSimulation(p);
    updateVis();
    updateStats();
}

// --- Play / Pause ---
function togglePlay() {
    if (playInterval) {
        stopPlay();
    } else {
        startPlay();
    }
}

function startPlay() {
    const btn = document.getElementById('btnPlay');
    btn.textContent = 'Pause';
    btn.classList.add('playing');
    const speed = parseInt(document.getElementById('speed').value);
    playInterval = setInterval(() => {
        const p = params();
        if (simCount >= p.n_simulations) {
            stopPlay();
            return;
        }
        stepSim();
    }, speed);
}

function stopPlay() {
    if (playInterval) {
        clearInterval(playInterval);
        playInterval = null;
    }
    const btn = document.getElementById('btnPlay');
    btn.textContent = 'Play';
    btn.classList.remove('playing');
}

// --- Run All ---
function runAll() {
    stopPlay();
    const p = params();
    while (simCount < p.n_simulations) {
        simCount++;
        runSimulation(p);
    }
    updateVis();
    updateStats();
    network.fit();
}

// --- Tooltip ---
function showTooltip(nodeId, domEvent) {
    const node = allNodes[nodeId];
    if (!node) return;

    const tt = document.getElementById('nodeTooltip');
    document.getElementById('ttId').textContent = node.id;
    document.getElementById('ttType').textContent = node.nodeType;
    document.getElementById('ttDepth').textContent = node.depth;
    document.getElementById('ttValue').textContent = node.value.toFixed(4);
    document.getElementById('ttVisits').textContent = node.visitCount;

    // Compute UCB relative to parent
    let ucbStr = '-';
    if (node.parent) {
        const p = params();
        ucbStr = ucbScore(node.parent, node, p.c_puct).toFixed(4);
    }
    document.getElementById('ttUCB').textContent = ucbStr;
    document.getElementById('ttChildren').textContent = node.children.length +
        (node.maxChildren > 0 ? ` / max ${node.maxChildren}` : '');
    document.getElementById('ttActions').textContent = node.actionCount;

    tt.style.display = 'block';
    // Position near the pointer
    const x = domEvent.center ? domEvent.center.x : (domEvent.clientX || 200);
    const y = domEvent.center ? domEvent.center.y : (domEvent.clientY || 200);
    tt.style.left = (x + 15) + 'px';
    tt.style.top = (y + 15) + 'px';

    // Keep tooltip on screen
    const rect = tt.getBoundingClientRect();
    if (rect.right > window.innerWidth) {
        tt.style.left = (x - rect.width - 15) + 'px';
    }
    if (rect.bottom > window.innerHeight) {
        tt.style.top = (y - rect.height - 15) + 'px';
    }
}

function hideTooltip() {
    document.getElementById('nodeTooltip').style.display = 'none';
}

// --- Initialize ---
window.addEventListener('load', function() {
    if (typeof vis === 'undefined') {
        document.body.innerHTML = '<div style="color:red;padding:20px;"><h2>Error</h2><p>vis-network library failed to load. Check your internet connection.</p></div>';
        return;
    }

    visNodes = new vis.DataSet();
    visEdges = new vis.DataSet();

    const container = document.getElementById('network');
    const data = { nodes: visNodes, edges: visEdges };
    const options = {
        layout: {
            hierarchical: {
                direction: 'UD',
                sortMethod: 'directed',
                levelSeparation: 80,
                nodeSpacing: 50,
                treeSpacing: 60,
            }
        },
        physics: false,
        interaction: {
            hover: true,
            selectConnectedEdges: false,
            tooltipDelay: 0,
        },
        edges: {
            color: { color: '#555' },
            width: 1,
            smooth: false,
        },
    };

    network = new vis.Network(container, data, options);

    // Click → tooltip
    network.on('click', function(evtParams) {
        if (evtParams.nodes.length > 0) {
            showTooltip(evtParams.nodes[0], evtParams.event);
        } else {
            hideTooltip();
        }
    });

    // Hover → tooltip
    network.on('hoverNode', function(evtParams) {
        showTooltip(evtParams.node, evtParams.event);
    });
    network.on('blurNode', function() {
        hideTooltip();
    });

    // Buttons
    document.getElementById('btnStep').addEventListener('click', stepSim);
    document.getElementById('btnPlay').addEventListener('click', togglePlay);
    document.getElementById('btnReset').addEventListener('click', resetTree);
    document.getElementById('btnRunAll').addEventListener('click', runAll);

    // Speed slider updates play interval if playing
    document.getElementById('speed').addEventListener('input', function() {
        if (playInterval) {
            stopPlay();
            startPlay();
        }
    });

    // Initialize tree
    resetTree();
});
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive MCTS tree growth simulator HTML.'
    )
    parser.add_argument(
        '-o', '--output',
        default='mcts_growth_sim.html',
        help='Output HTML file path (default: mcts_growth_sim.html)'
    )
    parser.add_argument(
        '--no-open',
        action='store_true',
        help='Do not auto-open in browser'
    )

    args = parser.parse_args()

    html_content = generate_html()

    with open(args.output, 'w') as f:
        f.write(html_content)

    abs_path = os.path.abspath(args.output)
    print(f"Generated: {abs_path}")

    if not args.no_open:
        try:
            webbrowser.open(f'file://{abs_path}')
            print("Opened in browser.")
        except Exception:
            print("Could not auto-open browser. Open the file manually.")


if __name__ == '__main__':
    main()
