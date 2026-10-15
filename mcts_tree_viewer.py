#!/usr/bin/env python3
"""
MCTS Tree Viewer - Interactive HTML visualization for MCTS tree data.

Creates a standalone HTML file that displays:
- Left panel (60%): Screenshot with action coordinate overlay (red dot)
- Right panel (40%): Interactive tree using vis-network

Usage:
    python visualizers/mcts_tree_viewer.py /path/to/mcts.jsonl -o output.html
"""

import argparse
import json
import html
from pathlib import Path


def load_jsonl(path: str) -> dict:
    """Load the first line of a JSONL file."""
    with open(path, 'r') as f:
        return json.loads(f.readline())


def parse_coords(used_coords: list) -> tuple:
    """
    Parse used_coords like ['(0.618, 0.362)'] into (x, y) normalized floats.
    Returns (None, None) if no valid coords.
    """
    if not used_coords or len(used_coords) == 0:
        return None, None

    coord_str = used_coords[0]  # Take the first coordinate
    # Parse "(0.618, 0.362)" format
    try:
        coord_str = coord_str.strip().strip('()')
        parts = coord_str.split(',')
        if len(parts) == 2:
            x = float(parts[0].strip())
            y = float(parts[1].strip())
            # Skip (0, 0) as it usually indicates no action
            if x == 0.0 and y == 0.0:
                return None, None
            return x, y
    except (ValueError, AttributeError):
        pass
    return None, None


def build_vis_data(tree: dict) -> tuple:
    """
    Convert tree structure to vis-network nodes and edges.
    Returns (nodes_list, edges_list, node_data_map).
    """
    nodes = []
    edges = []
    node_data = {}  # Maps node_id to full node data
    node_id_counter = [0]  # Use list for closure mutation

    def get_node_color(value: float) -> str:
        """Return color based on node value."""
        if value > 0.7:
            return '#4CAF50'  # Green
        elif value > 0.3:
            return '#FFC107'  # Yellow/Amber
        else:
            return '#F44336'  # Red

    def get_abbreviated_action(raw_prediction: str) -> str:
        """Extract abbreviated action description."""
        if not raw_prediction:
            return "root"

        # Look for action patterns like click, type, scroll
        raw = raw_prediction.lower()
        if 'click' in raw:
            return 'click'
        elif 'type' in raw or 'fill' in raw:
            return 'type'
        elif 'scroll' in raw:
            return 'scroll'
        elif 'press' in raw:
            return 'press'
        elif 'hover' in raw:
            return 'hover'
        elif 'goto' in raw or 'navigate' in raw:
            return 'goto'
        else:
            # Return first 10 chars
            return raw_prediction[:10].strip()

    def traverse(node: dict, parent_id: int = None):
        """Recursively traverse tree and build vis data."""
        current_id = node_id_counter[0]
        node_id_counter[0] += 1

        value = node.get('value', 0)
        visit_count = node.get('visit_count', 0)
        raw_prediction = node.get('raw_prediction', '')
        depth = node.get('depth', 0)

        # Create abbreviated label
        action_abbrev = get_abbreviated_action(raw_prediction)
        label = f"{action_abbrev}\nv={value:.2f}\nn={visit_count}"

        # Node styling
        color = get_node_color(value)

        nodes.append({
            'id': current_id,
            'label': label,
            'color': {
                'background': color,
                'border': '#333',
                'highlight': {
                    'background': color,
                    'border': '#000'
                }
            },
            'font': {'size': 12, 'color': '#000'},
            'shape': 'box',
            'level': depth
        })

        # Store full node data for display
        node_data[current_id] = {
            'node_type': node.get('node_type', ''),
            'value': value,
            'visit_count': visit_count,
            'depth': depth,
            'raw_prediction': raw_prediction,
            'thought_text': node.get('thought_text', ''),
            'browser_url': node.get('browser_url', ''),
            'action_success': node.get('action_success', True),
            'is_terminal': node.get('is_terminal', False),
            'image': node.get('image', ''),
            'used_coords': node.get('used_coords', [])
        }

        # Create edge from parent
        if parent_id is not None:
            edges.append({
                'from': parent_id,
                'to': current_id,
                'arrows': 'to'
            })

        # Process children
        children = node.get('children', [])
        for child in children:
            traverse(child, current_id)

    traverse(tree)
    return nodes, edges, node_data


def generate_html(data: dict, output_path: str):
    """Generate the HTML visualization file."""

    question = data.get('question', 'Unknown')
    initial_image = data.get('image', data.get('tree', {}).get('image', ''))

    # Build vis-network data
    nodes, edges, node_data = build_vis_data(data['tree'])

    # Convert to JSON for embedding
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    node_data_json = json.dumps(node_data)

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCTS Tree Viewer</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a2e;
            color: #eee;
            height: 100vh;
            overflow: hidden;
        }}

        .header {{
            background: #16213e;
            padding: 10px 20px;
            border-bottom: 1px solid #0f3460;
        }}

        .header h1 {{
            font-size: 18px;
            color: #e94560;
        }}

        .header .question {{
            font-size: 14px;
            color: #aaa;
            margin-top: 4px;
        }}

        .container {{
            display: flex;
            height: calc(100vh - 60px);
        }}

        .left-panel {{
            width: 60%;
            padding: 15px;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #0f3460;
        }}

        .right-panel {{
            width: 40%;
            display: flex;
            flex-direction: column;
        }}

        .screenshot-container {{
            flex: 1;
            position: relative;
            background: #0f0f23;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .screenshot-container canvas {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}

        .node-info {{
            background: #16213e;
            padding: 12px;
            border-radius: 8px;
            margin-top: 10px;
            font-size: 13px;
            max-height: 150px;
            overflow-y: auto;
        }}

        .node-info .label {{
            color: #e94560;
            font-weight: bold;
        }}

        .node-info .value {{
            color: #ddd;
            margin-left: 8px;
        }}

        .node-info .row {{
            margin-bottom: 6px;
            display: flex;
            flex-wrap: wrap;
        }}

        .tree-container {{
            flex: 1;
            background: #0f0f23;
            position: relative;
            min-height: 400px;
        }}

        #network {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            width: 100%;
            height: 100%;
        }}

        .legend {{
            padding: 10px;
            background: #16213e;
            font-size: 12px;
            display: flex;
            gap: 15px;
            justify-content: center;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}

        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}

        .action-text {{
            background: #1a1a2e;
            padding: 8px;
            border-radius: 4px;
            margin-top: 6px;
            font-family: monospace;
            font-size: 12px;
            word-break: break-all;
            max-height: 60px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>MCTS Tree Viewer</h1>
        <div class="question">Question: {html.escape(question)}</div>
    </div>

    <div class="container">
        <div class="left-panel">
            <div class="screenshot-container">
                <canvas id="screenshotCanvas"></canvas>
            </div>
            <div class="node-info" id="nodeInfo">
                <div class="row"><span class="label">Node:</span><span class="value" id="infoNode">Click a node to view details</span></div>
                <div class="row"><span class="label">Value:</span><span class="value" id="infoValue">-</span></div>
                <div class="row"><span class="label">Visits:</span><span class="value" id="infoVisits">-</span></div>
                <div class="row"><span class="label">Depth:</span><span class="value" id="infoDepth">-</span></div>
                <div class="row"><span class="label">URL:</span><span class="value" id="infoUrl">-</span></div>
                <div class="action-text" id="actionText">-</div>
            </div>
        </div>

        <div class="right-panel">
            <div class="tree-container">
                <div id="network"></div>
            </div>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: #4CAF50;"></div>
                    <span>High value (&gt;0.7)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #FFC107;"></div>
                    <span>Medium (0.3-0.7)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #F44336;"></div>
                    <span>Low (&lt;0.3)</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Error handling
        window.onerror = function(msg, url, lineNo, columnNo, error) {{
            console.error('Error:', msg, 'at line', lineNo);
            document.body.innerHTML = '<div style="color:red;padding:20px;"><h2>JavaScript Error</h2><pre>' + msg + '\\nLine: ' + lineNo + '</pre></div>';
            return false;
        }};

        // Check if vis library loaded
        if (typeof vis === 'undefined') {{
            document.body.innerHTML = '<div style="color:red;padding:20px;"><h2>Error</h2><p>vis-network library failed to load. Check your internet connection.</p></div>';
            throw new Error('vis-network not loaded');
        }}

        console.log('Initializing MCTS Tree Viewer...');

        // Embedded data
        const nodes = new vis.DataSet({nodes_json});
        const edges = new vis.DataSet({edges_json});
        const nodeData = {node_data_json};
        const initialImage = "{initial_image}";

        // Initialize network
        const container = document.getElementById('network');
        const data = {{ nodes: nodes, edges: edges }};
        const options = {{
            layout: {{
                hierarchical: {{
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 100,
                    nodeSpacing: 80
                }}
            }},
            physics: false,
            interaction: {{
                hover: true,
                selectConnectedEdges: false
            }},
            edges: {{
                color: '#666',
                width: 1.5
            }}
        }};

        console.log('Container element:', container);
        console.log('Container size:', container.offsetWidth, 'x', container.offsetHeight);
        console.log('Nodes count:', nodes.length, 'Edges count:', edges.length);

        const network = new vis.Network(container, data, options);

        // Ensure network fits and is visible
        network.once('afterDrawing', function() {{
            console.log('Network drawn successfully');
            network.fit();
        }});

        // Canvas and image handling
        const canvas = document.getElementById('screenshotCanvas');
        const ctx = canvas.getContext('2d');
        let currentImage = null;
        let currentCoords = null;

        function drawScreenshot(imageDataUrl, coords) {{
            if (!imageDataUrl) {{
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.fillStyle = '#333';
                ctx.font = '16px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText('No screenshot available', canvas.width / 2, canvas.height / 2);
                return;
            }}

            const img = new Image();
            img.onload = function() {{
                // Set canvas size to match image
                canvas.width = img.width;
                canvas.height = img.height;

                // Draw image
                ctx.drawImage(img, 0, 0);

                // Draw red dot at action coordinate
                if (coords && coords.x !== null && coords.y !== null) {{
                    const pixelX = coords.x * img.width;
                    const pixelY = coords.y * img.height;

                    // Draw outer glow
                    ctx.beginPath();
                    ctx.arc(pixelX, pixelY, 20, 0, 2 * Math.PI);
                    ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
                    ctx.fill();

                    // Draw red dot
                    ctx.beginPath();
                    ctx.arc(pixelX, pixelY, 12, 0, 2 * Math.PI);
                    ctx.fillStyle = '#e94560';
                    ctx.fill();
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 2;
                    ctx.stroke();

                    // Draw crosshair
                    ctx.beginPath();
                    ctx.moveTo(pixelX - 20, pixelY);
                    ctx.lineTo(pixelX - 8, pixelY);
                    ctx.moveTo(pixelX + 8, pixelY);
                    ctx.lineTo(pixelX + 20, pixelY);
                    ctx.moveTo(pixelX, pixelY - 20);
                    ctx.lineTo(pixelX, pixelY - 8);
                    ctx.moveTo(pixelX, pixelY + 8);
                    ctx.lineTo(pixelX, pixelY + 20);
                    ctx.strokeStyle = '#fff';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }}
            }};
            img.src = imageDataUrl;
        }}

        function parseCoords(usedCoords) {{
            if (!usedCoords || usedCoords.length === 0) {{
                return {{ x: null, y: null }};
            }}
            const coordStr = usedCoords[0].trim().replace(/[()]/g, '');
            const parts = coordStr.split(',');
            if (parts.length === 2) {{
                const x = parseFloat(parts[0].trim());
                const y = parseFloat(parts[1].trim());
                // Skip (0, 0) as it usually indicates no action
                if (x === 0 && y === 0) {{
                    return {{ x: null, y: null }};
                }}
                return {{ x, y }};
            }}
            return {{ x: null, y: null }};
        }}

        function updateNodeInfo(nodeId) {{
            const data = nodeData[nodeId];
            if (!data) return;

            document.getElementById('infoNode').textContent =
                data.node_type + ' (ID: ' + nodeId + ')';
            document.getElementById('infoValue').textContent =
                data.value.toFixed(3);
            document.getElementById('infoVisits').textContent =
                data.visit_count;
            document.getElementById('infoDepth').textContent =
                data.depth;
            document.getElementById('infoUrl').textContent =
                data.browser_url || '-';

            const actionText = data.raw_prediction || data.thought_text || '-';
            document.getElementById('actionText').textContent = actionText;

            // Draw screenshot with coordinate
            const coords = parseCoords(data.used_coords);
            drawScreenshot(data.image, coords);
        }}

        // Event handlers
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                updateNodeInfo(nodeId);
            }}
        }});

        // Initialize with root node (ID 0)
        setTimeout(() => {{
            // Set initial canvas size
            canvas.width = 800;
            canvas.height = 600;

            // Show root node info
            updateNodeInfo(0);

            // Select root node visually
            network.selectNodes([0]);
        }}, 100);
    </script>
</body>
</html>
'''

    with open(output_path, 'w') as f:
        f.write(html_content)

    print(f"Generated visualization: {output_path}")
    print(f"  - Total nodes: {len(nodes)}")
    print(f"  - Question: {question[:50]}...")


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive MCTS tree visualization from JSONL data.'
    )
    parser.add_argument(
        'input',
        help='Path to the MCTS JSONL file'
    )
    parser.add_argument(
        '-o', '--output',
        default='mcts_tree.html',
        help='Output HTML file path (default: mcts_tree.html)'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading JSONL: {args.input}")
    data = load_jsonl(args.input)

    # Generate HTML
    generate_html(data, args.output)


if __name__ == '__main__':
    main()
