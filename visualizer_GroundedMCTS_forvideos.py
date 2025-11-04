# visualizer_enhanced.py - Enhanced MCTS Visualizer with Tabbed Interface and Retrieval Support
# Version: 29 - FIXED BASIC/FULL PROMPT TOGGLE FOR STRATEGY CONTENT
# Last Updated: 2025-08-14
# CHANGES:
# - Added images_locations.json support for configurable image paths and scaling
# - Implemented automatic image scaling to match VLM-processed images
# - Fixed coordinate alignment for experiments with different scale factors
# - Enhanced backward compatibility for experiments without scaling configuration
# - Added debug coordinate display with scale factor information

"""Enhanced MCTS Visualizer (v29)

Overview
========
This module renders an interactive, self-contained HTML visualization of a
Monte-Carlo Tree Search (MCTS) run with optional retrieval-augmented prompts and
rollout details. The HTML uses vis-network for the tree on the right and a
tabbed inspector on the left to view the question image (with coordinate
highlights), input prompts (with ability to hide/show retrieved examples),
retrieved examples (with images/links), per-node/rollout output, aggregate
statistics, and a legend.

Command-line usage
------------------
- Basic:
    python -m vlmsearch.visualization.visualizer_enhanced input.jsonl

- With options:
    python -m vlmsearch.visualization.visualizer_enhanced \
        input.jsonl \
        --output_filename mcts_visualization_enhanced \
        --max_depth 3 \
        --debug

Arguments
---------
- input_file (positional):
    Path to a JSON Lines (JSONL) file. Only the first line is read and must
    contain a single JSON object that matches the schema defined below. If you
    have a multi-line JSON file, minify it to a single line or adapt the loader
    to read the whole file.

- --output_filename (str, default: "mcts_visualization_enhanced"):
    Base filename for outputs. Produces `{output_filename}.html` and
    `{output_filename}_stats.txt` in the current working directory.

- --max_depth (int, default: None):
    Limits how deep the tree is drawn. Use None for full depth.

- --debug (flag):
    When set, shows live image coordinates under the cursor in the image tab.

Expected input schema
---------------------
Top-level object keys:
- id (str|int, optional): Experiment/sample identifier. Not used by the visualizer
  but commonly present in experiment outputs.
- question (str, optional): A human-readable question or task description.
- true_answer (str, optional): Ground-truth or reference answer.
- image (str, optional): Absolute/relative file path or URL of the question
  image. If omitted, the image tab shows a drop zone. If provided and a local
  file ≤ 10 MB, the image is embedded as base64 for portability. URLs are left
  as-is and may be subject to CORS.
- system_prompt (str, optional): The system prompt used by the search. Not used
  by the visualizer directly, but often included for completeness.
- global_search_time (float, optional): Total search time in seconds. If
  provided, it is formatted with `:.2f` in the HTML/stats, so it should be
  numeric.
- tree (Node, required): The root node of the MCTS tree.

Node schema (recursive):
    {
      "thought_text": str,            # Display text shown on the node label
      "visit_count": int,             # Optional, defaults to 0
      "value": float,                 # Optional, defaults to 0.0
      "is_terminal": bool,            # Optional, defaults to False
      "prompt_for_node": str,         # Optional, raw prompt used to expand node
      "rollouts": [Rollout, ...],     # Optional
      "children": [Node, ...],        # Optional
      "used_coords": [str, ...]       # Optional, accumulated (x,y) strings
    }

Rollout schema:
    {
      "final_answer": str,                # Typically an action token or decision
      "reward": float,                    # Reward for the rollout
      "rollout_prompts": [str, ...],      # Optional, prompt trace per step
      "ephemeral_depth": int,             # Optional, depth for ephemeral context
      "ephemeral_retrieved_paths": [str, ...],  # Optional, image URLs or paths
      "ephemeral_texts": [str, ...],      # Optional, texts for retrieved items
      "depth": int,                       # Optional, total steps in rollout
      "num_same_points": int,             # Optional, repeated coords count
      "num_different_points": int         # Optional, new coords count
    }

Additional keys
---------------
Your pipeline may include additional keys at the top level or inside nodes/rollouts.
The visualizer ignores unknown fields and only relies on the keys described above.

Coordinate extraction and image highlights
-----------------------------------------
The visualizer extracts coordinates from node thoughts and rollout final
answers using the pattern "(x, y)" with integers or floats, for example:
"click (245, 380)". These coordinates are assumed to be pixel coordinates in
the original image space. When available, circular spotlights are cut into a
dark overlay to highlight the referenced regions. If no image is provided or no
coordinates are detected, the overlay remains empty.

Retrieved examples (how they are parsed)
---------------------------------------
Two sources are supported:
1) From `prompt_for_node`, when examples are delimited using lines that start
   with `--- Example ... ---` and end with `--- End Example ---`. Inside the
   example body, a line containing `url: ...` will be treated as an image path
   or URL.
2) From rollout fields `ephemeral_retrieved_paths` (and corresponding
   `ephemeral_texts`).
Duplicates are suppressed via a per-node uniqueness key composed from
title/text/path so that the examples tab does not repeat the same item.

Outputs
-------
- {output_filename}.html
    The interactive visualization file with all assets inlined except external
    image URLs (which are linked and attempted to be rendered subject to CORS).

- {output_filename}_stats.txt
    A compact, human-readable dump of aggregate counts and distributions.

Minimal working input (single line JSON; formatted here for clarity)
-------------------------------------------------------------------
    {
      "question": "What number is in the red square?",
      "true_answer": "42",
      "image": "/abs/path/to/question.png",
      "global_search_time": 12.34,
      "tree": {
        "thought_text": "<think> Start",
        "visit_count": 10,
        "value": 0.12,
        "children": [
          {
            "thought_text": "click (123, 456)",
            "visit_count": 3,
            "value": 0.8,
            "rollouts": [
              {
                "final_answer": "stop",
                "reward": 1.0,
                "rollout_prompts": ["...prompt step 1...", "...step 2..."]
              }
            ]
          }
        ]
      }
    }

Adapting to other tasks
-----------------------
- If there is no visual context, omit `image`; coordinate highlights are simply
  not drawn.
- You can use arbitrary action vocabularies in `thought_text` and
  `final_answer`. The visualizer heuristically colors nodes (e.g., strings
  starting with `<think>` become "Thinking", actions like `click`, `scroll`,
  `go_back`, `stop` become "Action"). Any other text is shown as "Other".
- To show retrieval examples, either embed them in `prompt_for_node` using the
  `--- Example ---` / `--- End Example ---` delimiters or provide
  `ephemeral_retrieved_paths` and `ephemeral_texts` on rollouts.
- If you provide `global_search_time`, ensure it is numeric (float seconds), as
  it is formatted with `:.2f` in both the HTML and the stats text file.

Programmatic use
----------------
Key functions you may reuse:
- get_tree_statistics(tree_data) -> dict
    Walks the tree to compute counts, maxima, and rollout action distribution.
- build_graph_data(tree_data, max_depth) -> dict
    Produces the nodes/edges for vis-network and an interactive data map.
- generate_html_file(tree_data, graph_data, stats, output_filename, debug)
    Renders a single HTML file and writes a companion stats text file.

Note
----
This script currently reads ONLY the first line from `input_file`. Supply a
JSONL file whose first line is the desired record, or ensure your JSON object
is minified onto a single line.
"""

import argparse
import json
import os
import html
import re
import base64
import mimetypes
from copy import deepcopy
from pathlib import Path

try:
    from utils.vis_utils import vis_all
    HAS_VIS_UTILS = True
except ImportError:
    HAS_VIS_UTILS = False
    print("Warning: utils.vis_utils not found. Visualization features will be limited.")


skill_map = {
    1: "Grasp",
    2: "Place"
}


def load_or_create_images_config(experiment_dir):
    """
    Load or create images_locations.json in the experiment directory.
    
    Schema (all fields optional):
    {
        "images_location": "/path/to/images" (optional - if not specified, uses JSONL path),
        "scale": 1.5 (optional - scale factor, mutually exclusive with resolution),
        "resolution": "1920x1080" (optional - target resolution, mutually exclusive with scale)
    }
    
    If scale is provided, it takes precedence over resolution.
    All fields are optional - missing fields maintain backward compatibility.
    """
    config_path = Path(experiment_dir) / "images_locations.json"
    
    # If no config file exists, return empty config (full backward compatibility)
    if not config_path.exists():
        # Try to infer configuration from experiment directory name for convenience
        dir_name = Path(experiment_dir).name
        inferred_config = {}
        
        # Only infer if directory name matches known patterns
        if "HIGH_RES" in dir_name.upper():
            inferred_config["images_location"] = "/data/user_data/adityaku/data/web_action/images"
            print(f"Inferred HIGH_RES images location from directory name: {dir_name}")
        elif "LOW_RES" in dir_name.upper():
            inferred_config["images_location"] = "/data/user_data/adityaku/data/web_action/images_backup"
            print(f"Inferred LOW_RES images location from directory name: {dir_name}")
        
        # Extract scale factor from directory name if present
        scale_match = re.search(r'(\d+(?:\.\d+)?)X_UPSAMPLE', dir_name.upper())
        if scale_match:
            try:
                inferred_config["scale"] = float(scale_match.group(1))
                print(f"Inferred scale factor {inferred_config['scale']} from directory name: {dir_name}")
            except ValueError:
                pass
        
        # Only create config file if we inferred something useful
        if inferred_config:
            try:
                with open(config_path, 'w') as f:
                    json.dump(inferred_config, f, indent=2)
                print(f"Created images config at: {config_path}")
                print(f"Configuration: {inferred_config}")
            except Exception as e:
                print(f"Warning: Could not create images config file: {e}")
        
        return inferred_config
    
    # Load existing config file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            print(f"Loaded images config from: {config_path}")
            
            # Validate mutually exclusive fields
            if config.get("scale") is not None and config.get("resolution") is not None:
                print("Warning: Both 'scale' and 'resolution' specified. Using 'scale' and ignoring 'resolution'.")
                config.pop("resolution", None)
            
            return config
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error reading images config {config_path}: {e}")
        print("Falling back to backward compatibility mode (no configuration)")
        return {}

def resolve_image_path(original_image_path, images_config):
    """
    Resolve the actual image path based on configuration.
    Note: Coordinates are NOT scaled - the VLM sees scaled images so coordinates are already correct.
    
    Args:
        original_image_path: Original image path from JSONL
        images_config: Configuration dict (can be empty for backward compatibility)
    
    Returns:
        str: resolved_image_path
    """
    # If no config or no images_location specified, use original path (backward compatibility)
    if not images_config or not images_config.get("images_location"):
        return original_image_path
    
    resolved_path = original_image_path
    new_base = images_config["images_location"]
    original_path = Path(original_image_path)
    
    # Extract the relative path from the original (everything after 'images')
    parts = original_path.parts
    try:
        images_index = next(i for i, part in enumerate(parts) if 'images' in part.lower())
        relative_parts = parts[images_index + 1:]  # Get parts after 'images' or 'images_backup'
        resolved_path = str(Path(new_base) / Path(*relative_parts))
        print(f"Resolved image path: {original_image_path} -> {resolved_path}")
    except (StopIteration, IndexError):
        print(f"Warning: Could not resolve path structure for {original_image_path}, using original")
        resolved_path = original_image_path
    
    return resolved_path

def _extract_coords(text):
    """Extracts all (x, y) coordinates from a string, supporting integers and floats."""
    if not isinstance(text, str):
        return []

    # This pattern supports floats and is from the original visualizer
    pattern = r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)'
    matches = re.findall(pattern, text)

    coords_list = []
    for match in matches:
        try:
            # match is a tuple of strings, e.g., ('123.4', '.4', '567', '')
            # The actual numbers are in group 1 and 2 of the regex, which correspond to match[0] and match[1]
            x, y = float(match[0]), float(match[1])

            if 0 <= x <= 10000 and 0 <= y <= 10000:
                coords_list.append([x, y])
        except (ValueError, IndexError):
            continue

    return coords_list

def _extract_bounding_boxes(text):
    """
    Extracts bounding boxes from text patterns.
    Supports multiple formats:
    - box(x1, y1, x2, y2)
    - bbox(x1, y1, x2, y2)
    - [x1, y1, x2, y2]
    - rect(x, y, width, height)

    Returns list of dicts: [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": "..."}, ...]
    """
    if not isinstance(text, str):
        return []

    boxes = []

    # Pattern 1: box(x1, y1, x2, y2) or bbox(x1, y1, x2, y2)
    pattern1 = r'(?:box|bbox)\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)'
    for match in re.finditer(pattern1, text, re.IGNORECASE):
        try:
            x1, y1, x2, y2 = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
            if 0 <= x1 <= 10000 and 0 <= y1 <= 10000 and 0 <= x2 <= 10000 and 0 <= y2 <= 10000:
                boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "label": match.group(0)})
        except (ValueError, IndexError):
            continue

    # Pattern 2: rect(x, y, width, height)
    pattern2 = r'rect\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)'
    for match in re.finditer(pattern2, text, re.IGNORECASE):
        try:
            x, y, w, h = float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))
            if 0 <= x <= 10000 and 0 <= y <= 10000 and w > 0 and h > 0:
                boxes.append({"x1": x, "y1": y, "x2": x + w, "y2": y + h, "label": match.group(0)})
        except (ValueError, IndexError):
            continue

    # Pattern 3: [x1, y1, x2, y2] (array notation)
    pattern3 = r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]'
    for match in re.finditer(pattern3, text):
        try:
            coords = [float(match.group(i)) for i in range(1, 5)]
            # Check if this looks like a bounding box (not just any 4-element array)
            if all(0 <= c <= 10000 for c in coords) and coords[2] > coords[0] and coords[3] > coords[1]:
                boxes.append({"x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3], "label": match.group(0)})
        except (ValueError, IndexError):
            continue

    return boxes

def extract_examples_from_system_prompt(system_prompt):
    """Extract examples from the system prompt text using the correct format."""
    examples = []
    lines = system_prompt.split('\n')
    
    current_example = None
    current_example_text = []
    in_example = False
    
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.startswith('--- Example') and line_stripped.endswith('---'):
            # Start of a new example
            if current_example is not None:
                # Save the previous example
                current_example['text'] = '\n'.join(current_example_text).strip()
                examples.append(current_example)
            
            # Start new example
            current_example = {
                'title': line_stripped,
                'text': '',
                'image_path': None
            }
            current_example_text = []
            in_example = True
        elif line_stripped.startswith('--- End Example') and line_stripped.endswith('---'):
            # End of example
            if current_example is not None:
                current_example['text'] = '\n'.join(current_example_text).strip()
                examples.append(current_example)
            current_example = None
            current_example_text = []
            in_example = False
        elif in_example and current_example is not None:
            # Collect example text, look for image paths
            current_example_text.append(line)
            # Check if this line contains an image path
            if 'url:' in line and ('.jpg' in line or '.png' in line or '.jpeg' in line or '.gif' in line):
                # Extract image path from URL
                url_start = line.find('url:') + 4
                url_part = line[url_start:].strip()
                # Remove any trailing characters like ]
                if url_part.endswith(']'):
                    url_part = url_part[:-1].strip()
                current_example['image_path'] = url_part
    
    # Handle the last example if no end marker
    if current_example is not None:
        current_example['text'] = '\n'.join(current_example_text).strip()
        examples.append(current_example)
    
    return examples

def get_tree_statistics(tree_data):
    """Recursively traverse the tree to compute overall statistics."""
    stats = {
        'num_sim_round': tree_data.get("num_simulation_rounds", 0),
        'total_nodes': 0,
        'max_depth': 0,
        'terminal_nodes': 0,
        'total_rollouts': 0,
        'successful_rollouts_0': 0,
        'successful_rollouts_05': 0,
        'successful_rollouts_075': 0,
        'successful_rollouts_1': 0,
        'max_visit_count': 0,
        'most_visited_node_text': 'N/A',
        'action_distribution': {},
        'best_reward': -1.0,
        'best_reward_node_count': 0,
    }
    
    if 'tree' not in tree_data:
        return stats

    root = tree_data['tree']

    def _recursive_stats(node, depth):
        if not node:
            return

        stats['total_nodes'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)

        if node.get('is_terminal'):
            stats['terminal_nodes'] += 1
            reward = node.get('value', 0.0)
            if reward > stats['best_reward']:
                stats['best_reward'] = reward
                stats['best_reward_node_count'] = 1
            elif reward == stats['best_reward'] and stats['best_reward'] != -1.0:
                stats['best_reward_node_count'] += 1

            if reward > 0:
                stats['successful_rollouts_0'] += 1
            if reward >= 0.5:
                stats['successful_rollouts_05'] += 1
            if reward >= 0.75:
                stats['successful_rollouts_075'] += 1
            if reward == 1:
                stats['successful_rollouts_1'] += 1
            
            stats['total_rollouts'] += 1
            # if node.get('rollouts'):
            #     rewards = [r.get('reward', -1.0) for r in node['rollouts']]
            #     if rewards:
            #         max_reward_in_node = max(rewards)
            #         if max_reward_in_node > stats['best_reward']:
            #             stats['best_reward'] = max_reward_in_node
            #             stats['best_reward_node_count'] = 1
            #         elif max_reward_in_node == stats['best_reward'] and stats['best_reward'] != -1.0:
            #             stats['best_reward_node_count'] += 1

        if node.get('visit_count', 0) > stats['max_visit_count'] and depth > 0:
            stats['max_visit_count'] = node['visit_count']
            stats['most_visited_node_text'] = node.get('thought_text', 'N/A')

        # if 'rollouts' in node:
        #     num_rollouts = len(node['rollouts'])
        #     stats['total_rollouts'] += num_rollouts
        #     for rollout in node['rollouts']:
        #         reward = rollout.get('reward', 0)
        #         if reward > 0:
        #             stats['successful_rollouts_0'] += 1
        #         if reward >= 0.5:
        #             stats['successful_rollouts_05'] += 1
        #         if reward >= 0.75:
        #             stats['successful_rollouts_075'] += 1
        #         if reward == 1:
        #             stats['successful_rollouts_1'] += 1
        #         if 'final_answer' in rollout:
        #             action = rollout['final_answer']
        #             stats['action_distribution'][action] = stats['action_distribution'].get(action, 0) + 1
        
        for child in node.get('children', []):
            _recursive_stats(child, depth + 1)

    _recursive_stats(root, 0)
    return stats


def build_graph_data(tree_data, max_depth_vis):
    """
    Recursively builds the node and edge lists for vis.js.
    Each rollout is now a distinct leaf node.
    """
    nodes = []
    edges = []
    interactive_data_map = {}
    node_id_counter = 0

    root = tree_data.get('tree')
    if not root:
        return {"nodes": [], "edges": [], "interactive_data": {}}
    
    T = tree_data.get("T")
    scene_pose_matrix = tree_data.get("scene_pose_matrix")
    ixt = tree_data.get("ixt")
    ext = tree_data.get("ext")
    H = tree_data.get("H")
    W = tree_data.get("W")

    def _recursive_build(node, parent_node, parent_id, depth):
        nonlocal node_id_counter

        print(node_id_counter)
        
        current_id = f"node_{node_id_counter}"
        node_id_counter += 1
        
        if max_depth_vis is not None and depth > max_depth_vis:
            return

        # 1. Create the main MCTS node (thought or action)
        thought = node.get('thought_text', 'No Thought')
        visit_count = node.get('visit_count', 0)
        value = node.get('value', 0.0)
        is_terminal = node.get('is_terminal', False)
        skill = node.get('skill', 0)

        # Determine node type for better visualization
        node_type = "Unknown"
        if parent_id is None:
            node_type = "Root"
        # elif thought.startswith('<think>'):
        #     node_type = "Thinking"
        # elif thought.startswith('click') or thought.startswith('scroll') or thought.startswith('go_back') or thought.startswith('stop'):
        #     node_type = "Action"
        # elif 'OBJECTIVE:' in thought:
        #     node_type = "Objective"
        # else:
        #     node_type = "Other"
        # elif skill == 1:
        #     node_type = "Grasp"
        # elif skill == 2:
        #     node_type = "Place"
        elif is_terminal:
            if node["value"] >= 1.0:
                node_type = "Success"
            else:
                node_type = "Failure"
            # node_type = "Leaf"
        else:
            node_type = "Other"

        if thought:
            display_thought = html.escape((thought[:200] + '...') if len(thought) > 200 else thought)
            if parent_id is not None:
                # label = f"[{node_type}] {skill_map[skill]}\nVisits: {visit_count} | Value: {value:.3f}"
                label = f"[{node_type}]\nVisits: {visit_count} | Value: {value:.3f}"
            else:
                label = f"[{node_type}]\nVisits: {visit_count} | Value: {value:.3f}"
        else:
            display_thought = ""
            if parent_id is not None:
                # label = f"[{node_type}]\nVisits: {skill_map[skill]} | Value: {value:.3f}"
                label = f"[{node_type}]\nVisits: {visit_count} | Value: {value:.3f}"
            else:
                label = f"[{node_type}]\nVisits: {visit_count} | Value: {value:.3f}"
        
        # Color coding by node type
        if node_type == "Failure":
            color = '#ff9999'  # Red for failure
        elif node_type == "Success":
            color = '#90EE90'
        elif parent_id is None:
            color = '#4CAF50'  # Green for root
        # elif node_type == "Thinking":
        #     color = '#87CEEB'  # Sky blue for thinking nodes
        # elif node_type == "Action":
        #     color = '#FFD700'  # Gold for action nodes
        else:
            color = '#97C2FC'  # Default blue
        
        # coords = _extract_coords(thought)
        coords = [node["point"]]

        # Extract video/gif path from node (support both field names)
        video_path = node.get('video_path') or node.get('gif_path')

        # Extract bounding boxes from thought text and any other relevant text fields
        bounding_boxes = []
        if thought:
            bounding_boxes.extend(_extract_bounding_boxes(thought))

        # Extract from node_system_prompt (useful in exhaustive mode)
        node_system_prompt = node.get('node_system_prompt', '')
        if node_system_prompt:
            bounding_boxes.extend(_extract_bounding_boxes(node_system_prompt))

        # Also extract from rollout outputs if available
        if node.get('rollouts'):
            for rollout in node['rollouts']:
                for field in ['node_text', 'node_status', 'progress_text', 'reflection']:
                    text = rollout.get(field, '')
                    if text:
                        bounding_boxes.extend(_extract_bounding_boxes(text))

        nodes.append({
            "id": current_id, "label": label, "shape": 'box', "color": color,
            "margin": 10, "font": {"color": "#333"}
        })

        if parent_id:
            edges.append({"from": parent_id, "to": current_id})

        # 2. Store prompt and coordinates for the main node
        # prompt_for_node = node.get('prompt_for_node')
        prompt_for_node = node.get('node_system_prompt')
        if not prompt_for_node:
            prompt_for_node = "This is the root node." if parent_id is None else "Prompt not found."
        
        # Extract retrieved examples from prompt_for_node using the correct format
        retrieved_examples = []
        seen_examples = set()  # Track unique examples to avoid duplicates
        
        # First, check if prompt_for_node contains examples or strategies
        if prompt_for_node and ("--- Example" in prompt_for_node or "--- Strategy" in prompt_for_node):
            if "--- Example" in prompt_for_node:
                parsed_examples = extract_examples_from_system_prompt(prompt_for_node)
                for example in parsed_examples:
                    unique_key = f"{example['title']}:::{example['text'][:100]}"
                    if unique_key not in seen_examples:
                        retrieved_examples.append({
                            'image_path': example.get('image_path'),
                            'encoded_image': encode_image_to_base64(example.get('image_path')) if example.get('image_path') else None,
                            'text': example['text'],
                            'title': example['title'],
                            'example_index': len(retrieved_examples),
                            'source': 'prompt_for_node'
                        })
                        seen_examples.add(unique_key)
            
            # Handle strategies in abstractions format
            if "--- Strategy" in prompt_for_node:
                strategy_sections = []
                parts = prompt_for_node.split("--- Strategy")
                for i, part in enumerate(parts[1:], 1):  # Skip first part before any strategy
                    if "--- End Strategy" in part:
                        strategy_content = f"--- Strategy{part.split('--- End Strategy')[0]}--- End Strategy {i} ---"
                        strategy_sections.append(strategy_content.strip())
                
                for j, strategy in enumerate(strategy_sections):
                    unique_key = f"strategy_{j}:::{strategy[:100]}"
                    if unique_key not in seen_examples:
                        retrieved_examples.append({
                            'image_path': None,
                            'encoded_image': None,
                            'text': strategy,
                            'title': f'Retrieved Strategy {len(retrieved_examples) + 1}',
                            'example_index': len(retrieved_examples),
                            'source': 'prompt_for_node_strategy'
                        })
                        seen_examples.add(unique_key)
        
        # Also check rollouts for any additional examples (fallback)
        if node.get('rollouts'):
            for rollout_idx, rollout in enumerate(node['rollouts']):
                node_text = rollout.get('node_text', '')
                node_status = rollout.get('node_status', '')
                progress_text = rollout.get('progress_text', '')
                reflection = rollout.get('reflection', '')
                percentage = rollout.get('percentage', 0)
        #         # Check ephemeral_retrieved_paths and ephemeral_texts
        #         ephemeral_texts = rollout.get('ephemeral_texts', [])
        #         ephemeral_paths = rollout.get('ephemeral_retrieved_paths', [])
                
        #         if ephemeral_texts:
        #             for i, text_content in enumerate(ephemeral_texts):
        #                 # Check if this text contains retrieved examples/strategies
        #                 if text_content and ('--- Example' in text_content or '--- Strategy' in text_content):
        #                     path = ephemeral_paths[i] if i < len(ephemeral_paths) else None
        #                     # Handle both cases: with path (regular) and without path (abstractions)
        #                     if path and path.strip():
        #                         unique_key = f"{path}:::{text_content[:100]}"
        #                         title = f'Retrieved Example {len(retrieved_examples) + 1}'
        #                     else:
        #                         # Abstractions format or text-only examples
        #                         unique_key = f"text_only_{i}:::{text_content[:100]}"
        #                         title = f'Retrieved Strategy {len(retrieved_examples) + 1}' if '--- Strategy' in text_content else f'Retrieved Example {len(retrieved_examples) + 1}'
                            
        #                     if unique_key not in seen_examples:
        #                         retrieved_examples.append({
        #                             'image_path': path,
        #                             'encoded_image': encode_image_to_base64(path) if path else None,
        #                             'text': text_content,
        #                             'title': title,
        #                             'example_index': i,
        #                             'rollout_index': rollout_idx,
        #                             'source': 'ephemeral_paths'
        #                         })
        #                         seen_examples.add(unique_key)
        
        interactive_data_map[current_id] = {
            "type": "node_prompt",
            "data": prompt_for_node,
            "coords": coords,
            "node_type": node_type,
            "node_data": node,
            "rollout_data": node.get('rollouts', []),
            "retrieved_examples": retrieved_examples,
            "video_path": video_path,
            "bounding_boxes": bounding_boxes
        }
        
        # 3. Create a SEPARATE leaf node for EACH rollout
        # if node.get('rollouts'):
        #     for i, rollout in enumerate(node['rollouts']):
        #         rollout_id = f"{current_id}_rollout_{i}"
        #         skill = node.get("skill", 0)
        #         reward = rollout.get('reward', 0)
        #         final_answer = rollout.get('final_answer', 'N/A')
                
        #         # rollout_label = f"Rollout #{i+1}\nAction: {html.escape(final_answer)}\nReward: {reward:.3f}"
        #         rollout_label = f"Rollout #{i+1}\nReward: {reward:.3f}"
                
        #         # Color rollout nodes based on reward values
        #         if reward <= 0:
        #             rollout_color = '#E0E0E0'  # Gray for zero/negative reward
        #         elif reward < 0.5:
        #             rollout_color = '#FFCCCB'  # Light red for low positive reward
        #         elif reward < 0.75:
        #             rollout_color = '#FFE135'  # Yellow for medium reward
        #         elif reward < 1.0:
        #             rollout_color = '#90EE90'  # Light green for high reward
        #         else:
        #             rollout_color = '#00FF00'  # Bright green for perfect reward

        #         nodes.append({
        #             "id": rollout_id,
        #             "label": rollout_label,
        #             "shape": 'ellipse',
        #             "color": rollout_color,
        #             "font": {"size": 12}
        #         })
        #         edges.append({"from": current_id, "to": rollout_id, "dashes": True, "color": "#888"})
                
        #         # rollout_coords = _extract_coords(final_answer)
        #         interactive_data_map[rollout_id] = {
        #             "type": "rollout_prompts",
        #             "data": rollout.get('rollout_prompts', []),
        #             # "coords": rollout_coords,
        #             "rollout_data": rollout,
        #             "image": final_answer
        #         }

        # 4. Add extra visualization for node
        if parent_node is not None:
            vis_mask = False
            vis_goal_pose = False
            vis_goal_pose_0 = False
            # parent_node = node["parent"]
            vis_input_image = deepcopy(parent_node["image"])
            vis_elems = {
                "point": node.get("point"),
                "box": node.get("box"),
            }
            if "mask" in node and node["mask"] is not None:
                vis_mask = True
                vis_elems.update({"mask": node["mask"]})
            # if "grasp" in node and node["grasp"] is not None:
            #     vis_goal_pose = True
            #     vis_elems.update({"goal_pose": node["grasp"]})
            # if "goal_pose" in node and node["goal_pose"] is not None:
            #     vis_goal_pose = True
            #     vis_elems.update({"goal_pose": node["goal_pose"]})
            if "goal_pose_0" in node and node["goal_pose_0"] is not None:
                vis_goal_pose_0 = True
                vis_elems.update({"goal_pose_0": node["goal_pose_0"]})

            if HAS_VIS_UTILS:
                vis_result = vis_all(
                    vis_input_image,
                    vis_elems,
                    T=T,
                    scene_pose_matrix=scene_pose_matrix,
                    ixt=ixt,
                    ext=ext,
                    H=H,
                    W=W,
                    vis_point="point" in vis_elems and vis_elems["point"] is not None,
                    vis_box="box" in vis_elems and vis_elems["box"] is not None,
                    vis_mask=vis_mask,
                    vis_gripper=vis_goal_pose,
                    vis_gripper_0=vis_goal_pose_0
                )
                node["vis_result"] = vis_result

            vis_goal_pose_next = False
            vis_goal_pose_0_next = False
            if "image" not in node or node["image"] is None:
                return
            vis_input_image_next = deepcopy(node["image"])
            vis_elems_next = dict()
            if "grasp" in node and node["grasp"] is not None:
                vis_goal_pose_next = True
                vis_elems_next.update({"goal_pose": node["grasp"]})
            if "goal_pose" in node and node["goal_pose"] is not None:
                vis_goal_pose_next = True
                vis_elems_next.update({"goal_pose": node["goal_pose"]})
            # if "goal_pose_0" in node and node["goal_pose_0"] is not None:
            #     vis_goal_pose_0_next = True
            #     vis_elems_next.update({"goal_pose_0": node["goal_pose_0"]})

            if HAS_VIS_UTILS:
                vis_result_next = vis_all(
                    vis_input_image_next,
                    vis_elems_next,
                    T=T,
                    scene_pose_matrix=scene_pose_matrix,
                    ixt=ixt,
                    ext=ext,
                    H=H,
                    W=W,
                    vis_point=False,
                    vis_box=False,
                    vis_mask=False,
                    vis_gripper=vis_goal_pose_next,
                    vis_gripper_0=vis_goal_pose_0_next
                )
                node["vis_result_next"] = vis_result_next
        else:
            node["vis_result"] = node["image"]

        # 5. Recurse for children
        for child in node.get('children', []):
            _recursive_build(child, node, current_id, depth + 1)

    _recursive_build(root, None, None, 0)
    
    return {"nodes": nodes, "edges": edges, "interactive_data": interactive_data_map}


def encode_image_to_base64(image_path):
    """Convert an image file to base64 data URL with comprehensive error handling"""
    if not image_path or not isinstance(image_path, str):
        return None
    
    # Check if it's a URL - don't try to encode URLs
    if image_path.startswith(('http://', 'https://')):
        print(f"Skipping URL image (will display directly): {image_path}")
        return None  # Return None so it uses the URL directly
        
    if not os.path.exists(image_path):
        print(f"Warning: Local image file does not exist: {image_path}")
        return None
    
    # Get MIME type
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image/'):
        # Try to determine from file extension
        _, ext = os.path.splitext(image_path.lower())
        if ext in ['.jpg', '.jpeg']:
            mime_type = 'image/jpeg'
        elif ext in ['.png']:
            mime_type = 'image/png'
        elif ext in ['.gif']:
            mime_type = 'image/gif'
        elif ext in ['.webp']:
            mime_type = 'image/webp'
        else:
            mime_type = 'image/png'  # Default fallback
    
    try:
        file_size = os.path.getsize(image_path)
        if file_size > 10 * 1024 * 1024:  # 10MB limit
            print(f"Warning: Image file too large ({file_size / 1024 / 1024:.1f}MB): {image_path}")
            return None
            
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            data_url = f"data:{mime_type};base64,{encoded_string}"
            print(f"Successfully encoded local image: {image_path} ({file_size / 1024:.1f}KB -> {len(data_url)} chars)")
            return data_url
    except Exception as e:
        print(f"Error: Could not encode image {image_path}: {e}")
        return None
    

def generate_html_file(tree_data, graph_data, stats, output_dir, output_filename, debug_mode=False, images_config=None, exhaustive_mode=False):
    """Generates the final self-contained HTML file."""

    nodes_json = json.dumps(graph_data['nodes'])
    edges_json = json.dumps(graph_data['edges'])
    interactive_data_json = json.dumps(graph_data['interactive_data'])
    debug_mode_json = json.dumps(debug_mode)
    exhaustive_mode_json = json.dumps(exhaustive_mode)

    # Calculate display scale from images config
    display_scale = 1.0
    if images_config and images_config.get("scale"):
        display_scale = float(images_config["scale"])
    elif images_config and images_config.get("resolution"):
        # For resolution, we don't scale the display since resolution is handled differently
        display_scale = 1.0
    
    display_scale_json = json.dumps(display_scale)

    # Format stats for display
    success_rate_0 = (stats['successful_rollouts_0'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    success_rate_05 = (stats['successful_rollouts_05'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    success_rate_075 = (stats['successful_rollouts_075'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    success_rate_1 = (stats['successful_rollouts_1'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    stats_html = (
        f"--- Tree Statistics ---\n"
        f"Number of Simulation Rounds: {stats['num_sim_round']}\n"
        f"Total Nodes: {stats['total_nodes']}\n"
        f"Max Depth: {stats['max_depth']}\n"
        f"Terminal Nodes: {stats['terminal_nodes']}\n"
        f"Total Search Time: {tree_data.get('global_search_time', 0) if isinstance(tree_data.get('global_search_time'), (int, float)) else 'N/A'}{'s' if isinstance(tree_data.get('global_search_time'), (int, float)) else ''}\n\n"
        f"--- Rollout Statistics ---\n"
        f"Total Rollouts: {stats['total_rollouts']}\n"
        f"Successful Rollouts (R > 0): {stats['successful_rollouts_0']} ({success_rate_0:.2f}%)\n"
        f"Successful Rollouts (R >= 0.5): {success_rate_05:.2f}%)\n"
        f"Successful Rollouts (R >= 0.75): {success_rate_075:.2f}%)\n"
        f"Perfect Rollouts (R == 1): {stats['successful_rollouts_1']} ({success_rate_1:.2f}%)\n\n"
        f"--- Best Reward in Terminal Node ---\n"
        f"Highest Reward: {stats['best_reward'] if stats['best_reward'] != -1.0 else 'N/A'}\n"
        f"Number of Nodes with this Reward: {stats['best_reward_node_count']}\n\n"
        f"--- Most Visited Node ---\n"
        f"Visit Count: {stats['max_visit_count']}\n"
        #f"Thought: {html.escape(stats['most_visited_node_text'])}\n\n"
        f"--- Action Distribution in Rollouts ---\n"
        f"{html.escape(json.dumps(stats['action_distribution'], indent=2))}"
    ).replace('\n', '<br>')

    question = html.escape(tree_data.get('question', 'N/A')).replace('\n', '<br>')
    # true_answer = html.escape(tree_data.get('true_answer', 'N/A'))
    # image_path = tree_data.get('image', '')
    # escaped_image_path = html.escape(image_path)
    
    # Convert image to base64 for portability
    # base64_image = encode_image_to_base64(image_path)
    mime_type = "image/jpeg"
    base64_image = f"data:{mime_type};base64,{tree_data.get('image', '')}"
    # if base64_image:
    print(f"Image successfully encoded to base64 ({len(base64_image)} characters)")
    display_image_src = base64_image
    # else:
    #    print(f"Warning: Could not encode image, using original path: {image_path}")
    #    display_image_src = image_path

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enhanced MCTS Visualizer v29</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 0; background-color: #f0f2f5; }}
        .main-container {{ display: flex; flex-direction: column; height: 100vh; }}
        .header {{ background-color: #ffffff; padding: 15px 25px; border-bottom: 1px solid #ddd; box-shadow: 0 2px 4px rgba(0,0,0,0.1); z-index: 10; }}
        .header h1 {{ margin: 0; color: #1c1e21; font-size: 24px; }}
        .version-badge {{ background-color: #007bff; color: white; padding: 2px 8px; border-radius: 12px; font-size: 12px; margin-left: 10px; }}
        .content-area {{ display: flex; flex: 1; overflow: hidden; }}
        
        .panel-container {{ display: flex; position: relative; transition: min-width 0.3s, max-width 0.3s; }}
        .panel-container.collapsed {{ min-width: 25px !important; max-width: 25px !important; }}

        .panel {{ flex-grow: 1; background-color: #f8f9fa; overflow: hidden; display: flex; flex-direction: column; width: 100%; }}
        .panel-container.collapsed .panel {{ display: none; }}

        .panel-toggle {{
            position: absolute; top: 50%; right: 0; transform: translateY(-50%);
            width: 25px; height: 60px; background-color: #007bff; color: white;
            border: none; cursor: pointer; border-radius: 5px 0 0 5px;
            display: flex; align-items: center; justify-content: center; z-index: 20;
        }}
        .panel-toggle:hover {{ background-color: #0056b3; }}
        #left-panel-container .panel-toggle {{ left: 0; border-radius: 0 5px 5px 0; }}

        /* Left panel with tabs */
        #left-panel-container {{ width: 45%; max-width: 900px; min-width: 500px; border-right: 1px solid #ddd; display: flex; flex-direction: column; }}
        #left-panel-container.collapsed {{ width: 25px; min-width: 25px; max-width: 25px; border-right: none; }}
        
        .tabs-container {{ background-color: #f8f9fa; border-bottom: 1px solid #ddd; }}
        .tabs {{ display: flex; overflow-x: auto; }}
        .tab {{ background-color: #e9ecef; border: 1px solid #ddd; border-bottom: none; padding: 8px 16px; cursor: pointer; white-space: nowrap; transition: background-color 0.2s; }}
        .tab:hover {{ background-color: #dee2e6; }}
        .tab.active {{ background-color: #ffffff; border-bottom: 1px solid #ffffff; }}
        
        .tab-content {{ flex: 1; overflow: hidden; display: none; }}
        .tab-content.active {{ display: flex; flex-direction: column; }}
        
        /* Tab 1: Image with highlighting */
        #image-container {{ flex-grow: 1; position: relative; overflow: hidden; background-color: #e0e0e0; cursor: grab; }}
        #image-transform-wrapper {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; transform-origin: 0 0; }}
        #task-image, #highlight-canvas {{ position: absolute; top: 0; left: 0; }}
        #highlight-canvas {{ pointer-events: none; }}
        .drop-zone {{ border: 2px dashed #ccc; border-radius: 8px; padding: 40px; text-align: center; color: #888; width: 80%; max-width: 400px; height: 200px; display: flex; align-items: center; justify-content: center; flex-direction: column; cursor: pointer; margin: auto; position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(255,255,255,0.8); }}
        .drop-zone.hidden {{ display: none; }}

        /* Tab 2:  Video*/
        .video-container {{ padding: 20px; overflow-y: auto; flex: 1; }}
        .video-display {{ background-color: #f1f1f1; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 13px; }}
        
        /* Tab 3: Prompt */
        .prompt-container {{ padding: 20px; overflow-y: auto; flex: 1; }}
        .prompt-toggle {{ margin-bottom: 15px; }}
        .prompt-display {{ background-color: #f1f1f1; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 13px; max-height: 70vh; overflow-y: auto; }}
        
        /* Tab 4: Retrieved examples */
        .examples-container {{ padding: 20px; overflow-y: auto; flex: 1; }}
        .example-item {{ margin-bottom: 20px; border: 1px solid #ddd; border-radius: 5px; overflow: hidden; }}
        .example-header {{ background-color: #f8f9fa; padding: 10px 15px; border-bottom: 1px solid #ddd; font-weight: bold; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px; }}
        .example-content {{ padding: 15px; }}
        .example-image {{ max-width: 100%; height: auto; margin-top: 10px; border: 1px solid #ddd; }}
        .example-buttons {{ margin-top: 10px; display: flex; flex-wrap: wrap; gap: 5px; }}
        .example-btn {{ background-color: #6c757d; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer; font-size: 12px; }}
        .example-btn:hover {{ background-color: #5a6268; }}
        .example-image-container {{ margin-top: 10px; border: 1px solid #ddd; border-radius: 5px; padding: 10px; background-color: #f9f9f9; }}
        .image-display-section {{ display: flex; flex-direction: column; gap: 10px; }}
        .example-text {{ margin-top: 10px; }}
        .url-image-fallback {{ background-color: #e9ecef; border: 2px dashed #6c757d; padding: 20px; text-align: center; border-radius: 5px; margin-top: 10px; }}
        .url-image-fallback a {{ color: #007bff; text-decoration: none; font-weight: bold; }}
        .url-image-fallback a:hover {{ text-decoration: underline; }}
        
        /* Debug coordinate display */
        #debug-coords {{ 
            position: fixed; 
            background-color: rgba(0, 0, 0, 0.8); 
            color: white; 
            padding: 5px 10px; 
            border-radius: 5px; 
            font-family: monospace; 
            font-size: 12px; 
            pointer-events: none; 
            z-index: 1000; 
            display: none;
        }}
        
        /* Tab 5: Node output */
        .output-container {{ padding: 20px; overflow-y: auto; flex: 1; }}
        .output-display {{ background-color: #f1f1f1; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 13px; }}
        
        /* Tab 6: Statistics */
        .stats-container {{ padding: 20px; overflow-y: auto; flex: 1; }}
        .stats-section {{ margin-bottom: 20px; }}
        .stats-title {{ font-weight: bold; margin-bottom: 10px; color: #007bff; }}
        .stats-content {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 12px; white-space: pre-wrap; }}
        
        /* Tab 7: Legend */
        .legend-container {{ padding: 20px; overflow-y: auto; flex: 1; }}
        .legend-section {{ margin-bottom: 25px; }}
        .legend-title {{ font-weight: bold; margin-bottom: 15px; color: #007bff; font-size: 16px; }}
        .legend-items {{ margin-left: 10px; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 10px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 3px; margin-right: 12px; border: 1px solid #ccc; }}
        .legend-color.rollout-color {{ border-radius: 50%; }}
        .legend-edge {{ width: 40px; height: 3px; margin-right: 12px; }}
        .legend-edge.dashed {{ border-top: 3px dashed #888; }}
        .legend-edge.solid {{ background-color: #333; }}
        .legend-content {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
        .legend-content p {{ margin: 8px 0; }}
        
        /* Right panel for tree */
        #graph-container {{ flex: 1; background-color: #ffffff; position: relative; }}
        #mcts-graph {{ height: 100%; width: 100%; }}
        
        /* Extended view styles */
        .extended-view {{ }}
        .extended-view .content-area {{ flex-direction: row !important; }}
        .extended-view #left-panel-container {{ 
            width: 50% !important; 
            max-width: none !important; 
            min-width: 50% !important; 
            flex-direction: column !important; 
        }}
        .extended-view #left-panel {{ flex-direction: column !important; }}
        .extended-view .tabs-container {{ display: none !important; }}
        .extended-view .tab-content {{ display: none !important; }}
        .extended-view #image-tab {{ display: flex !important; flex: 1; }}
        .extended-view #graph-container {{ 
            width: 50% !important; 
            display: flex !important; 
            flex-direction: column !important; 
        }}
        .extended-view #extended-tab-container {{ 
            flex: 1; 
            background-color: #f8f9fa; 
            border-bottom: 1px solid #ddd; 
            display: flex !important; 
            flex-direction: column; 
        }}
        .extended-view #extended-tab-header {{ 
            background-color: #f8f9fa; 
            border-bottom: 1px solid #ddd; 
            padding: 10px 15px; 
            display: flex; 
            justify-content: space-between; 
            align-items: center; 
        }}
        .extended-view #extended-tab-content {{ 
            flex: 1; 
            overflow: auto; 
        }}
        .extended-view #extended-graph-container {{ 
            flex: 1; 
            min-height: 300px; 
        }}
        
        /* Extended view components */
        #extended-tab-container {{ display: none; }}
        #extended-graph-container {{ display: none; }}
        .restore-view-btn {{ 
            background-color: #28a745; 
            color: white; 
            border: none; 
            padding: 8px 15px; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 14px; 
        }}
        .restore-view-btn:hover {{ 
            background-color: #218838; 
        }}
        .extended-tab-title {{ 
            font-weight: bold; 
            color: #007bff; 
        }}
        
        .control-button {{ background-color: #007bff; color: white; border: none; padding: 10px 15px; border-radius: 5px; cursor: pointer; font-size: 14px; margin-top: 10px; width: 100%; transition: background-color 0.2s; }}
        .control-button:hover {{ background-color: #0056b3; }}
        
        .modal {{ display: none; position: fixed; z-index: 1000; left: 0; top: 0; width: 100%; height: 100%; overflow: auto; background-color: rgba(0,0,0,0.6); align-items: center; justify-content: center; }}
        .modal-content {{ background-color: #fefefe; margin: auto; padding: 0; border: 1px solid #888; width: 85%; max-width: 1200px; border-radius: 8px; position: relative; max-height: 90vh; display: flex; flex-direction: column; box-shadow: 0 5px 15px rgba(0,0,0,0.3); }}
        .modal-header {{ padding: 15px 25px; background-color: #007bff; color: white; border-bottom: 1px solid #dee2e6; border-top-left-radius: 8px; border-top-right-radius: 8px; }}
        .modal-header h2 {{ margin: 0; font-size: 20px; }}
        .modal-body {{ padding: 25px; overflow-y: auto; }}
        .close-btn {{ color: #fff; float: right; font-size: 28px; font-weight: bold; cursor: pointer; line-height: 1; }}
        .close-btn:hover, .close-btn:focus {{ color: #ccc; text-decoration: none; }}
        pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #f1f1f1; padding: 15px; border-radius: 5px; font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 13px; max-height: 60vh; overflow-y: auto; }}
        
        /* Rollout modal specific styling */
        .turn-selector {{ margin-bottom: 15px; }}
        .turn-btn {{ background-color: #17a2b8; color: white; border: none; padding: 8px 15px; border-radius: 4px; margin-right: 8px; cursor: pointer; }}
        .turn-btn:hover {{ background-color: #138496; }}
        .turn-btn.active {{ background-color: #007bff; }}
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header">
            <h1>Enhanced MCTS Search Visualization<span class="version-badge">v29</span></h1>
        </div>
        <div class="content-area">
            <div id="left-panel-container" class="panel-container">
                <button id="left-panel-toggle" class="panel-toggle">&gt;</button>
                <div id="left-panel" class="panel">
                    <div class="tabs-container">
                        <div class="tabs">
                            <div class="tab active" data-tab="image-tab">Question Image</div>
                            <div class="tab active" data-tab="image-tab">Video</div>
                            <div class="tab" data-tab="prompt-tab">Input Prompt</div>
                            <div class="tab" data-tab="examples-tab">Retrieved Examples</div>
                            <div class="tab" data-tab="output-tab">Node Output</div>
                            <div class="tab" data-tab="stats-tab">Tree Statistics</div>
                            <div class="tab" data-tab="legend-tab">Color Legend</div>
                        </div>
                    </div>
                    
                    <!-- Tab 1: Question Image -->
                    <div id="image-tab" class="tab-content active">
                        <div id="image-container">
                            <div id="image-transform-wrapper">
                                <img id="task-image" src="{display_image_src}" alt="Task Image" style="display: none;">
                                <canvas id="highlight-canvas"></canvas>
                            </div>
                            <div id="drop-zone" class="drop-zone">
                                <p>Image not found or failed to load.</p>
                                <p>Drag & drop an image file here, or click to select one.</p>
                                <input type="file" id="file-input" accept="image/*" style="display: none;">
                            </div>
                        </div>
                    </div>

                    <!-- New Video tab pane -->
                    <!-- Tab 2: Video -->
                    <div id="video-tab" class="tab-content">
                        <div id="video-container" style="padding:16px; overflow:auto;">
                            <div id="video-box" style="position: relative; max-width:100%; display: inline-block;">
                                <!-- Video or GIF element will be injected here -->
                                <video id="video-player" style="display: none; max-width: 100%; height: auto;" controls loop></video>
                                <img id="gif-player" style="display: none; max-width: 100%; height: auto;" alt="GIF"/>
                                <!-- Canvas overlay for bounding boxes -->
                                <canvas id="video-overlay-canvas" style="position: absolute; top: 0; left: 0; pointer-events: none;"></canvas>
                                <div id="video-display" class="video-display">Click on a node to load its video/GIF...</div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Tab 3: Input Prompt -->
                    <div id="prompt-tab" class="tab-content">
                        <div class="prompt-container">
                            <div class="prompt-toggle">
                                <button id="show-full-prompt" class="control-button">Show Full Prompt</button>
                            </div>
                            <div id="prompt-display" class="prompt-display">Click on a node to see its prompt...</div>
                        </div>
                    </div>
                    
                    <!-- Tab 4: Retrieved Examples -->
                    <div id="examples-tab" class="tab-content">
                        <div class="examples-container">
                            <div id="examples-display">Click on a node to see retrieved examples...</div>
                        </div>
                    </div>
                    
                    <!-- Tab 5: Node Output -->
                    <div id="output-tab" class="tab-content">
                        <div class="output-container">
                            <div id="output-display" class="output-display">Click on a node to see its output...</div>
                        </div>
                    </div>
                    
                    <!-- Tab 6: Tree Statistics -->
                    <div id="stats-tab" class="tab-content">
                        <div class="stats-container">
                            <div class="stats-section">
                                <div class="stats-title">Overall Tree Statistics</div>
                                <div class="stats-content">{stats_html}</div>
                            </div>
                            <div class="stats-section">
                                <div class="stats-title">Question Details</div>
                                <div class="stats-content">
                                    <strong>Question:</strong><br>{question}<br><br>
                                    <!-- <strong>True Answer:</strong><br>true_answer<br><br> -->
                                    <!-- <strong>Image Path:</strong><br>escaped_image_path -->
                                    -->
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Tab 7: Color Legend -->
                    <div id="legend-tab" class="tab-content">
                        <div class="legend-container">
                            <div class="legend-section">
                                <div class="legend-title">Node Types & Colors</div>
                                <div class="legend-items">
                                    <div class="legend-item">
                                        <div class="legend-color" style="background-color: #4CAF50;"></div>
                                        <span>Root Node - Starting point of the search tree</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color" style="background-color: #87CEEB;"></div>
                                        <span>Thinking Node - Internal reasoning step</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color" style="background-color: #FFD700;"></div>
                                        <span>Action Node - Executable action (click, scroll, etc.)</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color" style="background-color: #97C2FC;"></div>
                                        <span>Other Node - Miscellaneous node types</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color" style="background-color: #ff9999;"></div>
                                        <span>Terminal Node - End of search path</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="legend-section">
                                <div class="legend-title">Rollout Reward Colors</div>
                                <div class="legend-items">
                                    <div class="legend-item">
                                        <div class="legend-color rollout-color" style="background-color: #E0E0E0;"></div>
                                        <span>No Reward (≤ 0) - Failed or neutral outcome</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color rollout-color" style="background-color: #FFCCCB;"></div>
                                        <span>Low Reward (0 < r < 0.5) - Partial success</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color rollout-color" style="background-color: #FFE135;"></div>
                                        <span>Medium Reward (0.5 ≤ r < 0.75) - Good progress</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color rollout-color" style="background-color: #90EE90;"></div>
                                        <span>High Reward (0.75 ≤ r < 1.0) - Near success</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-color rollout-color" style="background-color: #00FF00;"></div>
                                        <span>Perfect Reward (= 1.0) - Complete success</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="legend-section">
                                <div class="legend-title">Visual Indicators</div>
                                <div class="legend-items">
                                    <div class="legend-item">
                                        <div class="legend-color" style="background-color: #fff; border: 4px solid #FF6B6B;"></div>
                                        <span>Selected Node - Red border indicates current selection</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-edge dashed"></div>
                                        <span>Rollout Edge - Dashed line connects nodes to rollouts</span>
                                    </div>
                                    <div class="legend-item">
                                        <div class="legend-edge solid"></div>
                                        <span>Tree Edge - Solid line shows parent-child relationships</span>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="legend-section">
                                <div class="legend-title">How to Use</div>
                                <div class="legend-content">
                                    <p><strong>Click</strong> any node to view its details in other tabs</p>
                                    <p><strong>Hover</strong> over nodes to see quick information</p>
                                    <p><strong>Drag</strong> to pan the graph view</p>
                                    <p><strong>Scroll</strong> to zoom in/out on the graph</p>
                                    <p><strong>Switch tabs</strong> to view different aspects of the selected node</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="graph-container">
                <div id="extended-tab-container">
                    <div id="extended-tab-header">
                        <div id="extended-tab-title" class="extended-tab-title">Tab Content</div>
                        <button id="restore-view-btn" class="restore-view-btn">Restore View</button>
                    </div>
                    <div id="extended-tab-content"></div>
                </div>
                <div id="extended-graph-container">
                    <div id="extended-mcts-graph" style="width: 100%; height: 100%;"></div>
                </div>
                <div id="mcts-graph"></div>
            </div>
        </div>
    </div>

    <!-- Debug coordinates display -->
    <div id="debug-coords"></div>

    <!-- Rollout Modal -->
    <div id="rollout-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <span class="close-btn" onclick="closeModal('rollout-modal')">&times;</span>
                <h2 id="rollout-modal-title">Rollout Prompts</h2>
            </div>
            <div class="modal-body">
                <div class="turn-selector" id="turn-selector"></div>
                <pre id="rollout-prompt-text"></pre>
            </div>
        </div>
    </div>

    <!-- Node Prompt Modal -->
    <div id="prompt-modal" class="modal">
        <div class="modal-content">
            <div class="modal-header">
                <span class="close-btn" onclick="closeModal('prompt-modal')">&times;</span>
                <h2 id="prompt-modal-title">Node Prompt</h2>
            </div>
            <div class="modal-body">
                <pre id="prompt-text"></pre>
            </div>
        </div>
    </div>

    <script type="text/javascript">
        // --- Data ---
        const nodesData = {nodes_json};
        const edgesData = {edges_json};
        const interactiveDataMap = {interactive_data_json};
        const debugMode = {debug_mode_json};
        const exhaustiveMode = {exhaustive_mode_json};
        const displayScale = {display_scale_json};

        // --- State Variables ---
        let currentSelectedNode = null;
        let currentSelectedNodeId = null;
        let showFullPrompt = false;
        let currentRolloutData = null;
        let currentTurn = 0;
        let isExtendedView = false;
        let currentExtendedTab = null;

        // --- DOM Elements ---
        const leftPanelContainer = document.getElementById('left-panel-container');
        const leftPanelToggle = document.getElementById('left-panel-toggle');
        const imageContainer = document.getElementById('image-container');
        const transformWrapper = document.getElementById('image-transform-wrapper');
        const taskImage = document.getElementById('task-image');
        const highlightCanvas = document.getElementById('highlight-canvas');
        const ctx = highlightCanvas.getContext('2d');
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const debugCoordsDiv = document.getElementById('debug-coords');
        
        // Tab elements
        const tabs = document.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');
        
        // Content display elements
        const promptDisplay = document.getElementById('prompt-display');
        const examplesDisplay = document.getElementById('examples-display');
        const outputDisplay = document.getElementById('output-display');
        const showFullPromptBtn = document.getElementById('show-full-prompt');
        
        // Extended view elements
        const mainContainer = document.querySelector('.main-container');
        const graphContainer = document.getElementById('graph-container');
        const extendedTabContainer = document.getElementById('extended-tab-container');
        const extendedTabContent = document.getElementById('extended-tab-content');
        const extendedTabTitle = document.getElementById('extended-tab-title');
        const extendedGraphContainer = document.getElementById('extended-graph-container');
        const restoreViewBtn = document.getElementById('restore-view-btn');

        // --- Image Zoom & Pan State ---
        let scale = 1, panX = 0, panY = 0;
        let isPanning = false, startPan = {{x: 0, y: 0}};

        // --- vis.js Network ---
        const nodesDataSet = new vis.DataSet(nodesData);
        const edgesDataSet = new vis.DataSet(edgesData);
        const networkOptions = {{
            layout: {{ hierarchical: {{ direction: "UD", sortMethod: "directed", levelSeparation: 150, nodeSpacing: 220 }} }},
            interaction: {{ dragNodes: true, dragView: true, hover: false, zoomView: true }},
            physics: {{ enabled: false }},
        }};
        
        const network = new vis.Network(
            document.getElementById('mcts-graph'),
            {{ nodes: nodesDataSet, edges: edgesDataSet }},
            networkOptions
        );
        
        let extendedNetwork = null;

        // --- Functions ---
        function highlightSelectedNode(nodeId, targetNetwork) {{
            targetNetwork = targetNetwork || network;
            
            // Reset all nodes to their original colors
            nodesDataSet.forEach(node => {{
                const originalNode = nodesData.find(n => n.id === node.id);
                if (originalNode) {{
                    nodesDataSet.update({{
                        id: node.id,
                        color: originalNode.color,
                        borderWidth: 1,
                        chosen: false
                    }});
                }}
            }});
            
            // Highlight the selected node
            if (nodeId) {{
                const selectedNode = nodesDataSet.get(nodeId);
                if (selectedNode) {{
                    nodesDataSet.update({{
                        id: nodeId,
                        borderWidth: 4,
                        color: {{
                            border: '#FF6B6B',
                            background: selectedNode.color,
                            highlight: {{
                                border: '#FF6B6B',
                                background: selectedNode.color
                            }}
                        }},
                        chosen: true
                    }});
                }}
            }}
        }}

        function switchTab(targetTab) {{
            tabs.forEach(tab => tab.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            
            document.querySelector(`[data-tab="${{targetTab}}"]`).classList.add('active');
            document.getElementById(targetTab).classList.add('active');
        }}

        function updateImageTransform() {{
            transformWrapper.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{scale}})`;
        }}

        function fitImageToContainer() {{
            if (!taskImage.complete || taskImage.naturalWidth === 0) return;
            
            // Calculate the effective image dimensions accounting for display scale
            const effectiveWidth = taskImage.naturalWidth * displayScale;
            const effectiveHeight = taskImage.naturalHeight * displayScale;
            
            const containerRatio = imageContainer.clientWidth / imageContainer.clientHeight;
            const imageRatio = effectiveWidth / effectiveHeight;
            
            let newScale;
            if (containerRatio > imageRatio) {{
                newScale = imageContainer.clientHeight / effectiveHeight;
            }} else {{
                newScale = imageContainer.clientWidth / effectiveWidth;
            }}
            scale = newScale;
            panX = (imageContainer.clientWidth - effectiveWidth * scale) / 2;
            panY = (imageContainer.clientHeight - effectiveHeight * scale) / 2;
            
            // Set wrapper dimensions to the effective (scaled) size
            transformWrapper.style.width = effectiveWidth + 'px';
            transformWrapper.style.height = effectiveHeight + 'px';
            
            // Set canvas dimensions to match the effective size for coordinate alignment
            highlightCanvas.width = effectiveWidth;
            highlightCanvas.height = effectiveHeight;
            
            // Apply display scale to the image element itself
            taskImage.style.transform = `scale(${{displayScale}})`;
            taskImage.style.transformOrigin = '0 0';
            
            updateImageTransform();
            taskImage.style.display = 'block';
            dropZone.classList.add('hidden');
            
            console.log(`Applied display scale: ${{displayScale}}, effective size: ${{effectiveWidth}}x${{effectiveHeight}}, natural size: ${{taskImage.naturalWidth}}x${{taskImage.naturalHeight}}`);
        }}

        function drawHighlight(coordsList) {{
            if (!coordsList || coordsList.length === 0 || !taskImage.complete || taskImage.naturalWidth === 0) {{
                ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);
                return;
            }}
            
            const radius = 50;

            ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(0, 0, highlightCanvas.width, highlightCanvas.height);
            
            ctx.globalCompositeOperation = 'destination-out';
            
            coordsList.forEach(coords => {{
                const [x, y] = coords;
                ctx.beginPath();
                ctx.arc(x, y, radius, 0, 2 * Math.PI);
                ctx.fill();
            }});

            ctx.globalCompositeOperation = 'source-over';
        }}

        function handleFile(file) {{
            if (file && file.type.startsWith('image/')) {{
                const reader = new FileReader();
                reader.onload = (e) => {{ taskImage.src = e.target.result; }};
                reader.readAsDataURL(file);
            }}
        }}

        function updatePromptDisplay(nodeData, displayElement) {{
            displayElement = displayElement || promptDisplay;
            if (!nodeData) return;
            
            let promptText = nodeData.data || "No prompt available";
            
            if (!showFullPrompt) {{
                // Extract basic prompt by removing retrieved examples and strategies section
                if ((promptText.includes("--- Example") && promptText.includes("--- End Example")) ||
                    (promptText.includes("--- Strategy") && promptText.includes("--- End Strategy"))) {{
                    const lines = promptText.split('\\n');
                    let basicPromptLines = [];
                    let inExample = false;
                    let inStrategy = false;
                    let retrievedContentFound = false;
                    
                    for (let i = 0; i < lines.length; i++) {{
                        const line = lines[i];
                        const lineStripped = line.trim();
                        
                        if (lineStripped.startsWith('--- Example') && lineStripped.endsWith('---')) {{
                            inExample = true;
                            retrievedContentFound = true;
                            if (!basicPromptLines.some(l => l.includes('[Retrieved content section hidden]'))) {{
                                basicPromptLines.push('');
                                basicPromptLines.push('[Retrieved content section hidden - click "Show Full Prompt" to view]');
                                basicPromptLines.push('');
                            }}
                            continue;
                        }}
                        
                        if (lineStripped.startsWith('--- Strategy') && lineStripped.endsWith('---')) {{
                            inStrategy = true;
                            retrievedContentFound = true;
                            if (!basicPromptLines.some(l => l.includes('[Retrieved content section hidden]'))) {{
                                basicPromptLines.push('');
                                basicPromptLines.push('[Retrieved content section hidden - click "Show Full Prompt" to view]');
                                basicPromptLines.push('');
                            }}
                            continue;
                        }}
                        
                        if (lineStripped.startsWith('--- End Example') && lineStripped.endsWith('---')) {{
                            inExample = false;
                            continue;
                        }}
                        
                        if (lineStripped.startsWith('--- End Strategy') && lineStripped.endsWith('---')) {{
                            inStrategy = false;
                            continue;
                        }}
                        
                        if (!inExample && !inStrategy) {{
                            basicPromptLines.push(line);
                        }}
                    }}
                    
                    if (retrievedContentFound) {{
                        promptText = basicPromptLines.join('\\n');
                    }}
                }} else if (promptText.includes("Here are some relevant examples")) {{
                    // Fallback for old format
                    const examplesStart = promptText.indexOf("Here are some relevant examples");
                    const beforeExamples = promptText.substring(0, examplesStart).trim();
                    
                    // Find the end of examples section
                    const lines = promptText.split('\\n');
                    let afterExamplesStart = -1;
                    let inExamplesSection = false;
                    
                    for (let i = 0; i < lines.length; i++) {{
                        const line = lines[i];
                        if (line.includes("Here are some relevant examples")) {{
                            inExamplesSection = true;
                            continue;
                        }}
                        if (inExamplesSection && (
                            line.includes("OBJECTIVE:") || 
                            line.includes("PREVIOUS ACTIONS:") ||
                            line.includes("IMAGE:") ||
                            (line.trim().length > 0 && !line.includes("Example") && !line.includes("---") && !line.includes("THOUGHT") && !line.includes("ACTION"))
                        )) {{
                            afterExamplesStart = i;
                            break;
                        }}
                    }}
                    
                    if (afterExamplesStart !== -1) {{
                        const afterExamples = lines.slice(afterExamplesStart).join('\\n');
                        promptText = beforeExamples + '\\n\\n[Retrieved content section hidden - click "Show Full Prompt" to view]\\n\\n' + afterExamples;
                    }} else {{
                        promptText = beforeExamples + '\\n\\n[Retrieved content section hidden - click "Show Full Prompt" to view]';
                    }}
                }}
            }}
            
            displayElement.textContent = promptText;
        }}

        function updateExamplesDisplay(nodeData, displayElement) {{
            displayElement = displayElement || examplesDisplay;
            if (!nodeData || !nodeData.retrieved_examples || nodeData.retrieved_examples.length === 0) {{
                displayElement.innerHTML = '<p>No retrieved examples for this node.</p>';
                return;
            }}
            
            let html = '';
            nodeData.retrieved_examples.forEach((example, index) => {{
                const exampleId = `example_${{index}}`;
                const imageId = `image_${{index}}`;
                
                html += `
                    <div class="example-item">
                        <div class="example-header">
                            ${{example.title || `Example ${{index + 1}}`}}
                            <div class="example-buttons">
                                <button class="example-btn" onclick="toggleExample('${{exampleId}}')">Show/Hide Details</button>
                                ${{example.image_path ? `<button class="example-btn" onclick="toggleImage('${{imageId}}')">Show/Hide Image</button>` : ''}}
                                <button class="example-btn" onclick="copyToClipboard('${{example.text || ''}}')">Copy Text</button>
                                ${{example.image_path ? `<button class="example-btn" onclick="openInNewTab('${{example.image_path}}')">Open Image</button>` : ''}}
                            </div>
                        </div>
                        <div id="${{exampleId}}" class="example-content" style="display: none;">
                            <div class="example-text">
                                <p><strong>Text:</strong></p>
                                <pre>${{example.text || JSON.stringify(example.text, null, 2)}}</pre>
                            </div>
                        </div>
                        ${{example.image_path ? `
                            <div id="${{imageId}}" class="example-image-container" style="display: none;">
                                <div class="image-display-section">
                                    <p><strong>Image Path:</strong> ${{example.image_path}}</p>
                                    ${{example.encoded_image ? `
                                        <img class="example-image" src="${{example.encoded_image}}" alt="Retrieved Example Image" 
                                             style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px;">
                                        <p style="font-size: 12px; color: #28a745; margin: 5px 0;">✓ Image encoded for portability (Base64)</p>
                                    ` : `
                                        <div class="url-image-fallback">
                                            <p><strong>🌐 External URL Image</strong></p>
                                            <p>Due to browser security restrictions (CORS), external images may not display directly in the HTML file.</p>
                                            <p><a href="${{example.image_path}}" target="_blank">Click here to open image in new tab</a></p>
                                            <small>URL: ${{example.image_path}}</small>
                                        </div>
                                        <img class="example-image" src="${{example.image_path}}" alt="Retrieved Example Image" 
                                             style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px;"
                                             onload="this.previousElementSibling.style.display='none';" 
                                             onerror="this.style.display='none';"
                                             crossorigin="anonymous">
                                    `}}
                                </div>
                            </div>
                        ` : ''}}
                    </div>
                `;
            }});
            
            displayElement.innerHTML = html;
        }}

        function toggleExample(exampleId) {{
            const element = document.getElementById(exampleId);
            if (element) {{
                element.style.display = element.style.display === 'none' ? 'block' : 'none';
            }}
        }}

        function toggleImage(imageId) {{
            const element = document.getElementById(imageId);
            if (element) {{
                element.style.display = element.style.display === 'none' ? 'block' : 'none';
            }}
        }}

        function updateOutputDisplay(nodeData, displayElement) {{
            displayElement = displayElement || outputDisplay;
            if (!nodeData) {{
                displayElement.textContent = "No node selected";
                return;
            }}
            
            if (nodeData.type === 'node_prompt') {{
                const node = nodeData.node_data;
                let output = `Node Type: ${{nodeData.node_type}}\\n`;
                output += `Thought Text: ${{node.thought_text || 'N/A'}}\\n`;
                output += `Is Terminal: ${{node.is_terminal || false}}\\n`;
                output += `Visit Count: ${{node.visit_count || 0}}\\n`;
                output += `Value: ${{node.value || 0}}\\n`;
                output += `Coordinates: ${{nodeData.coords ? nodeData.coords.join(', ') : 'None'}}\\n`;
                output += `Judge Descriptions: ${{node.judge_description ? node.judge_description.join('; ') : 'None'}}\\n`;
                
                if (node.rollouts && node.rollouts.length > 0) {{
                    output += `\\nRollouts (${{node.rollouts.length}}):\\n`;
                    node.rollouts.forEach((rollout, i) => {{
                        output += `  Rollout ${{i+1}}: \\n **Node Text**: \\n ${{rollout.node_text}} \\n\\n **Node Status**: \\n ${{rollout.node_status}} \\n\\n Progress Text:\\n ${{rollout.progress_text}} \\n\\n Reflection:\\n ${{rollout.reflection}} \\n\\n Reward: \\n ${{rollout.percentage}}\\n\\n`;
                    }});
                }}

                output += `===== Visual Results ======\\n`;
                output += `First Image: Grounding Elements;\\n`;
                output += `Second Image: Next State.\\n`;

                displayElement.textContent = output;
                renderNodeImage(nodeData, displayElement);
                
            }} else if (nodeData.type === 'rollout_prompts') {{
                const rollout = nodeData.rollout_data;
                let output = `Rollout Details:\\n`;
                output += `Final Answer: ${{rollout.final_answer || 'N/A'}}\\n`;
                output += `Reward: ${{rollout.reward || 0}}\\n`;
                output += `Ephemeral Depth: ${{rollout.ephemeral_depth || 'N/A'}}\\n`;
                output += `Number of Turns: ${{rollout.rollout_prompts ? rollout.rollout_prompts.length : 0}}\\n`;
                
                if (rollout.ephemeral_texts) {{
                    output += `\\nEphemeral Texts:\\n`;
                    rollout.ephemeral_texts.forEach((text, i) => {{
                        output += `  Step ${{i+1}}: ${{text}}\\n`;
                    }});
                }}
                
                displayElement.textContent = output;
            }}
        }}

        function updateVideoDisplay(nodeData) {{
            const videoPlayer = document.getElementById('video-player');
            const gifPlayer = document.getElementById('gif-player');
            const videoDisplay = document.getElementById('video-display');
            const canvas = document.getElementById('video-overlay-canvas');
            const ctx = canvas.getContext('2d');

            // Clear previous state
            videoPlayer.style.display = 'none';
            gifPlayer.style.display = 'none';
            videoDisplay.style.display = 'block';
            videoPlayer.pause();
            videoPlayer.src = '';
            gifPlayer.src = '';
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (!nodeData || !nodeData.video_path) {{
                videoDisplay.textContent = 'No video/GIF available for this node.';
                return;
            }}

            const videoPath = nodeData.video_path;
            const boundingBoxes = nodeData.bounding_boxes || [];

            // Determine if it's a GIF or video based on file extension
            const isGif = videoPath.toLowerCase().endsWith('.gif');

            if (isGif) {{
                // Display GIF
                gifPlayer.onload = function() {{
                    videoDisplay.style.display = 'none';
                    gifPlayer.style.display = 'block';

                    // Set canvas size to match GIF
                    canvas.width = gifPlayer.offsetWidth;
                    canvas.height = gifPlayer.offsetHeight;
                    canvas.style.display = 'block';

                    // Draw bounding boxes
                    drawBoundingBoxes(ctx, boundingBoxes, gifPlayer.offsetWidth, gifPlayer.offsetHeight);
                }};

                gifPlayer.onerror = function() {{
                    videoDisplay.textContent = `Failed to load GIF: ${{videoPath}}`;
                    videoDisplay.style.display = 'block';
                    gifPlayer.style.display = 'none';
                }};

                gifPlayer.src = videoPath;
            }} else {{
                // Display video
                videoPlayer.onloadedmetadata = function() {{
                    videoDisplay.style.display = 'none';
                    videoPlayer.style.display = 'block';

                    // Set canvas size to match video
                    canvas.width = videoPlayer.offsetWidth;
                    canvas.height = videoPlayer.offsetHeight;
                    canvas.style.display = 'block';

                    // Draw bounding boxes
                    drawBoundingBoxes(ctx, boundingBoxes, videoPlayer.offsetWidth, videoPlayer.offsetHeight);

                    videoPlayer.play();
                }};

                videoPlayer.onerror = function() {{
                    videoDisplay.textContent = `Failed to load video: ${{videoPath}}`;
                    videoDisplay.style.display = 'block';
                    videoPlayer.style.display = 'none';
                }};

                // Handle video resize during playback
                videoPlayer.onresize = function() {{
                    canvas.width = videoPlayer.offsetWidth;
                    canvas.height = videoPlayer.offsetHeight;
                    drawBoundingBoxes(ctx, boundingBoxes, videoPlayer.offsetWidth, videoPlayer.offsetHeight);
                }};

                videoPlayer.src = videoPath;
            }}
        }}

        function drawBoundingBoxes(ctx, boundingBoxes, displayWidth, displayHeight) {{
            if (!boundingBoxes || boundingBoxes.length === 0) {{
                return;
            }}

            // Assume bounding boxes are in the original coordinate space
            // We may need to scale them to match the display size
            // For now, we'll draw them as-is and let CSS handle scaling

            ctx.strokeStyle = '#00FF00';  // Green boxes
            ctx.lineWidth = 2;
            ctx.font = '12px Arial';
            ctx.fillStyle = '#00FF00';

            boundingBoxes.forEach((box, index) => {{
                const x1 = box.x1;
                const y1 = box.y1;
                const x2 = box.x2;
                const y2 = box.y2;
                const width = x2 - x1;
                const height = y2 - y1;

                // Draw rectangle
                ctx.strokeRect(x1, y1, width, height);

                // Draw label if available (in exhaustive mode or always)
                if (exhaustiveMode && box.label) {{
                    const labelText = `Box ${{index + 1}}: ${{box.label}}`;
                    ctx.fillText(labelText, x1, Math.max(y1 - 5, 10));
                }}
            }});

            // In exhaustive mode, also draw coordinates as circles
            if (exhaustiveMode) {{
                ctx.fillStyle = '#FF0000';  // Red for coordinates
                // Note: coordinates would come from nodeData.coords
                // This is handled by the existing highlight canvas for the main image
            }}
        }}

        function showRolloutModal(rolloutData) {{
            currentRolloutData = rolloutData;
            currentTurn = 0;

            const turnSelector = document.getElementById('turn-selector');
            const prompts = rolloutData.data || [];
            
            if (prompts.length === 0) {{
                turnSelector.innerHTML = '<p>No prompts available for this rollout.</p>';
                document.getElementById('rollout-prompt-text').textContent = 'No prompts available.';
            }} else {{
                let html = '<div style="margin-bottom: 10px;"><strong>Select Step to View:</strong></div>';
                prompts.forEach((_, index) => {{
                    html += `<button class="turn-btn ${{index === 0 ? 'active' : ''}}" onclick="selectTurn(${{index}})">Step ${{index + 1}}</button>`;
                }});
                turnSelector.innerHTML = html;
                
                document.getElementById('rollout-prompt-text').textContent = prompts[0] || 'No prompt text available.';
            }}
            
            document.getElementById('rollout-modal').style.display = 'flex';
        }}

        function selectTurn(turnIndex) {{
            currentTurn = turnIndex;
            const prompts = currentRolloutData.data || [];
            
            // Update button states
            document.querySelectorAll('.turn-btn').forEach((btn, index) => {{
                btn.classList.toggle('active', index === turnIndex);
            }});
            
            // Update prompt display in modal
            const promptText = prompts[turnIndex] || 'No prompt text available.';
            document.getElementById('rollout-prompt-text').textContent = promptText;
        }}

        function closeModal(modalId) {{
            document.getElementById(modalId).style.display = 'none';
        }}

        function copyToClipboard(text) {{
            navigator.clipboard.writeText(text).then(() => {{
                alert('Copied to clipboard!');
            }});
        }}

        function openInNewTab(url) {{
            if (url) {{
                window.open(url, '_blank');
            }}
        }}

        function showDebugCoords(clientX, clientY) {{
            if (!debugMode || !taskImage.complete || taskImage.naturalWidth === 0) {{
                debugCoordsDiv.style.display = 'none';
                return;
            }}

            // Calculate the image coordinates relative to the effective (scaled) image
            const rect = imageContainer.getBoundingClientRect();
            const containerX = clientX - rect.left;
            const containerY = clientY - rect.top;
            
            // Convert container coordinates to effective image coordinates accounting for pan and scale
            const effectiveImageX = (containerX - panX) / scale;
            const effectiveImageY = (containerY - panY) / scale;
            
            const effectiveWidth = taskImage.naturalWidth * displayScale;
            const effectiveHeight = taskImage.naturalHeight * displayScale;
            
            // Ensure coordinates are within effective image bounds
            if (effectiveImageX >= 0 && effectiveImageX <= effectiveWidth && effectiveImageY >= 0 && effectiveImageY <= effectiveHeight) {{
                const roundedX = Math.round(effectiveImageX);
                const roundedY = Math.round(effectiveImageY);
                
                debugCoordsDiv.textContent = `(${{roundedX}}, ${{roundedY}}) [scale: ${{displayScale.toFixed(1)}}x]`;
                debugCoordsDiv.style.left = (clientX + 10) + 'px';
                debugCoordsDiv.style.top = (clientY - 30) + 'px';
                debugCoordsDiv.style.display = 'block';
            }} else {{
                debugCoordsDiv.style.display = 'none';
            }}
        }}

        // --- Extended View Functions ---
        function enterExtendedView(tabElement) {{
            if (tabElement.dataset.tab === 'image-tab') return; // Don't allow extended view for image tab
            
            isExtendedView = true;
            currentExtendedTab = tabElement.dataset.tab;
            
            // Add extended view class to main container
            mainContainer.classList.add('extended-view');
            
            // Update extended tab title
            extendedTabTitle.textContent = tabElement.textContent;
            
            // Copy the content from the selected tab to extended tab content
            const sourceTab = document.getElementById(currentExtendedTab);
            const sourceContent = sourceTab.innerHTML;
            extendedTabContent.innerHTML = sourceContent;
            
            // Initialize extended network
            if (!extendedNetwork) {{
                extendedNetwork = new vis.Network(
                    document.getElementById('extended-mcts-graph'),
                    {{ nodes: nodesDataSet, edges: edgesDataSet }},
                    networkOptions
                );
                
                // Set up extended network events
                extendedNetwork.on("click", handleNetworkClick);
            }}
            
            // Sync network state
            if (currentSelectedNodeId) {{
                highlightSelectedNode(currentSelectedNodeId, extendedNetwork);
            }}
            
            // Re-apply event listeners for copied content
            setupExtendedTabEventListeners();
            
            setTimeout(() => {{
                extendedNetwork.redraw();
                network.redraw();
                fitImageToContainer();
            }}, 100);
        }}

        function exitExtendedView() {{
            isExtendedView = false;
            currentExtendedTab = null;
            
            // Remove extended view class
            mainContainer.classList.remove('extended-view');
            
            // Clear extended content
            extendedTabContent.innerHTML = '';
            
            setTimeout(() => {{
                network.redraw();
                fitImageToContainer();
            }}, 100);
        }}

        function setupExtendedTabEventListeners() {{
            if (currentExtendedTab === 'prompt-tab') {{
                const extendedPromptBtn = extendedTabContent.querySelector('#show-full-prompt');
                if (extendedPromptBtn) {{
                    extendedPromptBtn.addEventListener('click', () => {{
                        showFullPrompt = !showFullPrompt;
                        extendedPromptBtn.textContent = showFullPrompt ? 'Show Basic Prompt' : 'Show Full Prompt';
                        const extendedPromptDisplay = extendedTabContent.querySelector('#prompt-display');
                        if (extendedPromptDisplay && currentSelectedNode) {{
                            updatePromptDisplay(currentSelectedNode, extendedPromptDisplay);
                        }}
                    }});
                }}
            }}
        }}

        function handleNetworkClick(params) {{
            const nodes = params.nodes;
            if (!nodes.length) return;
            
            const nodeId = nodes[0];
            const data = interactiveDataMap[nodeId];
            currentSelectedNode = data;
            currentSelectedNodeId = nodeId;
            
            // Highlight the selected node on both networks
            highlightSelectedNode(nodeId, network);
            if (extendedNetwork) {{
                highlightSelectedNode(nodeId, extendedNetwork);
            }}
            
            if (data?.type === 'node_prompt') {{
                updatePromptDisplay(data);
                updateExamplesDisplay(data);
                updateOutputDisplay(data);
                updateVideoDisplay(data);
                drawHighlight(data.coords);

                // Update extended view if active
                if (isExtendedView) {{
                    updateExtendedContent(data);
                }}
            }} else if (data?.type === 'rollout_prompts') {{
                showRolloutModal(data);
                updateOutputDisplay(data);
                updateVideoDisplay(data);
                drawHighlight(data.coords);
            }}
        }}

        function updateExtendedContent(data) {{
            if (currentExtendedTab === 'prompt-tab') {{
                const extendedPromptDisplay = extendedTabContent.querySelector('#prompt-display');
                if (extendedPromptDisplay) {{
                    updatePromptDisplay(data, extendedPromptDisplay);
                }}
            }} else if (currentExtendedTab === 'examples-tab') {{
                const extendedExamplesDisplay = extendedTabContent.querySelector('#examples-display');
                if (extendedExamplesDisplay) {{
                    updateExamplesDisplay(data, extendedExamplesDisplay);
                }}
            }} else if (currentExtendedTab === 'output-tab') {{
                const extendedOutputDisplay = extendedTabContent.querySelector('#output-display');
                if (extendedOutputDisplay) {{
                    updateOutputDisplay(data, extendedOutputDisplay);
                }}
            }}
        }}

        function renderNodeImage(nodeData, displayElement) {{
            const node = nodeData && nodeData.node_data;
            const b64 = node && node.vis_result_next;
            const b64Res = node && node.vis_result;
            
            let imgBox = displayElement.nextElementSibling;
            if (!imgBox || !imgBox.classList || !imgBox.classList.contains('node-image-box')) {{
                imgBox = document.createElement('div');
                imgBox.className = 'node-image-box';
                imgBox.style.marginTop = '8px';
                imgBox.style.maxWidth = '100%';
                displayElement.insertAdjacentElement('afterend', imgBox);
            }}

            if (!b64 && !b64Res) {{
                imgBox.innerHTML = '';
                return;
            }}

            const toSrc = (s) => s.startsWith('data:') ? s : `data:image/png;base64,${{s}}`;
            const srcs = [b64Res, b64].filter(Boolean).map(toSrc);

            // Optional: skip re-render if already identical (checks first only)
            const old = imgBox.querySelectorAll('img');
            if (old.length === srcs.length && old[0] && old[0].getAttribute('src') === srcs[0]) return;

            imgBox.innerHTML = '';
            for (const src of srcs) {{
                const img = document.createElement('img');
                img.src = src;
                img.alt = 'node image';
                img.loading = 'lazy';
                img.decoding = 'async';
                img.style.display = 'block';
                img.style.maxWidth = '100%';
                img.style.borderRadius = '6px';
                img.style.marginTop = '6px';
                imgBox.appendChild(img);
            }}
        }}

        // --- Event Listeners ---
        
        // Panel toggle
        leftPanelToggle.addEventListener('click', () => {{
            const isCollapsed = leftPanelContainer.classList.toggle('collapsed');
            leftPanelToggle.innerHTML = isCollapsed ? '&lt;' : '&gt;';
            window.dispatchEvent(new Event('resize'));
        }});

        // Tab switching
        tabs.forEach(tab => {{
            tab.addEventListener('click', () => {{
                switchTab(tab.dataset.tab);
            }});
            
            // Double-click to enter extended view (except for image tab)
            tab.addEventListener('dblclick', () => {{
                if (tab.dataset.tab !== 'image-tab') {{
                    enterExtendedView(tab);
                }}
            }});
        }});
        
        // Restore view button
        restoreViewBtn.addEventListener('click', exitExtendedView);

        // Show full prompt toggle
        showFullPromptBtn.addEventListener('click', () => {{
            showFullPrompt = !showFullPrompt;
            showFullPromptBtn.textContent = showFullPrompt ? 'Show Basic Prompt' : 'Show Full Prompt';
            updatePromptDisplay(currentSelectedNode);
        }});

        // Image handling
        taskImage.onload = fitImageToContainer;
        taskImage.onerror = () => {{
            taskImage.style.display = 'none';
            dropZone.classList.remove('hidden');
        }};
        if (taskImage.complete && taskImage.naturalWidth) taskImage.onload();

        // Image interaction
        imageContainer.addEventListener('wheel', (e) => {{
            e.preventDefault();
            const rect = imageContainer.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const delta = -e.deltaY * 0.001;
            const newScale = Math.max(0.5, Math.min(scale + delta, 10));
            
            panX = x - (x - panX) * (newScale / scale);
            panY = y - (y - panY) * (newScale / scale);
            scale = newScale;
            updateImageTransform();
        }});

        imageContainer.addEventListener('mousedown', (e) => {{
            isPanning = true;
            startPan = {{ x: e.clientX - panX, y: e.clientY - panY }};
            imageContainer.style.cursor = 'grabbing';
        }});
        imageContainer.addEventListener('mousemove', (e) => {{
            if (isPanning) {{
                panX = e.clientX - startPan.x;
                panY = e.clientY - startPan.y;
                updateImageTransform();
            }}
            // Show debug coordinates regardless of panning state
            showDebugCoords(e.clientX, e.clientY);
        }});
        imageContainer.addEventListener('mouseup', () => {{
            isPanning = false;
            imageContainer.style.cursor = 'grab';
        }});
        imageContainer.addEventListener('mouseleave', () => {{
            isPanning = false;
            imageContainer.style.cursor = 'grab';
            debugCoordsDiv.style.display = 'none';
        }});

        // Network events
        network.on("click", handleNetworkClick);

        // File drag and drop
        dropZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {{
            dropZone.addEventListener(eventName, (e) => {{ e.preventDefault(); e.stopPropagation(); }}, false);
        }});
        dropZone.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));

        // Window resize
        window.addEventListener('resize', () => {{
            setTimeout(() => {{ 
                network.redraw(); 
                fitImageToContainer(); 
            }}, 350);
        }});

        // Close modals when clicking outside
        window.addEventListener('click', (e) => {{
            if (e.target.classList.contains('modal')) {{
                e.target.style.display = 'none';
            }}
        }});
    </script>
</body>
</html>
"""
    with open(f"{output_dir}/{output_filename}.html", 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    # Save statistics to a text file for easy reference
    success_rate_0 = (stats['successful_rollouts_0'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    success_rate_05 = (stats['successful_rollouts_05'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    success_rate_075 = (stats['successful_rollouts_075'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    success_rate_1 = (stats['successful_rollouts_1'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    stats_text_file = (
        f"--- Tree Statistics ---\n"
        f"Total Nodes: {stats['total_nodes']}\n"
        f"Max Depth: {stats['max_depth']}\n"
        f"Terminal Nodes: {stats['terminal_nodes']}\n"
        f"Total Search Time: {tree_data.get('global_search_time', 0) if isinstance(tree_data.get('global_search_time'), (int, float)) else 'N/A'}{'s' if isinstance(tree_data.get('global_search_time'), (int, float)) else ''}\n\n"
        f"--- Rollout Statistics ---\n"
        f"Total Rollouts: {stats['total_rollouts']}\n"
        f"Successful Rollouts (R > 0): {stats['successful_rollouts_0']} ({success_rate_0:.2f}%)\n"
        f"Successful Rollouts (R >= 0.5): {success_rate_05:.2f}%)\n"
        f"Successful Rollouts (R >= 0.75): {success_rate_075:.2f}%)\n"
        f"Perfect Rollouts (R == 1): {stats['successful_rollouts_1']} ({success_rate_1:.2f}%)\n\n"
        f"--- Best Reward in Terminal Node ---\n"
        f"Highest Reward: {stats['best_reward'] if stats['best_reward'] != -1.0 else 'N/A'}\n"
        f"Number of Nodes with this Reward: {stats['best_reward_node_count']}\n\n"
        f"--- Most Visited Node ---\n"
        f"Visit Count: {stats['max_visit_count']}\n"
        f"Thought: {stats['most_visited_node_text']}\n\n"
        f"--- Action Distribution in Rollouts ---\n"
        f"{json.dumps(stats['action_distribution'], indent=2)}"
    )
    stats_output_path = f"{output_dir}/{output_filename}_stats.txt"
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        f.write(stats_text_file)
    print(f"Statistics saved to {os.path.abspath(stats_output_path)}")

def main():
    VERSION = "29"
    print(f"🚀 Enhanced MCTS Visualizer v{VERSION} - Starting visualization...")
    
    parser = argparse.ArgumentParser(description='Visualize MCTS search data as an enhanced interactive HTML file with tabbed interface and retrieval support.')
    parser.add_argument('input_file', type=str, help='Path to the MCTS rollout JSONL file.')
    parser.add_argument('--output_filename', type=str, default='mcts_visualization_enhanced', help='Name of the output file (without extension).')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output HTML file.')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the tree to visualize.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to show coordinates on hover.')
    parser.add_argument('--exhaustive', action='store_true', help='Enable exhaustive mode to visualize all coordinates and bounding boxes from node outputs.')

    args = parser.parse_args()

    # Get the experiment directory from the input file path
    input_file_path = Path(args.input_file)
    experiment_dir = input_file_path.parent
    print(f"Experiment directory: {experiment_dir}")

    # Load or create images configuration
    images_config = load_or_create_images_config(experiment_dir)

    try:
        # with open(args.input_file, 'r', encoding='utf-8') as f:
        #     first_line = f.readline()
        #     if not first_line:
        #         print("Error: File is empty.")
        #         return
        #     rollout_data = json.loads(first_line)
        with open(args.input_file) as f:
            rollout_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading or parsing the input file '{args.input_file}': {e}")
        return

    # # Resolve image path based on configuration (if any)
    # if 'image' in rollout_data and rollout_data['image']:
    #     original_image_path = rollout_data['image']
    #     resolved_image_path = resolve_image_path(original_image_path, images_config)
        
    #     # Only update if path actually changed
    #     if resolved_image_path != original_image_path:
    #         rollout_data['image'] = resolved_image_path
    #         print(f"Image path resolution: {original_image_path} -> {resolved_image_path}")
    #     else:
    #         print(f"Using original image path: {original_image_path}")

    # 1. Compute overall statistics
    stats = get_tree_statistics(rollout_data)
    print("Successfully computed tree statistics.")

    # 2. Build the graph data for vis.js
    graph_data = build_graph_data(rollout_data, args.max_depth)
    print(f"Processed {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges for the graph.")

    # 3. Generate the final HTML file
    generate_html_file(rollout_data, graph_data, stats, args.output_dir, args.output_filename, args.debug, images_config, args.exhaustive)
    print(f"✅ Enhanced interactive visualization v{VERSION} saved to {args.output_dir}/{args.output_filename}.html")

if __name__ == '__main__':
    main()
