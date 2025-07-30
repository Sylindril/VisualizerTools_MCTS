# visualizer_enhanced.py - Enhanced MCTS Visualizer with Tabbed Interface and Retrieval Support
# Version: 19 - FIXED URL IMAGE DISPLAY IN HTML
# Last Updated: 2025-07-16
# CHANGES:
# - Fixed duplicate retrieved examples parsing with proper unique key detection
# - Fixed basic vs full prompt separation to properly hide retrieved examples
# - Enhanced image base64 encoding with better error handling and file size limits
# - Added rollout and example indices for better debugging

import argparse
import json
import os
import html
import re
import base64
import mimetypes

def _extract_coords(text):
    """Extracts (x, y) coordinates from a string, supporting integers and floats."""
    if not isinstance(text, str):
        return None
    
    # Priority order: look for coordinates in different contexts
    patterns = [
        # Pattern 1: Standard (x, y) format - prioritize numbers at end of lines
        r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)\s*$',
        # Pattern 2: Standard (x, y) format - anywhere in text
        r'\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',
        # Pattern 3: Square brackets [x, y] - at end of lines
        r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*$',
        # Pattern 4: Square brackets [x, y] - anywhere in text
        r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',
        # Pattern 5: Function call format like click(x,y)
        r'\w+\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',
        # Pattern 6: Action with coordinates like "click [123, 456]"
        r'click\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]',
        # Pattern 7: Coordinates with "coordinate" or "coord" keyword
        r'(?:coordinate|coord)[s]?\s*[:\-]?\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',
        # Pattern 8: Position or point references
        r'(?:position|point|at|located)\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',
        # Pattern 9: X,Y format without parentheses
        r'(?:^|\s)(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)(?:\s|$)',
        # Pattern 10: Near/around coordinate references
        r'(?:near|around|at)\s*\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)',
        # Pattern 11: Loose coordinate format in reasoning
        r'(?:x|X)\s*[=:]?\s*(\d+(?:\.\d+)?)\s*,?\s*(?:y|Y)\s*[=:]?\s*(\d+(?:\.\d+)?)',
    ]
    
    # Search multiline text by processing each line
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        for pattern in patterns:
            match = re.search(pattern, line, re.MULTILINE | re.IGNORECASE)
            if match:
                # Validate the coordinates are reasonable (e.g., not too large)
                x, y = float(match.group(1)), float(match.group(2))
                if 0 <= x <= 10000 and 0 <= y <= 10000:  # Reasonable screen coordinates
                    return [x, y]
    
    # Fallback: search in the entire text
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            # Validate the coordinates are reasonable
            x, y = float(match.group(1)), float(match.group(2))
            if 0 <= x <= 10000 and 0 <= y <= 10000:  # Reasonable screen coordinates
                return [x, y]
    
    return None

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
            if node.get('rollouts'):
                rewards = [r.get('reward', -1.0) for r in node['rollouts']]
                if rewards:
                    max_reward_in_node = max(rewards)
                    if max_reward_in_node > stats['best_reward']:
                        stats['best_reward'] = max_reward_in_node
                        stats['best_reward_node_count'] = 1
                    elif max_reward_in_node == stats['best_reward'] and stats['best_reward'] != -1.0:
                        stats['best_reward_node_count'] += 1

        if node.get('visit_count', 0) > stats['max_visit_count']:
            stats['max_visit_count'] = node['visit_count']
            stats['most_visited_node_text'] = node.get('thought_text', 'N/A')

        if 'rollouts' in node:
            num_rollouts = len(node['rollouts'])
            stats['total_rollouts'] += num_rollouts
            for rollout in node['rollouts']:
                reward = rollout.get('reward', 0)
                if reward > 0:
                    stats['successful_rollouts_0'] += 1
                if reward >= 0.5:
                    stats['successful_rollouts_05'] += 1
                if reward >= 0.75:
                    stats['successful_rollouts_075'] += 1
                if reward == 1:
                    stats['successful_rollouts_1'] += 1
                if 'final_answer' in rollout:
                    action = rollout['final_answer']
                    stats['action_distribution'][action] = stats['action_distribution'].get(action, 0) + 1
        
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

    def _recursive_build(node, parent_id, depth):
        nonlocal node_id_counter
        
        current_id = f"node_{node_id_counter}"
        node_id_counter += 1
        
        if max_depth_vis is not None and depth > max_depth_vis:
            return

        # 1. Create the main MCTS node (thought or action)
        thought = node.get('thought_text', 'No Thought')
        visit_count = node.get('visit_count', 0)
        value = node.get('value', 0.0)
        is_terminal = node.get('is_terminal', False)

        # Determine node type for better visualization
        node_type = "Unknown"
        if parent_id is None:
            node_type = "Root"
        elif thought.startswith('<think>'):
            node_type = "Thinking"
        elif thought.startswith('click') or thought.startswith('scroll') or thought.startswith('go_back') or thought.startswith('stop'):
            node_type = "Action"
        elif 'OBJECTIVE:' in thought:
            node_type = "Objective"
        else:
            node_type = "Other"

        display_thought = html.escape((thought[:100] + '...') if len(thought) > 100 else thought)
        label = f"[{node_type}] {display_thought}\nVisits: {visit_count} | Value: {value:.3f}"
        
        # Color coding by node type
        if is_terminal:
            color = '#ff9999'  # Red for terminal
        elif parent_id is None:
            color = '#4CAF50'  # Green for root
        elif node_type == "Thinking":
            color = '#87CEEB'  # Sky blue for thinking nodes
        elif node_type == "Action":
            color = '#FFD700'  # Gold for action nodes
        else:
            color = '#97C2FC'  # Default blue
        
        coords = _extract_coords(thought)
        coords_info = f"Coordinates: {coords}" if coords else "No coordinates found"
        
        hover_title = (
            f"--- Node Statistics ---\n"
            f"Type: {node_type}\n"
            f"Visits: {visit_count}\n"
            f"Value (Q): {value:.4f}\n"
            f"Is Terminal: {is_terminal}\n"
            f"Rollouts from this node: {len(node.get('rollouts', []))}\n"
            f"{coords_info}\n"
            f"Full Thought: {html.escape(thought)}"
        )

        nodes.append({
            "id": current_id, "label": label, "shape": 'box', "color": color,
            "title": hover_title, "margin": 10, "font": {"color": "#333"}
        })

        if parent_id:
            edges.append({"from": parent_id, "to": current_id})

        # 2. Store prompt and coordinates for the main node
        prompt_for_node = node.get('prompt_for_node')
        if not prompt_for_node:
            prompt_for_node = "This is the root node." if parent_id is None else "Prompt not found."
        
        # Extract retrieved examples from prompt_for_node using the correct format
        retrieved_examples = []
        seen_examples = set()  # Track unique examples to avoid duplicates
        
        # First, check if prompt_for_node contains examples
        if prompt_for_node and "--- Example" in prompt_for_node:
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
        
        # Also check rollouts for any additional examples (fallback)
        if node.get('rollouts'):
            for rollout_idx, rollout in enumerate(node['rollouts']):
                # Check ephemeral_retrieved_paths
                if rollout.get('ephemeral_retrieved_paths'):
                    ephemeral_texts = rollout.get('ephemeral_texts', [])
                    for i, path in enumerate(rollout['ephemeral_retrieved_paths']):
                        if path and path.strip():
                            text_content = ephemeral_texts[i] if i < len(ephemeral_texts) else f"Retrieved example {i+1}"
                            unique_key = f"{path}:::{text_content[:100]}"
                            
                            if unique_key not in seen_examples:
                                retrieved_examples.append({
                                    'image_path': path,
                                    'encoded_image': encode_image_to_base64(path),
                                    'text': text_content,
                                    'title': f'Retrieved Example {len(retrieved_examples) + 1}',
                                    'example_index': i,
                                    'rollout_index': rollout_idx,
                                    'source': 'ephemeral_paths'
                                })
                                seen_examples.add(unique_key)
        
        interactive_data_map[current_id] = {
            "type": "node_prompt", 
            "data": prompt_for_node,
            "coords": coords,
            "node_type": node_type,
            "node_data": node,
            "retrieved_examples": retrieved_examples
        }
        
        # 3. Create a SEPARATE leaf node for EACH rollout
        if node.get('rollouts'):
            for i, rollout in enumerate(node['rollouts']):
                rollout_id = f"{current_id}_rollout_{i}"
                reward = rollout.get('reward', 0)
                final_answer = rollout.get('final_answer', 'N/A')
                
                rollout_label = f"Rollout #{i+1}\nAction: {html.escape(final_answer)}\nReward: {reward:.3f}"
                
                rollout_hover_title = (
                    f"--- Rollout Details ---\n"
                    f"Final Answer: {html.escape(final_answer)}\n"
                    f"Reward: {reward}\n"
                    f"Ephemeral Depth: {rollout.get('ephemeral_depth', 'N/A')}\n"
                    f"Number of Turns: {len(rollout.get('rollout_prompts', []))}"
                )

                nodes.append({
                    "id": rollout_id,
                    "label": rollout_label,
                    "shape": 'ellipse',
                    "color": '#E0E0E0',
                    "title": rollout_hover_title,
                    "font": {"size": 12}
                })
                edges.append({"from": current_id, "to": rollout_id, "dashes": True, "color": "#888"})
                
                rollout_coords = _extract_coords(final_answer)
                interactive_data_map[rollout_id] = {
                    "type": "rollout_prompts",
                    "data": rollout.get('rollout_prompts', []),
                    "coords": rollout_coords,
                    "rollout_data": rollout
                }

        # 4. Recurse for children
        for child in node.get('children', []):
            _recursive_build(child, current_id, depth + 1)

    _recursive_build(root, None, 0)
    
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

def generate_html_file(tree_data, graph_data, stats, output_filename, debug_mode=False):
    """Generates the final self-contained HTML file."""

    nodes_json = json.dumps(graph_data['nodes'])
    edges_json = json.dumps(graph_data['edges'])
    interactive_data_json = json.dumps(graph_data['interactive_data'])
    debug_mode_json = json.dumps(debug_mode)

    # Format stats for display
    success_rate_0 = (stats['successful_rollouts_0'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    success_rate_05 = (stats['successful_rollouts_05'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    success_rate_075 = (stats['successful_rollouts_075'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    success_rate_1 = (stats['successful_rollouts_1'] / stats['total_rollouts'] * 100) if stats['total_rollouts'] > 0 else 0
    stats_html = (
        f"--- Tree Statistics ---\n"
        f"Total Nodes: {stats['total_nodes']}\n"
        f"Max Depth: {stats['max_depth']}\n"
        f"Terminal Nodes: {stats['terminal_nodes']}\n"
        f"Total Search Time: {tree_data.get('global_search_time', 'N/A'):.2f}s\n\n"
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
        f"Thought: {html.escape(stats['most_visited_node_text'])}\n\n"
        f"--- Action Distribution in Rollouts ---\n"
        f"{html.escape(json.dumps(stats['action_distribution'], indent=2))}"
    ).replace('\n', '<br>')

    question = html.escape(tree_data.get('question', 'N/A')).replace('\n', '<br>')
    true_answer = html.escape(tree_data.get('true_answer', 'N/A'))
    image_path = tree_data.get('image', '')
    escaped_image_path = html.escape(image_path)
    
    # Convert image to base64 for portability
    base64_image = encode_image_to_base64(image_path)
    if base64_image:
        print(f"Image successfully encoded to base64 ({len(base64_image)} characters)")
        display_image_src = base64_image
    else:
        print(f"Warning: Could not encode image, using original path: {image_path}")
        display_image_src = image_path

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Enhanced MCTS Visualizer v19-URL-FIXED</title>
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
        
        /* Tab 2: Prompt */
        .prompt-container {{ padding: 20px; overflow-y: auto; flex: 1; }}
        .prompt-toggle {{ margin-bottom: 15px; }}
        .prompt-display {{ background-color: #f1f1f1; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 13px; max-height: 70vh; overflow-y: auto; }}
        
        /* Tab 3: Retrieved examples */
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
        
        /* Tab 4: Node output */
        .output-container {{ padding: 20px; overflow-y: auto; flex: 1; }}
        .output-display {{ background-color: #f1f1f1; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 13px; }}
        
        /* Tab 5: Statistics */
        .stats-container {{ padding: 20px; overflow-y: auto; flex: 1; }}
        .stats-section {{ margin-bottom: 20px; }}
        .stats-title {{ font-weight: bold; margin-bottom: 10px; color: #007bff; }}
        .stats-content {{ background-color: #f8f9fa; padding: 10px; border-radius: 5px; font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 12px; white-space: pre-wrap; }}
        
        /* Right panel for tree */
        #graph-container {{ flex: 1; background-color: #ffffff; position: relative; }}
        #mcts-graph {{ height: 100%; width: 100%; }}
        
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
            <h1>Enhanced MCTS Search Visualization<span class="version-badge">v19-URL-FIXED</span></h1>
        </div>
        <div class="content-area">
            <div id="left-panel-container" class="panel-container">
                <button id="left-panel-toggle" class="panel-toggle">&gt;</button>
                <div id="left-panel" class="panel">
                    <div class="tabs-container">
                        <div class="tabs">
                            <div class="tab active" data-tab="image-tab">Question Image</div>
                            <div class="tab" data-tab="prompt-tab">Input Prompt</div>
                            <div class="tab" data-tab="examples-tab">Retrieved Examples</div>
                            <div class="tab" data-tab="output-tab">Node Output</div>
                            <div class="tab" data-tab="stats-tab">Tree Statistics</div>
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
                    
                    <!-- Tab 2: Input Prompt -->
                    <div id="prompt-tab" class="tab-content">
                        <div class="prompt-container">
                            <div class="prompt-toggle">
                                <button id="show-full-prompt" class="control-button">Show Full Prompt</button>
                            </div>
                            <div id="prompt-display" class="prompt-display">Click on a node to see its prompt...</div>
                        </div>
                    </div>
                    
                    <!-- Tab 3: Retrieved Examples -->
                    <div id="examples-tab" class="tab-content">
                        <div class="examples-container">
                            <div id="examples-display">Click on a node to see retrieved examples...</div>
                        </div>
                    </div>
                    
                    <!-- Tab 4: Node Output -->
                    <div id="output-tab" class="tab-content">
                        <div class="output-container">
                            <div id="output-display" class="output-display">Click on a node to see its output...</div>
                        </div>
                    </div>
                    
                    <!-- Tab 5: Tree Statistics -->
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
                                    <strong>True Answer:</strong><br>{true_answer}<br><br>
                                    <strong>Image Path:</strong><br>{escaped_image_path}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div id="graph-container">
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

        // --- State Variables ---
        let currentSelectedNode = null;
        let showFullPrompt = false;
        let currentRolloutData = null;
        let currentTurn = 0;

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

        // --- Image Zoom & Pan State ---
        let scale = 1, panX = 0, panY = 0;
        let isPanning = false, startPan = {{x: 0, y: 0}};

        // --- vis.js Network ---
        const network = new vis.Network(
            document.getElementById('mcts-graph'),
            {{ nodes: new vis.DataSet(nodesData), edges: new vis.DataSet(edgesData) }},
            {{
                layout: {{ hierarchical: {{ direction: "UD", sortMethod: "directed", levelSeparation: 150, nodeSpacing: 220 }} }},
                interaction: {{ dragNodes: true, dragView: true, hover: true, zoomView: true, tooltipDelay: 300 }},
                physics: {{ enabled: false }},
            }}
        );

        // --- Functions ---
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
            const containerRatio = imageContainer.clientWidth / imageContainer.clientHeight;
            const imageRatio = taskImage.naturalWidth / taskImage.naturalHeight;
            
            let newScale;
            if (containerRatio > imageRatio) {{
                newScale = imageContainer.clientHeight / taskImage.naturalHeight;
            }} else {{
                newScale = imageContainer.clientWidth / taskImage.naturalWidth;
            }}
            scale = newScale;
            panX = (imageContainer.clientWidth - taskImage.naturalWidth * scale) / 2;
            panY = (imageContainer.clientHeight - taskImage.naturalHeight * scale) / 2;
            
            transformWrapper.style.width = taskImage.naturalWidth + 'px';
            transformWrapper.style.height = taskImage.naturalHeight + 'px';
            highlightCanvas.width = taskImage.naturalWidth;
            highlightCanvas.height = taskImage.naturalHeight;
            
            updateImageTransform();
            taskImage.style.display = 'block';
            dropZone.classList.add('hidden');
        }}

        function drawHighlight(coords) {{
            if (!coords || !taskImage.complete || taskImage.naturalWidth === 0) return;
            const [x, y] = coords;
            const radius = 50;

            ctx.clearRect(0, 0, highlightCanvas.width, highlightCanvas.height);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
            ctx.fillRect(0, 0, highlightCanvas.width, highlightCanvas.height);
            
            ctx.globalCompositeOperation = 'destination-out';
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
            ctx.globalCompositeOperation = 'source-over';
        }}

        function handleFile(file) {{
            if (file && file.type.startsWith('image/')) {{
                const reader = new FileReader();
                reader.onload = (e) => {{ taskImage.src = e.target.result; }};
                reader.readAsDataURL(file);
            }}
        }}

        function updatePromptDisplay(nodeData) {{
            if (!nodeData) return;
            
            let promptText = nodeData.data || "No prompt available";
            
            if (!showFullPrompt) {{
                // Extract basic prompt by removing retrieved examples section
                if (promptText.includes("--- Example") && promptText.includes("--- End Example")) {{
                    const lines = promptText.split('\\n');
                    let basicPromptLines = [];
                    let inExample = false;
                    let examplesFound = false;
                    
                    for (let i = 0; i < lines.length; i++) {{
                        const line = lines[i];
                        const lineStripped = line.trim();
                        
                        if (lineStripped.startsWith('--- Example') && lineStripped.endsWith('---')) {{
                            inExample = true;
                            examplesFound = true;
                            if (!basicPromptLines.some(l => l.includes('[Examples section hidden]'))) {{
                                basicPromptLines.push('');
                                basicPromptLines.push('[Retrieved examples section hidden - click "Show Full Prompt" to view]');
                                basicPromptLines.push('');
                            }}
                            continue;
                        }}
                        
                        if (lineStripped.startsWith('--- End Example') && lineStripped.endsWith('---')) {{
                            inExample = false;
                            continue;
                        }}
                        
                        if (!inExample) {{
                            basicPromptLines.push(line);
                        }}
                    }}
                    
                    if (examplesFound) {{
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
                        promptText = beforeExamples + '\\n\\n[Retrieved examples section hidden - click "Show Full Prompt" to view]\\n\\n' + afterExamples;
                    }} else {{
                        promptText = beforeExamples + '\\n\\n[Retrieved examples section hidden - click "Show Full Prompt" to view]';
                    }}
                }}
            }}
            
            promptDisplay.textContent = promptText;
        }}

        function updateExamplesDisplay(nodeData) {{
            if (!nodeData || !nodeData.retrieved_examples || nodeData.retrieved_examples.length === 0) {{
                examplesDisplay.innerHTML = '<p>No retrieved examples for this node.</p>';
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
            
            examplesDisplay.innerHTML = html;
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

        function updateOutputDisplay(nodeData) {{
            if (!nodeData) {{
                outputDisplay.textContent = "No node selected";
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
                
                if (node.rollouts && node.rollouts.length > 0) {{
                    output += `\\nRollouts (${{node.rollouts.length}}):\\n`;
                    node.rollouts.forEach((rollout, i) => {{
                        output += `  Rollout ${{i+1}}: ${{rollout.final_answer}} (Reward: ${{rollout.reward}})\\n`;
                    }});
                }}
                
                outputDisplay.textContent = output;
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
                
                outputDisplay.textContent = output;
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

            // Calculate the image coordinates relative to the original image
            const rect = imageContainer.getBoundingClientRect();
            const containerX = clientX - rect.left;
            const containerY = clientY - rect.top;
            
            // Convert container coordinates to image coordinates accounting for pan and scale
            const imageX = (containerX - panX) / scale;
            const imageY = (containerY - panY) / scale;
            
            // Ensure coordinates are within image bounds
            if (imageX >= 0 && imageX <= taskImage.naturalWidth && imageY >= 0 && imageY <= taskImage.naturalHeight) {{
                const roundedX = Math.round(imageX);
                const roundedY = Math.round(imageY);
                
                debugCoordsDiv.textContent = `(${{roundedX}}, ${{roundedY}})`;
                debugCoordsDiv.style.left = (clientX + 10) + 'px';
                debugCoordsDiv.style.top = (clientY - 30) + 'px';
                debugCoordsDiv.style.display = 'block';
            }} else {{
                debugCoordsDiv.style.display = 'none';
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
        }});

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
        network.on("click", ({{nodes}}) => {{
            if (!nodes.length) return;
            
            const nodeId = nodes[0];
            const data = interactiveDataMap[nodeId];
            currentSelectedNode = data;
            
            if (data?.type === 'node_prompt') {{
                updatePromptDisplay(data);
                updateExamplesDisplay(data);
                updateOutputDisplay(data);
                drawHighlight(data.coords);
            }} else if (data?.type === 'rollout_prompts') {{
                showRolloutModal(data);
                updateOutputDisplay(data);
                drawHighlight(data.coords);
            }}
        }});

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
    with open(f"{output_filename}.html", 'w', encoding='utf-8') as f:
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
        f"Total Search Time: {tree_data.get('global_search_time', 'N/A'):.2f}s\n\n"
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
    stats_output_path = f"{output_filename}_stats.txt"
    with open(stats_output_path, 'w', encoding='utf-8') as f:
        f.write(stats_text_file)
    print(f"Statistics saved to {os.path.abspath(stats_output_path)}")

def main():
    VERSION = "19-URL-FIXED"
    print(f"🚀 Enhanced MCTS Visualizer v{VERSION} - Starting visualization...")
    
    parser = argparse.ArgumentParser(description='Visualize MCTS search data as an enhanced interactive HTML file with tabbed interface and retrieval support.')
    parser.add_argument('input_file', type=str, help='Path to the MCTS rollout JSONL file.')
    parser.add_argument('--output_filename', type=str, default='mcts_visualization_enhanced', help='Name of the output file (without extension).')
    parser.add_argument('--max_depth', type=int, default=None, help='Maximum depth of the tree to visualize.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to show coordinates on hover.')
    
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line:
                print("Error: File is empty.")
                return
            rollout_data = json.loads(first_line)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error reading or parsing the input file '{args.input_file}': {e}")
        return

    # 1. Compute overall statistics
    stats = get_tree_statistics(rollout_data)
    print("Successfully computed tree statistics.")

    # 2. Build the graph data for vis.js
    graph_data = build_graph_data(rollout_data, args.max_depth)
    print(f"Processed {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges for the graph.")

    # 3. Generate the final HTML file
    generate_html_file(rollout_data, graph_data, stats, args.output_filename, args.debug)
    print(f"✅ Enhanced interactive visualization v{VERSION} saved to {os.path.abspath(args.output_filename)}.html")

if __name__ == '__main__':
    main()