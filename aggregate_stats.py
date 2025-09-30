# aggregate_stats.py

import argparse
import json
import os
import numpy as np
from collections import Counter
import html


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
            # The original best_reward logic was here, but it was incorrect as it only
            # checked terminal nodes. It has been moved to the general 'rollouts' block below.

        if node.get('visit_count', 0) > stats['max_visit_count']:
            stats['max_visit_count'] = node['visit_count']
            stats['most_visited_node_text'] = node.get('thought_text', 'N/A')

        if 'rollouts' in node:
            num_rollouts = len(node['rollouts'])
            stats['total_rollouts'] += num_rollouts
            
            # --- Corrected Best Reward Logic ---
            # This now correctly checks rollouts from ALL nodes, not just terminal ones.
            rewards = [r.get('reward') for r in node.get('rollouts', []) if r.get('reward') is not None]
            if rewards:
                max_reward_in_node = max(rewards)
                if max_reward_in_node > stats['best_reward']:
                    stats['best_reward'] = max_reward_in_node
                    stats['best_reward_node_count'] = 1
                elif max_reward_in_node == stats['best_reward'] and stats['best_reward'] != -1.0:
                    stats['best_reward_node_count'] += 1
            # ------------------------------------

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


def aggregate_statistics(folder_path):
    """
    Scans a folder for MCTS rollout JSONL files, aggregates statistics,
    and writes a summary report.
    """
    master_stats = {
        'total_files': 0,
        'list_total_nodes': [],
        'list_max_depth': [],
        'list_terminal_nodes': [],
        'list_total_rollouts': [],
        'list_successful_rollouts_0': [],
        'list_successful_rollouts_05': [],
        'list_successful_rollouts_075': [],
        'list_successful_rollouts_1': [],
        'list_search_time': [],
        'list_best_reward': [],
        'action_distribution': Counter(),
        'list_success_rate_0': [],
        'list_success_rate_05': [],
        'list_success_rate_075': [],
        'list_success_rate_1': [],
    }

    print(f"Scanning folder: {folder_path}")
    
    try:
        filenames = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"Error: Folder not found at '{folder_path}'")
        return

    for filename in filenames:
        if filename.endswith(('.json', '.jsonl')) and filename != 'args.json':
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if not first_line:
                        print(f"Warning: Skipping empty file '{filename}'")
                        continue
                    data = json.loads(first_line)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not read or parse '{filename}'. Skipping. Error: {e}")
                continue

            stats = get_tree_statistics(data)
            
            master_stats['total_files'] += 1
            master_stats['list_total_nodes'].append(stats['total_nodes'])
            master_stats['list_max_depth'].append(stats['max_depth'])
            master_stats['list_terminal_nodes'].append(stats['terminal_nodes'])
            master_stats['list_total_rollouts'].append(stats['total_rollouts'])
            master_stats['list_successful_rollouts_0'].append(stats['successful_rollouts_0'])
            master_stats['list_successful_rollouts_05'].append(stats['successful_rollouts_05'])
            master_stats['list_successful_rollouts_075'].append(stats['successful_rollouts_075'])
            master_stats['list_successful_rollouts_1'].append(stats['successful_rollouts_1'])
            
            # Only append the best reward if it was actually found (i.e., not the initial -1.0)
            if stats['best_reward'] != -1.0:
                master_stats['list_best_reward'].append(stats['best_reward'])
            
            if 'global_search_time' in data:
                master_stats['list_search_time'].append(data['global_search_time'])

            master_stats['action_distribution'].update(stats['action_distribution'])
            
            if stats['total_rollouts'] > 0:
                master_stats['list_success_rate_0'].append(stats['successful_rollouts_0'] / stats['total_rollouts'])
                master_stats['list_success_rate_05'].append(stats['successful_rollouts_05'] / stats['total_rollouts'])
                master_stats['list_success_rate_075'].append(stats['successful_rollouts_075'] / stats['total_rollouts'])
                master_stats['list_success_rate_1'].append(stats['successful_rollouts_1'] / stats['total_rollouts'])

    if master_stats['total_files'] == 0:
        print("No valid JSON/JSONL files found to aggregate.")
        return

    # --- Generate Report ---
    num_files = master_stats['total_files']
    report = []
    report.append("--- Aggregate MCTS Statistics ---")
    report.append(f"Data from {num_files} file(s) in: {os.path.abspath(folder_path)}\n")

    def get_stats_str(name, data_list, is_integer_total=False):
        if not data_list:
            return f"{name}: N/A (no data)"
        mean = np.mean(data_list)
        std = np.std(data_list)
        total = np.sum(data_list)
        total_str = f"{int(total)}" if is_integer_total else f"{total:.2f}"
        return f"{name}: Mean={mean:.2f}, StdDev={std:.2f}, Total={total_str}"

    def get_rate_stats_str(name, rate_list):
        if not rate_list:
            return f"{name}: N/A (no data)"
        mean_rate = np.mean(rate_list) * 100
        std_rate = np.std(rate_list) * 100
        return f"{name}: Mean={mean_rate:.2f}%, StdDev={std_rate:.2f}%"

    report.append("--- Search Performance ---")
    report.append(get_stats_str("Search Time (s)", master_stats['list_search_time']))
    report.append("")

    report.append("--- Tree Structure ---")
    report.append(get_stats_str("Total Nodes", master_stats['list_total_nodes'], is_integer_total=True))
    report.append(get_stats_str("Max Depth", master_stats['list_max_depth'], is_integer_total=True))
    report.append(get_stats_str("Terminal Nodes", master_stats['list_terminal_nodes'], is_integer_total=True))
    report.append("")

    report.append("--- Rollout Counts ---")
    report.append(get_stats_str("Total Rollouts", master_stats['list_total_rollouts'], is_integer_total=True))
    report.append(get_stats_str("Success (R > 0)", master_stats['list_successful_rollouts_0'], is_integer_total=True))
    report.append(get_stats_str("Success (R >= 0.5)", master_stats['list_successful_rollouts_05'], is_integer_total=True))
    report.append(get_stats_str("Success (R >= 0.75)", master_stats['list_successful_rollouts_075'], is_integer_total=True))
    report.append(get_stats_str("Success (R == 1)", master_stats['list_successful_rollouts_1'], is_integer_total=True))
    report.append("")

    report.append("--- Success Rates (per file) ---")
    report.append(get_rate_stats_str("Success Rate (R > 0)", master_stats['list_success_rate_0']))
    report.append(get_rate_stats_str("Success Rate (R >= 0.5)", master_stats['list_success_rate_05']))
    report.append(get_rate_stats_str("Success Rate (R >= 0.75)", master_stats['list_success_rate_075']))
    report.append(get_rate_stats_str("Success Rate (R == 1)", master_stats['list_success_rate_1']))
    report.append("")
    
    report.append("--- Rewards ---")
    report.append(get_stats_str("Best Reward in Tree", master_stats['list_best_reward']))
    report.append("")

    report.append("--- Action Distribution (Total Counts) ---")
    total_actions = sum(master_stats['action_distribution'].values())
    if total_actions > 0:
        for action, count in master_stats['action_distribution'].most_common():
            percentage = (count / total_actions) * 100
            report.append(f"  - {action}: {count} ({percentage:.2f}%)")
    else:
        report.append("  No actions recorded.")
    report.append("")

    # --- Write to file ---
    output_path = os.path.join(folder_path, 'aggregate_summary.txt')
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report))
        print(f"Successfully generated aggregate report at: {output_path}")
    except IOError as e:
        print(f"Error writing report to file: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate aggregate statistics from multiple MCTS rollout JSONL files in a folder.'
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to the folder containing the MCTS rollout JSONL files.'
    )
    args = parser.parse_args()
    
    aggregate_statistics(args.folder_path)

if __name__ == '__main__':
    main() 