#!/usr/bin/env python3
"""
Tree Depth vs Success Rate Analysis for MCTS Experiments

This tool analyzes how tree depth (longest path from root to action node)
correlates with success rates across MCTS experiments.
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def is_thinking_node(node):
    """Check if a node is a thinking node (contains <think> tags)."""
    thought_text = node.get('thought_text', '')
    return '<think>' in thought_text and '</think>' in thought_text


def is_action_node(node):
    """Check if a node is an action node (not a thinking node and not root)."""
    thought_text = node.get('thought_text', '')

    # Root nodes typically contain "OBJECTIVE:" - not action nodes
    if 'OBJECTIVE:' in thought_text:
        return False

    # If it has <think> tags, it's a thinking node
    if is_thinking_node(node):
        return False

    # Check for action keywords
    action_keywords = ['click', 'scroll', 'type', 'go_back', 'stop', '<answer>']
    for keyword in action_keywords:
        if keyword in thought_text.lower():
            return True

    # If it's a leaf node (no children) and not a thinking node, consider it an action node
    if not node.get('children', []):
        return True

    return False


def get_max_depth_to_action(node, current_depth=0):
    """
    Calculate maximum depth from current node to any action node.
    Returns (max_depth, found_action_node).
    """
    # Check if current node is an action node
    if is_action_node(node):
        return current_depth, True

    # If no children, return current depth
    children = node.get('children', [])
    if not children:
        return current_depth, False

    # Recursively check children
    max_depth = current_depth
    found_any_action = False

    for child in children:
        child_depth, found_action = get_max_depth_to_action(child, current_depth + 1)
        if found_action or child_depth > max_depth:
            max_depth = child_depth
            if found_action:
                found_any_action = True

    return max_depth, found_any_action


def has_successful_rollout(tree, threshold=1.0):
    """
    Check if the tree has at least one successful rollout.
    Success is defined as reward >= threshold.
    """
    def check_node(node):
        # Check rollouts in current node
        for rollout in node.get('rollouts', []):
            if rollout.get('reward', 0) >= threshold:
                return True

        # Check children
        for child in node.get('children', []):
            if check_node(child):
                return True

        return False

    return check_node(tree)


def analyze_jsonl_file(filepath, threshold=1.0):
    """
    Analyze a single JSONL file.
    Returns (max_depth, has_success) or None if error.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            if not first_line.strip():
                return None

            data = json.loads(first_line)
            tree = data.get('tree')

            if not tree:
                return None

            # Get max depth to action node
            max_depth, found_action = get_max_depth_to_action(tree)

            # Check for successful rollout
            has_success = has_successful_rollout(tree, threshold)

            return max_depth, has_success

    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Error reading {filepath}: {e}", file=sys.stderr)
        return None


def analyze_experiment(experiment_dir, threshold=1.0):
    """
    Analyze all JSONL files in an experiment directory.
    Returns dict mapping depth -> (total_count, success_count).
    """
    depth_stats = defaultdict(lambda: {'total': 0, 'success': 0})

    # Find all JSONL files
    jsonl_files = list(Path(experiment_dir).glob('*.jsonl'))

    for filepath in jsonl_files:
        result = analyze_jsonl_file(filepath, threshold)
        if result:
            depth, has_success = result
            depth_stats[depth]['total'] += 1
            if has_success:
                depth_stats[depth]['success'] += 1

    return depth_stats


def print_experiment_stats(experiment_name, depth_stats):
    """Print statistics for a single experiment."""
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*80}")

    if not depth_stats:
        print("No data found.")
        return

    # Sort by depth
    sorted_depths = sorted(depth_stats.keys())

    print(f"\n{'Depth':<10} {'Total':<10} {'Successful':<15} {'Success Rate':<15}")
    print('-' * 55)

    for depth in sorted_depths:
        stats = depth_stats[depth]
        total = stats['total']
        success = stats['success']
        rate = (success / total * 100) if total > 0 else 0

        print(f"{depth:<10} {total:<10} {success:<15} {rate:>6.2f}%")

    # Print summary
    total_instances = sum(s['total'] for s in depth_stats.values())
    total_success = sum(s['success'] for s in depth_stats.values())
    overall_rate = (total_success / total_instances * 100) if total_instances > 0 else 0

    print('-' * 55)
    print(f"{'TOTAL':<10} {total_instances:<10} {total_success:<15} {overall_rate:>6.2f}%")


def aggregate_stats(all_experiment_stats):
    """Aggregate statistics across all experiments."""
    aggregated = defaultdict(lambda: {'total': 0, 'success': 0})

    for exp_stats in all_experiment_stats.values():
        for depth, stats in exp_stats.items():
            aggregated[depth]['total'] += stats['total']
            aggregated[depth]['success'] += stats['success']

    return aggregated


def plot_experiment_histogram(experiment_name, depth_stats, output_path):
    """Generate and save a histogram for a single experiment."""
    if not depth_stats:
        return

    sorted_depths = sorted(depth_stats.keys())
    success_rates = []

    for depth in sorted_depths:
        stats = depth_stats[depth]
        rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        success_rates.append(rate)

    # Create figure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_depths, success_rates, color='steelblue', alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for i, (depth, rate) in enumerate(zip(sorted_depths, success_rates)):
        stats = depth_stats[depth]
        plt.text(depth, rate + 2, f'{rate:.1f}%\n({stats["success"]}/{stats["total"]})',
                ha='center', va='bottom', fontsize=9)

    plt.xlabel('Tree Depth', fontsize=12, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
    plt.title(f'Success Rate by Tree Depth\n{experiment_name}', fontsize=14, fontweight='bold')
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved histogram: {output_path}")


def plot_aggregated_histogram(aggregated_stats, output_path):
    """Generate and save a histogram for aggregated data across all experiments."""
    if not aggregated_stats:
        return

    sorted_depths = sorted(aggregated_stats.keys())
    success_rates = []

    for depth in sorted_depths:
        stats = aggregated_stats[depth]
        rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
        success_rates.append(rate)

    # Create figure
    plt.figure(figsize=(12, 7))
    bars = plt.bar(sorted_depths, success_rates, color='forestgreen', alpha=0.8, edgecolor='black')

    # Add value labels on bars
    for i, (depth, rate) in enumerate(zip(sorted_depths, success_rates)):
        stats = aggregated_stats[depth]
        plt.text(depth, rate + 2, f'{rate:.1f}%\n({stats["success"]}/{stats["total"]})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.xlabel('Tree Depth', fontsize=14, fontweight='bold')
    plt.ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    plt.title('Success Rate by Tree Depth (Aggregated Across All Experiments)',
             fontsize=16, fontweight='bold')
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved aggregated histogram: {output_path}")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze tree depth vs success rate correlation in MCTS experiments.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python depth_analysis.py ../compilation --output-dir results --threshold 1.0
  python depth_analysis.py /path/to/experiments -o my_output -t 0.5
        """
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Path to directory containing experiment subdirectories (each with JSONL files)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='depth_analysis_output',
        help='Output directory for histograms and reports (default: depth_analysis_output)'
    )
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=1.0,
        help='Success reward threshold (default: 1.0)'
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    threshold = args.threshold

    # Validate input directory
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Using success threshold: reward >= {threshold}")

    # Get all experiment directories
    experiment_dirs = [d for d in input_dir.iterdir() if d.is_dir()]

    if not experiment_dirs:
        print(f"Error: No experiment directories found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(experiment_dirs)} experiment directories in {input_dir}")

    # Analyze each experiment
    all_stats = {}
    for exp_dir in sorted(experiment_dirs):
        exp_name = exp_dir.name
        print(f"Analyzing {exp_name}...", end='', flush=True)

        depth_stats = analyze_experiment(exp_dir, threshold)
        all_stats[exp_name] = depth_stats

        total = sum(s['total'] for s in depth_stats.values())
        print(f" {total} instances")

    # Redirect stdout to capture output for text file
    import io
    text_output = io.StringIO()
    original_stdout = sys.stdout

    try:
        # Duplicate output to both console and string buffer
        class TeeOutput:
            def __init__(self, *outputs):
                self.outputs = outputs

            def write(self, text):
                for output in self.outputs:
                    output.write(text)

            def flush(self):
                for output in self.outputs:
                    output.flush()

        sys.stdout = TeeOutput(original_stdout, text_output)

        # Print per-experiment results
        print(f"\n{'#'*80}")
        print(f"# PER-EXPERIMENT RESULTS")
        print(f"{'#'*80}")

        for exp_name in sorted(all_stats.keys()):
            print_experiment_stats(exp_name, all_stats[exp_name])

        # Print aggregated results
        print(f"\n{'#'*80}")
        print(f"# OVERALL STATISTICS (Across All Experiments)")
        print(f"{'#'*80}")

        aggregated = aggregate_stats(all_stats)
        print_experiment_stats("ALL EXPERIMENTS", aggregated)

        # Print depth correlation insights
        print(f"\n{'#'*80}")
        print(f"# DEPTH CORRELATION INSIGHTS")
        print(f"{'#'*80}\n")

        sorted_depths = sorted(aggregated.keys())
        if sorted_depths:
            print("Success rate by depth (aggregated):")
            for depth in sorted_depths:
                stats = aggregated[depth]
                rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
                bar_length = int(rate / 2)  # Scale to 50 chars max
                bar = '█' * bar_length
                print(f"  Depth {depth:2d}: {bar} {rate:6.2f}% ({stats['success']}/{stats['total']})")

            # Calculate correlation trend
            depth_rates = [(d, (aggregated[d]['success'] / aggregated[d]['total'] * 100)
                           if aggregated[d]['total'] > 0 else 0) for d in sorted_depths]

            if len(depth_rates) > 1:
                # Simple trend analysis
                first_half_avg = sum(r for _, r in depth_rates[:len(depth_rates)//2]) / (len(depth_rates)//2)
                second_half_avg = sum(r for _, r in depth_rates[len(depth_rates)//2:]) / (len(depth_rates) - len(depth_rates)//2)

                print(f"\nTrend Analysis:")
                print(f"  Shallow depths (first half) avg: {first_half_avg:.2f}%")
                print(f"  Deeper depths (second half) avg: {second_half_avg:.2f}%")

                if second_half_avg > first_half_avg + 5:
                    print(f"  → Positive correlation: Deeper trees show HIGHER success rates (+{second_half_avg - first_half_avg:.2f}%)")
                elif first_half_avg > second_half_avg + 5:
                    print(f"  → Negative correlation: Deeper trees show LOWER success rates (-{first_half_avg - second_half_avg:.2f}%)")
                else:
                    print(f"  → Weak/no correlation: Similar success rates across depths")

    finally:
        sys.stdout = original_stdout

    # Save text report
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(text_output.getvalue())
    print(f"\n{'='*80}")
    print(f"Saved text report: {report_path}")

    # Generate histograms
    print(f"\n{'='*80}")
    print("Generating histograms...")
    print(f"{'='*80}")

    for exp_name in sorted(all_stats.keys()):
        output_path = output_dir / f'{exp_name}_depth_histogram.png'
        plot_experiment_histogram(exp_name, all_stats[exp_name], output_path)

    # Generate aggregated histogram
    aggregated_path = output_dir / 'aggregated_depth_histogram.png'
    plot_aggregated_histogram(aggregated, aggregated_path)

    print(f"\n{'='*80}")
    print(f"Analysis complete! All outputs saved to: {output_dir.absolute()}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
