#!/usr/bin/env python3
"""
MCTS Experiment Comparison Tool

A comprehensive statistical analysis tool for comparing Monte Carlo Tree Search (MCTS) 
experiments with rigorous significance testing, compute scaling analysis, and interactive 
visualization capabilities.

====================
CORE FEATURES
====================

🔬 Statistical Analysis:
    - Independent samples t-test (parametric)
    - Mann-Whitney U test (non-parametric)
    - Wilcoxon rank-sum test (non-parametric)
    - Cohen's d effect size calculations
    - Multiple comparison corrections

📊 Visualization:
    - Interactive HTML reports with explanations
    - Line graphs for compute scaling analysis
    - Success rate comparisons across reward thresholds
    - Tree efficiency metrics visualization

🧩 Flexible Experiment Organization:
    - Automatic experiment grouping from folder names
    - Manual experiment grouping for ambiguous names
    - Baseline experiment designation
    - Series analysis for compute scaling studies

📈 Comprehensive Metrics:
    - Success rates at multiple thresholds (0, 0.5, 0.75, 1.0)
    - Search efficiency (rollouts per node, terminal ratios)
    - Tree exploration (depth, node counts, branching factors)
    - Timing and computational cost analysis

====================
USAGE PATTERNS
====================

1. BASIC COMPARISON (2 experiments):
   python experiment_comparison.py exp1/ exp2/ --output comparison.html

2. MULTIPLE EXPERIMENTS:
   python experiment_comparison.py baseline/ method1/ method2/ method3/
   
3. BASELINE COMPARISON:
   python experiment_comparison.py baseline/ exp1/ exp2/ --baseline baseline

4. SERIES ANALYSIS (compute scaling):
   python experiment_comparison.py ExpA_1x/ ExpA_2x/ ExpA_4x/ ExpB_1x/ ExpB_2x/ ExpB_4x/ --series-mode

5. MANUAL GROUPING (for ambiguous folder names):
   python experiment_comparison.py run1/ run2/ run3/ run4/ run5/ run6/ \\
       --groups "ExpA,ExpA,ExpA,ExpB,ExpB,ExpB" \\
       --x-axis "1000,2000,4000,1000,2000,4000" \\
       --series-mode

6. CUSTOM X-AXIS VALUES:
   python experiment_comparison.py exp1_500/ exp1_1000/ exp2_500/ exp2_1000/ \\
       --series-mode --x-axis "500,1000,500,1000"

====================
COMMAND LINE OPTIONS
====================

Required Arguments:
    experiment_dirs         Two or more experiment directories containing JSONL files

Optional Arguments:
    --output FILE           Output filename (auto-detects format from extension)
    --format {html,json}    Force output format (default: html)
    --series-mode           Enable compute scaling analysis with line graphs
    --groups LIST           Manual experiment grouping (comma-separated)
                           Example: "ExpA,ExpA,ExpB,ExpB"
    --x-axis VALUES         Custom x-axis values (comma-separated numbers)
                           Example: "1000,2000,3000,1000,2000,3000"
    --baseline NAME         Specify baseline experiment for comparisons
    --statistical-tests     Enable statistical significance testing (default: True)
    --verbose              Enable detailed progress output

====================
AUTOMATIC NAMING PATTERNS
====================

The tool automatically detects experiment series from folder names:

✅ Supported Patterns:
    - ExperimentName_2x, ExperimentName_4x (multiplier format)
    - ExperimentName_compute_1000, ExperimentName_compute_2000 
    - ExperimentName_steps_1000, ExperimentName_rollouts_500
    - Method_1x, Method_2x, Method_4x

❌ Ambiguous Names (use --groups):
    - run1/, run2/, run3/ (unclear which experiment)
    - config_v1/, config_v2/ (no compute indication)
    - baseline_trial1/, method2_test3/ (mixed naming)

====================
MANUAL GROUPING GUIDE
====================

When folder names don't clearly indicate experiment boundaries:

1. List directories in order: dir1/ dir2/ dir3/ dir4/ dir5/ dir6/

2. Specify groups in same order: --groups "A,A,A,B,B,B"

3. Optional x-axis values: --x-axis "1000,2000,4000,1000,2000,4000"

Example:
    Folders: baseline_run47/ baseline_run82/ method2_v3/ method2_v7/ method2_v12/
    Groups:  --groups "Baseline,Baseline,Method2,Method2,Method2"
    X-axis:  --x-axis "1000,2000,1000,2000,4000"

====================
OUTPUT FORMATS
====================

📄 HTML Report (default):
    - Interactive tabs: Overview, Series Analysis, Metrics, Statistical Tests, Visualizations
    - Line graphs for compute scaling (if --series-mode)
    - Statistical test explanations and interpretations
    - Metric descriptions and recommendations
    - Export-ready visualizations

📋 JSON Report:
    - Machine-readable format for further analysis
    - Complete statistical test results
    - Raw data and computed metrics
    - Programmatic access to all comparisons

====================
METRICS EXPLAINED
====================

🎯 Success Rate Metrics:
    - success_rate_0: % rollouts with any positive reward (> 0)
    - success_rate_05: % rollouts with good performance (≥ 0.5)
    - success_rate_075: % rollouts with very good performance (≥ 0.75)
    - success_rate_1: % rollouts with perfect performance (= 1.0)

🌳 Tree Search Metrics:
    - total_nodes: Total nodes explored in search tree
    - total_rollouts: Total rollout simulations performed
    - max_depth: Maximum depth reached in search
    - terminal_nodes: Number of leaf nodes (endpoints)

⚡ Efficiency Metrics:
    - rollouts_per_node: Search intensity per node
    - nodes_per_depth: Average branching factor
    - terminal_ratio: Proportion of terminal nodes
    - search_time: Total computation time (when available)

====================
STATISTICAL TESTS
====================

🧪 Tests Performed:
    - t-test: Compares means assuming normality
    - Mann-Whitney U: Non-parametric alternative to t-test
    - Wilcoxon rank-sum: Equivalent to Mann-Whitney
    - Cohen's d: Standardized effect size measure

📊 Interpretations:
    - p < 0.05: Statistically significant difference
    - Cohen's d: 0.2=small, 0.5=medium, 0.8=large effect
    - Effect sizes more important than p-values for practical significance

⚠️ Multiple Comparisons:
    - When testing many metrics simultaneously, consider:
    - Bonferroni correction: α = 0.05 / number_of_tests
    - Focus on effect sizes rather than just p-values
    - Interpret results in context of practical importance

====================
TROUBLESHOOTING
====================

❗ Common Issues:

1. "No JSONL files found":
   - Check directory paths are correct
   - Ensure JSONL files exist in each directory
   - Use absolute paths if relative paths fail

2. "Number of groups must match experiments":
   - Count your directories: ls dir1/ dir2/ dir3/
   - Provide same number of group names
   - Check for typos in --groups argument

3. "Insufficient data for comparison":
   - Need at least 2 problems per experiment
   - Check JSONL files contain valid data
   - Verify experiment directories have results

4. "Could not parse series info":
   - Use --groups for manual grouping
   - Or rename directories to follow supported patterns
   - Enable --verbose to see parsing attempts

====================
ADVANCED EXAMPLES
====================

🔬 Research Study with Multiple Methods:
python experiment_comparison.py \\
    baseline_mcts/ improved_mcts_v1/ improved_mcts_v2/ neural_mcts/ \\
    --baseline baseline_mcts \\
    --output research_comparison.html \\
    --verbose

📈 Compute Scaling Analysis:
python experiment_comparison.py \\
    method1_1000steps/ method1_2000steps/ method1_4000steps/ \\
    method2_1000steps/ method2_2000steps/ method2_4000steps/ \\
    --series-mode \\
    --x-axis "1000,2000,4000,1000,2000,4000" \\
    --output scaling_study.html

🧩 Ambiguous Folder Names:
python experiment_comparison.py \\
    run_batch1_config47/ run_batch2_config82/ run_batch3_config91/ \\
    experiment_trial3_v2/ experiment_trial7_v2/ experiment_trial12_v2/ \\
    --groups "Baseline,Baseline,Baseline,NewMethod,NewMethod,NewMethod" \\
    --x-axis "500,1000,2000,500,1000,2000" \\
    --series-mode \\
    --baseline Baseline \\
    --output method_comparison.html

Author: AI Assistant
Version: 2.1
Python: >=3.7
Dependencies: numpy, scipy, json, pathlib, argparse
License: MIT
"""

import argparse
import html
import json
import os
import sys
from pathlib import Path
from collections import defaultdict, Counter
import statistics
from typing import Dict, List, Tuple, Any, Optional
import math

# Statistical libraries
try:
    import numpy as np
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Some statistical tests will be skipped.")

# Add the src directory to the path to import visualizer functions
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
try:
    from vlmsearch.visualization.visualizer_enhanced import get_tree_statistics
except ImportError:
    print("Error: Cannot import visualizer_enhanced. Make sure the src directory is accessible.")
    sys.exit(1)


class ExperimentComparison:
    """Main class for comparing experiments"""
    
    def __init__(self, experiment_dirs: List[str], verbose: bool = False, x_axis_values: List[float] = None, series_mode: bool = False, baseline_exp: str = None, groups: List[str] = None):
        self.experiment_dirs = experiment_dirs
        self.verbose = verbose
        self.experiments = {}
        self.problem_data = defaultdict(dict)  # problem_id -> {exp_name: data}
        self.aggregate_stats = {}
        self.x_axis_values = x_axis_values  # Custom x-axis values for compute scaling
        self.series_mode = series_mode  # Whether to enable experiment series analysis
        self.experiment_series = {}  # Grouped experiments by series
        self.baseline_exp = baseline_exp  # Which experiment to use as baseline (if specified)
        self.groups = groups  # Manual experiment group assignments
        self.exp_name_to_global_index = {}  # Mapping from experiment name to global directory index
        
    def load_experiments(self):
        """Load all JSONL files from experiment directories"""
        print("Loading experiments...")
        
        for i, exp_dir in enumerate(self.experiment_dirs):
            exp_path = Path(exp_dir)
            if not exp_path.exists():
                print(f"Warning: Experiment directory {exp_dir} does not exist")
                continue
                
            exp_name = exp_path.name
            jsonl_files = list(exp_path.glob("*.jsonl"))
            
            if not jsonl_files:
                print(f"Warning: No JSONL files found in {exp_dir}")
                continue
                
            print(f"  Loading {exp_name}: {len(jsonl_files)} files")
            
            exp_data = {}
            for jsonl_file in jsonl_files:
                try:
                    with open(jsonl_file, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            data = json.loads(line)
                            # Extract problem identifier
                            problem_id = self._extract_problem_id(data, jsonl_file)
                            exp_data[problem_id] = data
                except Exception as e:
                    if self.verbose:
                        print(f"    Error loading {jsonl_file}: {e}")
                    continue
            
            self.experiments[exp_name] = exp_data
            self.exp_name_to_global_index[exp_name] = i  # Store mapping for x-axis indexing
            print(f"  Loaded {len(exp_data)} problems for {exp_name}")
            
        # If in series mode, group experiments by series
        if self.series_mode:
            if self.groups:
                self._group_experiments_manually()
            else:
                self._group_experiments_by_series()
    
    def _group_experiments_manually(self):
        """
        Group experiments based on manual group specifications.
        
        Uses the self.groups list to assign experiments to series.
        Groups should be provided in the same order as experiment directories.
        """
        print("Grouping experiments manually based on provided groups...")
        
        if len(self.groups) != len(self.experiment_dirs):
            raise ValueError(f"Number of groups ({len(self.groups)}) must match number of experiments ({len(self.experiment_dirs)})")
        
        exp_names = list(self.experiments.keys())
        
        # Create mapping from directory path to experiment name
        dir_to_exp = {}
        for exp_name in exp_names:
            # Find matching directory (experiment name is derived from directory)
            for exp_dir in self.experiment_dirs:
                if Path(exp_dir).name == exp_name:
                    dir_to_exp[exp_dir] = exp_name
                    break
        
        # Group experiments according to provided groups
        for i, (exp_dir, group_name) in enumerate(zip(self.experiment_dirs, self.groups)):
            exp_name = dir_to_exp.get(exp_dir)
            if not exp_name:
                print(f"  Warning: Could not find experiment for directory {exp_dir}")
                continue
                
            # Determine compute level
            if self.x_axis_values and i < len(self.x_axis_values):
                compute_level = self.x_axis_values[i]
            else:
                # Try to parse compute from name, fallback to index
                series_info = self._parse_experiment_series(exp_name)
                compute_level = series_info['compute'] if series_info else i + 1
            
            if group_name not in self.experiment_series:
                self.experiment_series[group_name] = []
            
            self.experiment_series[group_name].append({
                'name': exp_name,
                'compute': compute_level,
                'global_index': i  # Store global directory index for x-axis mapping
                # Don't store 'data' - it contains all raw experiment data with images
            })
        
        # Sort each series by compute level
        for series_name in self.experiment_series:
            self.experiment_series[series_name].sort(key=lambda x: x['compute'])
            print(f"  Series '{series_name}': {len(self.experiment_series[series_name])} experiments")
            if self.verbose:
                for exp in self.experiment_series[series_name]:
                    print(f"    - {exp['name']} (compute: {exp['compute']})")
    
    def _group_experiments_by_series(self):
        """Group experiments into series based on name patterns"""
        print("Grouping experiments by series...")
        
        # Parse experiment names to extract series and compute level
        for exp_name in self.experiments.keys():
            series_info = self._parse_experiment_series(exp_name)
            if series_info:
                series_name = series_info['series']
                compute_level = series_info['compute']
                
                if series_name not in self.experiment_series:
                    self.experiment_series[series_name] = []
                
                self.experiment_series[series_name].append({
                    'name': exp_name,
                    'compute': compute_level,
                    'global_index': self.exp_name_to_global_index.get(exp_name, 0)  # Get global directory index
                    # Don't store 'data' - it contains all raw experiment data
                })
        
        # Sort each series by compute level
        for series_name in self.experiment_series:
            self.experiment_series[series_name].sort(key=lambda x: x['compute'])
            print(f"  Series '{series_name}': {len(self.experiment_series[series_name])} experiments")
    
    def _parse_experiment_series(self, exp_name: str) -> dict:
        """Parse experiment name to extract series name and compute level"""
        import re
        
        # Pattern 1: ExperimentName_2x, ExperimentName_4x, etc.
        pattern1 = re.compile(r'^(.+?)_([0-9.]+)x?$')
        match1 = pattern1.match(exp_name)
        if match1:
            return {
                'series': match1.group(1),
                'compute': float(match1.group(2))
            }
        
        # Pattern 2: ExperimentName_compute_1000, ExperimentName_compute_2000, etc.
        pattern2 = re.compile(r'^(.+?)_compute[_-]?([0-9.]+)$')
        match2 = pattern2.match(exp_name)
        if match2:
            return {
                'series': match2.group(1),
                'compute': float(match2.group(2))
            }
        
        # Pattern 3: ExperimentName_steps_1000, ExperimentName_steps_2000, etc.
        pattern3 = re.compile(r'^(.+?)_steps[_-]?([0-9.]+)$')
        match3 = pattern3.match(exp_name)
        if match3:
            return {
                'series': match3.group(1),
                'compute': float(match3.group(2))
            }
        
        # Pattern 4: ExperimentName_rollouts_1000, etc.
        pattern4 = re.compile(r'^(.+?)_rollouts[_-]?([0-9.]+)$')
        match4 = pattern4.match(exp_name)
        if match4:
            return {
                'series': match4.group(1),
                'compute': float(match4.group(2))
            }
        
        # If no pattern matches, treat as single experiment
        if self.verbose:
            print(f"    Warning: Could not parse series info from '{exp_name}'")
        return None
    
    def _extract_problem_id(self, data: dict, jsonl_file: Path) -> str:
        """Extract a unique problem identifier from the data"""
        # Try to use question + image combination for uniqueness
        question = data.get('question', '')
        image = data.get('image', '')
        
        if question and image:
            # Create a hash of question + image path
            problem_key = f"{question[:100]}_{Path(image).name}"
            return problem_key
        
        # Fallback to filename if no question/image
        return jsonl_file.stem
    
    def compute_statistics(self):
        """Compute detailed statistics for each experiment"""
        print("Computing statistics...")
        
        for exp_name, exp_data in self.experiments.items():
            print(f"  Computing stats for {exp_name}")
            
            exp_stats = {
                'name': exp_name,
                'total_problems': len(exp_data),
                'problems': {},
                'aggregate': {}
            }
            
            all_metrics = defaultdict(list)
            
            for problem_id, data in exp_data.items():
                try:
                    # Use existing visualizer function to get tree statistics
                    tree_stats = get_tree_statistics(data)
                    
                    # Add additional metrics
                    metrics = self._compute_additional_metrics(data, tree_stats)
                    
                    # Store only numeric metrics, not raw data
                    numeric_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                    exp_stats['problems'][problem_id] = numeric_metrics
                    
                    # Collect for aggregate stats
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            all_metrics[key].append(value)
                            
                    # Store for cross-experiment comparison
                    self.problem_data[problem_id][exp_name] = metrics
                    
                except Exception as e:
                    if self.verbose:
                        print(f"    Error computing stats for {problem_id}: {e}")
                    continue
            
            # Compute aggregate statistics
            exp_stats['aggregate'] = self._compute_aggregate_stats(all_metrics)
            self.aggregate_stats[exp_name] = exp_stats
    
    def _compute_additional_metrics(self, data: dict, tree_stats: dict) -> dict:
        """Compute additional metrics beyond the standard tree statistics"""
        metrics = tree_stats.copy()
        
        # Add search efficiency metrics
        if tree_stats['total_rollouts'] > 0:
            metrics['rollouts_per_node'] = tree_stats['total_rollouts'] / max(tree_stats['total_nodes'], 1)
            metrics['success_rate_0'] = tree_stats['successful_rollouts_0'] / tree_stats['total_rollouts']
            metrics['success_rate_05'] = tree_stats['successful_rollouts_05'] / tree_stats['total_rollouts']
            metrics['success_rate_075'] = tree_stats['successful_rollouts_075'] / tree_stats['total_rollouts']
            metrics['success_rate_1'] = tree_stats['successful_rollouts_1'] / tree_stats['total_rollouts']
        else:
            metrics['rollouts_per_node'] = 0
            metrics['success_rate_0'] = 0
            metrics['success_rate_05'] = 0
            metrics['success_rate_075'] = 0
            metrics['success_rate_1'] = 0
        
        # Add search time if available
        metrics['search_time'] = data.get('global_search_time', 0)
        
        # Add tree efficiency metrics
        if tree_stats['max_depth'] > 0:
            metrics['nodes_per_depth'] = tree_stats['total_nodes'] / tree_stats['max_depth']
        else:
            metrics['nodes_per_depth'] = 0
            
        # Add terminal efficiency
        if tree_stats['total_nodes'] > 0:
            metrics['terminal_ratio'] = tree_stats['terminal_nodes'] / tree_stats['total_nodes']
        else:
            metrics['terminal_ratio'] = 0
            
        return metrics
    
    def _compute_aggregate_stats(self, metrics: dict) -> dict:
        """Compute aggregate statistics from individual problem metrics"""
        aggregate = {}
        
        for metric_name, values in metrics.items():
            if not values:
                continue
                
            aggregate[metric_name] = {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std': statistics.stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
            
        return aggregate
    
    def _create_lightweight_stats(self) -> dict:
        """Create a lightweight version of aggregate_stats without images or large data"""
        lightweight_stats = {}
        
        for exp_name, exp_data in self.aggregate_stats.items():
            lightweight_stats[exp_name] = {
                'name': exp_data['name'],
                'total_problems': exp_data['total_problems'],
                'aggregate': exp_data['aggregate']
                # Exclude 'problems' dict which contains all the raw data including images
            }
        
        return lightweight_stats
    
    def perform_statistical_comparisons(self) -> dict:
        """Perform statistical comparisons between experiments"""
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not available. Skipping statistical tests.")
            return {}
            
        print("Performing statistical comparisons...")
        
        exp_names = list(self.experiments.keys())
        if len(exp_names) < 2:
            print("Need at least 2 experiments for comparison")
            return {}
        
        comparisons = {}
        
        # Compare each pair of experiments
        for i in range(len(exp_names)):
            for j in range(i + 1, len(exp_names)):
                exp1_name, exp2_name = exp_names[i], exp_names[j]
                comparison_key = f"{exp1_name}_vs_{exp2_name}"
                
                print(f"  Comparing {exp1_name} vs {exp2_name}")
                
                comparison = self._compare_two_experiments(exp1_name, exp2_name)
                comparisons[comparison_key] = comparison
        
        return comparisons
    
    def _compare_two_experiments(self, exp1_name: str, exp2_name: str, common_problems: set = None) -> dict:
        """Compare two specific experiments"""
        comparison = {
            'experiment_1': exp1_name,
            'experiment_2': exp2_name,
            'common_problems': [],
            'metrics_comparison': {},
            'statistical_tests': {}
        }
        
        # Find common problems if not provided
        if common_problems is None:
            exp1_problems = set(self.aggregate_stats[exp1_name]['problems'].keys())
            exp2_problems = set(self.aggregate_stats[exp2_name]['problems'].keys())
            common_problems = exp1_problems.intersection(exp2_problems)
        
        comparison['common_problems'] = list(common_problems)
        comparison['total_common'] = len(common_problems)
        
        if len(common_problems) < 2:
            print(f"    Warning: Only {len(common_problems)} common problems found")
            return comparison
        
        print(f"    Found {len(common_problems)} common problems")
        
        # Extract metrics for common problems
        metrics_data = defaultdict(lambda: {'exp1': [], 'exp2': []})
        
        for problem_id in common_problems:
            exp1_metrics = self.aggregate_stats[exp1_name]['problems'][problem_id]
            exp2_metrics = self.aggregate_stats[exp2_name]['problems'][problem_id]
            
            for metric_name in exp1_metrics:
                if metric_name in exp2_metrics:
                    val1 = exp1_metrics[metric_name]
                    val2 = exp2_metrics[metric_name]
                    
                    if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                        metrics_data[metric_name]['exp1'].append(val1)
                        metrics_data[metric_name]['exp2'].append(val2)
        
        # Perform statistical tests for each metric
        for metric_name, data in metrics_data.items():
            if len(data['exp1']) < 2 or len(data['exp2']) < 2:
                continue
                
            metric_comparison = self._compare_metric(data['exp1'], data['exp2'], metric_name)
            comparison['metrics_comparison'][metric_name] = metric_comparison
        
        return comparison
    
    def _compare_metric(self, values1: List[float], values2: List[float], metric_name: str) -> dict:
        """Compare a specific metric between two experiments with detailed explanations"""
        if not SCIPY_AVAILABLE:
            return {}
            
        comparison = {
            'metric_name': metric_name,
            'metric_description': self._get_metric_description(metric_name),
            'experiment_1': {
                'mean': float(np.mean(values1)),
                'std': float(np.std(values1)),
                'median': float(np.median(values1)),
                'count': len(values1)
            },
            'experiment_2': {
                'mean': float(np.mean(values2)),
                'std': float(np.std(values2)),
                'median': float(np.median(values2)),
                'count': len(values2)
            }
        }
        
        # Calculate effect size (Cohen's d) with explanation
        pooled_std = np.sqrt(((len(values1) - 1) * np.var(values1, ddof=1) + 
                             (len(values2) - 1) * np.var(values2, ddof=1)) / 
                            (len(values1) + len(values2) - 2))
        
        if pooled_std > 0:
            cohens_d = (comparison['experiment_1']['mean'] - comparison['experiment_2']['mean']) / pooled_std
            comparison['effect_size'] = {
                'cohens_d': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d),
                'explanation': "Cohen's d measures the standardized difference between two means. Values: 0.2=small, 0.5=medium, 0.8=large effect."
            }
        
        # Perform t-test (parametric test assuming normal distribution)
        try:
            t_stat, t_pvalue = stats.ttest_ind(values1, values2)
            comparison['t_test'] = {
                'statistic': float(t_stat),
                'p_value': float(t_pvalue),
                'significant': bool(t_pvalue < 0.05),  # Convert numpy bool to Python bool
                'explanation': "Independent t-test compares means assuming normal distributions. Use when data is approximately normal."
            }
            if self.verbose:
                print(f"      t-test for {metric_name}: p={t_pvalue:.6f}, significant={t_pvalue < 0.05}")
                print(f"      stored significant value: {comparison['t_test']['significant']} (type: {type(comparison['t_test']['significant'])})")
        except Exception as e:
            if self.verbose:
                print(f"    T-test failed for {metric_name}: {e}")
        
        # Perform Mann-Whitney U test (non-parametric alternative to t-test)
        try:
            u_stat, u_pvalue = stats.mannwhitneyu(values1, values2, alternative='two-sided')
            comparison['mann_whitney'] = {
                'statistic': float(u_stat),
                'p_value': float(u_pvalue),
                'significant': bool(u_pvalue < 0.05),  # Convert numpy bool to Python bool
                'explanation': "Mann-Whitney U test compares distributions without assuming normality. More robust than t-test for non-normal data."
            }
        except Exception as e:
            if self.verbose:
                print(f"    Mann-Whitney test failed for {metric_name}: {e}")
        
        # Perform Wilcoxon rank-sum test (same as Mann-Whitney but different implementation)
        try:
            w_stat, w_pvalue = stats.ranksums(values1, values2)
            comparison['wilcoxon'] = {
                'statistic': float(w_stat),
                'p_value': float(w_pvalue),
                'significant': bool(w_pvalue < 0.05),  # Convert numpy bool to Python bool
                'explanation': "Wilcoxon rank-sum test is equivalent to Mann-Whitney U, comparing median ranks between groups."
            }
        except Exception as e:
            if self.verbose:
                print(f"    Wilcoxon test failed for {metric_name}: {e}")
        
        return comparison
    
    def _get_metric_description(self, metric_name: str) -> str:
        """Get human-readable description for each metric"""
        descriptions = {
            # Original rollout success metrics
            'success_rate_0': 'Percentage of rollouts that achieved any positive reward (> 0)',
            'success_rate_05': 'Percentage of rollouts that achieved good performance (reward ≥ 0.5)',
            'success_rate_075': 'Percentage of rollouts that achieved very good performance (reward ≥ 0.75)',
            'success_rate_1': 'Percentage of rollouts that achieved perfect performance (reward = 1.0)',
            'successful_rollouts_0': 'Number of rollouts with reward > 0',
            'successful_rollouts_05': 'Number of rollouts with reward ≥ 0.5',
            'successful_rollouts_075': 'Number of rollouts with reward ≥ 0.75',
            'successful_rollouts_1': 'Number of rollouts with perfect reward (1.0)',

            # Tree structure metrics
            'total_nodes': 'Total number of nodes explored in the search tree',
            'total_rollouts': 'Total number of rollout simulations performed',
            'terminal_nodes': 'Number of leaf nodes (endpoints) in the search tree',
            'max_depth': 'Maximum depth reached in the search tree',
            'branching_factor': 'Average number of children per non-terminal node (tree breadth)',

            # Key performance indicators
            'tree_success': '🏆 TREE SUCCESS: Binary indicator (1/0) if tree found at least one perfect solution',
            'tree_has_any_success': 'Binary indicator (1/0) if tree found any positive reward',
            'tree_has_good_success': 'Binary indicator (1/0) if tree found any good solution (≥ 0.5)',

            # Early success and efficiency metrics
            'depth_to_first_success': 'Tree depth where first perfect solution was discovered (-1 if none)',
            'rollouts_to_first_success': 'Number of rollouts needed to find first perfect solution (-1 if none)',
            'nodes_to_first_success': 'Number of nodes explored before finding first perfect solution (-1 if none)',
            'perfect_success_density': 'Perfect rollouts per node explored (efficiency indicator)',
            'search_efficiency_score': 'Efficiency score: perfect_rollouts / (nodes × depth)',
            'rollout_efficiency': 'Fraction of rollouts that achieved perfect performance',
            'node_efficiency': 'Perfect rollouts per node (computational efficiency)',

            # Tree exploration quality
            'exploration_to_exploitation_ratio': 'Ratio of unique nodes to total rollouts (exploration vs exploitation)',
            'unique_states_explored': 'Estimated number of unique states explored (based on thought diversity)',
            'terminal_success_rate': 'Fraction of terminal nodes that contain perfect rollouts',
            'successful_paths': 'Number of root-to-leaf paths containing perfect solutions',

            # Search strategy analysis
            'visit_concentration': 'Visit concentration index (0=uniform, 1=highly concentrated)',
            'depth_distribution_variance': 'Variance in node depth distribution (tree balance indicator)',

            # Derived efficiency metrics
            'rollouts_per_node': 'Average number of rollouts performed per node (search intensity)',
            'nodes_per_depth': 'Average branching factor (nodes per depth level)',
            'terminal_ratio': 'Proportion of nodes that are terminal (leaf nodes / total nodes)',
            'search_time': 'Total time spent on search in seconds',

            # Legacy metrics
            'max_visit_count': 'Highest visit count among all nodes',
            'best_reward': 'Best reward found across all rollouts',
            'best_reward_node_count': 'Number of nodes that achieved the best reward'
        }
        return descriptions.get(metric_name, f'Metric: {metric_name} (description not available)')
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def generate_html_report(self, output_file: str, comparisons: dict):
        """Generate an interactive HTML report with comprehensive analysis"""
        print(f"Generating HTML report: {output_file}")
        
        html_content = self._create_html_template()
        
        # Insert experiment data with better error handling
        try:
            # Create lightweight version of data without images or large objects
            lightweight_stats = self._create_lightweight_stats()
            
            # Debug: Check sizes
            print(f"Debug: lightweight_stats size: {len(str(lightweight_stats))} characters")
            print(f"Debug: comparisons size: {len(str(comparisons))} characters")
            print(f"Debug: series_data size: {len(str(self.experiment_series))} characters")
            
            experiments_json = json.dumps(lightweight_stats, indent=2, default=str, ensure_ascii=False)
            comparisons_json = json.dumps(comparisons, indent=2, ensure_ascii=False)
            series_json = json.dumps(self.experiment_series, indent=2, default=str, ensure_ascii=False)
            x_axis_json = json.dumps(self.x_axis_values or [], indent=2, default=str, ensure_ascii=False)
            
            print(f"Debug: experiments_json size: {len(experiments_json)} characters")
            
            html_content = html_content.replace('/*EXPERIMENTS_DATA*/', f'const experimentsData = {experiments_json};')
            html_content = html_content.replace('/*COMPARISONS_DATA*/', f'const comparisonsData = {comparisons_json};')
            html_content = html_content.replace('/*SERIES_DATA*/', f'const seriesData = {series_json};')
            html_content = html_content.replace('/*X_AXIS_DATA*/', f'const xAxisData = {x_axis_json};')
            html_content = html_content.replace('/*SERIES_MODE*/', f'const seriesMode = {str(self.series_mode).lower()};')
            
            with open(output_file, 'w') as f:
                f.write(html_content)
            
            print(f"✅ HTML report saved to {os.path.abspath(output_file)}")
            print(f"📊 Report includes:")
            print(f"   - {len(self.experiments)} experiments compared")
            if self.series_mode:
                print(f"   - {len(self.experiment_series)} experiment series")
                print(f"   - Line graphs showing compute scaling")
            print(f"   - {len(comparisons)} pairwise comparisons")
            print(f"   - Statistical significance tests with explanations")
            print(f"   - Interactive visualizations with metric descriptions")
            
        except Exception as e:
            print(f"❌ Error generating HTML report: {e}")
            raise
    
    def _create_html_template(self) -> str:
        """Create the HTML template for the report"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCTS Experiment Comparison Report</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        h1 {
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 20px;
            background-color: #f8f9fa;
        }
        .metric-comparison {
            margin: 20px 0;
            border: 1px solid #ddd;
            border-radius: 8px;
            overflow: hidden;
        }
        .metric-header {
            background-color: #e9ecef;
            padding: 15px;
            font-weight: bold;
        }
        .metric-content {
            padding: 15px;
        }
        .significance {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            margin-top: 4px;
        }
        .significant {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .not-significant {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .metric-description {
            font-style: italic;
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
        .explanation-box {
            background-color: #e7f3ff;
            border-left: 4px solid #007bff;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }
        .effect-size {
            margin: 10px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 4px;
        }
        .plot-container {
            margin: 20px 0;
            height: 400px;
        }
        .tabs {
            display: flex;
            border-bottom: 1px solid #ddd;
            margin: 20px 0;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            border-bottom: 2px solid transparent;
        }
        .tab.active {
            border-bottom-color: #007bff;
            color: #007bff;
        }
        .tab-content {
            display: none;
            padding: 20px 0;
        }
        .tab-content.active {
            display: block;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f8f9fa;
        }
        .number {
            text-align: right;
        }

        /* Progressive disclosure styles for large experiment sets */
        .statistical-controls {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #ddd;
        }
        .search-container {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 10px;
        }
        .search-input {
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
            flex: 1;
            min-width: 200px;
        }
        .filter-buttons {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .filter-btn {
            padding: 6px 12px;
            border: 1px solid #007bff;
            background-color: white;
            color: #007bff;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }
        .filter-btn:hover {
            background-color: #007bff;
            color: white;
        }
        .filter-btn.active {
            background-color: #007bff;
            color: white;
        }
        .comparison-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 10px;
            background-color: white;
            overflow: hidden;
            transition: box-shadow 0.2s;
        }
        .comparison-card:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .comparison-header {
            background-color: #f8f9fa;
            padding: 12px 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #eee;
        }
        .comparison-header:hover {
            background-color: #e9ecef;
        }
        .comparison-title {
            font-weight: bold;
            color: #333;
            margin: 0;
        }
        .comparison-summary {
            font-size: 12px;
            color: #666;
            margin: 2px 0 0 0;
        }
        .comparison-expand-btn {
            background: none;
            border: none;
            font-size: 16px;
            color: #007bff;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .comparison-expand-btn.expanded {
            transform: rotate(90deg);
        }
        .comparison-content {
            display: none;
            padding: 15px;
            background-color: white;
        }
        .comparison-content.expanded {
            display: block;
        }
        .comparison-pagination {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        .pagination-btn {
            padding: 8px 12px;
            border: 1px solid #ddd;
            background-color: white;
            color: #333;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .pagination-btn:hover:not(:disabled) {
            background-color: #007bff;
            color: white;
        }
        .pagination-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .pagination-info {
            font-size: 14px;
            color: #666;
        }
        .loading-indicator {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        .no-results {
            text-align: center;
            padding: 40px;
            color: #999;
            font-style: italic;
        }
        .significance-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
            margin-left: 8px;
        }
        .high-effect {
            background-color: #d4edda;
            color: #155724;
        }
        .medium-effect {
            background-color: #fff3cd;
            color: #856404;
        }
        .low-effect {
            background-color: #f8d7da;
            color: #721c24;
        }

        /* Enhanced pagination controls styling */
        .pagination-controls {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #ddd;
        }

        .pagination-controls button {
            padding: 8px 16px;
            border: 1px solid #007bff;
            background-color: white;
            color: #007bff;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
        }

        .pagination-controls button:hover:not(:disabled) {
            background-color: #007bff;
            color: white;
        }

        .pagination-controls button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            border-color: #ddd;
            color: #999;
        }

        /* Loading animation */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 8px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Enhanced metrics comparison styling */
        .metrics-cards-container {
            min-height: 200px;
        }

        .comparison-header h3 {
            margin: 0;
            flex: 1;
        }

        .comparison-summary {
            display: flex;
            gap: 15px;
            font-size: 12px;
            color: #666;
            margin: 5px 0 0 0;
        }

        .comparison-summary span {
            padding: 2px 6px;
            background-color: #e9ecef;
            border-radius: 3px;
        }

        /* Improved responsive design */
        @media (max-width: 768px) {
            .comparison-header {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }

            .comparison-summary {
                flex-direction: column;
                gap: 5px;
            }

            .pagination-controls {
                flex-direction: column;
                gap: 10px;
            }

            .filter-buttons {
                justify-content: center;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>MCTS Experiment Comparison Report</h1>
        
        <div class="tabs">
            <button class="tab active" onclick="showTab('overview', event)">Overview</button>
            <button class="tab" onclick="showTab('series', event)" id="series-tab" style="display: none;">Series Analysis</button>
            <button class="tab" onclick="showTab('metrics', event)">Metric Comparisons</button>
            <button class="tab" onclick="showTab('statistical', event)">Statistical Analysis</button>
            <button class="tab" onclick="showTab('visualizations', event)">Key Visualizations</button>
            <button class="tab" onclick="showTab('exhaustive', event)">📊 Exhaustive Graphs</button>
        </div>
        
        <div id="overview" class="tab-content active">
            <h2>Experiment Overview</h2>
            <div id="experiment-summary"></div>
        </div>
        
        <div id="series" class="tab-content">
            <h2>Series Analysis - Compute Scaling</h2>
            <div id="series-analysis"></div>
        </div>
        
        <div id="metrics" class="tab-content">
            <h2>Metric Comparisons</h2>
            <div id="metrics-comparison"></div>
        </div>
        
        <div id="statistical" class="tab-content">
            <h2>Statistical Analysis</h2>
            <div id="statistical-tests"></div>
        </div>
        
        <div id="visualizations" class="tab-content">
            <h2>Key Visualizations</h2>
            <div id="plots-container"></div>
        </div>

        <div id="exhaustive" class="tab-content">
            <h2>📊 Exhaustive Graphs: Complete Metric Analysis</h2>
            <div id="exhaustive-container"></div>
        </div>
    </div>

    <script>
        // Global error handler
        window.onerror = function(msg, url, lineNo, columnNo, error) {
            console.error('JavaScript Error:', msg, 'at line', lineNo);
            document.body.innerHTML += '<div style="color: red; background: #ffe6e6; padding: 15px; margin: 10px; border: 1px solid red;">❌ JavaScript Error: ' + msg + ' at line ' + lineNo + '</div>';
            return false;
        };

        // Debug logging
        console.log('🚀 Script loading started');

        /*EXPERIMENTS_DATA*/
        /*COMPARISONS_DATA*/
        /*SERIES_DATA*/
        /*X_AXIS_DATA*/
        /*SERIES_MODE*/

        function showTab(tabName, event) {
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));

            const tabButtons = document.querySelectorAll('.tab');
            tabButtons.forEach(button => button.classList.remove('active'));

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            if (event && event.target) {
                event.target.classList.add('active');
            }
            
            // Generate content for the tab if needed
            if (tabName === 'visualizations') {
                generatePlots();
            } else if (tabName === 'series') {
                generateSeriesAnalysis();
            } else if (tabName === 'exhaustive') {
                generateExhaustiveGraphs();
            }
        }
        
        function generateOverview() {
            const container = document.getElementById('experiment-summary');
            
            let html = '<div class="summary-grid">';
            
            for (const [expName, expData] of Object.entries(experimentsData)) {
                html += `
                    <div class="summary-card">
                        <h3>${expName}</h3>
                        <p><strong>Total Problems:</strong> ${expData.total_problems}</p>
                        <h4>Key Metrics (Average):</h4>
                        <ul>
                `;
                
                if (expData.aggregate.success_rate_1) {
                    html += `<li>Perfect Success Rate: ${(expData.aggregate.success_rate_1.mean * 100).toFixed(1)}%</li>`;
                }
                if (expData.aggregate.success_rate_05) {
                    html += `<li>Success Rate (≥0.5): ${(expData.aggregate.success_rate_05.mean * 100).toFixed(1)}%</li>`;
                }
                if (expData.aggregate.total_nodes) {
                    html += `<li>Avg Nodes: ${expData.aggregate.total_nodes.mean.toFixed(1)}</li>`;
                }
                if (expData.aggregate.total_rollouts) {
                    html += `<li>Avg Total Rollouts: ${expData.aggregate.total_rollouts.mean.toFixed(1)}</li>`;
                }
                
                html += '</ul></div>';
            }
            
            html += '</div>';
            container.innerHTML = html;
        }
        
        // Progressive disclosure system for metrics comparison
        let metricsState = {
            searchTerm: '',
            currentPage: 1,
            itemsPerPage: 10,
            expandedComparisons: new Set()
        };

        function generateMetricsComparison() {
            const container = document.getElementById('metrics-comparison');

            // Add controls for search and pagination
            let html = `
                <div class="explanation-box">
                    <h4>📊 Metrics Comparison</h4>
                    <p>Detailed comparison of performance metrics between experiment pairs. Use search and pagination to navigate through many experiments efficiently.</p>
                </div>
                <div class="statistical-controls">
                    <div class="search-container">
                        <input type="text" class="search-input" placeholder="Search metrics by experiment names..."
                               id="metrics-search" onkeyup="filterMetricsComparisons()">
                        <span class="pagination-info" id="metrics-results-count">Loading...</span>
                    </div>
                    <div class="filter-buttons">
                        <button class="filter-btn" onclick="expandAllMetrics()">
                            Expand All
                        </button>
                        <button class="filter-btn" onclick="collapseAllMetrics()">
                            Collapse All
                        </button>
                    </div>
                </div>
                <div id="metrics-cards-container"></div>
                <div class="pagination-controls">
                    <button onclick="changeMetricsPage(-1)" id="metrics-prev-btn">Previous</button>
                    <span id="metrics-page-info">Page 1</span>
                    <button onclick="changeMetricsPage(1)" id="metrics-next-btn">Next</button>
                </div>
            `;

            container.innerHTML = html;
            renderMetricsCards();
        }

        function renderMetricsCards() {
            const container = document.getElementById('metrics-cards-container');
            if (!container) return;

            // Show loading indicator
            container.innerHTML = '<div class="loading-indicator"><div class="loading-spinner"></div>Loading metrics comparisons...</div>';

            // Use setTimeout to allow the loading indicator to render
            setTimeout(() => {
                // Filter comparisons based on search term
                const filteredComparisons = Object.entries(comparisonsData).filter(([compKey, compData]) => {
                    if (metricsState.searchTerm) {
                        const searchLower = metricsState.searchTerm.toLowerCase();
                        const titleText = `${compData.experiment_1} vs ${compData.experiment_2}`.toLowerCase();
                        if (!titleText.includes(searchLower)) {
                            return false;
                        }
                    }
                    return true;
                });

                // Calculate pagination
                const totalItems = filteredComparisons.length;
                const totalPages = Math.ceil(totalItems / metricsState.itemsPerPage);
                const startIndex = (metricsState.currentPage - 1) * metricsState.itemsPerPage;
                const endIndex = Math.min(startIndex + metricsState.itemsPerPage, totalItems);
                const pageComparisons = filteredComparisons.slice(startIndex, endIndex);

                // Update pagination controls
                updateMetricsResultsCount(totalItems, startIndex + 1, endIndex);

                const prevBtn = document.getElementById('metrics-prev-btn');
                const nextBtn = document.getElementById('metrics-next-btn');
                const pageInfo = document.getElementById('metrics-page-info');

                if (prevBtn) prevBtn.disabled = metricsState.currentPage === 1;
                if (nextBtn) nextBtn.disabled = metricsState.currentPage === totalPages || totalPages === 0;
                if (pageInfo) pageInfo.textContent = totalPages > 0 ? `Page ${metricsState.currentPage} of ${totalPages}` : 'No results';

                // Handle no results case
                if (totalItems === 0) {
                    container.innerHTML = '<div class="no-results">No metrics comparisons found. Try adjusting your search terms.</div>';
                    return;
                }

                // Render comparison cards
                let cardsHtml = '';
                for (const [compKey, compData] of pageComparisons) {
                    const isExpanded = metricsState.expandedComparisons.has(compKey);

                    cardsHtml += `
                        <div class="comparison-card">
                            <div class="comparison-header" onclick="toggleMetricsComparison('${compKey}')">
                                <h3>${compData.experiment_1} vs ${compData.experiment_2}</h3>
                                <div class="comparison-summary">
                                    <span>📊 ${Object.keys(compData.metrics_comparison || {}).length} metrics</span>
                                    <span>🔍 ${compData.total_common} problems</span>
                                </div>
                                <button class="comparison-expand-btn ${isExpanded ? 'expanded' : ''}"
                                        data-comparison-key="${compKey}">
                                    ${isExpanded ? '▼' : '▶'}
                                </button>
                            </div>
                            <div class="comparison-content ${isExpanded ? 'expanded' : ''}" id="metrics-content-${compKey}">
                                ${isExpanded ? generateMetricsDetailedContent(compData) : '<p>Click to load detailed metrics comparison...</p>'}
                            </div>
                        </div>
                    `;
                }

                container.innerHTML = cardsHtml;
            }, 100); // Small delay to show loading indicator
        }

        function generateMetricsDetailedContent(compData) {
            if (!compData.metrics_comparison) {
                return '<p>No detailed metrics comparison available.</p>';
            }

            let html = `<p><strong>Common Problems:</strong> ${compData.total_common}</p>`;

            for (const [metricName, metricData] of Object.entries(compData.metrics_comparison)) {
                html += `
                    <div class="metric-comparison">
                        <div class="metric-header">
                            <strong>${metricName}</strong>
                            ${metricData.metric_description ? `<br><em style="font-size: 0.9em; color: #666;">${metricData.metric_description}</em>` : ''}
                        </div>
                        <div class="metric-content">
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 15px;">
                                <div>
                                    <h4>${compData.experiment_1}</h4>
                                    <p><strong>Mean:</strong> ${metricData.experiment_1.mean.toFixed(4)}</p>
                                    <p><strong>Std:</strong> ${metricData.experiment_1.std.toFixed(4)}</p>
                                    <p><strong>Median:</strong> ${metricData.experiment_1.median.toFixed(4)}</p>
                                    <p><strong>Count:</strong> ${metricData.experiment_1.count}</p>
                                </div>
                                <div>
                                    <h4>${compData.experiment_2}</h4>
                                    <p><strong>Mean:</strong> ${metricData.experiment_2.mean.toFixed(4)}</p>
                                    <p><strong>Std:</strong> ${metricData.experiment_2.std.toFixed(4)}</p>
                                    <p><strong>Median:</strong> ${metricData.experiment_2.median.toFixed(4)}</p>
                                    <p><strong>Count:</strong> ${metricData.experiment_2.count}</p>
                                </div>
                            </div>
                `;

                if (metricData.effect_size) {
                    html += `
                        <div class="effect-size">
                            <strong>Effect Size (Cohen\\'s d):</strong> ${metricData.effect_size.cohens_d.toFixed(4)}
                            (${metricData.effect_size.interpretation})
                            ${metricData.effect_size.explanation ? `<br><small>${metricData.effect_size.explanation}</small>` : ''}
                        </div>
                    `;
                }

                html += '</div></div>';
            }

            return html;
        }

        function filterMetricsComparisons() {
            const searchInput = document.getElementById('metrics-search');
            metricsState.searchTerm = searchInput.value;
            metricsState.currentPage = 1; // Reset to first page
            renderMetricsCards();
        }

        function changeMetricsPage(direction) {
            metricsState.currentPage += direction;
            renderMetricsCards();
        }

        function toggleMetricsComparison(compKey) {
            const contentElement = document.getElementById(`metrics-content-${compKey}`);
            const buttonElement = document.querySelector(`[data-comparison-key="${compKey}"] .comparison-expand-btn`);

            if (metricsState.expandedComparisons.has(compKey)) {
                // Collapse
                metricsState.expandedComparisons.delete(compKey);
                contentElement.classList.remove('expanded');
                buttonElement.classList.remove('expanded');
                buttonElement.textContent = '▶';
            } else {
                // Expand
                metricsState.expandedComparisons.add(compKey);
                contentElement.classList.add('expanded');
                buttonElement.classList.add('expanded');
                buttonElement.textContent = '▼';

                // Lazy load detailed content if not already loaded
                if (contentElement.innerHTML.includes('Click to load')) {
                    const compData = comparisonsData[compKey];
                    contentElement.innerHTML = generateMetricsDetailedContent(compData);
                }
            }
        }

        function expandAllMetrics() {
            for (const compKey in comparisonsData) {
                if (!metricsState.expandedComparisons.has(compKey)) {
                    metricsState.expandedComparisons.add(compKey);
                }
            }
            renderMetricsCards();
        }

        function collapseAllMetrics() {
            metricsState.expandedComparisons.clear();
            renderMetricsCards();
        }

        function updateMetricsResultsCount(total, start, end) {
            const countElement = document.getElementById('metrics-results-count');
            if (countElement) {
                countElement.textContent = `Showing ${start}-${end} of ${total} comparisons`;
            }
        }
        
        // Progressive disclosure system for statistical analysis
        let statisticalState = {
            searchTerm: '',
            filters: {
                significantOnly: false,
                largeEffectOnly: false,
                mediumEffectOnly: false
            },
            currentPage: 1,
            itemsPerPage: 20,
            expandedComparisons: new Set()
        };

        function generateStatisticalTests() {
            const container = document.getElementById('statistical-tests');

            // Create header with explanations
            let html = `
                <div style="background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin-bottom: 25px; border-left: 5px solid #007bff;">
                    <h4>📊 Statistical Significance Testing for Tree Search Performance</h4>
                    <p><strong>Understanding the Tests:</strong></p>
                    <ul>
                        <li><strong>📈 Student\\'s t-test (Parametric):</strong> Compares average performance assuming normal distributions. Best for well-behaved metrics like success rates and efficiency scores.</li>
                        <li><strong>🔄 Mann-Whitney U test (Non-parametric):</strong> Compares overall distributions without normality assumptions. More robust for tree metrics that may have skewed distributions.</li>
                        <li><strong>📏 Effect Size (Cohen\\'s d):</strong> Measures practical significance beyond p-values. Small (0.2), Medium (0.5), Large (0.8+) effects indicate real-world importance.</li>
                        <li><strong>🎯 Statistical Significance:</strong> p < 0.05 means the difference is unlikely due to chance. However, focus on effect sizes for practical importance!</li>
                    </ul>
                    <div style="background-color: #fff3cd; padding: 10px; border-radius: 4px; margin-top: 10px;">
                        <strong>💡 Key Insight:</strong> For tree search evaluation, focus on <em>tree_success</em> and efficiency metrics. Large effect sizes matter more than p-values for practical algorithm improvements.
                    </div>
                </div>
            `;

            // Add search and filter controls
            html += `
                <div class="statistical-controls">
                    <div class="search-container">
                        <input type="text" class="search-input" placeholder="Search comparisons by experiment names..."
                               id="comparison-search" onkeyup="filterComparisons()">
                        <span class="pagination-info" id="results-count">Loading...</span>
                    </div>
                    <div class="filter-buttons">
                        <button class="filter-btn" id="filter-significant" onclick="toggleFilter('significantOnly')">
                            Significant Only
                        </button>
                        <button class="filter-btn" id="filter-large" onclick="toggleFilter('largeEffectOnly')">
                            Large Effect (d≥0.8)
                        </button>
                        <button class="filter-btn" id="filter-medium" onclick="toggleFilter('mediumEffectOnly')">
                            Medium+ Effect (d≥0.5)
                        </button>
                        <button class="filter-btn" onclick="expandAllComparisons()">
                            Expand All
                        </button>
                        <button class="filter-btn" onclick="collapseAllComparisons()">
                            Collapse All
                        </button>
                    </div>
                </div>
            `;

            // Container for comparison cards
            html += '<div id="comparison-cards-container"></div>';

            // Pagination controls
            html += `
                <div class="comparison-pagination" id="pagination-controls" style="display: none;">
                    <button class="pagination-btn" id="prev-page" onclick="changePage(-1)">Previous</button>
                    <span class="pagination-info" id="page-info"></span>
                    <button class="pagination-btn" id="next-page" onclick="changePage(1)">Next</button>
                </div>
            `;

            container.innerHTML = html;

            // Initialize the comparison list
            renderComparisonCards();
        }

        function renderComparisonCards() {
            const container = document.getElementById('comparison-cards-container');
            const comparisons = Object.entries(comparisonsData);

            if (comparisons.length === 0) {
                container.innerHTML = '<div class="no-results">No statistical comparisons available.</div>';
                return;
            }

            // Apply filters
            const filteredComparisons = filterComparisonData(comparisons);

            // Update results count
            updateResultsCount(filteredComparisons.length, comparisons.length);

            // Calculate pagination
            const totalPages = Math.ceil(filteredComparisons.length / statisticalState.itemsPerPage);
            const startIndex = (statisticalState.currentPage - 1) * statisticalState.itemsPerPage;
            const endIndex = startIndex + statisticalState.itemsPerPage;
            const pageComparisons = filteredComparisons.slice(startIndex, endIndex);

            // Show/hide pagination
            const paginationControls = document.getElementById('pagination-controls');
            if (totalPages > 1) {
                paginationControls.style.display = 'flex';
                updatePaginationControls(totalPages);
            } else {
                paginationControls.style.display = 'none';
            }

            // Generate comparison cards
            let html = '';
            if (pageComparisons.length === 0) {
                html = '<div class="no-results">No comparisons match your current filters.</div>';
            } else {
                pageComparisons.forEach(([compKey, compData]) => {
                    html += generateComparisonCard(compKey, compData);
                });
            }

            container.innerHTML = html;
        }

        function generateComparisonCard(compKey, compData) {
            const isExpanded = statisticalState.expandedComparisons.has(compKey);
            const summary = generateComparisonSummary(compData);

            return `
                <div class="comparison-card" data-comparison-key="${compKey}">
                    <div class="comparison-header" onclick="toggleComparison('${compKey}')">
                        <div>
                            <h4 class="comparison-title">${compData.experiment_1} vs ${compData.experiment_2}</h4>
                            <p class="comparison-summary">${summary}</p>
                        </div>
                        <button class="comparison-expand-btn ${isExpanded ? 'expanded' : ''}">▶</button>
                    </div>
                    <div class="comparison-content ${isExpanded ? 'expanded' : ''}" id="content-${compKey}">
                        ${isExpanded ? generateDetailedComparison(compData) : '<div class="loading-indicator">Click to load detailed analysis...</div>'}
                    </div>
                </div>
            `;
        }

        function generateComparisonSummary(compData) {
            if (!compData.metrics_comparison) {
                return 'No metrics comparison available';
            }

            const metrics = Object.entries(compData.metrics_comparison);
            const significantTests = metrics.filter(([_, data]) =>
                data.t_test?.significant || data.mann_whitney?.significant
            ).length;

            const largeEffects = metrics.filter(([_, data]) =>
                data.effect_size && Math.abs(data.effect_size.cohens_d) >= 0.8
            ).length;

            const badges = [];
            if (significantTests > 0) {
                badges.push(`<span class="significance-badge significant">${significantTests} significant</span>`);
            }
            if (largeEffects > 0) {
                badges.push(`<span class="significance-badge high-effect">${largeEffects} large effects</span>`);
            }

            return `${compData.total_common} common problems • ${metrics.length} metrics compared${badges.length > 0 ? ' • ' + badges.join(' ') : ''}`;
        }

        function generateDetailedComparison(compData) {
            if (!compData.metrics_comparison) {
                return '<p>No detailed metrics comparison available.</p>';
            }

            let html = `<p><strong>Common Problems:</strong> ${compData.total_common}</p>`;
            html += '<table><thead><tr><th>🔍 Performance Metric</th><th>📋 Description</th><th>📈 Student\\'s t-test<br><small>(parametric)</small></th><th>🔄 Mann-Whitney U<br><small>(non-parametric)</small></th><th>📏 Effect Size<br><small>(practical impact)</small></th></tr></thead><tbody>';

            for (const [metricName, metricData] of Object.entries(compData.metrics_comparison)) {
                html += `<tr><td><strong>${metricName}</strong></td>`;
                html += '<td style="max-width: 200px; font-size: 0.9em;">' + (metricData.metric_description || 'N/A') + '</td>';

                // t-test
                if (metricData.t_test) {
                    const significance = metricData.t_test.significant ? 'significant' : 'not-significant';
                    html += '<td>p=' + metricData.t_test.p_value.toFixed(4) + '<br><span class="significance ' + significance + '">' + (metricData.t_test.significant ? 'Significant' : 'Not Significant') + '</span></td>';
                } else {
                    html += '<td>N/A</td>';
                }

                // Mann-Whitney
                if (metricData.mann_whitney) {
                    const significance = metricData.mann_whitney.significant ? 'significant' : 'not-significant';
                    html += '<td>p=' + metricData.mann_whitney.p_value.toFixed(4) + '<br><span class="significance ' + significance + '">' + (metricData.mann_whitney.significant ? 'Significant' : 'Not Significant') + '</span></td>';
                } else {
                    html += '<td>N/A</td>';
                }

                // Effect size
                if (metricData.effect_size) {
                    html += `<td>d=${metricData.effect_size.cohens_d.toFixed(3)}<br><em>(${metricData.effect_size.interpretation})</em></td>`;
                } else {
                    html += '<td>N/A</td>';
                }

                html += '</tr>';
            }

            html += '</tbody></table>';
            return html;
        }

        function filterComparisonData(comparisons) {
            return comparisons.filter(([compKey, compData]) => {
                // Search filter
                if (statisticalState.searchTerm) {
                    const searchLower = statisticalState.searchTerm.toLowerCase();
                    const titleText = `${compData.experiment_1} vs ${compData.experiment_2}`.toLowerCase();
                    if (!titleText.includes(searchLower)) {
                        return false;
                    }
                }

                // Statistical significance filter
                if (statisticalState.filters.significantOnly && compData.metrics_comparison) {
                    const hasSignificant = Object.values(compData.metrics_comparison).some(data =>
                        data.t_test?.significant || data.mann_whitney?.significant
                    );
                    if (!hasSignificant) return false;
                }

                // Effect size filters
                if ((statisticalState.filters.largeEffectOnly || statisticalState.filters.mediumEffectOnly) && compData.metrics_comparison) {
                    const threshold = statisticalState.filters.largeEffectOnly ? 0.8 : 0.5;
                    const hasLargeEffect = Object.values(compData.metrics_comparison).some(data =>
                        data.effect_size && Math.abs(data.effect_size.cohens_d) >= threshold
                    );
                    if (!hasLargeEffect) return false;
                }

                return true;
            });
        }

        function updateResultsCount(filtered, total) {
            const element = document.getElementById('results-count');
            element.textContent = `Showing ${filtered} of ${total} comparisons`;
        }

        function updatePaginationControls(totalPages) {
            document.getElementById('page-info').textContent = `Page ${statisticalState.currentPage} of ${totalPages}`;
            document.getElementById('prev-page').disabled = statisticalState.currentPage === 1;
            document.getElementById('next-page').disabled = statisticalState.currentPage === totalPages;
        }

        // Event handlers
        function filterComparisons() {
            const searchInput = document.getElementById('comparison-search');
            statisticalState.searchTerm = searchInput.value;
            statisticalState.currentPage = 1; // Reset to first page
            renderComparisonCards();
        }

        function toggleFilter(filterName) {
            statisticalState.filters[filterName] = !statisticalState.filters[filterName];

            // Map filter names to button IDs
            const filterButtonMap = {
                'significantOnly': 'filter-significant',
                'largeEffectOnly': 'filter-large',
                'mediumEffectOnly': 'filter-medium'
            };

            // Update button appearance
            const buttonId = filterButtonMap[filterName];
            const button = document.getElementById(buttonId);
            if (button) {
                if (statisticalState.filters[filterName]) {
                    button.classList.add('active');
                } else {
                    button.classList.remove('active');
                }
            }

            // Handle mutual exclusivity for effect size filters
            if (filterName === 'largeEffectOnly' && statisticalState.filters[filterName]) {
                statisticalState.filters.mediumEffectOnly = false;
                const mediumButton = document.getElementById('filter-medium');
                if (mediumButton) {
                    mediumButton.classList.remove('active');
                }
            } else if (filterName === 'mediumEffectOnly' && statisticalState.filters[filterName]) {
                statisticalState.filters.largeEffectOnly = false;
                const largeButton = document.getElementById('filter-large');
                if (largeButton) {
                    largeButton.classList.remove('active');
                }
            }

            statisticalState.currentPage = 1; // Reset to first page
            renderComparisonCards();
        }

        function changePage(direction) {
            statisticalState.currentPage += direction;
            renderComparisonCards();
        }

        function toggleComparison(compKey) {
            const contentElement = document.getElementById(`content-${compKey}`);
            const buttonElement = document.querySelector(`[data-comparison-key="${compKey}"] .comparison-expand-btn`);

            if (statisticalState.expandedComparisons.has(compKey)) {
                // Collapse
                statisticalState.expandedComparisons.delete(compKey);
                contentElement.classList.remove('expanded');
                buttonElement.classList.remove('expanded');
            } else {
                // Expand
                statisticalState.expandedComparisons.add(compKey);
                contentElement.classList.add('expanded');
                buttonElement.classList.add('expanded');

                // Lazy load detailed content if not already loaded
                if (contentElement.innerHTML.includes('Click to load')) {
                    contentElement.innerHTML = '<div class="loading-indicator">Loading detailed analysis...</div>';
                    setTimeout(() => {
                        const compData = comparisonsData[compKey];
                        contentElement.innerHTML = generateDetailedComparison(compData);
                    }, 100); // Small delay for better UX
                }
            }
        }

        function expandAllComparisons() {
            const comparisons = Object.entries(comparisonsData);
            const filteredComparisons = filterComparisonData(comparisons);
            const startIndex = (statisticalState.currentPage - 1) * statisticalState.itemsPerPage;
            const endIndex = startIndex + statisticalState.itemsPerPage;
            const pageComparisons = filteredComparisons.slice(startIndex, endIndex);

            pageComparisons.forEach(([compKey, _]) => {
                if (!statisticalState.expandedComparisons.has(compKey)) {
                    toggleComparison(compKey);
                }
            });
        }

        function collapseAllComparisons() {
            const visibleCards = document.querySelectorAll('.comparison-card');
            visibleCards.forEach(card => {
                const compKey = card.getAttribute('data-comparison-key');
                if (statisticalState.expandedComparisons.has(compKey)) {
                    toggleComparison(compKey);
                }
            });
        }
        
        function generatePlots() {
            const container = document.getElementById('plots-container');
            container.innerHTML = '';
            
            // Safely extract experiment names
            const expNames = Object.keys(experimentsData || {});
            if (expNames.length === 0) {
                container.innerHTML = '<p>No experiment data available for plotting.</p>';
                return;
            }
            
            // Helper function to safely extract metric values
            function safeExtractMetric(expName, metricPath) {
                try {
                    const pathParts = metricPath.split('.');
                    let value = experimentsData[expName];
                    for (const part of pathParts) {
                        value = value[part];
                        if (value === undefined) return 0;
                    }
                    return value;
                } catch (e) {
                    console.warn(`Failed to extract ${metricPath} for ${expName}:`, e);
                    return 0;
                }
            }
            
            // Extract key tree search performance metrics
            const treeSuccess = expNames.map(name => safeExtractMetric(name, 'aggregate.tree_success.mean') * 100);
            const successRate0 = expNames.map(name => safeExtractMetric(name, 'aggregate.success_rate_0.mean') * 100);
            const successRate05 = expNames.map(name => safeExtractMetric(name, 'aggregate.success_rate_05.mean') * 100);
            const successRate075 = expNames.map(name => safeExtractMetric(name, 'aggregate.success_rate_075.mean') * 100);
            const successRate1 = expNames.map(name => safeExtractMetric(name, 'aggregate.success_rate_1.mean') * 100);

            // Extract efficiency metrics
            const rolloutEfficiency = expNames.map(name => safeExtractMetric(name, 'aggregate.rollout_efficiency.mean') * 100);
            const nodeEfficiency = expNames.map(name => safeExtractMetric(name, 'aggregate.node_efficiency.mean') * 100);
            const searchEfficiency = expNames.map(name => safeExtractMetric(name, 'aggregate.search_efficiency_score.mean') * 1000); // Scale for visibility

            // Extract early success metrics
            const depthToSuccess = expNames.map(name => safeExtractMetric(name, 'aggregate.depth_to_first_success.mean'));
            const rolloutsToSuccess = expNames.map(name => safeExtractMetric(name, 'aggregate.rollouts_to_first_success.mean'));
            
            // 🏆 PRIORITY PLOT: Tree Success Rate (Most Important Metric)
            if (treeSuccess.some(val => val > 0) || treeSuccess.some(val => val >= 0)) {
                const treePlotDiv = document.createElement('div');
                treePlotDiv.id = 'tree-success-plot';
                treePlotDiv.className = 'plot-container';
                container.appendChild(treePlotDiv);

                Plotly.newPlot('tree-success-plot', [{
                    x: expNames,
                    y: treeSuccess,
                    type: 'bar',
                    name: '🏆 Tree Success Rate',
                    marker: {
                        color: treeSuccess.map(val => val > 50 ? '#28a745' : val > 25 ? '#ffc107' : '#dc3545'),
                        line: { color: '#000', width: 1 }
                    },
                    text: treeSuccess.map(v => v.toFixed(1) + '%'),
                    textposition: 'auto',
                    textfont: { color: 'white', size: 12, family: 'Arial Black' }
                }], {
                    title: {
                        text: '🏆 TREE SUCCESS RATE: Fraction of Problems Where Tree Found Perfect Solution',
                        font: { size: 18, color: '#2c3e50' }
                    },
                    xaxis: {
                        title: 'Experiment Method',
                        tickangle: -45,
                        tickfont: { size: 12 }
                    },
                    yaxis: {
                        title: 'Tree Success Rate (%)',
                        range: [0, Math.max(...treeSuccess, 100) * 1.1],
                        tickformat: '.1f'
                    },
                    margin: { b: 150, t: 80 },
                    paper_bgcolor: '#f8f9fa',
                    plot_bgcolor: '#ffffff'
                });

                // Add explanation
                const explanation1 = document.createElement('div');
                explanation1.style.cssText = 'background-color: #e7f3ff; padding: 15px; border-left: 4px solid #007bff; margin: 10px 0; border-radius: 0 5px 5px 0;';
                explanation1.innerHTML = `
                    <h5>🔍 Understanding Tree Success</h5>
                    <p><strong>Tree Success Rate</strong> is the most important metric: it shows the percentage of problems where the tree search found at least one perfect solution (reward = 1.0). This indicates the fundamental capability of the method to solve problems.</p>
                `;
                container.appendChild(explanation1);
            }

            // Only create other plots if we have valid data
            if (successRate1.some(val => val > 0) || successRate05.some(val => val > 0)) {
                // Create grouped bar chart for all success rates
                const plotDiv = document.createElement('div');
                plotDiv.id = 'success-rates-plot';
                plotDiv.className = 'plot-container';
                container.appendChild(plotDiv);
                
                const traces = [];
                if (successRate0.some(val => val > 0)) {
                    traces.push({
                        x: expNames,
                        y: successRate0,
                        type: 'bar',
                        name: 'Any Success (> 0)',
                        marker: { color: '#17a2b8' },
                        text: successRate0.map(v => v.toFixed(1) + '%'),
                        textposition: 'auto'
                    });
                }
                if (successRate05.some(val => val > 0)) {
                    traces.push({
                        x: expNames,
                        y: successRate05,
                        type: 'bar',
                        name: 'Good Success (≥ 0.5)',
                        marker: { color: '#28a745' },
                        text: successRate05.map(v => v.toFixed(1) + '%'),
                        textposition: 'auto'
                    });
                }
                if (successRate075.some(val => val > 0)) {
                    traces.push({
                        x: expNames,
                        y: successRate075,
                        type: 'bar',
                        name: 'Very Good Success (≥ 0.75)',
                        marker: { color: '#ffc107' },
                        text: successRate075.map(v => v.toFixed(1) + '%'),
                        textposition: 'auto'
                    });
                }
                if (successRate1.some(val => val > 0)) {
                    traces.push({
                        x: expNames,
                        y: successRate1,
                        type: 'bar',
                        name: 'Perfect Success (= 1.0)',
                        marker: { color: '#dc3545' },
                        text: successRate1.map(v => v.toFixed(1) + '%'),
                        textposition: 'auto'
                    });
                }
                
                Plotly.newPlot('success-rates-plot', traces, {
                    title: {
                        text: 'Success Rates Comparison Across All Thresholds',
                        font: { size: 16 }
                    },
                    xaxis: { 
                        title: 'Experiment',
                        tickangle: -45
                    },
                    yaxis: { 
                        title: 'Success Rate (%)',
                        range: [0, Math.max(...successRate0, ...successRate05, ...successRate075, ...successRate1) * 1.1]
                    },
                    barmode: 'group',
                    margin: { b: 150 }  // Extra margin for rotated labels
                });
            }
            
            // Create separate plot for perfect success rates (most important)
            if (successRate1.some(val => val > 0)) {
                const plotDiv2 = document.createElement('div');
                plotDiv2.id = 'perfect-success-plot';
                plotDiv2.className = 'plot-container';
                container.appendChild(plotDiv2);
                
                Plotly.newPlot('perfect-success-plot', [{
                    x: expNames,
                    y: successRate1,
                    type: 'bar',
                    name: 'Perfect Success Rate (%)',
                    marker: { color: '#007bff' },
                    text: successRate1.map(v => v.toFixed(1) + '%'),
                    textposition: 'auto'
                }], {
                    title: {
                        text: 'Perfect Success Rate (Reward = 1.0) - Key Performance Indicator',
                        font: { size: 16 }
                    },
                    xaxis: { 
                        title: 'Experiment',
                        tickangle: -45
                    },
                    yaxis: { 
                        title: 'Perfect Success Rate (%)',
                        range: [0, Math.max(...successRate1) * 1.2]
                    },
                    margin: { b: 150 }
                });
            }
            
            // Create tree efficiency comparison
            const totalNodes = expNames.map(name => safeExtractMetric(name, 'aggregate.total_nodes.mean'));
            const totalRollouts = expNames.map(name => safeExtractMetric(name, 'aggregate.total_rollouts.mean'));
            
            if (totalNodes.some(n => n > 0) && totalRollouts.some(r => r > 0)) {
                const plotDiv3 = document.createElement('div');
                plotDiv3.id = 'efficiency-plot';
                plotDiv3.className = 'plot-container';
                container.appendChild(plotDiv3);
                
                // Create subplot with separate y-axes
                Plotly.newPlot('efficiency-plot', [
                    {
                        x: expNames,
                        y: totalNodes,
                        type: 'bar',
                        name: 'Total Nodes',
                        marker: { color: '#6610f2' },
                        text: totalNodes.map(v => v.toFixed(0)),
                        textposition: 'auto',
                        yaxis: 'y'
                    },
                    {
                        x: expNames,
                        y: totalRollouts,
                        type: 'bar',
                        name: 'Total Rollouts',
                        marker: { color: '#e83e8c' },
                        text: totalRollouts.map(v => v.toFixed(0)),
                        textposition: 'auto',
                        yaxis: 'y2'
                    }
                ], {
                    title: {
                        text: 'Search Efficiency: Tree Nodes vs Rollout Simulations',
                        font: { size: 16 }
                    },
                    xaxis: { 
                        title: 'Experiment',
                        tickangle: -45
                    },
                    yaxis: { 
                        title: 'Average Total Nodes',
                        side: 'left',
                        color: '#6610f2'
                    },
                    yaxis2: { 
                        title: 'Average Total Rollouts',
                        side: 'right',
                        overlaying: 'y',
                        color: '#e83e8c'
                    },
                    barmode: 'group',
                    margin: { b: 150 }
                });
            }

            // ⚡ EFFICIENCY METRICS PLOT
            if (rolloutEfficiency.some(val => val > 0) || nodeEfficiency.some(val => val > 0)) {
                const efficiencyPlotDiv = document.createElement('div');
                efficiencyPlotDiv.id = 'efficiency-metrics-plot';
                efficiencyPlotDiv.className = 'plot-container';
                container.appendChild(efficiencyPlotDiv);

                const efficiencyTraces = [];
                if (rolloutEfficiency.some(val => val > 0)) {
                    efficiencyTraces.push({
                        x: expNames,
                        y: rolloutEfficiency,
                        type: 'bar',
                        name: '⚡ Rollout Efficiency',
                        marker: { color: '#20c997' },
                        text: rolloutEfficiency.map(v => v.toFixed(1) + '%'),
                        textposition: 'auto',
                        yaxis: 'y'
                    });
                }
                if (nodeEfficiency.some(val => val > 0)) {
                    efficiencyTraces.push({
                        x: expNames,
                        y: nodeEfficiency.map(v => v * 100), // Scale for better visibility
                        type: 'bar',
                        name: '🎯 Node Efficiency',
                        marker: { color: '#fd7e14' },
                        text: nodeEfficiency.map(v => (v * 100).toFixed(2) + '%'),
                        textposition: 'auto',
                        yaxis: 'y'
                    });
                }

                Plotly.newPlot('efficiency-metrics-plot', efficiencyTraces, {
                    title: {
                        text: '⚡ SEARCH EFFICIENCY: How Effectively Methods Convert Computation to Success',
                        font: { size: 16, color: '#2c3e50' }
                    },
                    xaxis: {
                        title: 'Experiment Method',
                        tickangle: -45
                    },
                    yaxis: {
                        title: 'Efficiency (%)',
                        side: 'left'
                    },
                    barmode: 'group',
                    margin: { b: 150, t: 70 },
                    showlegend: true
                });

                const explanation2 = document.createElement('div');
                explanation2.style.cssText = 'background-color: #e8f5e8; padding: 15px; border-left: 4px solid #28a745; margin: 10px 0; border-radius: 0 5px 5px 0;';
                explanation2.innerHTML = `
                    <h5>⚡ Understanding Efficiency</h5>
                    <p><strong>Rollout Efficiency:</strong> % of rollouts that achieved perfect success. <strong>Node Efficiency:</strong> Perfect successes per node explored. Higher values mean better computational efficiency.</p>
                `;
                container.appendChild(explanation2);
            }

            // 🕰️ EARLY SUCCESS METRICS PLOT
            if (depthToSuccess.some(val => val > 0 && val !== -1) || rolloutsToSuccess.some(val => val > 0 && val !== -1)) {
                const earlyPlotDiv = document.createElement('div');
                earlyPlotDiv.id = 'early-success-plot';
                earlyPlotDiv.className = 'plot-container';
                container.appendChild(earlyPlotDiv);

                // Filter out -1 values (no success found)
                const validDepths = depthToSuccess.map((val, idx) => val > 0 && val !== -1 ? val : null);
                const validRollouts = rolloutsToSuccess.map((val, idx) => val > 0 && val !== -1 ? val : null);

                const earlyTraces = [];
                if (validDepths.some(val => val !== null)) {
                    earlyTraces.push({
                        x: expNames,
                        y: validDepths,
                        type: 'bar',
                        name: '🌊 Depth to First Success',
                        marker: { color: '#6f42c1' },
                        text: validDepths.map(v => v !== null ? v.toFixed(1) : 'No Success'),
                        textposition: 'auto',
                        yaxis: 'y'
                    });
                }
                if (validRollouts.some(val => val !== null)) {
                    earlyTraces.push({
                        x: expNames,
                        y: validRollouts,
                        type: 'bar',
                        name: '🎲 Rollouts to First Success',
                        marker: { color: '#e83e8c' },
                        text: validRollouts.map(v => v !== null ? v.toFixed(0) : 'No Success'),
                        textposition: 'auto',
                        yaxis: 'y2'
                    });
                }

                if (earlyTraces.length > 0) {
                    Plotly.newPlot('early-success-plot', earlyTraces, {
                        title: {
                            text: '🕰️ EARLY SUCCESS: How Quickly Methods Find First Perfect Solution',
                            font: { size: 16, color: '#2c3e50' }
                        },
                        xaxis: {
                            title: 'Experiment Method',
                            tickangle: -45
                        },
                        yaxis: {
                            title: 'Depth to Success',
                            side: 'left',
                            color: '#6f42c1'
                        },
                        yaxis2: {
                            title: 'Rollouts to Success',
                            side: 'right',
                            overlaying: 'y',
                            color: '#e83e8c'
                        },
                        barmode: 'group',
                        margin: { b: 150, t: 70 },
                        showlegend: true
                    });

                    const explanation3 = document.createElement('div');
                    explanation3.style.cssText = 'background-color: #f3e5f5; padding: 15px; border-left: 4px solid #6f42c1; margin: 10px 0; border-radius: 0 5px 5px 0;';
                    explanation3.innerHTML = `
                        <h5>🕰️ Understanding Early Success</h5>
                        <p>These metrics show how quickly methods find their first perfect solution. Lower values indicate faster discovery. Methods that find solutions quickly are more efficient and practical.</p>
                    `;
                    container.appendChild(explanation3);
                }
            }

            // Add enhanced explanation text
            const explanationDiv = document.createElement('div');
            explanationDiv.style.cssText = 'background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 20px;';
            explanationDiv.innerHTML = `
                <h4>📈 Complete Tree Search Performance Guide:</h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-top: 15px;">
                    <div>
                        <h5>🏆 Key Performance Indicators</h5>
                        <ul>
                            <li><strong>Tree Success Rate:</strong> THE most important metric - % of problems where tree found perfect solution</li>
                            <li><strong>Perfect Success Rate:</strong> % of rollouts achieving reward = 1.0</li>
                            <li><strong>Rollout Efficiency:</strong> How many rollouts lead to perfect success</li>
                            <li><strong>Node Efficiency:</strong> Perfect successes per computational node</li>
                        </ul>
                    </div>
                    <div>
                        <h5>⚡ Efficiency & Speed</h5>
                        <ul>
                            <li><strong>Early Success Metrics:</strong> How quickly methods find first solution</li>
                            <li><strong>Search Efficiency Score:</strong> Balances success vs computational cost</li>
                            <li><strong>Exploration/Exploitation:</strong> Balance between broad search vs deep investigation</li>
                            <li><strong>Tree Structure:</strong> Branching patterns and depth distributions</li>
                        </ul>
                    </div>
                </div>
                <div style="background-color: #e7f3ff; padding: 10px; border-radius: 4px; margin-top: 15px;">
                    <strong>💡 Analysis Tip:</strong> Focus on Tree Success Rate first, then examine efficiency metrics to understand computational trade-offs. Methods with high tree success but low rollout efficiency may need optimization.
                </div>
            `;
            container.appendChild(explanationDiv);
        }
        
        function generateSeriesAnalysis() {
            const container = document.getElementById('series-analysis');
            
            if (!seriesMode || !seriesData || Object.keys(seriesData).length === 0) {
                container.innerHTML = '<p>No series data available. Use --series-mode to enable series analysis.</p>';
                return;
            }
            
            let html = `
                <div class="explanation-box">
                    <h4>📈 Series Analysis</h4>
                    <p>This analysis shows how different experiment types (A, B, C...) perform as compute resources increase. Each line represents a different experimental approach, allowing you to compare scaling behavior.</p>
                </div>
            `;
            
            // Create line plots for each important metric
            const metrics = ['tree_success', 'success_rate_1', 'success_rate_05', 'rollout_efficiency', 'total_rollouts', 'total_nodes'];
            
            for (const metric of metrics) {
                html += `<div id="series-${metric}-plot" class="plot-container"></div>`;
            }
            
            container.innerHTML = html;
            
            // Generate the actual plots
            generateSeriesPlots();
        }
        
        function generateSeriesPlots() {
            if (!seriesMode || !seriesData) return;
            
            const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'];
            
            const metrics = [
                { key: 'tree_success', title: '🏆 Tree Success Rate (Most Important)', yaxis: 'Tree Success (%)', multiplier: 100 },
                { key: 'success_rate_1', title: 'Perfect Rollout Success Rate', yaxis: 'Success Rate (%)', multiplier: 100 },
                { key: 'success_rate_05', title: 'Good Rollout Success Rate (≥ 0.5)', yaxis: 'Success Rate (%)', multiplier: 100 },
                { key: 'rollout_efficiency', title: '⚡ Rollout Efficiency', yaxis: 'Efficiency (%)', multiplier: 100 },
                { key: 'total_rollouts', title: 'Total Rollouts (Computational Cost)', yaxis: 'Average Rollouts', multiplier: 1 },
                { key: 'total_nodes', title: 'Total Nodes Explored', yaxis: 'Average Nodes', multiplier: 1 }
            ];
            
            metrics.forEach((metric, metricIndex) => {
                const traces = [];
                
                Object.entries(seriesData).forEach(([seriesName, experiments], seriesIndex) => {
                    const xValues = [];
                    const yValues = [];
                    
                    experiments.forEach(exp => {
                        const expData = experimentsData[exp.name];
                        if (expData && expData.aggregate[metric.key]) {
                            // Use custom x-axis values if provided, otherwise use computed compute levels
                            const xVal = xAxisData.length > 0 && xAxisData[exp.global_index] !== undefined
                                       ? xAxisData[exp.global_index]
                                       : exp.compute;
                            
                            xValues.push(xVal);
                            yValues.push(expData.aggregate[metric.key].mean * metric.multiplier);
                        }
                    });
                    
                    if (xValues.length > 0) {
                        // Use different display modes based on number of points
                        const traceMode = xValues.length === 1 ? 'markers' : 'lines+markers';
                        const markerSize = xValues.length === 1 ? 12 : 8; // Larger markers for single points

                        const trace = {
                            x: xValues,
                            y: yValues,
                            type: 'scatter',
                            mode: traceMode,
                            name: seriesName,
                            marker: {
                                size: markerSize,
                                color: colors[seriesIndex % colors.length]
                            }
                        };

                        // Only add line properties for multi-point series
                        if (xValues.length > 1) {
                            trace.line = {
                                color: colors[seriesIndex % colors.length],
                                width: 3
                            };
                        }

                        traces.push(trace);
                    }
                });
                
                if (traces.length > 0) {
                    Plotly.newPlot(`series-${metric.key}-plot`, traces, {
                        title: {
                            text: `${metric.title} vs Compute`,
                            font: { size: 16 }
                        },
                        xaxis: {
                            title: xAxisData.length > 0 ? 'Number of Simulations' : 'Compute Level',
                            type: 'linear'
                        },
                        yaxis: { 
                            title: metric.yaxis
                        },
                        showlegend: true,
                        legend: {
                            x: 1.02,
                            y: 0.98,
                            xanchor: 'left',
                            bgcolor: 'rgba(255,255,255,0.8)',
                            bordercolor: 'rgba(0,0,0,0.2)',
                            borderwidth: 1
                        },
                        margin: { l: 60, r: 150, t: 60, b: 60 }
                    });
                }
            });
        }
        
        function generateExhaustiveGraphs() {
            const container = document.getElementById('exhaustive-container');

            // Check if already generated
            if (container.children.length > 0) {
                return;
            }

            container.innerHTML =
                '<div style="background-color: #e7f3ff; padding: 20px; border-radius: 8px; margin-bottom: 25px; border-left: 5px solid #007bff;">' +
                    '<h4>📊 Complete Metric Analysis</h4>' +
                    '<p>This section provides exhaustive visualizations for all available tree search performance metrics. Each graph includes detailed explanations of what the metric calculates and how to interpret it for algorithm evaluation.</p>' +
                    '<p><strong>🎯 Purpose:</strong> Deep-dive analysis for researchers who want to understand every aspect of tree search performance and behavior patterns.</p>' +
                '</div>' +
                '<div id="exhaustive-plots"></div>';

            const plotsContainer = document.getElementById('exhaustive-plots');

            // Get all available metrics from the first experiment
            const expNames = Object.keys(experimentsData || {});
            if (expNames.length === 0) {
                plotsContainer.innerHTML = '<p>No experiment data available.</p>';
                return;
            }

            const firstExp = experimentsData[expNames[0]];
            if (!firstExp || !firstExp.aggregate) {
                plotsContainer.innerHTML = '<p>No aggregate data available.</p>';
                return;
            }

            // Define metric categories with comprehensive descriptions
            const metricCategories = {
                'Core Success Metrics': {
                    color: '#dc3545',
                    description: 'Fundamental measures of whether the tree search successfully solves problems',
                    metrics: [
                        {
                            key: 'tree_success',
                            title: '🏆 Tree Success Rate',
                            description: 'The most important metric: percentage of problems where the search tree found at least one perfect solution (reward = 1.0). This binary metric indicates the fundamental problem-solving capability of the method.',
                            calculation: 'For each problem: 1 if any rollout achieved reward = 1.0, else 0. Then average across all problems.',
                            interpretation: 'Higher is better. Values above 50% indicate strong performance. This metric answers: "Can the method solve problems?"',
                            multiplier: 100,
                            format: '%.1f%%'
                        },
                        {
                            key: 'tree_has_any_success',
                            title: '🎯 Tree Has Any Success',
                            description: 'Percentage of problems where the tree found any positive reward (> 0). Less strict than perfect success.',
                            calculation: 'For each problem: 1 if any rollout achieved reward > 0, else 0. Then average across all problems.',
                            interpretation: 'Shows how often the method finds partial solutions. Gap with tree_success indicates room for improvement.',
                            multiplier: 100,
                            format: '%.1f%%'
                        },
                        {
                            key: 'tree_has_good_success',
                            title: '📈 Tree Has Good Success',
                            description: 'Percentage of problems where the tree found good solutions (reward ≥ 0.5). Intermediate success level.',
                            calculation: 'For each problem: 1 if any rollout achieved reward ≥ 0.5, else 0. Then average across all problems.',
                            interpretation: 'Measures consistent good performance. Bridge between any success and perfect success.',
                            multiplier: 100,
                            format: '%.1f%%'
                        }
                    ]
                },
                'Rollout Performance': {
                    color: '#28a745',
                    description: 'Detailed analysis of individual rollout simulation performance',
                    metrics: [
                        {
                            key: 'success_rate_1',
                            title: '🏅 Perfect Rollout Success Rate',
                            description: 'Percentage of individual rollouts that achieved perfect performance (reward = 1.0).',
                            calculation: 'Count rollouts with reward = 1.0, divide by total rollouts, multiply by 100.',
                            interpretation: 'Measures rollout quality. Higher values mean more rollouts lead to perfect solutions.',
                            multiplier: 100,
                            format: '%.1f%%'
                        },
                        {
                            key: 'success_rate_05',
                            title: '🥈 Good Rollout Success Rate',
                            description: 'Percentage of rollouts achieving good performance (reward ≥ 0.5).',
                            calculation: 'Count rollouts with reward ≥ 0.5, divide by total rollouts, multiply by 100.',
                            interpretation: 'Broader measure of rollout effectiveness. Shows consistency of good outcomes.',
                            multiplier: 100,
                            format: '%.1f%%'
                        },
                        {
                            key: 'rollout_efficiency',
                            title: '⚡ Rollout Efficiency',
                            description: 'Same as perfect rollout success rate - measures computational efficiency of rollouts.',
                            calculation: 'successful_rollouts_1 / total_rollouts × 100',
                            interpretation: 'Key efficiency metric. Higher values mean less wasted computation.',
                            multiplier: 100,
                            format: '%.2f%%'
                        }
                    ]
                },
                'Search Efficiency': {
                    color: '#007bff',
                    description: 'Metrics measuring how efficiently the search process uses computational resources',
                    metrics: [
                        {
                            key: 'node_efficiency',
                            title: '🎯 Node Efficiency',
                            description: 'Perfect rollouts achieved per node explored. Measures computational efficiency.',
                            calculation: 'successful_rollouts_1 / total_nodes',
                            interpretation: 'Higher values indicate better use of exploration. More solutions per node explored.',
                            multiplier: 100,
                            format: '%.3f%%'
                        },
                        {
                            key: 'search_efficiency_score',
                            title: '📊 Search Efficiency Score',
                            description: 'Comprehensive efficiency metric balancing success against computational cost (nodes × depth).',
                            calculation: 'successful_rollouts_1 / (total_nodes × max_depth)',
                            interpretation: 'Accounts for both breadth and depth cost. Higher scores indicate more efficient search strategies.',
                            multiplier: 1000,
                            format: '%.3f'
                        },
                        {
                            key: 'perfect_success_density',
                            title: '💎 Perfect Success Density',
                            description: 'Perfect rollouts per node explored. Alternative view of node efficiency.',
                            calculation: 'successful_rollouts_1 / total_nodes',
                            interpretation: 'Same as node efficiency but different perspective. Focus on solution density.',
                            multiplier: 1,
                            format: '%.4f'
                        }
                    ]
                },
                'Early Discovery': {
                    color: '#6f42c1',
                    description: 'Metrics measuring how quickly methods discover their first perfect solution',
                    metrics: [
                        {
                            key: 'depth_to_first_success',
                            title: '🌊 Depth to First Success',
                            description: 'Tree depth where the first perfect solution was discovered. Lower is better.',
                            calculation: 'Record depth when first rollout achieves reward = 1.0. -1 if no success found.',
                            interpretation: 'Shows search strategy effectiveness. Lower depths indicate efficient exploration paths.',
                            multiplier: 1,
                            format: '%.1f',
                            special: 'exclude_negative'
                        },
                        {
                            key: 'rollouts_to_first_success',
                            title: '🎲 Rollouts to First Success',
                            description: 'Number of rollouts performed before finding the first perfect solution.',
                            calculation: 'Count total rollouts up to and including first perfect rollout. -1 if no success.',
                            interpretation: 'Computational cost to first solution. Lower values indicate faster discovery.',
                            multiplier: 1,
                            format: '%.0f',
                            special: 'exclude_negative'
                        },
                        {
                            key: 'nodes_to_first_success',
                            title: '🗺️ Nodes to First Success',
                            description: 'Number of nodes explored before finding the first perfect solution.',
                            calculation: 'Count total nodes explored up to first perfect rollout. -1 if no success.',
                            interpretation: 'Exploration cost to first solution. Indicates search strategy efficiency.',
                            multiplier: 1,
                            format: '%.0f',
                            special: 'exclude_negative'
                        }
                    ]
                },
                'Tree Structure': {
                    color: '#fd7e14',
                    description: 'Analysis of tree shape, exploration patterns, and structural characteristics',
                    metrics: [
                        {
                            key: 'total_nodes',
                            title: '🌳 Total Nodes Explored',
                            description: 'Total number of nodes in the search tree. Indicates exploration breadth.',
                            calculation: 'Count all nodes in the search tree structure.',
                            interpretation: 'Computational cost indicator. More nodes = more exploration but higher cost.',
                            multiplier: 1,
                            format: '%.0f'
                        },
                        {
                            key: 'max_depth',
                            title: '📏 Maximum Depth',
                            description: 'Deepest level reached in the search tree. Indicates exploration depth.',
                            calculation: 'Maximum distance from root to any leaf node.',
                            interpretation: 'Shows how deep the search goes. Higher depths may indicate thorough exploration.',
                            multiplier: 1,
                            format: '%.0f'
                        },
                        {
                            key: 'branching_factor',
                            title: '🌿 Branching Factor',
                            description: 'Average number of children per non-terminal node. Measures tree breadth.',
                            calculation: 'total_children / non_terminal_nodes',
                            interpretation: 'Higher values indicate broader exploration at each step.',
                            multiplier: 1,
                            format: '%.2f'
                        },
                        {
                            key: 'terminal_nodes',
                            title: '🍃 Terminal Nodes',
                            description: 'Number of leaf nodes (endpoints) in the search tree.',
                            calculation: 'Count nodes with no children.',
                            interpretation: 'Shows exploration endpoints. More terminals may indicate thorough search.',
                            multiplier: 1,
                            format: '%.0f'
                        },
                        {
                            key: 'terminal_ratio',
                            title: '📊 Terminal Ratio',
                            description: 'Proportion of nodes that are terminals. Indicates tree shape.',
                            calculation: 'terminal_nodes / total_nodes',
                            interpretation: 'Higher ratios indicate more leaf-heavy trees. Lower ratios suggest deeper exploration.',
                            multiplier: 100,
                            format: '%.1f%%'
                        }
                    ]
                },
                'Exploration Quality': {
                    color: '#20c997',
                    description: 'Advanced metrics analyzing the quality and patterns of tree exploration',
                    metrics: [
                        {
                            key: 'exploration_to_exploitation_ratio',
                            title: '🔄 Exploration vs Exploitation',
                            description: 'Ratio of unique nodes explored to total rollouts. Measures exploration breadth.',
                            calculation: 'total_nodes / total_rollouts',
                            interpretation: 'Higher values indicate more exploration. Lower values suggest more exploitation.',
                            multiplier: 1,
                            format: '%.3f'
                        },
                        {
                            key: 'unique_states_explored',
                            title: '🗺️ Unique States Explored',
                            description: 'Estimated number of unique states based on thought text diversity.',
                            calculation: 'Count unique thought_text values across all nodes.',
                            interpretation: 'Proxy for search diversity. More unique states indicate broader exploration.',
                            multiplier: 1,
                            format: '%.0f'
                        },
                        {
                            key: 'visit_concentration',
                            title: '🎯 Visit Concentration',
                            description: 'Gini coefficient-style measure of how concentrated visits are across nodes.',
                            calculation: 'Modified Gini coefficient on visit counts. 0=uniform, 1=highly concentrated.',
                            interpretation: 'Higher values indicate focused search. Lower values suggest broader exploration.',
                            multiplier: 1,
                            format: '%.3f'
                        },
                        {
                            key: 'depth_distribution_variance',
                            title: '📊 Depth Distribution Variance',
                            description: 'Variance in node depth distribution. Measures tree balance.',
                            calculation: 'Variance of depth values across all nodes.',
                            interpretation: 'Higher values indicate unbalanced trees. Lower values suggest more uniform exploration.',
                            multiplier: 1,
                            format: '%.2f'
                        }
                    ]
                },
                'Success Pathways': {
                    color: '#e83e8c',
                    description: 'Analysis of successful solution paths and terminal node performance',
                    metrics: [
                        {
                            key: 'successful_paths',
                            title: '🛤️ Successful Paths',
                            description: 'Number of root-to-leaf paths containing at least one perfect rollout.',
                            calculation: 'Count paths from root to terminal nodes that contain rollouts with reward = 1.0.',
                            interpretation: 'More successful paths indicate multiple solution routes. Redundancy can be good.',
                            multiplier: 1,
                            format: '%.0f'
                        },
                        {
                            key: 'terminal_success_rate',
                            title: '🎯 Terminal Success Rate',
                            description: 'Fraction of terminal nodes that contain perfect rollouts.',
                            calculation: 'Count terminal nodes with perfect rollouts / total terminal nodes.',
                            interpretation: 'Shows how often leaf exploration leads to success. Higher is better.',
                            multiplier: 100,
                            format: '%.1f%%'
                        }
                    ]
                },
                'Computational Metrics': {
                    color: '#6c757d',
                    description: 'Basic computational cost and resource usage metrics',
                    metrics: [
                        {
                            key: 'total_rollouts',
                            title: '🎲 Total Rollouts',
                            description: 'Total number of rollout simulations performed across all nodes.',
                            calculation: 'Sum of rollouts across all nodes in the tree.',
                            interpretation: 'Primary computational cost metric. More rollouts = higher computational cost.',
                            multiplier: 1,
                            format: '%.0f'
                        },
                        {
                            key: 'rollouts_per_node',
                            title: '🔢 Rollouts per Node',
                            description: 'Average number of rollouts performed per node. Search intensity measure.',
                            calculation: 'total_rollouts / total_nodes',
                            interpretation: 'Higher values indicate more intensive local search at each node.',
                            multiplier: 1,
                            format: '%.1f'
                        },
                        {
                            key: 'search_time',
                            title: '⏱️ Search Time',
                            description: 'Total time spent on search in seconds (when available).',
                            calculation: 'Sum of search time across all problems.',
                            interpretation: 'Wall-clock computational cost. Should be considered alongside success metrics.',
                            multiplier: 1,
                            format: '%.2f sec'
                        }
                    ]
                }
            };

            // Generate graphs for each category
            for (const [categoryName, categoryInfo] of Object.entries(metricCategories)) {
                generateMetricCategoryGraphs(plotsContainer, categoryName, categoryInfo, expNames);
            }
        }

        function generateMetricCategoryGraphs(container, categoryName, categoryInfo, expNames) {
            // Create category section
            const categoryDiv = document.createElement('div');
            categoryDiv.style.cssText = 'margin: 30px 0; border: 1px solid #ddd; border-radius: 8px; overflow: hidden;';

            // Category header
            const headerDiv = document.createElement('div');
            headerDiv.style.cssText = `background-color: ${categoryInfo.color}; color: white; padding: 20px; font-weight: bold; font-size: 18px;`;
            headerDiv.innerHTML = `${categoryName}<br><small style="opacity: 0.9; font-weight: normal;">${categoryInfo.description}</small>`;
            categoryDiv.appendChild(headerDiv);

            // Category content
            const contentDiv = document.createElement('div');
            contentDiv.style.cssText = 'padding: 20px; background-color: #f8f9fa;';

            // Generate graphs for each metric in category
            categoryInfo.metrics.forEach((metric, index) => {
                generateSingleMetricGraph(contentDiv, metric, expNames, categoryInfo.color, index);
            });

            categoryDiv.appendChild(contentDiv);
            container.appendChild(categoryDiv);
        }

        function generateSingleMetricGraph(container, metric, expNames, color, index) {
            // Extract metric values
            const values = expNames.map(name => {
                const value = safeExtractMetric(name, `aggregate.${metric.key}.mean`);
                // Handle special cases
                if (metric.special === 'exclude_negative' && value < 0) {
                    return null;
                }
                return value * metric.multiplier;
            });

            // Check if we have valid data
            if (values.every(v => v === null || v === 0)) {
                return; // Skip if no data
            }

            // Create metric section
            const metricDiv = document.createElement('div');
            metricDiv.style.cssText = 'margin: 25px 0; padding: 20px; background-color: white; border-radius: 8px; border-left: 4px solid ' + color + ';';

            // Metric header with description
            const descDiv = document.createElement('div');
            descDiv.style.cssText = 'margin-bottom: 20px;';
            descDiv.innerHTML =
                '<h4 style="margin: 0 0 10px 0; color: ' + color + ';">' + metric.title + '</h4>' +
                '<p style="margin: 5px 0; font-size: 14px;"><strong>What it measures:</strong> ' + metric.description + '</p>' +
                '<p style="margin: 5px 0; font-size: 14px; color: #666;"><strong>How it\\'s calculated:</strong> ' + metric.calculation + '</p>' +
                '<p style="margin: 5px 0; font-size: 14px; color: #444;"><strong>How to interpret:</strong> ' + metric.interpretation + '</p>';
            metricDiv.appendChild(descDiv);

            // Create plot
            const plotDiv = document.createElement('div');
            plotDiv.id = 'exhaustive-' + metric.key + '-plot';
            plotDiv.className = 'plot-container';
            plotDiv.style.height = '300px';
            metricDiv.appendChild(plotDiv);

            // Add to container BEFORE calling Plotly to ensure DOM element exists
            container.appendChild(metricDiv);

            const trace = {
                x: expNames,
                y: values,
                type: 'bar',
                name: metric.title,
                marker: {
                    color: color,
                    opacity: 0.8,
                    line: { color: color, width: 1 }
                },
                text: values.map(v => {
                    if (v === null) return 'N/A';
                    try {
                        // Simple format parsing for %.Nf patterns
                        const formatMatch = metric.format.match(/%.(\\d+)f/);
                        if (formatMatch) {
                            const precision = parseInt(formatMatch[1]);
                            const formatted = v.toFixed(precision);
                            return metric.format.includes('%%') ? formatted + '%' : formatted;
                        }
                        return v.toString();
                    } catch (e) {
                        return v.toString();
                    }
                }),
                textposition: 'auto',
                textfont: { color: 'white', size: 11 }
            };

            Plotly.newPlot(plotDiv, [trace], {
                title: {
                    text: metric.title,
                    font: { size: 14, color: '#2c3e50' }
                },
                xaxis: {
                    title: 'Experiment',
                    tickangle: -45,
                    tickfont: { size: 10 }
                },
                yaxis: {
                    title: metric.title.replace(/🏆|📊|⚡|🎯|🌊|🎲|🗺️|🌳|📏|🌿|🍃|🔄|🎯|🛤️|🔢|⏱️|💎|🥈|🏅/g, '').trim(),
                    tickformat: metric.format.includes('%') ? '.1f' : '.2f'
                },
                margin: { b: 100, t: 50, l: 80, r: 40 },
                height: 280,
                paper_bgcolor: '#ffffff',
                plot_bgcolor: '#fafafa'
            });
        }

        function safeExtractMetric(expName, metricPath) {
            try {
                const pathParts = metricPath.split('.');
                let value = experimentsData[expName];
                for (const part of pathParts) {
                    value = value[part];
                    if (value === undefined) return 0;
                }
                return value;
            } catch (e) {
                return 0;
            }
        }

        // Initialize the page
        document.addEventListener('DOMContentLoaded', function() {
            try {
                generateOverview();
                generateMetricsComparison();
                generateStatisticalTests();

                // Show series tab if series mode is enabled
                if (seriesMode) {
                    document.getElementById('series-tab').style.display = 'block';
                    generateSeriesAnalysis();
                }
            } catch (error) {
                console.error('Error initializing page:', error);
                document.body.innerHTML += '<div style="color: red; padding: 20px; border: 1px solid red; margin: 20px;">Error loading data. Check console for details.</div>';
            }
        });
    </script>
</body>
</html>'''
    
    def generate_json_report(self, output_file: str, comparisons: dict):
        """Generate a JSON report with all data"""
        print(f"Generating JSON report: {output_file}")
        
        report = {
            'experiments': self.aggregate_stats,
            'comparisons': comparisons,
            'metadata': {
                'experiment_directories': self.experiment_dirs,
                'total_experiments': len(self.experiments),
                'scipy_available': SCIPY_AVAILABLE
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ JSON report saved to {os.path.abspath(output_file)}")
    
    def run_comparison(self, output_file: str = None, format: str = 'html', statistical_tests: bool = True):
        """Run the complete comparison workflow"""
        self.load_experiments()
        
        if len(self.experiments) == 0:
            print("Error: No experiments loaded")
            return
            
        self.compute_statistics()
        
        comparisons = {}
        if statistical_tests and len(self.experiments) >= 2:
            comparisons = self.perform_statistical_comparisons()
        
        # Generate output
        if output_file is None:
            if self.series_mode:
                output_file = f"experiment_series_comparison.{format}"
            else:
                exp_names = "_vs_".join([Path(d).name for d in self.experiment_dirs[:2]])
                output_file = f"experiment_comparison_{exp_names}.{format}"
        
        if format == 'html':
            self.generate_html_report(output_file, comparisons)
        elif format == 'json':
            self.generate_json_report(output_file, comparisons)
        else:
            print(f"Unknown format: {format}")
            return
        
        # Print summary
        print("\n=== Comparison Summary ===")
        
        if self.series_mode and self.experiment_series:
            print("\nExperiment Series Analysis:")
            for series_name, experiments in self.experiment_series.items():
                print(f"\n{series_name} Series:")
                for exp in experiments:
                    exp_data = self.aggregate_stats.get(exp['name'])
                    if exp_data:
                        success_rate = exp_data['aggregate'].get('success_rate_1', {}).get('mean', 0) * 100
                        print(f"  Compute {exp['compute']}: {success_rate:.1f}% perfect success")
        else:
            for exp_name, exp_data in self.aggregate_stats.items():
                print(f"\n{exp_name}:")
                print(f"  Problems: {exp_data['total_problems']}")
                
                if 'success_rate_1' in exp_data['aggregate']:
                    success_rate = exp_data['aggregate']['success_rate_1']['mean'] * 100
                    print(f"  Perfect Success Rate: {success_rate:.1f}%")
                
                if 'success_rate_05' in exp_data['aggregate']:
                    success_rate_05 = exp_data['aggregate']['success_rate_05']['mean'] * 100
                    print(f"  Success Rate ≥ 0.5: {success_rate_05:.1f}%")
                    
                if 'total_rollouts' in exp_data['aggregate']:
                    rollouts = exp_data['aggregate']['total_rollouts']['mean']
                    print(f"  Avg Total Rollouts: {rollouts:.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare MCTS experiment results with detailed statistical analysis",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "experiment_dirs",
        nargs='+',
        help="Two or more experiment directories to compare"
    )
    
    parser.add_argument(
        "--output",
        help="Output filename (extension determines format if --format not specified)"
    )
    
    parser.add_argument(
        "--groups",
        help="Manual experiment group assignments (comma-separated, e.g., 'ExpA,ExpA,ExpA,ExpB,ExpB,ExpB')"
    )
    
    parser.add_argument(
        "--format",
        choices=['html', 'json'],
        default='html',
        help="Output format (default: html)"
    )
    
    parser.add_argument(
        "--statistical-tests",
        action="store_true",
        default=True,
        help="Perform statistical significance tests (default: True)"
    )
    
    parser.add_argument(
        "--series-mode",
        action="store_true",
        help="Enable compute scaling analysis with line graphs"
    )
    
    parser.add_argument(
        "--x-axis",
        help="Custom x-axis values for series mode (comma-separated numbers)"
    )
    
    parser.add_argument(
        "--baseline",
        help="Specify baseline experiment name for comparisons"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output for debugging"
    )
    
    args = parser.parse_args()
    
    if len(args.experiment_dirs) < 2:
        print("Error: Need at least 2 experiment directories for comparison")
        return 1
    
    # Parse groups if provided
    groups = None
    if args.groups:
        groups = [g.strip() for g in args.groups.split(',')]
        if len(groups) != len(args.experiment_dirs):
            print(f"Error: Number of groups ({len(groups)}) must match number of experiments ({len(args.experiment_dirs)})")
            return 1
    
    # Parse x-axis values if provided
    x_axis_values = None
    if args.x_axis:
        try:
            x_axis_values = [float(x.strip()) for x in args.x_axis.split(',')]
            if len(x_axis_values) != len(args.experiment_dirs):
                print(f"Error: Number of x-axis values ({len(x_axis_values)}) must match number of experiments ({len(args.experiment_dirs)})")
                return 1
        except ValueError as e:
            print(f"Error parsing x-axis values: {e}")
            return 1
    
    # Auto-detect format from output filename if not specified
    if args.output and not args.format:
        if args.output.endswith('.json'):
            args.format = 'json'
        else:
            args.format = 'html'
    
    print("=== MCTS Experiment Comparison Tool ===")
    if args.series_mode:
        print("📈 Series Mode: Compute scaling analysis enabled")
    if groups:
        print("🧩 Manual Grouping Mode: Using specified experiment groups")
    if x_axis_values:
        print(f"📊 Custom X-axis: {x_axis_values}")
    print(f"Comparing {len(args.experiment_dirs)} experiments:")
    for i, exp_dir in enumerate(args.experiment_dirs):
        group_str = f" (group: {groups[i]})" if groups else ""
        x_str = f" (x: {x_axis_values[i]})" if x_axis_values else ""
        print(f"  - {exp_dir}{group_str}{x_str}")
    print(f"Output format: {args.format}")
    print(f"Statistical tests: {args.statistical_tests}")
    
    comparison = ExperimentComparison(
        args.experiment_dirs, 
        verbose=args.verbose,
        x_axis_values=x_axis_values,
        series_mode=args.series_mode,
        baseline_exp=args.baseline,
        groups=groups
    )
    comparison.run_comparison(
        output_file=args.output,
        format=args.format,
        statistical_tests=args.statistical_tests
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())