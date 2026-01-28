"""
Executive summary generation for eye tracking analysis.
"""
from __future__ import annotations

from typing import Dict, List
import pandas as pd
import numpy as np
from datetime import datetime

from state import state
from analysis import calculate_rankings, generate_statistics_table


def generate_executive_summary(
    aggregated_data: Dict,
    selected_groups: List[str],
    selected_tasks: List[str],
    active_parameters: List[str],
    mode: str
) -> str:
    """
    Generate a text summary of the analysis results.
    
    Returns a formatted string with key findings.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("EYE TRACKING ANALYSIS - EXECUTIVE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Aggregate data across all groups (ignore group boundaries)
    aggregated_all_data = {}
    for group_id, group_data in aggregated_data.items():
        for task_id, task_data in group_data.items():
            if task_id not in aggregated_all_data:
                aggregated_all_data[task_id] = {}
            
            # Collect all parameter values across all groups for this task
            if isinstance(task_data, dict):
                # Check if it's group mean mode or individual participant mode
                is_individual = any(
                    isinstance(k, str) and isinstance(v, dict) and 
                    any(isinstance(vv, dict) for vv in v.values() if isinstance(vv, dict))
                    for k, v in task_data.items() if k != "_group_stats"
                )
                
                if is_individual:
                    # Individual participant mode: collect all participant values
                    for participant, participant_data in task_data.items():
                        if participant == "_group_stats":
                            continue
                        if isinstance(participant_data, dict):
                            for parameter, metrics in participant_data.items():
                                if parameter not in aggregated_all_data[task_id]:
                                    aggregated_all_data[task_id][parameter] = []
                                if isinstance(metrics, dict) and 'mean' in metrics:
                                    aggregated_all_data[task_id][parameter].append(metrics['mean'])
                else:
                    # Group mean mode: collect group mean values
                    for parameter, metrics in task_data.items():
                        if parameter == "_group_stats":
                            continue
                        if isinstance(metrics, dict) and 'mean' in metrics:
                            if parameter not in aggregated_all_data[task_id]:
                                aggregated_all_data[task_id][parameter] = []
                            aggregated_all_data[task_id][parameter].append(metrics['mean'])
    
    # Calculate overall statistics (ignoring groups)
    overall_data = {}
    for task_id, task_params in aggregated_all_data.items():
        overall_data[task_id] = {}
        for parameter, values in task_params.items():
            if values:
                overall_data[task_id][parameter] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
    
    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"Analysis Mode: {mode}")
    lines.append(f"Number of Groups: {len(selected_groups)} (aggregated)")
    lines.append(f"Number of Tasks: {len(selected_tasks)}")
    lines.append(f"Parameters Analyzed: {len(active_parameters)}")
    lines.append("")
    lines.append("Note: Statistics are calculated across all participants, ignoring group boundaries.")
    lines.append("")
    
    # Task information
    lines.append("TASKS ANALYZED:")
    for task_id in selected_tasks:
        task_label = state.format_task(task_id)
        lines.append(f"  - {task_label}")
    lines.append("")
    
    # Key findings for each parameter
    lines.append("KEY FINDINGS")
    lines.append("-" * 80)
    
    for parameter in active_parameters:
        if parameter == "Standard Deviation of TCT":
            continue  # Skip - calculated differently
        
        lines.append(f"\n{parameter}:")
        
        # Calculate rankings from overall aggregated data (ignoring groups)
        task_values = {}
        for task_id, task_data in overall_data.items():
            if parameter in task_data:
                task_values[task_id] = task_data[parameter]['mean']
        
        if not task_values:
            lines.append("  No data available for this parameter.")
            continue
        
        # Find highest and lowest tasks
        if task_values:
            sorted_tasks = sorted(task_values.items(), key=lambda x: x[1], reverse=True)
            highest_task_id, value_h = sorted_tasks[0]
            lowest_task_id, value_l = sorted_tasks[-1]
            
            task_label_h = state.format_task(highest_task_id)
            task_label_l = state.format_task(lowest_task_id)
            
            lines.append(f"  Highest: {task_label_h} - {value_h:.4f}")
            lines.append(f"  Lowest: {task_label_l} - {value_l:.4f}")
            
            # Calculate range
            if value_h > value_l:
                range_pct = ((value_h - value_l) / value_l * 100) if value_l > 0 else 0
                lines.append(f"  Range: {value_h - value_l:.4f} ({range_pct:.1f}% difference)")
    
    lines.append("")
    
    # Narrative summary (universal formulations filled with actual values)
    lines.append("NARRATIVE SUMMARY")
    lines.append("-" * 80)
    tct_name = "Task Completion Time (TCT)"
    std_tct_name = "Standard Deviation of TCT"
    params_for_variation = [p for p in active_parameters if p != std_tct_name]
    
    # Longest / shortest TCT (TCT is in ms in overall_data)
    if overall_data and any(tct_name in pdata for pdata in overall_data.values()):
        task_tct_means = [
            (tid, overall_data[tid][tct_name]["mean"])
            for tid in overall_data
            if tct_name in overall_data[tid]
        ]
        if task_tct_means:
            longest_tid, tct_ms_max = max(task_tct_means, key=lambda x: x[1])
            shortest_tid, tct_ms_min = min(task_tct_means, key=lambda x: x[1])
            longest_label = state.format_task(longest_tid)
            shortest_label = state.format_task(shortest_tid)
            lines.append(
                f"Task {longest_label} had the longest completion time ({tct_ms_max / 1000.0:.1f} s), "
                "consistent with its high cognitive demand."
            )
            if longest_tid != shortest_tid:
                lines.append(
                    f"Task {shortest_label} had the shortest completion time ({tct_ms_min / 1000.0:.1f} s), "
                    "indicating relatively low cognitive demand."
                )
            lines.append(
                f"The largest difference in completion time was between {longest_label} (most demanding) "
                f"and {shortest_label} (least demanding)."
            )
    else:
        lines.append("Task completion time data not available for narrative summary.")
    
    # Most varied parameter across tasks
    if overall_data and params_for_variation:
        best_param = None
        best_range = -1.0
        task_high = task_low = None
        for param in params_for_variation:
            means = [
                (tid, overall_data[tid][param]["mean"])
                for tid in overall_data
                if param in overall_data[tid]
            ]
            if len(means) < 2:
                continue
            vals = [m[1] for m in means]
            r = max(vals) - min(vals)
            if r > best_range:
                best_range = r
                best_param = param
                task_high = max(means, key=lambda x: x[1])[0]
                task_low = min(means, key=lambda x: x[1])[0]
        if best_param is not None:
            lines.append(
                f"The metric that varied most across tasks was {best_param}, with the highest value "
                f"in {state.format_task(task_high)} and the lowest in {state.format_task(task_low)}."
            )
    
    lines.append(
        f"Analysis included {len(selected_groups)} group(s); statistics are aggregated across all participants."
    )
    lines.append("")
    
    # Overall statistics summary (across all tasks and groups)
    lines.append("OVERALL STATISTICS SUMMARY")
    lines.append("-" * 80)
    
    for parameter in active_parameters:
        if parameter == "Standard Deviation of TCT":
            continue
        
        lines.append(f"\n{parameter}:")
        
        # Calculate overall mean across all tasks
        all_values = []
        for task_id, task_data in overall_data.items():
            if parameter in task_data:
                all_values.append(task_data[parameter]['mean'])
        
        if all_values:
            overall_mean = float(np.mean(all_values))
            overall_std = float(np.std(all_values, ddof=1)) if len(all_values) > 1 else 0.0
            overall_min = float(np.min(all_values))
            overall_max = float(np.max(all_values))
            
            lines.append(f"  Overall Mean: {overall_mean:.4f}")
            lines.append(f"  Overall Std Dev: {overall_std:.4f}")
            lines.append(f"  Range: {overall_min:.4f} to {overall_max:.4f}")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def format_statistics_table_for_summary(
    aggregated_data: Dict,
    selected_tasks: List[str],
    selected_groups: List[str],
    mode: str
) -> str:
    """
    Format statistics table as a text string for inclusion in summary.
    Note: This function aggregates data across all groups (ignoring group boundaries).
    """
    # Aggregate data across all groups
    aggregated_all_data = {}
    for group_id, group_data in aggregated_data.items():
        for task_id, task_data in group_data.items():
            if task_id not in aggregated_all_data:
                aggregated_all_data[task_id] = {}
            
            if isinstance(task_data, dict):
                is_individual = any(
                    isinstance(k, str) and isinstance(v, dict) and 
                    any(isinstance(vv, dict) for vv in v.values() if isinstance(vv, dict))
                    for k, v in task_data.items() if k != "_group_stats"
                )
                
                if is_individual:
                    for participant, participant_data in task_data.items():
                        if participant == "_group_stats":
                            continue
                        if isinstance(participant_data, dict):
                            for parameter, metrics in participant_data.items():
                                if parameter not in aggregated_all_data[task_id]:
                                    aggregated_all_data[task_id][parameter] = []
                                if isinstance(metrics, dict) and 'mean' in metrics:
                                    aggregated_all_data[task_id][parameter].append(metrics['mean'])
                else:
                    for parameter, metrics in task_data.items():
                        if parameter == "_group_stats":
                            continue
                        if isinstance(metrics, dict) and 'mean' in metrics:
                            if parameter not in aggregated_all_data[task_id]:
                                aggregated_all_data[task_id][parameter] = []
                            aggregated_all_data[task_id][parameter].append(metrics['mean'])
    
    # Create a simplified aggregated structure for generate_statistics_table
    # Create a single "ALL" group with aggregated means
    simplified_data = {"ALL": {}}
    for task_id, task_params in aggregated_all_data.items():
        simplified_data["ALL"][task_id] = {}
        for parameter, values in task_params.items():
            if values:
                simplified_data["ALL"][task_id][parameter] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'q1': float(np.percentile(values, 25)),
                    'q3': float(np.percentile(values, 75)),
                }
    
    stats_df = generate_statistics_table(
        simplified_data,
        selected_tasks,
        ["ALL"],
        "Only group mean"
    )
    
    if stats_df.empty:
        return "No statistics available."
    
    # Convert to string representation
    return stats_df.to_string(index=False)


def export_summary_to_text(
    summary_text: str,
    statistics_text: str,
    filename: str
) -> None:
    """
    Export executive summary to a text file.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(summary_text)
        f.write("\n\n")
        f.write("DETAILED STATISTICS")
        f.write("\n")
        f.write("-" * 80)
        f.write("\n\n")
        f.write(statistics_text)


def export_summary_to_pdf(
    summary_text: str,
    statistics_text: str,
    filename: str
) -> None:
    """
    Export executive summary to PDF using matplotlib's PDF backend.
    """
    try:
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib.figure import Figure
        import matplotlib.pyplot as plt
        
        with PdfPages(filename) as pdf:
            # Create a figure for the summary text
            fig = Figure(figsize=(8.5, 11))  # Letter size
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            # Split text into pages if needed
            lines = summary_text.split('\n')
            current_page_lines = []
            page_height = 0
            max_height = 10.5  # Leave margin
            
            for line in lines:
                line_height = 0.15  # Approximate line height
                if page_height + line_height > max_height and current_page_lines:
                    # Save current page
                    ax.text(0.1, 1.0, '\n'.join(current_page_lines), 
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top', family='monospace',
                           wrap=True)
                    pdf.savefig(fig, bbox_inches='tight')
                    
                    # Start new page
                    fig = Figure(figsize=(8.5, 11))
                    ax = fig.add_subplot(111)
                    ax.axis('off')
                    current_page_lines = []
                    page_height = 0
                
                current_page_lines.append(line)
                page_height += line_height
            
            # Save last page of summary
            if current_page_lines:
                ax.text(0.1, 1.0, '\n'.join(current_page_lines),
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', family='monospace',
                       wrap=True)
                pdf.savefig(fig, bbox_inches='tight')
            
            # Add statistics table
            stats_lines = statistics_text.split('\n')
            current_page_lines = []
            page_height = 0
            
            fig = Figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis('off')
            
            for line in stats_lines:
                line_height = 0.12
                if page_height + line_height > max_height and current_page_lines:
                    ax.text(0.05, 1.0, '\n'.join(current_page_lines),
                           transform=ax.transAxes, fontsize=8,
                           verticalalignment='top', family='monospace',
                           wrap=True)
                    pdf.savefig(fig, bbox_inches='tight')
                    
                    fig = Figure(figsize=(8.5, 11))
                    ax = fig.add_subplot(111)
                    ax.axis('off')
                    current_page_lines = []
                    page_height = 0
                
                current_page_lines.append(line)
                page_height += line_height
            
            if current_page_lines:
                ax.text(0.05, 1.0, '\n'.join(current_page_lines),
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', family='monospace',
                       wrap=True)
                pdf.savefig(fig, bbox_inches='tight')
    
    except ImportError:
        raise ImportError("matplotlib is required for PDF export")
