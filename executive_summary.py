"""
Executive summary generation for eye tracking analysis.
"""
from __future__ import annotations

from typing import Dict, List
import pandas as pd
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
    
    # Overview
    lines.append("OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"Analysis Mode: {mode}")
    lines.append(f"Number of Groups: {len(selected_groups)}")
    lines.append(f"Number of Tasks: {len(selected_tasks)}")
    lines.append(f"Parameters Analyzed: {len(active_parameters)}")
    lines.append("")
    
    # Group information
    group_names = state.get_effective_group_names()
    lines.append("GROUPS ANALYZED:")
    for group_id in selected_groups:
        group_name = group_names.get(group_id, group_id)
        if group_id in aggregated_data:
            num_tasks = len([t for t in selected_tasks if t in aggregated_data[group_id]])
            lines.append(f"  - {group_name}: {num_tasks} task(s)")
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
        
        # Get rankings
        rankings = calculate_rankings(aggregated_data, parameter)
        
        if not rankings:
            lines.append("  No data available for this parameter.")
            continue
        
        # Find highest and lowest
        if rankings:
            highest = rankings[0]  # First is highest rank (highest value)
            lowest = rankings[-1]   # Last is lowest rank (lowest value)
            
            group_name_h = state.get_effective_group_names().get(highest[0], highest[0])
            task_label_h = state.format_task(highest[1])
            value_h = highest[2]
            
            group_name_l = state.get_effective_group_names().get(lowest[0], lowest[0])
            task_label_l = state.format_task(lowest[1])
            value_l = lowest[2]
            
            lines.append(f"  Highest: {task_label_h} ({group_name_h}) - {value_h:.4f}")
            lines.append(f"  Lowest: {task_label_l} ({group_name_l}) - {value_l:.4f}")
            
            # Calculate range
            if value_h > value_l:
                range_pct = ((value_h - value_l) / value_l * 100) if value_l > 0 else 0
                lines.append(f"  Range: {value_h - value_l:.4f} ({range_pct:.1f}% difference)")
    
    lines.append("")
    
    # Group comparisons
    if len(selected_groups) > 1:
        lines.append("GROUP COMPARISONS")
        lines.append("-" * 80)
        
        for parameter in active_parameters:
            if parameter == "Standard Deviation of TCT":
                continue
            
            lines.append(f"\n{parameter}:")
            
            # Calculate group means
            group_means = {}
            for group_id in selected_groups:
                if group_id not in aggregated_data:
                    continue
                
                group_data = aggregated_data[group_id]
                group_name = group_names.get(group_id, group_id)
                
                values = []
                for task_id in selected_tasks:
                    if task_id not in group_data:
                        continue
                    
                    task_data = group_data[task_id]
                    if isinstance(task_data, dict) and parameter in task_data:
                        values.append(task_data[parameter].get('mean', 0))
                
                if values:
                    group_means[group_name] = sum(values) / len(values)
            
            if group_means:
                sorted_groups = sorted(group_means.items(), key=lambda x: x[1], reverse=True)
                for group_name, mean_value in sorted_groups:
                    lines.append(f"  {group_name}: {mean_value:.4f}")
    
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
    """
    stats_df = generate_statistics_table(
        aggregated_data,
        selected_tasks,
        selected_groups,
        mode
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
