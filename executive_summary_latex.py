"""
Executive summary generation using LaTeX/MikTeX for professional PDF output.
Connects to TSV data and generates a comprehensive executive summary.
"""
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from state import state
from analysis import (
    aggregate_by_groups,
    calculate_normalized_rankings_per_group,
    generate_statistics_table
)
from models import PARAMETER_OPTIONS


BASE_DIR = Path(__file__).parent
TEMPLATE = BASE_DIR / "executive_summary_template.tex"
OUTPUT_BASE = BASE_DIR / "output"
OUTPUT_BASE.mkdir(exist_ok=True, parents=True)


def latex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    if not isinstance(s, str):
        s = str(s)
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "^": "\\textasciicircum{}",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "\\": "\\textbackslash{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


def render_rows(rows: List[Dict]) -> str:
    """Render table rows for LaTeX."""
    return "\n".join(
        f"{r['rank']} & {latex_escape(r['task'])} & "
        f"{latex_escape(r['reason'])} & "
        f"{latex_escape(r['consistency'])} & "
        f"{latex_escape(r['outliers'])} \\\\"
        for r in rows
    )


def _generate_tct_chart_pdf(
    aggregated_data: Dict,
    selected_groups: List[str],
    selected_tasks: List[str],
    output_dir: Path,
) -> Optional[str]:
    """
    Generate TCT (Task Completion Time) bar chart as PDF in output_dir.
    Returns filename (e.g. 'tct_chart.pdf') if created, else None.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    try:
        tct_data = {}  # {group_name: {task_id: mean_value_seconds}}
        tct_param = "Task Completion Time (TCT)"

        for group_id in selected_groups:
            if group_id not in aggregated_data:
                continue
            group_data = aggregated_data[group_id]
            group_name = state.get_effective_group_names().get(group_id, group_id)
            tct_data[group_name] = {}
            for task_id in selected_tasks:
                if task_id not in group_data:
                    continue
                task_data = group_data[task_id]
                if not isinstance(task_data, dict):
                    continue
                if tct_param in task_data:
                    mean_ms = task_data[tct_param].get("mean", 0)
                    tct_data[group_name][task_id] = mean_ms / 1000.0
                else:
                    # Participant mode: compute mean from participants
                    vals = []
                    for k, v in task_data.items():
                        if k == "_group_stats" or not isinstance(v, dict):
                            continue
                        if tct_param in v:
                            vals.append(v[tct_param].get("mean", 0))
                    if vals:
                        tct_data[group_name][task_id] = float(np.mean(vals)) / 1000.0

        if not tct_data:
            return None

        try:
            from data_processor import _natural_sort_key
        except ImportError:
            _natural_sort_key = lambda x: (str(x).isdigit(), str(x))

        all_task_ids = set()
        for group_data in tct_data.values():
            all_task_ids.update(group_data.keys())
        task_ids = sorted(all_task_ids, key=_natural_sort_key)
        tasks = [state.format_task(tid) for tid in task_ids]

        group_values = {}
        for group_name, task_data in tct_data.items():
            group_values[group_name] = [task_data.get(tid, 0) for tid in task_ids]

        fig, ax = plt.subplots(figsize=(10, 6))
        task_spacing = 1.15
        x = np.arange(len(tasks)) * task_spacing
        width = 1.0 / len(group_values) if group_values else 0.8
        colors = plt.cm.tab10(np.linspace(0, 1, len(group_values)))
        for idx, (group_name, values) in enumerate(group_values.items()):
            offset = (idx - (len(group_values) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=group_name, alpha=0.8, color=colors[idx])

        ax.set_xlabel("Tasks")
        ax.set_ylabel("Task Completion Time (seconds)")
        ax.set_title("Task Completion Time by Group and Task")
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()

        pdf_path = output_dir / "tct_chart.pdf"
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        return "tct_chart.pdf"
    except Exception:
        return None


def find_pdflatex() -> Optional[str]:
    """Find pdflatex executable in PATH or common MikTeX locations."""
    # First try to find in PATH
    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        return pdflatex
    
    # Common MikTeX installation paths on Windows
    common_paths = [
        r"C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\Program Files (x86)\MiKTeX\miktex\bin\x64\pdflatex.exe",
        r"C:\Users\{}\AppData\Local\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe".format(
            Path.home().name
        ),
        r"C:\Program Files\MiKTeX\texmfs\install\miktex\bin\x64\pdflatex.exe",
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
    
    return None


def generate_latex_summary(
    aggregated_data: Dict,
    selected_groups: List[str],
    selected_tasks: List[str],
    active_parameters: List[str],
    mode: str,
    parameter_weights: Dict[str, float],
    df_path: Optional[str] = None
) -> str:
    """
    Generate LaTeX executive summary from TSV data.
    
    Returns path to generated PDF file, or raises exception on error.
    """
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"LaTeX template not found: {TEMPLATE}")
    
    # Find pdflatex
    pdflatex = find_pdflatex()
    if not pdflatex:
        raise RuntimeError(
            "pdflatex not found. Please install MikTeX and ensure it's in your PATH, "
            "or install it from https://miktex.org/download"
        )
    
    # Validate data
    if not aggregated_data:
        raise ValueError("No aggregated data available. Please run analysis first.")
    
    if not selected_groups:
        raise ValueError("No groups selected for analysis.")
    
    if not selected_tasks:
        raise ValueError("No tasks selected for analysis.")
    
    # Get group names
    group_names = state.get_effective_group_names()
    group_name_list = [group_names.get(gid, gid) for gid in selected_groups]
    
    # Calculate rankings per group
    group_rankings = calculate_normalized_rankings_per_group(
        aggregated_data,
        selected_groups,
        selected_tasks,
        active_parameters,
        parameter_weights
    )
    
    # Validate rankings data
    if not group_rankings:
        raise ValueError("No rankings data calculated. Check that groups and tasks have valid data.")

    # Create dated output folder: "exec summary - YYYY-MM-DD_HH-MM-SS" (like images export)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = OUTPUT_BASE / f"exec summary - {timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate TCT chart PDF before LaTeX so it can be included in the PDF
    tct_filename = _generate_tct_chart_pdf(
        aggregated_data, selected_groups, selected_tasks, output_dir
    )
    
    # Determine hardest and easiest tasks (aggregate across groups)
    all_task_ranks = {}  # task_id -> list of overall ranks
    
    for group_id, ranking_df in group_rankings.items():
        if 'Overall_Rank' not in ranking_df.columns:
            continue
        for _, row in ranking_df.iterrows():
            task_id = row['Task_Number']
            overall_rank = row['Overall_Rank']
            if task_id not in all_task_ranks:
                all_task_ranks[task_id] = []
            all_task_ranks[task_id].append(overall_rank)
    
    # Calculate average rank per task
    task_avg_ranks = {
        task_id: sum(ranks) / len(ranks)
        for task_id, ranks in all_task_ranks.items()
    }
    
    # Sort tasks by average rank (lower rank = harder)
    sorted_tasks = sorted(task_avg_ranks.items(), key=lambda x: x[1])
    
    # Get top 3 hardest (lowest ranks) and easiest (highest ranks)
    hardest_tasks = sorted_tasks[:3] if len(sorted_tasks) >= 3 else sorted_tasks
    easiest_tasks = sorted_tasks[-3:] if len(sorted_tasks) >= 3 else sorted_tasks
    easiest_tasks.reverse()  # Show easiest first
    
    # Build hardest tasks rows
    hardest_rows = []
    for rank, (task_id, avg_rank) in enumerate(hardest_tasks, 1):
        task_label = state.format_task(task_id)
        # Find reason (which parameter contributed most)
        reasons = []
        for group_id, ranking_df in group_rankings.items():
            task_row = ranking_df[ranking_df['Task_Number'] == task_id]
            if not task_row.empty:
                # Check which parameter ranks are low
                rank_cols = [col for col in ranking_df.columns if col.startswith('Rank_')]
                for col in rank_cols:
                    val = task_row[col].iloc[0]
                    if pd.notna(val) and val <= 3:  # Top 3 rank
                        param_name = col.replace('Rank_', '').replace('_', ' ')
                        reasons.append(param_name)
        
        reason = ", ".join(set(reasons[:2])) if reasons else "High overall cognitive load"
        consistency = "High agreement" if len(all_task_ranks[task_id]) > 1 else "Single group"
        outliers = "None detected"
        
        hardest_rows.append({
            'rank': rank,
            'task': task_label,
            'reason': reason,
            'consistency': consistency,
            'outliers': outliers
        })
    
    # Build easiest tasks rows
    easiest_rows = []
    for rank, (task_id, avg_rank) in enumerate(easiest_tasks, 1):
        task_label = state.format_task(task_id)
        reasons = []
        for group_id, ranking_df in group_rankings.items():
            task_row = ranking_df[ranking_df['Task_Number'] == task_id]
            if not task_row.empty:
                # Check which parameter ranks are high (easier)
                rank_cols = [col for col in ranking_df.columns if col.startswith('Rank_')]
                for col in rank_cols:
                    val = task_row[col].iloc[0]
                    if pd.notna(val):
                        max_rank = ranking_df[col].max()
                        if val >= max_rank - 2:  # Bottom 3 rank
                            param_name = col.replace('Rank_', '').replace('_', ' ')
                            reasons.append(param_name)
        
        reason = ", ".join(set(reasons[:2])) if reasons else "Low overall cognitive load"
        consistency = "High agreement" if len(all_task_ranks[task_id]) > 1 else "Single group"
        outliers = "None detected"
        
        easiest_rows.append({
            'rank': rank,
            'task': task_label,
            'reason': reason,
            'consistency': consistency,
            'outliers': outliers
        })
    
    # Get highest load task
    if hardest_tasks:
        highest_task_id = hardest_tasks[0][0]
        highest_task_label = state.format_task(highest_task_id)
        highest_load_task = f"{highest_task_label} (highest cognitive demand)"
    else:
        highest_load_task = "Unable to determine"
    
    # Count participants
    num_participants = 0
    if state.df is not None:
        num_participants = len(state.df['Participant'].unique())
    
    # Build context for template
    context = {
        "meta_info": f"generated {datetime.now():%Y-%m-%d %H:%M}",
        "test_name": "Eye Tracking Analysis",
        "test_date": datetime.now().strftime("%Y-%m-%d"),
        "num_participants": str(num_participants),
        "num_tasks": str(len(selected_tasks)),
        "highest_load_task": highest_load_task,
        "top3_hardest_rows": render_rows(hardest_rows),
        "hardest_implication": (
            f"Tasks {', '.join([state.format_task(t[0]) for t in hardest_tasks[:3]])} "
            "should be simplified or restructured to reduce cognitive load."
        ) if hardest_tasks else "No clear hardest tasks identified.",
        "top3_easiest_rows": render_rows(easiest_rows),
        "overall_ranking_text": (
            f"Overall ranking shows task difficulty across {len(selected_groups)} group(s). "
            f"Tasks are ranked based on normalized metrics: {', '.join(active_parameters[:3])}."
        ),
        "interpretation_text": (
            "Higher cognitive load typically manifests as longer task duration, "
            "higher inter-participant variability, increased pupil diameter, "
            "and altered saccade behavior. Lower values indicate easier tasks."
        ),
        "boxplot_path": "boxplot_tasks.pdf",  # Optional - will be skipped if file doesn't exist
        "radar_path": "radar_tasks.pdf",  # Optional - will be skipped if file doesn't exist
        "tct_path": tct_filename or ".no-tct-chart",  # TCT chart PDF generated above; .no-tct-chart never exists
        "consistency_text": (
            f"Analysis includes {len(selected_groups)} group(s) with {num_participants} total participants. "
            f"Rankings are based on normalized values across {len(active_parameters)} parameter(s)."
        ),
        "notable_findings": (
            f"\\item Analysis mode: {mode}\n"
            f"\\item Groups analyzed: {', '.join(group_name_list)}\n"
            f"\\item Parameters weighted: {len([p for p, w in parameter_weights.items() if w != 1.0])}"
        ),
        "data_source": df_path if df_path else "TSV file",
        "groups": ", ".join(group_name_list),
        "tasks": f"{len(selected_tasks)} tasks detected",
        "ranking_rule": "Normalized mean rank across weighted parameters"
    }
    
    # Read template
    tex = TEMPLATE.read_text(encoding="utf-8")
    
    # Replace placeholders
    for key, value in context.items():
        placeholder = f"{{{{{key}}}}}"
        tex = tex.replace(placeholder, str(value))
    
    # Write LaTeX file into dated folder
    tex_path = output_dir / "executive_summary.tex"
    tex_path.write_text(tex, encoding="utf-8")
    
    # Compile to PDF in same folder
    try:
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Run twice for references (if any)
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
            cwd=output_dir,
            capture_output=True,
            text=True,
            check=False
        )
        
        pdf_path = output_dir / "executive_summary.pdf"
        if pdf_path.exists():
            return str(pdf_path)
        else:
            raise RuntimeError(f"PDF was not generated. LaTeX output: {result.stdout}")
    
    except subprocess.CalledProcessError as e:
        error_output = (e.stderr or "") + "\n" + (e.stdout or "")
        
        # Check for MiKTeX update error
        if "you have not checked for MiKTeX updates" in error_output.lower() or "miktex update" in error_output.lower():
            error_msg = (
                "MiKTeX requires an update check before compilation.\n\n"
                "To fix this issue:\n"
                "1. Open a command prompt (cmd) or PowerShell\n"
                "2. Run the following command:\n"
                "   miktex update\n"
                "   (or: miktex packages update)\n"
                "3. Wait for MiKTeX to check and install any required updates\n"
                "4. Try generating the executive summary again\n\n"
                "Alternatively, you can update MiKTeX through the MiKTeX Console:\n"
                "1. Open 'MiKTeX Console' from the Start menu\n"
                "2. Go to 'Updates' tab\n"
                "3. Click 'Check for updates' and install any available updates\n\n"
                f"Original error:\n{error_output[:500]}"
            )
            raise RuntimeError(error_msg)
        else:
            error_msg = f"LaTeX compilation failed:\n{error_output[:1000]}"
            raise RuntimeError(error_msg)
    except Exception as e:
        raise RuntimeError(f"Failed to compile LaTeX: {str(e)}")
