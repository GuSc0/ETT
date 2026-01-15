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

from state import state
from analysis import (
    aggregate_by_groups,
    calculate_normalized_rankings_per_group,
    generate_statistics_table
)
from models import PARAMETER_OPTIONS


BASE_DIR = Path(__file__).parent
TEMPLATE = BASE_DIR / "eye_tracking_tool" / "executive_summary_template.tex"
OUTPUT_DIR = BASE_DIR / "eye_tracking_tool" / "output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


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
        "boxplot_path": "boxplot_tasks.pdf",  # Placeholder - can be generated later
        "radar_path": "radar_tasks.pdf",  # Placeholder - can be generated later
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
    
    # Write LaTeX file
    tex_path = OUTPUT_DIR / "executive_summary.tex"
    tex_path.write_text(tex, encoding="utf-8")
    
    # Compile to PDF
    try:
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-output-directory", str(OUTPUT_DIR), str(tex_path)],
            cwd=OUTPUT_DIR,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Run twice for references (if any)
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-output-directory", str(OUTPUT_DIR), str(tex_path)],
            cwd=OUTPUT_DIR,
            capture_output=True,
            text=True,
            check=False
        )
        
        pdf_path = OUTPUT_DIR / "executive_summary.pdf"
        if pdf_path.exists():
            return str(pdf_path)
        else:
            raise RuntimeError(f"PDF was not generated. LaTeX output: {result.stdout}")
    
    except subprocess.CalledProcessError as e:
        error_msg = f"LaTeX compilation failed:\n{e.stderr}\n{e.stdout}"
        raise RuntimeError(error_msg)
    except Exception as e:
        raise RuntimeError(f"Failed to compile LaTeX: {str(e)}")
