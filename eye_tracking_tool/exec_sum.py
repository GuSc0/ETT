import subprocess
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent
TEMPLATE = BASE_DIR / "executive_summary_template.tex"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

def latex_escape(s: str) -> str:
    return s.replace("&", "\\&").replace("%", "\\%")

def render_rows(rows):
    return "\n".join(
        f"{r['rank']} & {latex_escape(r['task'])} & "
        f"{latex_escape(r['reason'])} & "
        f"{latex_escape(r['consistency'])} & "
        f"{latex_escape(r['outliers'])} \\\\"
        for r in rows
    )

def main():
    context = {
        "meta_info": f"generated {datetime.now():%Y-%m-%d %H:%M}",
        "test_name": "Visual Search Experiment",
        "test_date": "2026-01-10",
        "num_participants": "24",
        "num_tasks": "12",
        "highest_load_task": "Task 7 (complex decision + time pressure)",

        "top3_hardest_rows": render_rows([
            dict(rank=1, task="Task 7", reason="Longest completion time",
                 consistency="high agreement", outliers="none"),
            dict(rank=2, task="Task 3", reason="High variability",
                 consistency="mixed", outliers="experts differ"),
            dict(rank=3, task="Task 10", reason="Large pupil dilation",
                 consistency="high agreement", outliers="none"),
        ]),

        "hardest_implication": "Tasks 7, 3 and 10 should be simplified or restructured.",

        "top3_easiest_rows": render_rows([
            dict(rank=1, task="Task 1", reason="Short time",
                 consistency="high", outliers="none"),
            dict(rank=2, task="Task 4", reason="Low variability",
                 consistency="high", outliers="none"),
            dict(rank=3, task="Task 6", reason="Low pupil response",
                 consistency="high", outliers="none"),
        ]),

        "overall_ranking_text":
            "Overall ranking shows a clear separation between navigation-heavy "
            "and recognition-heavy tasks. Confidence is high for the top half.",

        "interpretation_text":
            "Higher cognitive load typically manifests as longer task duration, "
            "higher inter-participant variability, increased pupil diameter, "
            "and altered saccade behavior.",

        "boxplot_path": "boxplot_tasks.pdf",
        "radar_path": "radar_tasks.pdf",

        "consistency_text":
            "Approximately 75\\% of participants ranked Task 7 among the top-3 hardest. "
            "Experts consistently showed lower load on Task 4.",

        "notable_findings":
            "\\item Task 11 has low sample size\n"
            "\\item One participant shows inverted ranking pattern",

        "data_source": "normalized_df",
        "groups": "experts, novices",
        "tasks": "12 tasks detected",
        "ranking_rule": "Mean rank across normalized metrics"
    }

    tex = TEMPLATE.read_text(encoding="utf-8")
    for key, value in context.items():
        tex = tex.replace(f"{{{{{key}}}}}", value)

    tex_path = OUTPUT_DIR / "executive_summary.tex"
    tex_path.write_text(tex, encoding="utf-8")

    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=OUTPUT_DIR,
        check=True
    )

    print("PDF generated:", OUTPUT_DIR / "executive_summary.pdf")

if __name__ == "__main__":
    main()

