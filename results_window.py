"""
Results display window for eye tracking analysis.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QMessageBox,
    QFileDialog, QHeaderView, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from state import state
from analysis import (
    calculate_rankings, normalize_for_radar, generate_statistics_table,
    calculate_normalized_rankings_per_group
)


class ResultsWindow(QMainWindow):
    """Window displaying analysis results with tabs for different views."""
    
    def __init__(
        self,
        aggregated_data: Dict,
        selected_groups: List[str],
        selected_tasks: List[str],
        active_parameters: List[str],
        mode: str,
        domains: List[str],
        show_statistics: bool = True,
        parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Eye Tracking Analysis Results")
        self.setMinimumSize(1200, 800)
        
        self.aggregated_data = aggregated_data
        self.selected_groups = selected_groups
        self.selected_tasks = selected_tasks
        self.active_parameters = active_parameters
        self.mode = mode
        self.domains = domains
        self.show_statistics = show_statistics
        
        # Store canvas references for export
        self.radar_canvas: Optional[FigureCanvas] = None
        self.tct_canvas: Optional[FigureCanvas] = None
        
        # Filter parameters based on domains
        self._filter_parameters_by_domains()
        
        self._setup_ui()
        self._populate_tabs()
    
    def _filter_parameters_by_domains(self) -> None:
        """Filter active parameters based on selected domains."""
        # If "Rank" is selected, include all parameters
        # If "Radar Chart" is selected, include all parameters
        # If "Task Completion Time" is selected, only include TCT
        if "Task Completion Time" in self.domains and len(self.domains) == 1:
            # Only TCT domain selected
            self.active_parameters = [p for p in self.active_parameters if "TCT" in p]
        # Otherwise, keep all active parameters
    
    def _setup_ui(self) -> None:
        """Set up the UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Info label
        info_label = QLabel(
            f"Mode: {self.mode} | "
            f"Groups: {len(self.selected_groups)} | "
            f"Tasks: {len(self.selected_tasks)} | "
            f"Parameters: {len(self.active_parameters)}"
        )
        info_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(info_label)
        
        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Export buttons
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        self.export_csv_btn = QPushButton("Export Statistics to CSV")
        self.export_csv_btn.clicked.connect(self._export_csv)
        export_layout.addWidget(self.export_csv_btn)
        
        self.export_png_btn = QPushButton("Export Charts to PNG")
        self.export_png_btn.clicked.connect(self._export_png)
        export_layout.addWidget(self.export_png_btn)
        
        layout.addLayout(export_layout)
    
    def _populate_tabs(self) -> None:
        """Populate tabs based on selected domains."""
        if "Rank" in self.domains:
            self._add_rank_tab()
        
        if "Radar Chart" in self.domains:
            self._add_radar_tab()
        
        if "Task Completion Time" in self.domains:
            self._add_tct_tab()
        
        # Conditionally add statistics tab
        if self.show_statistics:
            self._add_statistics_tab()
    
    def _add_rank_tab(self) -> None:
        """Add ranking tab with per-group ranking tables."""
        rank_widget = QWidget()
        rank_layout = QVBoxLayout(rank_widget)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Get parameter weights from state
        parameter_weights = state.parameter_weights.copy()
        
        # Calculate normalized rankings per group
        group_rankings = calculate_normalized_rankings_per_group(
            self.aggregated_data,
            self.selected_groups,
            self.selected_tasks,
            self.active_parameters,
            parameter_weights
        )
        
        if not group_rankings:
            scroll_layout.addWidget(QLabel("No ranking data available."))
            scroll.setWidget(scroll_widget)
            rank_layout.addWidget(scroll)
            self.tabs.addTab(rank_widget, "Rankings")
            return
        
        # Create one table per group
        for group_id in self.selected_groups:
            if group_id not in group_rankings:
                continue
            
            group_name = state.get_effective_group_names().get(group_id, group_id)
            ranking_df = group_rankings[group_id]
            
            # Group label
            group_label = QLabel(f"<b>{group_name}</b>")
            group_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px 5px 5px 5px;")
            scroll_layout.addWidget(group_label)
            
            # Create table
            table = QTableWidget()
            table.setRowCount(len(ranking_df))
            
            # Determine columns to display
            # Always show: Overall_Rank, Sum_of_Ranks, Task_Number, indications, contraindications, neither
            # Plus individual parameter ranks
            base_columns = ['Overall_Rank', 'Sum_of_Ranks', 'Task_Number', 
                          'indications', 'contraindications', 'neither']
            rank_columns = [col for col in ranking_df.columns if col.startswith('Rank_')]
            all_columns = base_columns + rank_columns
            
            table.setColumnCount(len(all_columns))
            table.setHorizontalHeaderLabels(all_columns)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            
            # Enable word wrapping for text columns
            text_columns = ['indications', 'contraindications', 'neither']
            
            for row_idx, (_, row) in enumerate(ranking_df.iterrows()):
                for col_idx, col_name in enumerate(all_columns):
                    value = row[col_name]
                    
                    if col_name == 'Task_Number':
                        # Format task with label
                        task_label = state.format_task(str(value))
                        item_text = task_label
                    elif isinstance(value, (int, float)):
                        if col_name in ['Overall_Rank', 'Sum_of_Ranks'] or col_name.startswith('Rank_'):
                            item_text = str(int(value))
                        else:
                            item_text = f"{value:.4f}"
                    else:
                        item_text = str(value)
                    
                    item = QTableWidgetItem(item_text)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    
                    # Enable word wrapping for text columns
                    if col_name in text_columns:
                        item.setTextAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
                        # Set word wrap
                        table.setWordWrap(True)
                    
                    table.setItem(row_idx, col_idx, item)
            
            # Resize rows to fit content for text columns
            table.resizeRowsToContents()
            
            # Set minimum column widths for text columns to ensure readability
            for col_idx, col_name in enumerate(all_columns):
                if col_name in text_columns:
                    table.setColumnWidth(col_idx, max(200, table.columnWidth(col_idx)))
            
            # Remove height constraint so table shows all rows - page will scroll instead
            table.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
            # Disable vertical scrollbar on table - let the page scroll instead
            table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            # Set size policy to ensure table expands vertically to show all rows
            table.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            scroll_layout.addWidget(table)
            
            # Add spacing between groups
            spacer = QLabel("")
            spacer.setMinimumHeight(10)
            scroll_layout.addWidget(spacer)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        rank_layout.addWidget(scroll)
        self.tabs.addTab(rank_widget, "Rankings")
    
    def _add_radar_tab(self) -> None:
        """Add radar chart tab."""
        radar_widget = QWidget()
        radar_layout = QVBoxLayout(radar_widget)
        
        # Filter parameters for radar chart: exclude TCT, keep Standard Deviation of TCT
        radar_parameters = [
            p for p in self.active_parameters 
            if p != "Task Completion Time (TCT)"  # Exclude TCT from radar chart
        ]
        
        if not radar_parameters:
            radar_layout.addWidget(QLabel("No parameters available for radar chart (TCT excluded)."))
            self.tabs.addTab(radar_widget, "Radar Chart")
            return
        
        # Normalize data for radar chart
        normalized_data = normalize_for_radar(self.aggregated_data, radar_parameters)
        
        if not normalized_data:
            radar_layout.addWidget(QLabel("No data available for radar chart."))
            self.tabs.addTab(radar_widget, "Radar Chart")
            return
        
        # Set up angles for radar chart
        num_params = len(radar_parameters)
        if num_params == 0:
            radar_layout.addWidget(QLabel("No parameters available for radar chart."))
            self.tabs.addTab(radar_widget, "Radar Chart")
            return
        
        angles = np.linspace(0, 2 * np.pi, num_params, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Scroll area for the entire chart
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        # Enable mouse wheel scrolling
        scroll.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(10, 10, 10, 10)
        scroll_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        # TOP SECTION: Aggregated view (one chart per group, all tasks overlaid)
        aggregated_label = QLabel("Aggregated View (All Tasks)")
        aggregated_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        scroll_layout.addWidget(aggregated_label)
        
        num_groups = len(self.selected_groups)
        if num_groups > 0:
            agg_cols = min(3, num_groups)
            agg_rows = (num_groups + agg_cols - 1) // agg_cols
            agg_fig = Figure(figsize=(14, 5 * agg_rows))
            agg_canvas = FigureCanvas(agg_fig)
            # Set minimum height and size policy for aggregated view
            agg_canvas.setMinimumHeight(500)
            agg_canvas.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            agg_canvas.setMinimumWidth(800)
            agg_canvas.setMaximumWidth(1400)  # Limit maximum width
            # Ensure canvas doesn't intercept wheel events - let scroll area handle them
            agg_canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            
            for idx, group_id in enumerate(self.selected_groups):
                if group_id not in normalized_data:
                    continue
                
                ax = agg_fig.add_subplot(agg_rows, agg_cols, idx + 1, projection='polar')
                group_data = normalized_data[group_id]
                group_name = state.get_effective_group_names().get(group_id, group_id)
                
                # Plot each task
                for task_id in self.selected_tasks:
                    if task_id not in group_data:
                        continue
                    
                    task_data = group_data[task_id]
                    task_label = state.format_task(task_id)
                    
                    values = []
                    for param in radar_parameters:
                        if param in task_data:
                            values.append(task_data[param])
                        elif isinstance(task_data, dict) and "_group_stats" in task_data and param in task_data["_group_stats"]:
                            # Handle Standard Deviation of TCT from _group_stats
                            values.append(task_data["_group_stats"][param])
                        else:
                            values.append(0.0)
                    
                    values += values[:1]  # Complete the circle
                    
                    ax.plot(angles, values, 'o-', linewidth=2, label=task_label)
                    ax.fill(angles, values, alpha=0.25)
                
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(radar_parameters, fontsize=12)
                ax.set_ylim(0, 1)
                ax.set_title(group_name, fontsize=14, fontweight='bold', pad=30)
                ax.grid(True)
                ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.2), fontsize=10)
            
            agg_fig.tight_layout()
            scroll_layout.addWidget(agg_canvas)
        
        # Separator
        separator = QLabel("─" * 80)
        separator.setStyleSheet("color: #ccc; padding: 10px;")
        scroll_layout.addWidget(separator)
        
        # BOTTOM SECTION: Individual charts listed vertically (larger, scrollable)
        grid_label = QLabel("Individual Task Analysis")
        grid_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
        scroll_layout.addWidget(grid_label)
        
        # Create individual charts for each task, listed vertically
        for task_idx, task_id in enumerate(self.selected_tasks):
            task_label = state.format_task(task_id)
            
            # Task section header
            task_header = QLabel(f"Task: {task_label}")
            task_header.setStyleSheet("font-weight: bold; font-size: 11px; padding: 10px 5px 5px 5px;")
            scroll_layout.addWidget(task_header)
            
            # Create horizontal layout for this task row (group charts side by side)
            task_row_layout = QHBoxLayout()
            task_row_layout.setContentsMargins(0, 0, 0, 0)
            task_row_layout.setSpacing(10)
            task_row_widget = QWidget()
            task_row_widget.setLayout(task_row_layout)
            # Ensure the widget has proper size policy for scrolling
            task_row_widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
            task_row_widget.setMinimumWidth(600)  # Minimum width for the row
            
            # Individual group charts for this task (side by side)
            for group_id in self.selected_groups:
                if group_id not in normalized_data:
                    continue
                
                group_data = normalized_data[group_id]
                if task_id not in group_data:
                    continue
                
                task_data = group_data[task_id]
                group_name = state.get_effective_group_names().get(group_id, group_id)
                
                # Create individual chart
                # Adjust figure size based on number of groups (fewer groups = smaller width)
                num_groups_in_row = len([g for g in self.selected_groups if g in normalized_data and task_id in normalized_data[g]])
                if num_groups_in_row == 1:
                    # Single column: limit width to reasonable size
                    fig_width = 8
                    max_width = 800
                else:
                    # Multiple columns: use smaller width per chart
                    fig_width = 10
                    max_width = 1200
                
                ind_fig = Figure(figsize=(fig_width, 10))
                ind_canvas = FigureCanvas(ind_fig)
                # Set size constraints
                ind_canvas.setMinimumHeight(500)
                ind_canvas.setMinimumWidth(600)
                ind_canvas.setMaximumWidth(max_width)  # Limit maximum width
                ind_canvas.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                # Ensure canvas doesn't intercept wheel events - let scroll area handle them
                ind_canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                ax = ind_fig.add_subplot(111, projection='polar')
                
                values = []
                for param in radar_parameters:
                    if param in task_data:
                        values.append(task_data[param])
                    elif isinstance(task_data, dict) and "_group_stats" in task_data and param in task_data["_group_stats"]:
                        # Handle Standard Deviation of TCT from _group_stats
                        values.append(task_data["_group_stats"][param])
                    else:
                        values.append(0.0)
                
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=3, color='steelblue', markersize=8)
                ax.fill(angles, values, alpha=0.3, color='steelblue')
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(radar_parameters, fontsize=14)
                ax.set_ylim(0, 1)
                ax.set_title(f"{task_label} - {group_name}", fontsize=16, fontweight='bold', pad=30)
                ax.grid(True)
                
                ind_fig.tight_layout()
                task_row_layout.addWidget(ind_canvas)
            
            # Add the task row to the scroll layout
            scroll_layout.addWidget(task_row_widget)
            
            # Store reference to last canvas for export (from the last group of the last task)
            if task_idx == len(self.selected_tasks) - 1:
                # ind_canvas will be the last one created in the inner loop
                if 'ind_canvas' in locals():
                    self.radar_canvas = ind_canvas
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        radar_layout.addWidget(scroll)
        
        self.tabs.addTab(radar_widget, "Radar Chart")
    
    def _add_tct_tab(self) -> None:
        """Add Task Completion Time analysis tab."""
        tct_widget = QWidget()
        tct_layout = QVBoxLayout(tct_widget)
        
        # Extract TCT data
        # Store with task_id as key for proper sorting, then format for display
        tct_data = {}  # {group_name: {task_id: mean_value}}
        tct_participant_data = {}  # {group_name: {task_id: {participant: value}}}
        
        for group_id in self.selected_groups:
            if group_id not in self.aggregated_data:
                continue
            
            group_data = self.aggregated_data[group_id]
            group_name = state.get_effective_group_names().get(group_id, group_id)
            tct_data[group_name] = {}
            tct_participant_data[group_name] = {}
            
            for task_id in self.selected_tasks:
                if task_id not in group_data:
                    continue
                
                task_data = group_data[task_id]
                tct_param = "Task Completion Time (TCT)"
                
                if isinstance(task_data, dict):
                    # Group mean mode
                    if tct_param in task_data:
                        tct_value = task_data[tct_param].get('mean', 0)
                        tct_data[group_name][task_id] = tct_value
                    # Individual participant mode
                    elif any(isinstance(v, dict) for v in task_data.values() if isinstance(v, dict)):
                        participant_values_dict = {}
                        tct_values_list = []
                        for participant, participant_data in task_data.items():
                            if participant == "_group_stats":
                                continue
                            if isinstance(participant_data, dict) and tct_param in participant_data:
                                p_value = participant_data[tct_param].get('mean', 0)
                                participant_values_dict[participant] = p_value
                                tct_values_list.append(p_value)
                        
                        if participant_values_dict:
                            tct_participant_data[group_name][task_id] = participant_values_dict
                            # Also store mean for group mean display
                            if tct_values_list:
                                tct_data[group_name][task_id] = float(np.mean(tct_values_list))
        
        if not tct_data and not tct_participant_data:
            tct_layout.addWidget(QLabel("No TCT data available."))
            self.tabs.addTab(tct_widget, "Task Completion Time")
            return
        
        # Create bar chart
        fig = Figure(figsize=(12, 8))
        canvas = FigureCanvas(fig)
        self.tct_canvas = canvas  # Store reference early
        ax = fig.add_subplot(111)
        
        # Check if we should show individual participants
        show_individual = self.mode in ["Each participant for selected groups", "Group mean and individual participants"]
        
        if show_individual and tct_participant_data:
            # Show individual participants
            task_ids = []
            all_participants = set()
            
            # Collect all task IDs and participants
            for group_name, task_data in tct_participant_data.items():
                task_ids.extend(task_data.keys())
                for participant_dict in task_data.values():
                    all_participants.update(participant_dict.keys())
            
            # Use natural sorting for tasks
            from data_processor import _natural_sort_key
            task_ids = sorted(set(task_ids), key=_natural_sort_key)
            # Format task IDs for display
            tasks = [state.format_task(tid) for tid in task_ids]
            all_participants = sorted(list(all_participants))
            
            x = np.arange(len(tasks))
            num_bars = len(self.selected_groups) * (len(all_participants) + (1 if self.mode == "Group mean and individual participants" else 0))
            width = 0.8 / max(num_bars, 1) if num_bars > 0 else 0.8
            
            bar_idx = 0
            colors = plt.cm.tab20(np.linspace(0, 1, len(all_participants) + len(self.selected_groups)))
            color_idx = 0
            
            for group_id in self.selected_groups:
                group_name = state.get_effective_group_names().get(group_id, group_id)
                
                # Show group mean if mode includes it
                if self.mode == "Group mean and individual participants":
                    mean_values = []
                    for task_id in task_ids:
                        if group_name in tct_data and task_id in tct_data[group_name]:
                            mean_values.append(tct_data[group_name][task_id])
                        else:
                            mean_values.append(0)
                    
                    offset = (bar_idx - num_bars / 2 + 0.5) * width
                    ax.bar(x + offset, mean_values, width, 
                          label=f"{group_name} (Mean)", 
                          alpha=0.8, color=colors[color_idx % len(colors)])
                    bar_idx += 1
                    color_idx += 1
                
                # Show individual participants
                if group_name in tct_participant_data:
                    for participant in all_participants:
                        participant_values = []
                        for task_id in task_ids:
                            if task_id in tct_participant_data[group_name]:
                                participant_values.append(
                                    tct_participant_data[group_name][task_id].get(participant, 0)
                                )
                            else:
                                participant_values.append(0)
                        
                        offset = (bar_idx - num_bars / 2 + 0.5) * width
                        ax.bar(x + offset, participant_values, width,
                              label=f"{group_name} - {participant}",
                              alpha=0.6, color=colors[color_idx % len(colors)])
                        bar_idx += 1
                        color_idx += 1
        
        else:
            # Show only group means
            tasks = []
            group_values = {}
            
            # Collect all task IDs and sort them naturally
            from data_processor import _natural_sort_key
            all_task_ids = set()
            for group_name, task_data in tct_data.items():
                all_task_ids.update(task_data.keys())
            task_ids = sorted(all_task_ids, key=_natural_sort_key)
            tasks = [state.format_task(tid) for tid in task_ids]  # Format for display
            
            for group_name, task_data in tct_data.items():
                group_values[group_name] = []
                for task_id in task_ids:
                    group_values[group_name].append(task_data.get(task_id, 0))
            
            x = np.arange(len(tasks))
            width = 0.8 / len(group_values) if group_values else 0.8
            
            for idx, (group_name, values) in enumerate(group_values.items()):
                offset = (idx - len(group_values) / 2 + 0.5) * width
                ax.bar(x + offset, values, width, label=group_name, alpha=0.8)
        
        ax.set_xlabel('Tasks')
        ax.set_ylabel('Task Completion Time (ms)')
        ax.set_title('Task Completion Time by Group and Task')
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        
        fig.tight_layout()
        tct_layout.addWidget(canvas)
        # Canvas reference already stored above
        
        self.tabs.addTab(tct_widget, "Task Completion Time")
    
    def _add_statistics_tab(self) -> None:
        """Add statistics table tab."""
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        
        # Generate statistics table
        stats_df = generate_statistics_table(
            self.aggregated_data,
            self.selected_tasks,
            self.selected_groups,
            self.mode
        )
        
        if stats_df.empty:
            stats_layout.addWidget(QLabel("No statistics available."))
            self.tabs.addTab(stats_widget, "Statistics")
            return
        
        # Create table widget
        table = QTableWidget()
        table.setRowCount(len(stats_df))
        table.setColumnCount(len(stats_df.columns))
        table.setHorizontalHeaderLabels(stats_df.columns.tolist())
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        for row_idx, row in stats_df.iterrows():
            for col_idx, col_name in enumerate(stats_df.columns):
                value = row[col_name]
                if isinstance(value, (int, float)):
                    if abs(value) < 0.0001:
                        item_text = "0.0000"
                    else:
                        item_text = f"{value:.4f}"
                else:
                    item_text = str(value)
                
                item = QTableWidgetItem(item_text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(row_idx, col_idx, item)
        
        stats_layout.addWidget(table)
        self.tabs.addTab(stats_widget, "Statistics")
    
    def _export_csv(self) -> None:
        """Export statistics to CSV."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Statistics to CSV",
            "",
            "CSV files (*.csv);;All files (*.*)"
        )
        
        if not filename:
            return
        
        try:
            stats_df = generate_statistics_table(
                self.aggregated_data,
                self.selected_tasks,
                self.selected_groups,
                self.mode
            )
            
            if stats_df.empty:
                QMessageBox.warning(self, "Export Error", "No data to export.")
                return
            
            stats_df.to_csv(filename, index=False)
            QMessageBox.information(self, "Export Success", f"Statistics exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def _export_png(self) -> None:
        """Export charts to PNG."""
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Charts to PNG",
            "",
            "PNG files (*.png);;All files (*.*)"
        )
        
        if not filename:
            return
        
        try:
            # Export each chart from stored references
            saved_count = 0
            base_name = filename.rsplit('.', 1)[0] if '.' in filename else filename
            
            # Export radar chart if available
            if self.radar_canvas is not None:
                self.radar_canvas.figure.savefig(f"{base_name}_radar.png", dpi=300, bbox_inches='tight')
                saved_count += 1
            
            # Export TCT chart if available
            if self.tct_canvas is not None:
                self.tct_canvas.figure.savefig(f"{base_name}_tct.png", dpi=300, bbox_inches='tight')
                saved_count += 1
            
            if saved_count > 0:
                QMessageBox.information(self, "Export Success", f"Exported {saved_count} chart(s).")
            else:
                QMessageBox.warning(self, "Export Warning", "No charts available to export.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
