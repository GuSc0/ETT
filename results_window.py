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
    QFileDialog, QHeaderView, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from state import state
from analysis import calculate_rankings, normalize_for_radar, generate_statistics_table


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
        
        # Always add statistics tab
        self._add_statistics_tab()
    
    def _add_rank_tab(self) -> None:
        """Add ranking tab."""
        rank_widget = QWidget()
        rank_layout = QVBoxLayout(rank_widget)
        
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Create ranking table for each parameter
        for parameter in self.active_parameters:
            if parameter == "Standard Deviation of TCT":
                continue  # Skip - not ranked individually
            
            rankings = calculate_rankings(self.aggregated_data, parameter)
            
            if not rankings:
                continue
            
            # Parameter label
            param_label = QLabel(f"<b>{parameter}</b>")
            param_label.setStyleSheet("font-size: 12px; padding: 5px;")
            scroll_layout.addWidget(param_label)
            
            # Ranking table
            table = QTableWidget()
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(["Rank", "Group", "Task", "Mean Value", "Std Dev"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            
            table.setRowCount(len(rankings))
            
            for row, (group_id, task_id, value, rank) in enumerate(rankings):
                group_name = state.get_effective_group_names().get(group_id, group_id)
                task_label = state.format_task(task_id)
                
                # Get std dev if available
                std_dev = 0.0
                if group_id in self.aggregated_data and task_id in self.aggregated_data[group_id]:
                    task_data = self.aggregated_data[group_id][task_id]
                    if isinstance(task_data, dict) and parameter in task_data:
                        std_dev = task_data[parameter].get('std', 0.0)
                
                table.setItem(row, 0, QTableWidgetItem(str(rank)))
                table.setItem(row, 1, QTableWidgetItem(group_name))
                table.setItem(row, 2, QTableWidgetItem(task_label))
                table.setItem(row, 3, QTableWidgetItem(f"{value:.4f}"))
                table.setItem(row, 4, QTableWidgetItem(f"{std_dev:.4f}"))
            
            table.setMaximumHeight(200)
            scroll_layout.addWidget(table)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        
        rank_layout.addWidget(scroll)
        self.tabs.addTab(rank_widget, "Rankings")
    
    def _add_radar_tab(self) -> None:
        """Add radar chart tab."""
        radar_widget = QWidget()
        radar_layout = QVBoxLayout(radar_widget)
        
        # Normalize data for radar chart
        normalized_data = normalize_for_radar(self.aggregated_data, self.active_parameters)
        
        if not normalized_data:
            radar_layout.addWidget(QLabel("No data available for radar chart."))
            self.tabs.addTab(radar_widget, "Radar Chart")
            return
        
        # Create matplotlib figure
        fig = Figure(figsize=(10, 8))
        canvas = FigureCanvas(fig)
        
        # Determine number of subplots needed
        num_plots = len(self.selected_groups)
        if num_plots == 0:
            num_plots = 1
        
        cols = min(2, num_plots)
        rows = (num_plots + cols - 1) // cols
        
        for idx, group_id in enumerate(self.selected_groups):
            if group_id not in normalized_data:
                continue
            
            ax = fig.add_subplot(rows, cols, idx + 1, projection='polar')
            
            group_data = normalized_data[group_id]
            group_name = state.get_effective_group_names().get(group_id, group_id)
            
            # Set up angles for radar chart
            num_params = len(self.active_parameters)
            if num_params == 0:
                continue
            
            angles = np.linspace(0, 2 * np.pi, num_params, endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            # Plot each task
            for task_id in self.selected_tasks:
                if task_id not in group_data:
                    continue
                
                task_data = group_data[task_id]
                task_label = state.format_task(task_id)
                
                values = []
                for param in self.active_parameters:
                    if param in task_data:
                        values.append(task_data[param])
                    else:
                        values.append(0.0)
                
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=task_label)
                ax.fill(angles, values, alpha=0.25)
            
            # Set labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self.active_parameters, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(group_name, fontsize=10, fontweight='bold', pad=20)
            ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=7)
        
        fig.tight_layout()
        radar_layout.addWidget(canvas)
        # Canvas reference already stored above
        
        self.tabs.addTab(radar_widget, "Radar Chart")
    
    def _add_tct_tab(self) -> None:
        """Add Task Completion Time analysis tab."""
        tct_widget = QWidget()
        tct_layout = QVBoxLayout(tct_widget)
        
        # Extract TCT data
        tct_data = {}
        for group_id in self.selected_groups:
            if group_id not in self.aggregated_data:
                continue
            
            group_data = self.aggregated_data[group_id]
            group_name = state.get_effective_group_names().get(group_id, group_id)
            tct_data[group_name] = {}
            
            for task_id in self.selected_tasks:
                if task_id not in group_data:
                    continue
                
                task_data = group_data[task_id]
                task_label = state.format_task(task_id)
                
                # Get TCT value
                tct_param = "Task Completion Time (TCT)"
                if isinstance(task_data, dict) and tct_param in task_data:
                    tct_value = task_data[tct_param].get('mean', 0)
                    tct_data[group_name][task_label] = tct_value
        
        if not tct_data:
            tct_layout.addWidget(QLabel("No TCT data available."))
            self.tabs.addTab(tct_widget, "Task Completion Time")
            return
        
        # Create bar chart
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        self.tct_canvas = canvas  # Store reference early
        ax = fig.add_subplot(111)
        
        # Prepare data for plotting
        tasks = []
        group_values = {}
        
        for group_name, task_data in tct_data.items():
            group_values[group_name] = []
            if not tasks:
                tasks = list(task_data.keys())
            
            for task in tasks:
                group_values[group_name].append(task_data.get(task, 0))
        
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
        ax.legend()
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
