"""
Results display window for eye tracking analysis.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel, QMessageBox,
    QFileDialog, QHeaderView, QScrollArea, QSizePolicy, QDialog,
    QCheckBox, QDialogButtonBox, QApplication
)
from PyQt6.QtCore import Qt, QObject, QEvent
from PyQt6.QtGui import QFont, QWheelEvent
from pathlib import Path

from state import state
from dialogs import LoadingDialog
from analysis import (
    calculate_rankings, normalize_for_radar, generate_statistics_table,
    calculate_normalized_rankings_per_group, calculate_normalized_rankings_per_participant
)


class WheelEventFilter(QObject):
    """Event filter to forward wheel events from canvas to scroll area."""
    
    def __init__(self, scroll_area: QScrollArea) -> None:
        super().__init__()
        self.scroll_area = scroll_area
    
    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Forward wheel events to scroll area."""
        if event.type() == QEvent.Type.Wheel:
            # Forward wheel event to scroll area's viewport
            if isinstance(event, QWheelEvent):
                # Create a new event for the scroll area viewport
                from PyQt6.QtCore import QPoint
                viewport = self.scroll_area.viewport()
                # Get the global position and convert to viewport coordinates
                global_pos = event.globalPosition()
                viewport_pos = viewport.mapFromGlobal(global_pos.toPoint())
                # Create new wheel event for viewport
                new_event = QWheelEvent(
                    viewport_pos,
                    global_pos,
                    event.pixelDelta(),
                    event.angleDelta(),
                    event.buttons(),
                    event.modifiers(),
                    event.phase(),
                    event.inverted(),
                    event.source()
                )
                QApplication.sendEvent(viewport, new_event)
            return True  # Event handled
        return False  # Let other events pass through


class ExportDialog(QDialog):
    """Dialog for selecting what to export."""
    
    def __init__(self, parent: QWidget, has_statistics: bool, has_rankings: bool, 
                 has_radar: bool, has_tct: bool) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        self.setMinimumSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        instructions = QLabel("Select items to export:")
        instructions.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(instructions)
        
        # Checkboxes
        self.export_stats_csv = QCheckBox("Export Statistics to CSV")
        self.export_stats_csv.setChecked(has_statistics)
        self.export_stats_csv.setEnabled(has_statistics)
        layout.addWidget(self.export_stats_csv)
        
        self.export_rankings_csv = QCheckBox("Export Rankings to CSV")
        self.export_rankings_csv.setChecked(has_rankings)
        self.export_rankings_csv.setEnabled(has_rankings)
        layout.addWidget(self.export_rankings_csv)
        
        self.export_radar_png = QCheckBox("Export Radar Chart(s) to PNG")
        self.export_radar_png.setChecked(has_radar)
        self.export_radar_png.setEnabled(has_radar)
        layout.addWidget(self.export_radar_png)
        
        self.export_tct_png = QCheckBox("Export TCT Chart to PNG")
        self.export_tct_png.setChecked(has_tct)
        self.export_tct_png.setEnabled(has_tct)
        layout.addWidget(self.export_tct_png)
        
        self.export_all_charts_png = QCheckBox("Print all images to one PNG")
        self.export_all_charts_png.setChecked(False)
        self.export_all_charts_png.setEnabled(has_radar or has_tct)
        layout.addWidget(self.export_all_charts_png)
        
        layout.addStretch()
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_selections(self) -> dict:
        """Return dictionary of selected export options."""
        return {
            'stats_csv': self.export_stats_csv.isChecked(),
            'rankings_csv': self.export_rankings_csv.isChecked(),
            'radar_png': self.export_radar_png.isChecked(),
            'tct_png': self.export_tct_png.isChecked(),
            'all_charts_png': self.export_all_charts_png.isChecked(),
        }


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
        # Store as list of tuples: (canvas, task_id or None for aggregated)
        self.radar_canvases: List[tuple] = []  # Store all radar charts with task info
        self.tct_canvas: Optional[FigureCanvas] = None
        self.rankings_data: Optional[Dict] = None  # Store rankings data for CSV export
        
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
        
        # Export button
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._show_export_dialog)
        export_layout.addWidget(self.export_btn)
        
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
        
        # For "Group mean and individual participants" mode, show both group mean and participant rankings
        if self.mode == "Group mean and individual participants":
            # First, show group mean rankings
            group_rankings = calculate_normalized_rankings_per_group(
                self.aggregated_data,
                self.selected_groups,
                self.selected_tasks,
                self.active_parameters,
                parameter_weights
            )
            
            if group_rankings:
                # Section header for group mean rankings
                group_mean_header = QLabel("<b>Group Mean Rankings</b>")
                group_mean_header.setStyleSheet("font-size: 18px; font-weight: bold; padding: 20px 5px 15px 5px; color: #0066cc; background-color: #e6f2ff; border-radius: 4px;")
                scroll_layout.addWidget(group_mean_header)
                
                # Create one table per group for group mean rankings
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
                                table.setWordWrap(True)
                            
                            table.setItem(row_idx, col_idx, item)
                    
                    # Resize rows to fit content for text columns
                    table.resizeRowsToContents()
                    
                    # Set minimum column widths for text columns to ensure readability
                    for col_idx, col_name in enumerate(all_columns):
                        if col_name in text_columns:
                            table.setColumnWidth(col_idx, max(200, table.columnWidth(col_idx)))
                    
                    table.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
                    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                    table.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                    scroll_layout.addWidget(table)
                    
                    # Add spacing between groups
                    spacer = QLabel("")
                    spacer.setMinimumHeight(10)
                    scroll_layout.addWidget(spacer)
                
                # Add separator between group mean and participant rankings
                separator = QLabel("─" * 80)
                separator.setStyleSheet("color: #ccc; padding: 20px;")
                scroll_layout.addWidget(separator)
                
                # Section header for individual participant rankings
                participant_header = QLabel("<b>Individual Participant Rankings</b>")
                participant_header.setStyleSheet("font-size: 18px; font-weight: bold; padding: 20px 5px 15px 5px; color: #0066cc; background-color: #e6f2ff; border-radius: 4px;")
                scroll_layout.addWidget(participant_header)
            
            # Now show individual participant rankings
            participant_rankings = calculate_normalized_rankings_per_participant(
                self.aggregated_data,
                self.selected_groups,
                self.selected_tasks,
                self.active_parameters,
                parameter_weights
            )
            
            if not participant_rankings:
                if not group_rankings:
                    scroll_layout.addWidget(QLabel("No ranking data available."))
                scroll.setWidget(scroll_widget)
                rank_layout.addWidget(scroll)
                self.tabs.addTab(rank_widget, "Rankings")
                return
            
            # Store rankings data for export (combine both if available)
            if group_rankings:
                # Store both types for export
                self.rankings_data = {'group_means': group_rankings, 'participants': participant_rankings}
            else:
                self.rankings_data = participant_rankings
            
            # Create one table per participant per group
            for group_id in self.selected_groups:
                if group_id not in participant_rankings:
                    continue
                
                group_name = state.get_effective_group_names().get(group_id, group_id)
                
                # Group header
                group_header = QLabel(f"<b>{group_name}</b>")
                group_header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 15px 5px 10px 5px; color: #0066cc;")
                scroll_layout.addWidget(group_header)
                
                # Sort participants for consistent display
                participants = sorted(participant_rankings[group_id].keys())
                
                for participant_id in participants:
                    ranking_df = participant_rankings[group_id][participant_id]
                    
                    # Participant label
                    participant_label = QLabel(f"<b>Participant: {participant_id}</b>")
                    participant_label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 10px 5px 5px 20px;")
                    scroll_layout.addWidget(participant_label)
                    
                    # Create table
                    table = QTableWidget()
                    table.setRowCount(len(ranking_df))
                    
                    # Determine columns to display
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
                                table.setWordWrap(True)
                            
                            table.setItem(row_idx, col_idx, item)
                    
                    # Resize rows to fit content
                    table.resizeRowsToContents()
                    
                    # Set minimum column widths for text columns
                    for col_idx, col_name in enumerate(all_columns):
                        if col_name in text_columns:
                            table.setColumnWidth(col_idx, max(200, table.columnWidth(col_idx)))
                    
                    table.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
                    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                    table.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                    scroll_layout.addWidget(table)
                    
                    # Add spacing between participants
                    spacer = QLabel("")
                    spacer.setMinimumHeight(10)
                    scroll_layout.addWidget(spacer)
                
                # Add spacing between groups
                group_spacer = QLabel("")
                group_spacer.setMinimumHeight(20)
                scroll_layout.addWidget(group_spacer)
        
        elif self.mode == "Each participant for selected groups":
            # Calculate rankings per participant
            participant_rankings = calculate_normalized_rankings_per_participant(
                self.aggregated_data,
                self.selected_groups,
                self.selected_tasks,
                self.active_parameters,
                parameter_weights
            )
            
            if not participant_rankings:
                scroll_layout.addWidget(QLabel("No ranking data available."))
                scroll.setWidget(scroll_widget)
                rank_layout.addWidget(scroll)
                self.tabs.addTab(rank_widget, "Rankings")
                return
            
            # Store rankings data for export
            self.rankings_data = participant_rankings
            
            # Create one table per participant per group
            for group_id in self.selected_groups:
                if group_id not in participant_rankings:
                    continue
                
                group_name = state.get_effective_group_names().get(group_id, group_id)
                
                # Group header
                group_header = QLabel(f"<b>{group_name}</b>")
                group_header.setStyleSheet("font-size: 16px; font-weight: bold; padding: 15px 5px 10px 5px; color: #0066cc;")
                scroll_layout.addWidget(group_header)
                
                # Sort participants for consistent display
                participants = sorted(participant_rankings[group_id].keys())
                
                for participant_id in participants:
                    ranking_df = participant_rankings[group_id][participant_id]
                    
                    # Participant label
                    participant_label = QLabel(f"<b>Participant: {participant_id}</b>")
                    participant_label.setStyleSheet("font-size: 13px; font-weight: bold; padding: 10px 5px 5px 20px;")
                    scroll_layout.addWidget(participant_label)
                    
                    # Create table
                    table = QTableWidget()
                    table.setRowCount(len(ranking_df))
                    
                    # Determine columns to display
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
                                table.setWordWrap(True)
                            
                            table.setItem(row_idx, col_idx, item)
                    
                    # Resize rows to fit content
                    table.resizeRowsToContents()
                    
                    # Set minimum column widths for text columns
                    for col_idx, col_name in enumerate(all_columns):
                        if col_name in text_columns:
                            table.setColumnWidth(col_idx, max(200, table.columnWidth(col_idx)))
                    
                    table.setSizeAdjustPolicy(QTableWidget.SizeAdjustPolicy.AdjustToContents)
                    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
                    table.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                    scroll_layout.addWidget(table)
                    
                    # Add spacing between participants
                    spacer = QLabel("")
                    spacer.setMinimumHeight(10)
                    scroll_layout.addWidget(spacer)
                
                # Add spacing between groups
                group_spacer = QLabel("")
                group_spacer.setMinimumHeight(20)
                scroll_layout.addWidget(group_spacer)
        else:
            # Calculate normalized rankings per group (group mean mode)
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
            
            # Store rankings data for export
            self.rankings_data = group_rankings
            
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
        
        # Normalize data for radar chart (with baseline subtraction, like notebook)
        from state import state
        normalized_data = normalize_for_radar(self.aggregated_data, radar_parameters, df=state.df)
        
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
        
        # Sort tasks chronologically using natural sort
        from data_processor import _natural_sort_key
        sorted_tasks = sorted(self.selected_tasks, key=_natural_sort_key)
        
        # Add explanation label for radar chart normalization
        explanation_label = QLabel(
            "Values are normalized using baseline subtraction + min-max scaling (0-1). "
            "First, each participant's baseline (Task 0) is subtracted from their values. "
            "Then, baseline-subtracted values are scaled to 0-1 (0 = minimum deviation from baseline, "
            "1 = maximum deviation from baseline across all tasks and groups)."
        )
        explanation_label.setWordWrap(True)
        explanation_label.setStyleSheet("padding: 8px; background-color: #f0f0f0; border-radius: 4px; font-size: 10pt; color: #666;")
        radar_layout.addWidget(explanation_label)
        
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
            # Layout: all groups side-by-side (1 row, num_groups columns)
            agg_cols = num_groups
            agg_rows = 1
            # Calculate figure size: wider for more groups, consistent height
            fig_width = 6 * num_groups + 2  # Extra space for legend on left
            fig_height = 7
            agg_fig = Figure(figsize=(fig_width, fig_height))
            agg_canvas = FigureCanvas(agg_fig)
            # Set minimum height and size policy for aggregated view
            agg_canvas.setMinimumHeight(500)
            agg_canvas.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
            agg_canvas.setMinimumWidth(800)
            agg_canvas.setMaximumWidth(2500)  # Increased limit to allow expansion
            # Install event filter to forward wheel events to scroll area
            wheel_filter = WheelEventFilter(scroll)
            agg_canvas.installEventFilter(wheel_filter)
            agg_canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            
            # Use gridspec for better control - create once before loop
            from matplotlib import gridspec
            gs = gridspec.GridSpec(1, num_groups + 1, figure=agg_fig, 
                                 width_ratios=[0.15] + [1] * num_groups,
                                 wspace=0.3, hspace=0.2)
            
            # Collect all labels for shared legend (only from first group since all groups have same tasks)
            shared_labels = []
            shared_handles = []
            
            for idx, group_id in enumerate(self.selected_groups):
                if group_id not in normalized_data:
                    continue
                
                ax = agg_fig.add_subplot(gs[0, idx + 1], projection='polar')
                group_data = normalized_data[group_id]
                group_name = state.get_effective_group_names().get(group_id, group_id)
                
                # Plot each task (in chronological order)
                for task_id in sorted_tasks:
                    if task_id not in group_data:
                        continue
                    
                    task_data = group_data[task_id]
                    task_label = state.format_task(task_id)
                    
                    # Extract values using consistent logic
                    values = []
                    for param in radar_parameters:
                        value = None
                        if isinstance(task_data, dict):
                            if param in task_data:
                                value = task_data[param]
                            elif "_group_stats" in task_data and param in task_data["_group_stats"]:
                                # Handle Standard Deviation of TCT from _group_stats
                                value = task_data["_group_stats"][param]
                        
                        # Use 0.0 for missing values (will show as no data point)
                        values.append(value if value is not None else 0.0)
                    
                    # Only plot if we have at least one non-zero value
                    if any(v > 0.0 for v in values):
                        values += values[:1]  # Complete the circle
                        # Use color palette similar to notebook (viridis-like)
                        color_map = plt.cm.viridis
                        task_num = sorted_tasks.index(task_id) if task_id in sorted_tasks else 0
                        color = color_map(task_num / max(len(sorted_tasks) - 1, 1))
                        line = ax.plot(angles, values, 'o-', linewidth=2, label=task_label, color=color)[0]
                        ax.fill(angles, values, alpha=0.25, color=color)
                        
                        # Collect labels and handles for shared legend (only from first group)
                        if idx == 0 and task_label not in shared_labels:
                            shared_labels.append(task_label)
                            shared_handles.append(line)
                
                # Fix text overflow: reduce font size and use shorter parameter names
                ax.set_xticks(angles[:-1])
                # Create shorter labels to prevent overflow
                short_labels = []
                for param in radar_parameters:
                    if len(param) > 15:
                        # Abbreviate long parameter names
                        if "Saccade Velocity" in param:
                            short_labels.append("Mean Saccade Vel.")
                        elif "Peak Saccade Velocity" in param:
                            short_labels.append("Peak Saccade Vel.")
                        elif "Saccade Amplitude" in param:
                            short_labels.append("Saccade Amp.")
                        elif "Standard Deviation" in param:
                            short_labels.append("Std Dev TCT")
                        elif "Pupil Diameter" in param:
                            short_labels.append("Pupil Diam.")
                        else:
                            short_labels.append(param[:15])
                    else:
                        short_labels.append(param)
                ax.set_xticklabels(short_labels, fontsize=9)
                ax.tick_params(pad=25)  # Move labels further from chart center
                ax.set_ylim(0, 1)
                ax.set_title(group_name, fontsize=12, fontweight='bold', pad=35)
                ax.grid(True)
                # Don't add individual legend - will use shared legend on the left
            
            # Add shared legend vertically to the left of the first chart (only if we have labels)
            if shared_handles:
                # Create a single shared legend on the left side, vertically oriented
                legend_ax = agg_fig.add_subplot(gs[0, 0])
                legend_ax.axis('off')  # Hide axes
                agg_fig.legend(shared_handles, shared_labels, 
                              loc='center left', bbox_to_anchor=(0, 0.5),
                              fontsize=9, frameon=True, ncol=1)
            
            # Use subplots_adjust to add explicit spacing between subplots
            agg_fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.1, wspace=0.3)
            agg_fig.tight_layout(pad=2.0, rect=[0.12, 0.1, 0.98, 0.95])  # Leave space for legend on left
            scroll_layout.addWidget(agg_canvas)
            # Store aggregated canvas for export (None = aggregated view, not task-specific)
            self.radar_canvases.append((agg_canvas, None))
        
        # Separator
        separator = QLabel("─" * 80)
        separator.setStyleSheet("color: #ccc; padding: 10px;")
        scroll_layout.addWidget(separator)
        
        # Check if we're in participant mode
        is_participant_mode = self.mode in ["Each participant for selected groups", "Group mean and individual participants"]
        
        if is_participant_mode:
            # PARTICIPANT MODE: Show one radar chart per task per group, with each participant as a line
            # Similar structure to group mean mode, but with participant lines instead of group mean line
            participant_label = QLabel("Individual Participant Analysis")
            participant_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
            scroll_layout.addWidget(participant_label)
            
            # Get all participants from aggregated data for each group
            all_participants_by_group = {}
            for group_id in self.selected_groups:
                if group_id not in self.aggregated_data:
                    continue
                group_data = self.aggregated_data[group_id]
                participants = set()
                for task_data in group_data.values():
                    if isinstance(task_data, dict):
                        for key in task_data.keys():
                            if key != "_group_stats" and isinstance(task_data[key], dict):
                                participants.add(key)
                if participants:
                    all_participants_by_group[group_id] = sorted(participants)
            
            # Create individual charts for each task, listed vertically (in chronological order)
            for task_idx, task_id in enumerate(sorted_tasks):
                task_label = state.format_task(task_id)
                
                # Task section header
                task_header = QLabel(f"Task: {task_label}")
                task_header.setStyleSheet("font-weight: bold; font-size: 11px; padding: 10px 5px 5px 5px;")
                scroll_layout.addWidget(task_header)
                
                # Create horizontal layout for this task row (group charts side by side)
                task_row_layout = QHBoxLayout()
                task_row_layout.setContentsMargins(0, 0, 0, 0)
                task_row_layout.setSpacing(30)  # Increased spacing to prevent overlap
                task_row_widget = QWidget()
                task_row_widget.setLayout(task_row_layout)
                task_row_widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
                task_row_widget.setMinimumWidth(600)
                
                # Individual group charts for this task (side by side)
                for group_id in self.selected_groups:
                    if group_id not in self.aggregated_data or task_id not in self.aggregated_data[group_id]:
                        continue
                    
                    task_data = self.aggregated_data[group_id][task_id]
                    if not isinstance(task_data, dict):
                        continue
                    
                    group_name = state.get_effective_group_names().get(group_id, group_id)
                    
                    # Get participants for this group
                    participants = all_participants_by_group.get(group_id, [])
                    if not participants:
                        continue
                    
                    # Create individual chart for this group-task
                    num_groups_in_row = len([g for g in self.selected_groups if g in self.aggregated_data and task_id in self.aggregated_data[g]])
                    if num_groups_in_row == 1:
                        fig_width = 8
                        max_width = 800
                    else:
                        fig_width = 10
                        max_width = 1200
                    
                    ind_fig = Figure(figsize=(fig_width, 10))
                    ind_canvas = FigureCanvas(ind_fig)
                    ind_canvas.setMinimumHeight(500)
                    ind_canvas.setMinimumWidth(600)
                    ind_canvas.setMaximumWidth(max_width)
                    ind_canvas.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
                    # Install event filter to forward wheel events to scroll area
                    wheel_filter = WheelEventFilter(scroll)
                    ind_canvas.installEventFilter(wheel_filter)
                    ind_canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                    ax = ind_fig.add_subplot(111, projection='polar')
                    
                    # Calculate baselines for all participants
                    from analysis import _calculate_participant_baselines
                    baselines = _calculate_participant_baselines(state.df, radar_parameters) if state.df is not None else {}
                    
                    # Collect baseline-subtracted values for this group-task to normalize together
                    all_participant_values_for_task = {}
                    for p_id in participants:
                        if p_id in task_data and isinstance(task_data[p_id], dict):
                            participant_baseline = baselines.get(p_id, {})
                            for param in radar_parameters:
                                if param in task_data[p_id]:
                                    if param not in all_participant_values_for_task:
                                        all_participant_values_for_task[param] = []
                                    value = task_data[p_id][param].get('mean', 0)
                                    baseline = participant_baseline.get(param, 0)
                                    baseline_subtracted = value - baseline
                                    all_participant_values_for_task[param].append(baseline_subtracted)
                    
                    # Plot each participant as a separate line
                    color_map = plt.cm.get_cmap('tab20', len(participants))
                    for part_idx, participant_id in enumerate(participants):
                        if participant_id not in task_data or not isinstance(task_data[participant_id], dict):
                            continue
                        
                        participant_data = task_data[participant_id]
                        participant_baseline = baselines.get(participant_id, {})
                        
                        # Extract and normalize values for this participant (with baseline subtraction)
                        values = []
                        for param in radar_parameters:
                            value = None
                            if param in participant_data:
                                value = participant_data[param].get('mean', 0)
                                baseline = participant_baseline.get(param, 0)
                                baseline_subtracted = value - baseline
                                value = baseline_subtracted
                            
                            if value is not None and param in all_participant_values_for_task:
                                # Normalize using min/max across all participants for this task (baseline-subtracted values)
                                param_values = all_participant_values_for_task[param]
                                if param_values:
                                    min_val = min(param_values)
                                    max_val = max(param_values)
                                    if max_val > min_val:
                                        normalized_value = (value - min_val) / (max_val - min_val)
                                    else:
                                        normalized_value = 0.5 if value > 0 else 0.0
                                    values.append(normalized_value)
                                else:
                                    values.append(0.0)
                            else:
                                values.append(0.0)
                        
                        # Only plot if we have at least one non-zero value
                        if any(v > 0.0 for v in values):
                            values += values[:1]  # Complete the circle
                            color = color_map(part_idx)
                            ax.plot(angles, values, 'o-', linewidth=2, label=participant_id, color=color)
                            ax.fill(angles, values, alpha=0.25, color=color)
                    
                    # Fix text overflow: reduce font size and use shorter parameter names
                    ax.set_xticks(angles[:-1])
                    short_labels = []
                    for param in radar_parameters:
                        if len(param) > 15:
                            # Abbreviate long parameter names
                            if "Saccade Velocity" in param:
                                short_labels.append("Mean Saccade Vel.")
                            elif "Peak Saccade Velocity" in param:
                                short_labels.append("Peak Saccade Vel.")
                            elif "Saccade Amplitude" in param:
                                short_labels.append("Saccade Amp.")
                            elif "Standard Deviation" in param:
                                short_labels.append("Std Dev TCT")
                            elif "Pupil Diameter" in param:
                                short_labels.append("Pupil Diam.")
                            else:
                                short_labels.append(param[:15])
                        else:
                            short_labels.append(param)
                    ax.set_xticklabels(short_labels, fontsize=8)
                    ax.set_ylim(0, 1)
                    ax.set_title(f"{task_label} - {group_name}", fontsize=13, fontweight='bold', pad=20)
                    ax.grid(True)
                    # Place legend below chart to prevent overlap with adjacent charts
                    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(len(participants), 4), fontsize=8, frameon=True)
                    
                    ind_fig.tight_layout(pad=3.0)  # Increased padding for better spacing and legend visibility
                    task_row_layout.addWidget(ind_canvas)
                    # Store individual canvas for export with task_id
                    self.radar_canvases.append((ind_canvas, task_id))
                
                # Add the task row to the scroll layout
                scroll_layout.addWidget(task_row_widget)
        else:
            # GROUP MEAN MODE: Show individual charts for each task
            # BOTTOM SECTION: Individual charts listed vertically (larger, scrollable)
            grid_label = QLabel("Individual Task Analysis")
            grid_label.setStyleSheet("font-weight: bold; font-size: 12px; padding: 5px;")
            scroll_layout.addWidget(grid_label)
            
            # Create individual charts for each task, listed vertically (in chronological order)
            for task_idx, task_id in enumerate(sorted_tasks):
                task_label = state.format_task(task_id)
                
                # Task section header
                task_header = QLabel(f"Task: {task_label}")
                task_header.setStyleSheet("font-weight: bold; font-size: 11px; padding: 10px 5px 5px 5px;")
                scroll_layout.addWidget(task_header)
                
                # Create horizontal layout for this task row (group charts side by side)
                task_row_layout = QHBoxLayout()
                task_row_layout.setContentsMargins(0, 0, 0, 0)
                task_row_layout.setSpacing(30)  # Increased spacing to prevent overlap
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
                    # Install event filter to forward wheel events to scroll area
                    wheel_filter = WheelEventFilter(scroll)
                    ind_canvas.installEventFilter(wheel_filter)
                    ind_canvas.setFocusPolicy(Qt.FocusPolicy.NoFocus)
                    ax = ind_fig.add_subplot(111, projection='polar')
                    
                    # Extract values using same logic as aggregated view
                    values = []
                    for param in radar_parameters:
                        value = None
                        if isinstance(task_data, dict):
                            if param in task_data:
                                value = task_data[param]
                            elif "_group_stats" in task_data and param in task_data["_group_stats"]:
                                # Handle Standard Deviation of TCT from _group_stats
                                value = task_data["_group_stats"][param]
                        
                        # Use 0.0 for missing values
                        values.append(value if value is not None else 0.0)
                    
                    values += values[:1]
                    
                    # Use viridis color scheme for group mean mode
                    color_map = plt.cm.viridis
                    task_num = sorted_tasks.index(task_id) if task_id in sorted_tasks else 0
                    color = color_map(task_num / max(len(sorted_tasks) - 1, 1))
                    ax.plot(angles, values, 'o-', linewidth=3, color=color, markersize=8)
                    ax.fill(angles, values, alpha=0.3, color=color)
                    # Fix text overflow: reduce font size and use shorter parameter names
                    ax.set_xticks(angles[:-1])
                    short_labels = []
                    for param in radar_parameters:
                        if len(param) > 15:
                            # Abbreviate long parameter names
                            if "Saccade Velocity" in param:
                                short_labels.append("Mean Saccade Vel.")
                            elif "Peak Saccade Velocity" in param:
                                short_labels.append("Peak Saccade Vel.")
                            elif "Saccade Amplitude" in param:
                                short_labels.append("Saccade Amp.")
                            elif "Standard Deviation" in param:
                                short_labels.append("Std Dev TCT")
                            elif "Pupil Diameter" in param:
                                short_labels.append("Pupil Diam.")
                            else:
                                short_labels.append(param[:15])
                        else:
                            short_labels.append(param)
                    ax.set_xticklabels(short_labels, fontsize=8)
                    ax.set_ylim(0, 1)
                    ax.set_title(f"{task_label} - {group_name}", fontsize=13, fontweight='bold', pad=20)
                    ax.grid(True)
                    
                    ind_fig.tight_layout(pad=3.0)  # Increased padding for better spacing
                    task_row_layout.addWidget(ind_canvas)
                    # Store individual canvas for export with task_id
                    self.radar_canvases.append((ind_canvas, task_id))
                
                # Add the task row to the scroll layout
                scroll_layout.addWidget(task_row_widget)
        
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
        tct_std_data = {}  # {group_name: {task_id: std_value}}
        tct_participant_data = {}  # {group_name: {task_id: {participant: value}}}
        
        for group_id in self.selected_groups:
            if group_id not in self.aggregated_data:
                continue
            
            group_data = self.aggregated_data[group_id]
            group_name = state.get_effective_group_names().get(group_id, group_id)
            tct_data[group_name] = {}
            tct_std_data[group_name] = {}
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
                        tct_std = task_data[tct_param].get('std', 0)
                        # Convert from milliseconds to seconds
                        tct_data[group_name][task_id] = tct_value / 1000.0
                        tct_std_data[group_name][task_id] = tct_std / 1000.0
                    # Individual participant mode
                    elif any(isinstance(v, dict) for v in task_data.values() if isinstance(v, dict)):
                        participant_values_dict = {}
                        tct_values_list = []
                        for participant, participant_data in task_data.items():
                            if participant == "_group_stats":
                                continue
                            if isinstance(participant_data, dict) and tct_param in participant_data:
                                p_value = participant_data[tct_param].get('mean', 0)
                                # Convert from milliseconds to seconds
                                participant_values_dict[participant] = p_value / 1000.0
                                tct_values_list.append(p_value / 1000.0)
                        
                        if participant_values_dict:
                            tct_participant_data[group_name][task_id] = participant_values_dict
                            # Also store mean and std for group mean display
                            if tct_values_list:
                                tct_data[group_name][task_id] = float(np.mean(tct_values_list))
                                if len(tct_values_list) > 1:
                                    tct_std_data[group_name][task_id] = float(np.std(tct_values_list, ddof=1))
                                else:
                                    tct_std_data[group_name][task_id] = 0.0
        
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
            
            # Add small spacing between tasks (not between groups within a task)
            # Use a spacing factor: 1.0 for bar width + 0.15 for gap between tasks
            task_spacing = 1.15  # 1.0 for bars + 0.15 gap between task clusters
            x = np.arange(len(tasks)) * task_spacing
            
            # Calculate number of bars per task
            if self.mode == "Group mean and individual participants":
                # For combined mode, we'll show group means separately or make them distinct
                # Count: group means + all participants
                bars_per_group = 1 + len(all_participants)  # 1 group mean + participants
                num_bars = len(self.selected_groups) * bars_per_group
            else:
                # For participant-only mode, just participants
                num_bars = len(self.selected_groups) * len(all_participants)
            
            # Calculate width to make bars completely adjacent (no gaps)
            # Each task position has num_bars bars that should fill the space between positions
            # Total space per task = 1.0, so each bar width = 1.0 / num_bars
            width = 1.0 / max(num_bars, 1) if num_bars > 0 else 0.8
            
            # Use distinct colors for each bar - generate enough colors
            num_colors_needed = num_bars
            colors = plt.cm.tab20(np.linspace(0, 1, min(num_colors_needed, 20)))
            if num_colors_needed > 20:
                # Extend color palette if needed
                colors = np.vstack([colors, plt.cm.Set3(np.linspace(0, 1, num_colors_needed - 20))])
            
            bar_idx = 0
            
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
                    
                    # Get std values for error bars
                    std_values = []
                    for task_id in task_ids:
                        if group_name in tct_std_data and task_id in tct_std_data[group_name]:
                            std_values.append(tct_std_data[group_name][task_id])
                        else:
                            std_values.append(0)
                    
                    # Position group mean bar - calculate offset so bars are adjacent
                    # Bars fill space from x-0.5 to x+0.5, so offset = (bar_idx - (num_bars-1)/2) * width
                    offset = (bar_idx - (num_bars - 1) / 2) * width
                    # Use thicker bar for group mean (width * 1.2) and different style
                    ax.bar(x + offset, mean_values, width * 1.2, 
                          yerr=std_values,
                          label=f"{group_name} (Mean)", 
                          alpha=0.9, color=colors[bar_idx % len(colors)], 
                          edgecolor='black', linewidth=2,
                          capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
                    bar_idx += 1
                
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
                        
                        # Position participant bar - calculate offset so bars are adjacent
                        # Bars fill space from x-0.5 to x+0.5, so offset = (bar_idx - (num_bars-1)/2) * width
                        offset = (bar_idx - (num_bars - 1) / 2) * width
                        ax.bar(x + offset, participant_values, width,
                              label=f"{group_name} - {participant}",
                              alpha=0.7, color=colors[bar_idx % len(colors)])
                        bar_idx += 1
        
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
            
            # Add small spacing between tasks (not between groups within a task)
            task_spacing = 1.15  # 1.0 for bars + 0.15 gap between task clusters
            x = np.arange(len(tasks)) * task_spacing
            # Make bars adjacent (no gaps) - use full width divided by number of groups
            width = 1.0 / len(group_values) if group_values else 0.8
            
            # Use distinct colors for each group
            group_colors = plt.cm.tab10(np.linspace(0, 1, len(group_values)))
            for idx, (group_name, values) in enumerate(group_values.items()):
                # Get std values for error bars
                std_data = tct_std_data.get(group_name, {})
                std_values = [std_data.get(tid, 0) for tid in task_ids]
                
                # Calculate offset so bars are adjacent (no gaps)
                offset = (idx - (len(group_values) - 1) / 2) * width
                ax.bar(x + offset, values, width, yerr=std_values,
                      label=group_name, alpha=0.8, color=group_colors[idx],
                      capsize=5, error_kw={'elinewidth': 1.5, 'capthick': 1.5})
        
        ax.set_xlabel('Tasks')
        ax.set_ylabel('Task Completion Time (seconds)')
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
    
    def _show_export_dialog(self) -> None:
        """Show export dialog and handle exports."""
        # Determine what's available
        has_statistics = self.show_statistics
        has_rankings = self.rankings_data is not None and "Rank" in self.domains
        has_radar = len(self.radar_canvases) > 0 and "Radar Chart" in self.domains
        has_tct = self.tct_canvas is not None and "Task Completion Time" in self.domains
        
        dialog = ExportDialog(self, has_statistics, has_rankings, has_radar, has_tct)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        selections = dialog.get_selections()
        
        # Create date-time stamped output folder in ETT repository root
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Ensure we're using the ETT repository root directory
        ett_root = Path(__file__).parent
        output_dir = ett_root / "output" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export selected items
        exported_items = []
        
        if selections['stats_csv']:
            if self._export_statistics_csv(output_dir):
                exported_items.append("Statistics CSV")
        
        if selections['rankings_csv']:
            if self._export_rankings_csv(output_dir):
                exported_items.append("Rankings CSV")
        
        # Check if any image exports are selected
        has_image_exports = (selections['all_charts_png'] or 
                            selections['radar_png'] or 
                            selections['tct_png'])
        
        # Show loading dialog for image exports
        loading = None
        if has_image_exports:
            loading = LoadingDialog(self, "Exporting images...")
            loading.show()
            QApplication.processEvents()
        
        try:
            if selections['all_charts_png']:
                # Export both combined and separate when "Print all images to one PNG" is selected
                if loading:
                    loading.setMessage("Exporting combined chart...")
                    QApplication.processEvents()
                if self._export_all_charts_png(output_dir):
                    exported_items.append("All Charts Combined PNG")
                # Also export separately
                if loading:
                    loading.setMessage("Exporting radar charts...")
                    QApplication.processEvents()
                if self._export_radar_charts_png(output_dir):
                    exported_items.append("Radar Charts PNG")
                if loading:
                    loading.setMessage("Exporting TCT chart...")
                    QApplication.processEvents()
                if self._export_tct_chart_png(output_dir):
                    exported_items.append("TCT Chart PNG")
            else:
                if selections['radar_png']:
                    if loading:
                        loading.setMessage("Exporting radar charts...")
                        QApplication.processEvents()
                    if self._export_radar_charts_png(output_dir):
                        exported_items.append("Radar Charts PNG")
                if selections['tct_png']:
                    if loading:
                        loading.setMessage("Exporting TCT chart...")
                        QApplication.processEvents()
                    if self._export_tct_chart_png(output_dir):
                        exported_items.append("TCT Chart PNG")
        finally:
            if loading:
                loading.close()
                QApplication.processEvents()
        
        if exported_items:
            QMessageBox.information(
                self, 
                "Export Success", 
                f"Exported: {', '.join(exported_items)}\n\nFiles saved to: {output_dir}"
            )
        else:
            QMessageBox.warning(self, "Export Warning", "No items were exported.")
    
    def _export_statistics_csv(self, output_dir: Path) -> bool:
        """Export statistics to CSV."""
        try:
            stats_df = generate_statistics_table(
                self.aggregated_data,
                self.selected_tasks,
                self.selected_groups,
                self.mode
            )
            
            if stats_df.empty:
                QMessageBox.warning(self, "Export Error", "No statistics data to export.")
                return False
            
            filename = output_dir / "statistics.csv"
            stats_df.to_csv(filename, index=False)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export statistics: {str(e)}")
            return False
    
    def _export_rankings_csv(self, output_dir: Path) -> bool:
        """Export rankings to CSV."""
        try:
            if self.rankings_data is None:
                QMessageBox.warning(self, "Export Error", "No rankings data to export.")
                return False
            
            # Combine all rankings into one DataFrame
            all_rankings = []
            
            # Check if rankings_data is the combined structure (for "Group mean and individual participants" mode)
            if isinstance(self.rankings_data, dict) and 'group_means' in self.rankings_data and 'participants' in self.rankings_data:
                # Export group means first
                for group_id, ranking_df in self.rankings_data['group_means'].items():
                    ranking_df = ranking_df.copy()
                    ranking_df.insert(0, 'Group', state.get_effective_group_names().get(group_id, group_id))
                    ranking_df.insert(1, 'Participant', 'Group Mean')
                    all_rankings.append(ranking_df)
                
                # Then export individual participants
                for group_id, group_data in self.rankings_data['participants'].items():
                    if isinstance(group_data, dict):
                        for participant_id, ranking_df in group_data.items():
                            ranking_df = ranking_df.copy()
                            ranking_df.insert(0, 'Group', state.get_effective_group_names().get(group_id, group_id))
                            ranking_df.insert(1, 'Participant', participant_id)
                            all_rankings.append(ranking_df)
            else:
                # Standard structure (either group means or participants only)
                for group_id, group_data in self.rankings_data.items():
                    if isinstance(group_data, dict):
                        # Participant mode: group_data is {participant_id: DataFrame}
                        for participant_id, ranking_df in group_data.items():
                            ranking_df = ranking_df.copy()
                            ranking_df.insert(0, 'Group', state.get_effective_group_names().get(group_id, group_id))
                            ranking_df.insert(1, 'Participant', participant_id)
                            all_rankings.append(ranking_df)
                    else:
                        # Group mean mode: group_data is DataFrame
                        ranking_df = group_data.copy()
                        ranking_df.insert(0, 'Group', state.get_effective_group_names().get(group_id, group_id))
                        all_rankings.append(ranking_df)
            
            if not all_rankings:
                QMessageBox.warning(self, "Export Error", "No rankings data to export.")
                return False
            
            import pandas as pd
            combined_df = pd.concat(all_rankings, ignore_index=True)
            filename = output_dir / "rankings.csv"
            combined_df.to_csv(filename, index=False)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export rankings: {str(e)}")
            return False
    
    def _export_radar_charts_png(self, output_dir: Path) -> bool:
        """Export radar charts to PNG."""
        try:
            if not self.radar_canvases:
                QMessageBox.warning(self, "Export Error", "No radar charts to export.")
                return False
            
            saved_count = 0
            for idx, canvas_info in enumerate(self.radar_canvases):
                # Handle both old format (just canvas) and new format (tuple with task_id)
                if isinstance(canvas_info, tuple):
                    canvas, task_id = canvas_info
                else:
                    canvas = canvas_info
                    task_id = None
                
                if task_id is None:
                    filename = output_dir / f"radar_chart_aggregated.png"
                else:
                    task_label = state.format_task(task_id)
                    filename = output_dir / f"radar_chart_task_{task_label.replace(' ', '_')}_{idx + 1}.png"
                
                canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
                saved_count += 1
            
            return saved_count > 0
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export radar charts: {str(e)}")
            return False
    
    def _export_tct_chart_png(self, output_dir: Path) -> bool:
        """Export TCT chart to PNG."""
        try:
            if self.tct_canvas is None:
                QMessageBox.warning(self, "Export Error", "No TCT chart to export.")
                return False
            
            filename = output_dir / "tct_chart.png"
            self.tct_canvas.figure.savefig(filename, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export TCT chart: {str(e)}")
            return False
    
    def _export_all_charts_png(self, output_dir: Path) -> bool:
        """Export all charts combined into one PNG image, organized by task."""
        try:
            import tempfile
            from PIL import Image
            from collections import defaultdict
            
            if not self.radar_canvases and not self.tct_canvas:
                QMessageBox.warning(self, "Export Error", "No charts to export.")
                return False
            
            # Organize radar charts by task_id
            # Group charts: aggregated (None) first, then by task_id
            aggregated_charts = []
            charts_by_task = defaultdict(list)
            
            for canvas_info in self.radar_canvases:
                # Handle both old format (just canvas) and new format (tuple with task_id)
                if isinstance(canvas_info, tuple):
                    canvas, task_id = canvas_info
                else:
                    canvas = canvas_info
                    task_id = None
                
                if task_id is None:
                    aggregated_charts.append(canvas)
                else:
                    charts_by_task[task_id].append(canvas)
            
            # Collect all chart images using temporary files
            chart_images = []
            chart_positions = []  # (row, col) for each chart
            temp_files = []
            
            try:
                # Process aggregated charts first (one per row, full width)
                for canvas in aggregated_charts:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_files.append(temp_file.name)
                    temp_file.close()
                    canvas.figure.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight')
                    chart_images.append(Image.open(temp_file.name))
                    chart_positions.append((len(chart_positions), 0))  # Full width, one per row
                
                # Process task charts - group by task, same task on same row
                # Sort tasks naturally
                from data_processor import _natural_sort_key
                sorted_task_ids = sorted(charts_by_task.keys(), key=_natural_sort_key)
                
                current_row = len(aggregated_charts)
                max_cols_per_row = 0
                
                for task_id in sorted_task_ids:
                    task_charts = charts_by_task[task_id]
                    # All charts for this task go on the same row
                    for col, canvas in enumerate(task_charts):
                        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                        temp_files.append(temp_file.name)
                        temp_file.close()
                        canvas.figure.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight')
                        chart_images.append(Image.open(temp_file.name))
                        chart_positions.append((current_row, col))
                        max_cols_per_row = max(max_cols_per_row, col + 1)
                    current_row += 1
                
                # Add TCT chart at the end (full width)
                if self.tct_canvas:
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_files.append(temp_file.name)
                    temp_file.close()
                    self.tct_canvas.figure.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight')
                    chart_images.append(Image.open(temp_file.name))
                    chart_positions.append((current_row, 0))
                    current_row += 1
                
                # Calculate dimensions for combined image
                max_width = max(img.width for img in chart_images) if chart_images else 800
                max_height = max(img.height for img in chart_images) if chart_images else 600
                
                # Determine grid: max columns per row, total rows
                total_rows = current_row
                cols = max(max_cols_per_row, 1)  # At least 1 column
                
                # Create combined image
                combined_width = cols * max_width
                combined_height = total_rows * max_height
                combined_img = Image.new('RGB', (combined_width, combined_height), color='white')
                
                # Paste charts into combined image at their positions
                for idx, (img, (row, col)) in enumerate(zip(chart_images, chart_positions)):
                    x_offset = col * max_width + (max_width - img.width) // 2
                    y_offset = row * max_height + (max_height - img.height) // 2
                    combined_img.paste(img, (x_offset, y_offset))
                
                # Save combined image
                filename = output_dir / "all_charts_combined.png"
                combined_img.save(filename, 'PNG', dpi=(300, 300))
                
                return True
            finally:
                # Clean up: close images and delete temporary files
                for img in chart_images:
                    img.close()
                for temp_file_path in temp_files:
                    try:
                        Path(temp_file_path).unlink(missing_ok=True)
                    except Exception:
                        pass  # Ignore errors when cleaning up temp files
            
        except ImportError:
            # Fallback if PIL is not available - use simpler matplotlib approach
            try:
                # Count total charts
                num_radar = len(self.radar_canvases) if self.radar_canvases else 0
                num_tct = 1 if self.tct_canvas else 0
                total_charts = num_radar + num_tct
                
                if total_charts == 0:
                    QMessageBox.warning(self, "Export Error", "No charts to export.")
                    return False
                
                # Calculate grid layout
                cols = 2
                rows = (total_charts + cols - 1) // cols
                
                # Create combined figure
                combined_fig = Figure(figsize=(cols * 10, rows * 8))
                
                chart_idx = 0
                
                # Add radar charts by saving and loading as images
                if self.radar_canvases:
                    for canvas_info in self.radar_canvases:
                        # Handle both old format (just canvas) and new format (tuple with task_id)
                        if isinstance(canvas_info, tuple):
                            radar_canvas, _ = canvas_info
                        else:
                            radar_canvas = canvas_info
                        
                        row = chart_idx // cols
                        col = chart_idx % cols
                        subplot_idx = row * cols + col + 1
                        ax = combined_fig.add_subplot(rows, cols, subplot_idx)
                        ax.axis('off')
                        ax.text(0.5, 0.5, f'Radar Chart {chart_idx + 1}\n(Use PIL for full image combination)', 
                               ha='center', va='center', transform=ax.transAxes)
                        chart_idx += 1
                
                # Add TCT chart
                if self.tct_canvas:
                    row = chart_idx // cols
                    col = chart_idx % cols
                    subplot_idx = row * cols + col + 1
                    ax = combined_fig.add_subplot(rows, cols, subplot_idx)
                    ax.axis('off')
                    ax.text(0.5, 0.5, 'TCT Chart\n(Use PIL for full image combination)', 
                           ha='center', va='center', transform=ax.transAxes)
                
                combined_fig.tight_layout()
                filename = output_dir / "all_charts_combined.png"
                combined_fig.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.warning(
                    self, 
                    "Export Notice", 
                    "PIL/Pillow not available. Charts exported separately.\n"
                    "Install Pillow (pip install Pillow) for full image combination."
                )
                return True
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export combined charts: {str(e)}")
                return False
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export combined charts: {str(e)}")
            return False
