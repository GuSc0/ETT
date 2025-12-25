"""
Main window for the Eye Tracking Tool using PyQt6.
"""
from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QLineEdit, QComboBox, QCheckBox, QGroupBox, QGridLayout,
    QFrame, QMessageBox, QFileDialog, QMenu, QMenuBar, QSplitter
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction

from state import state
from models import EXPECTED_COLUMNS, PARAMETER_OPTIONS
from data_processor import validate_tsv_format, extract_participants, extract_tasks_from_toi
from dialogs import GroupParticipantsDialog, GroupTasksDialog


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Eye Tracking Analysis Tool")
        self.setMinimumSize(1200, 720)
        
        # Result domain checkboxes
        self.result_domain_vars: Dict[str, QCheckBox] = {}
        
        # Result group/task checkboxes
        self.result_group_checkboxes: Dict[str, QCheckBox] = {}
        self.result_task_checkboxes: Dict[str, QCheckBox] = {}
        
        # Deselect parameters
        self.deselect_param_checkboxes: Dict[str, QCheckBox] = {}
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Top bar
        topbar = QHBoxLayout()
        
        self.load_btn = QPushButton("Load TSV")
        self.load_btn.setFixedWidth(120)
        self.load_btn.clicked.connect(self._load_tsv)
        topbar.addWidget(self.load_btn)
        
        self.group_participants_btn = QPushButton("Group Participants")
        self.group_participants_btn.setFixedWidth(150)
        self.group_participants_btn.setEnabled(False)
        self.group_participants_btn.clicked.connect(self._open_group_participants)
        topbar.addWidget(self.group_participants_btn)
        
        self.group_tasks_btn = QPushButton("Group Tasks")
        self.group_tasks_btn.setFixedWidth(120)
        self.group_tasks_btn.setEnabled(False)
        self.group_tasks_btn.clicked.connect(self._open_group_tasks)
        topbar.addWidget(self.group_tasks_btn)
        
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("No file loaded")
        topbar.addWidget(self.file_path_edit)
        
        main_layout.addLayout(topbar)
        
        # Listboxes section
        listbox_layout = QHBoxLayout()
        
        # Participants listbox
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel('Detected participants (from "Participant"):'))
        self.participants_list = QListWidget()
        self.participants_list.setMaximumHeight(120)
        left_layout.addWidget(self.participants_list)
        listbox_layout.addWidget(left_widget)
        
        # Tasks listbox
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel('Detected tasks (from "TOI" suffix):'))
        self.tasks_list = QListWidget()
        self.tasks_list.setMaximumHeight(120)
        right_layout.addWidget(self.tasks_list)
        listbox_layout.addWidget(right_widget)
        
        main_layout.addLayout(listbox_layout)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator)
        
        # Results section
        results_label = QLabel("Results")
        results_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        main_layout.addWidget(results_label)
        
        # Mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Show results for:"))
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Only group mean",
            "Each participant for selected groups",
            "Group mean and individual participants",
        ])
        self.mode_combo.setFixedWidth(350)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        main_layout.addLayout(mode_layout)
        
        # Groups and Tasks frames
        sel_layout = QHBoxLayout()
        
        self.groups_group = QGroupBox("Groups")
        self.groups_layout = QVBoxLayout(self.groups_group)
        sel_layout.addWidget(self.groups_group)
        
        self.tasks_group = QGroupBox("Tasks")
        self.tasks_layout = QGridLayout(self.tasks_group)
        sel_layout.addWidget(self.tasks_group)
        
        main_layout.addLayout(sel_layout)
        
        # Bottom row: Result domain and Deselect parameters
        bottom_layout = QHBoxLayout()
        
        # Result domain
        domain_group = QGroupBox("Result domain")
        domain_layout = QHBoxLayout(domain_group)
        
        for name in ["Rank", "Radar Chart", "Task Completion Time"]:
            cb = QCheckBox(name)
            cb.setChecked(True)
            self.result_domain_vars[name] = cb
            domain_layout.addWidget(cb)
        
        domain_layout.addStretch()
        bottom_layout.addWidget(domain_group)
        
        # Deselect parameters
        deselect_group = QGroupBox("Deselect Parameters")
        deselect_layout = QHBoxLayout(deselect_group)
        
        self.deselect_enabled_cb = QCheckBox("Enable Deselect")
        self.deselect_enabled_cb.stateChanged.connect(self._toggle_deselect_menu)
        deselect_layout.addWidget(self.deselect_enabled_cb)
        
        self.deselect_btn = QPushButton("None")
        self.deselect_btn.setEnabled(False)
        self.deselect_btn.clicked.connect(self._show_deselect_menu)
        deselect_layout.addWidget(self.deselect_btn)
        
        deselect_layout.addStretch()
        bottom_layout.addWidget(deselect_group)
        
        main_layout.addLayout(bottom_layout)
        
        # Action buttons
        action_layout = QHBoxLayout()
        
        self.show_results_btn = QPushButton("Show Results")
        self.show_results_btn.setFixedWidth(150)
        self.show_results_btn.setEnabled(False)
        self.show_results_btn.clicked.connect(self._on_show_results)
        action_layout.addWidget(self.show_results_btn)
        
        self.exec_summary_btn = QPushButton("Print Executive Summary")
        self.exec_summary_btn.setFixedWidth(180)
        self.exec_summary_btn.setEnabled(False)
        self.exec_summary_btn.clicked.connect(self._on_print_exec_summary)
        action_layout.addWidget(self.exec_summary_btn)
        
        action_layout.addStretch()
        main_layout.addLayout(action_layout)
        
        main_layout.addStretch()
        
        # Initialize deselect menu
        self._build_deselect_menu()
        
        # Initial population
        self._refresh_group_task_toggles()
    
    def _load_tsv(self) -> None:
        """Load and validate a TSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select TSV file",
            "",
            "TSV files (*.tsv);;All files (*.*)"
        )
        
        if not file_path:
            return
        
        # Validate format
        result = validate_tsv_format(file_path, EXPECTED_COLUMNS)
        if not result.ok:
            QMessageBox.critical(self, "Format error", result.message or "Unknown format error.")
            return
        
        # Load data
        try:
            state.df = pd.read_csv(file_path, sep="\t")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return
        
        # Update UI with file path
        state.loaded_file_path = file_path
        self.file_path_edit.setText(file_path)
        
        # Reset caches
        state.participants_cache = []
        state.tasks_cache = []
        
        # Clear listboxes
        self.participants_list.clear()
        self.tasks_list.clear()
        
        # Disable action buttons during loading
        self.group_participants_btn.setEnabled(False)
        self.group_tasks_btn.setEnabled(False)
        self._set_main_action_buttons_enabled(False)
        
        # Extract participants
        try:
            state.participants_cache = extract_participants(state.df, column="Participant")
        except Exception as e:
            QMessageBox.critical(self, "Participant scan failed", str(e))
            state.df = None
            return
        
        # Extract tasks
        try:
            state.tasks_cache = extract_tasks_from_toi(state.df, column="TOI")
        except Exception as e:
            QMessageBox.critical(self, "Task scan failed", str(e))
            state.df = None
            return
        
        # Keep only labels for tasks that exist
        state.task_labels = {k: v for k, v in state.task_labels.items() if k in set(state.tasks_cache)}
        
        # Update listboxes
        for p in state.participants_cache:
            self.participants_list.addItem(p)
        
        self._refresh_tasks_listbox()
        
        # Enable group buttons
        self.group_participants_btn.setEnabled(bool(state.participants_cache))
        self.group_tasks_btn.setEnabled(bool(state.tasks_cache))
        
        self._set_main_action_buttons_enabled(True)
        self._refresh_group_task_toggles()
        
        QMessageBox.information(
            self, 
            "Success", 
            f"Loaded {len(state.df)} rows.\n"
            f"Found {len(state.participants_cache)} participants and {len(state.tasks_cache)} tasks."
        )
    
    def _refresh_tasks_listbox(self) -> None:
        """Refresh the tasks listbox with formatted task names."""
        self.tasks_list.clear()
        for t in state.tasks_cache:
            self.tasks_list.addItem(state.format_task(t))
    
    def _set_main_action_buttons_enabled(self, enabled: bool) -> None:
        """Enable or disable main action buttons."""
        self.show_results_btn.setEnabled(enabled)
        self.exec_summary_btn.setEnabled(enabled)
    
    def _refresh_group_task_toggles(self) -> None:
        """Refresh the checkbox lists in the Results section."""
        # Clear existing group checkboxes
        while self.groups_layout.count():
            item = self.groups_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Clear existing task checkboxes
        while self.tasks_layout.count():
            item = self.tasks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.result_group_checkboxes.clear()
        self.result_task_checkboxes.clear()
        
        # Groups toggles
        effective_groups = state.get_effective_participant_groups()
        effective_names = state.get_effective_group_names()
        
        if not effective_groups:
            label = QLabel("(Load a TSV file first)")
            label.setStyleSheet("color: #666;")
            self.groups_layout.addWidget(label)
        else:
            for gid in effective_groups.keys():
                name = effective_names.get(gid, gid)
                cb = QCheckBox(name)
                cb.setChecked(True)
                self.result_group_checkboxes[gid] = cb
                self.groups_layout.addWidget(cb)
        
        self.groups_layout.addStretch()
        
        # Tasks toggles
        if not state.tasks_cache:
            label = QLabel("(Load a TSV file first)")
            label.setStyleSheet("color: #666;")
            self.tasks_layout.addWidget(label, 0, 0)
            return
        
        cols = 8
        row = 0
        col = 0
        for tid in state.tasks_cache:
            cb = QCheckBox(state.format_task(tid))
            cb.setChecked(True)
            self.result_task_checkboxes[tid] = cb
            self.tasks_layout.addWidget(cb, row, col)
            col += 1
            if col >= cols:
                col = 0
                row += 1
    
    def _build_deselect_menu(self) -> None:
        """Build the deselect parameters menu."""
        self.deselect_menu = QMenu(self)
        self.deselect_param_checkboxes.clear()
        
        for name in PARAMETER_OPTIONS:
            action = QAction(name, self)
            action.setCheckable(True)
            action.triggered.connect(self._update_deselect_label)
            self.deselect_menu.addAction(action)
            self.deselect_param_checkboxes[name] = action
    
    def _toggle_deselect_menu(self, state_val: int) -> None:
        """Toggle the deselect parameters menu button."""
        self.deselect_btn.setEnabled(state_val == Qt.CheckState.Checked.value)
    
    def _show_deselect_menu(self) -> None:
        """Show the deselect parameters menu."""
        self.deselect_menu.exec(self.deselect_btn.mapToGlobal(self.deselect_btn.rect().bottomLeft()))
    
    def _update_deselect_label(self) -> None:
        """Update the deselect button label."""
        count = sum(1 for action in self.deselect_param_checkboxes.values() if action.isChecked())
        self.deselect_btn.setText("None" if count == 0 else f"{count} deselected")
    
    def _open_group_participants(self) -> None:
        """Open the group participants dialog."""
        if not state.participants_cache:
            QMessageBox.information(self, "No data", "Load a TSV file first.")
            return
        
        dialog = GroupParticipantsDialog(self)
        if dialog.exec() == GroupParticipantsDialog.DialogCode.Accepted:
            self._refresh_group_task_toggles()
    
    def _open_group_tasks(self) -> None:
        """Open the group tasks dialog."""
        if not state.tasks_cache:
            QMessageBox.information(self, "No data", "Load a TSV file first.")
            return
        
        dialog = GroupTasksDialog(self)
        if dialog.exec() == GroupTasksDialog.DialogCode.Accepted:
            self._refresh_tasks_listbox()
            self._refresh_group_task_toggles()
    
    def _on_show_results(self) -> None:
        """Handle Show Results button click."""
        if state.df is None:
            QMessageBox.information(self, "No data", "Load a TSV file first.")
            return
        
        selected_groups = [gid for gid, cb in self.result_group_checkboxes.items() if cb.isChecked()]
        selected_tasks = [tid for tid, cb in self.result_task_checkboxes.items() if cb.isChecked()]
        domains = [name for name, cb in self.result_domain_vars.items() if cb.isChecked()]
        mode = self.mode_combo.currentText()
        
        deselected = []
        if self.deselect_enabled_cb.isChecked():
            deselected = [name for name, action in self.deselect_param_checkboxes.items() if action.isChecked()]
        
        msg = (
            f"Mode: {mode}\n"
            f"Groups selected: {len(selected_groups)}\n"
            f"Tasks selected: {len(selected_tasks)}\n"
            f"Result domains: {', '.join(domains) if domains else '(none)'}\n"
            f"Deselected parameters: {', '.join(deselected) if deselected else '(none)'}"
        )
        QMessageBox.information(self, "Show Results (placeholder)", msg)
    
    def _on_print_exec_summary(self) -> None:
        """Handle Print Executive Summary button click."""
        if state.df is None:
            QMessageBox.information(self, "No data", "Load a TSV file first.")
            return
        QMessageBox.information(self, "Executive Summary (placeholder)", 
                                "This will generate the executive summary later.")
