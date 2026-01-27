"""
Main window for the Eye Tracking Tool using PyQt6.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import pandas as pd

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QLineEdit, QComboBox, QCheckBox, QGroupBox, QGridLayout,
    QFrame, QMessageBox, QFileDialog, QMenu, QMenuBar, QSplitter,
    QDialog, QTextEdit, QSlider, QScrollArea, QProgressDialog, QApplication
)
from PyQt6.QtCore import Qt, QSize, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QAction, QPainter, QColor

from state import state
from models import EXPECTED_COLUMNS, PARAMETER_OPTIONS
from data_processor import (
    validate_tsv_format,
    extract_participants,
    extract_tasks_from_toi,
    normalize_by_participant_baseline,
)
from dialogs import GroupParticipantsDialog, GroupTasksDialog
from analysis import aggregate_by_groups
from results_window import ResultsWindow
from executive_summary import (
    generate_executive_summary,
    format_statistics_table_for_summary,
    export_summary_to_text,
    export_summary_to_pdf
)
from executive_summary_latex import generate_latex_summary, find_pdflatex


class LoadingDialog(QDialog):
    """Loading dialog with animated spinner."""
    
    def __init__(self, parent: Optional[QWidget] = None, message: str = "Loading...") -> None:
        super().__init__(parent)
        self.setWindowTitle("Loading")
        self.setModal(True)
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowTitleHint)
        self.setMinimumSize(250, 120)
        self.setMaximumSize(250, 120)
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(15)
        
        # Spinner label with rotating text
        self.spinner_label = QLabel()
        self.spinner_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinner_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.spinner_label.setText(message)
        layout.addWidget(self.spinner_label)
        
        # Animation for spinner
        self.rotation = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_animation)
        self.timer.start(100)  # Update every 100ms
    
    def _update_animation(self) -> None:
        """Update animation frame."""
        self.rotation = (self.rotation + 45) % 360
        # Create rotating spinner characters
        spinner_chars = ['|', '/', '-', '\\']
        char_idx = (self.rotation // 90) % len(spinner_chars)
        current_text = self.spinner_label.text()
        # Extract message part (before spinner char)
        if ' ' in current_text:
            message_part = current_text.rsplit(' ', 1)[0]
        else:
            message_part = "Loading"
        self.spinner_label.setText(f"{message_part} {spinner_chars[char_idx]}")
    
    def showEvent(self, event) -> None:
        """Start animation when shown."""
        super().showEvent(event)
        self.timer.start()
    
    def hideEvent(self, event) -> None:
        """Stop animation when hidden."""
        super().hideEvent(event)
        self.timer.stop()
    
    def setMessage(self, message: str) -> None:
        """Update the loading message."""
        self.spinner_label.setText(message)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Eye Tracking Analysis Tool")
        self.setMinimumSize(1200, 720)

        # Ensure output folder exists in ETT repository root
        from pathlib import Path
        ett_root = Path(__file__).parent
        output_dir = ett_root / "output"
        output_dir.mkdir(exist_ok=True)

        # Result domain checkboxes
        self.result_domain_vars: Dict[str, QCheckBox] = {}

        # Result group/task checkboxes
        self.result_group_checkboxes: Dict[str, QCheckBox] = {}
        self.result_task_checkboxes: Dict[str, QCheckBox] = {}

        # Deselect parameters
        self.deselect_param_checkboxes: Dict[str, QCheckBox] = {}
        
        # Parameter weighting
        self.parameter_weight_sliders: Dict[str, QSlider] = {}
        self.parameter_weight_labels: Dict[str, QLabel] = {}
        self.weight_rows: List = []  # List of (combo, slider, label) tuples (old system - kept for compatibility)
        self.weighting_enabled_cb = None  # Old system checkbox - no longer used
        
        self._setup_ui()

    # --- Weighting slider mapping (QSlider is int-based) ---
    def _weight_slider_to_float(self, slider_val: int) -> float:
        # 2..10 -> 1.0..5.0 in 0.5 steps
        return float(slider_val) / 2.0

    def _weight_float_to_slider(self, weight: float) -> int:
        # clamp + round to nearest 0.5 step, then map to int
        w = max(1.0, min(5.0, float(weight)))
        return int(round(w * 2.0))

    def _format_weight(self, weight: float) -> str:
        return f"{weight:.2f}".rstrip("0").rstrip(".")

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

        # Button-Text wie gewünscht: "Name Tasks"
        self.group_tasks_btn = QPushButton("Name Tasks")
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
        
        # Statistics tab toggle
        stats_cb = QCheckBox("Statistics")
        stats_cb.setChecked(True)
        self.result_domain_vars["Statistics"] = stats_cb
        domain_layout.addWidget(stats_cb)
        
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

        # Weighting parameters - HIDDEN (using "Parameter Weighting" section below instead)
        # Old system removed - keeping only the new "Parameter Weighting" with sliders

        main_layout.addLayout(bottom_layout)
        
        # Parameter Weighting section
        weighting_group = QGroupBox("Parameter Weighting")
        weighting_layout = QVBoxLayout(weighting_group)
        weighting_layout.setSpacing(8)
        
        # Header with instructions and reset button
        header_layout = QHBoxLayout()
        weight_instructions = QLabel(
            "Adjust parameter weights (0-300%, default 100%). Higher weights increase the parameter's influence on rankings."
        )
        weight_instructions.setWordWrap(True)
        weight_instructions.setStyleSheet("padding: 6px; color: #666; font-size: 10pt;")
        header_layout.addWidget(weight_instructions)
        
        reset_weights_btn = QPushButton("Reset")
        reset_weights_btn.setFixedWidth(80)
        reset_weights_btn.clicked.connect(self._reset_all_weights)
        header_layout.addWidget(reset_weights_btn)
        
        weighting_layout.addLayout(header_layout)
        
        # Scrollable area for weight sliders
        weight_scroll = QScrollArea()
        weight_scroll.setWidgetResizable(True)
        weight_scroll.setMaximumHeight(250)
        weight_scroll.setMinimumHeight(200)
        weight_scroll.setFrameShape(QFrame.Shape.NoFrame)
        weight_scroll_widget = QWidget()
        weight_scroll_layout = QVBoxLayout(weight_scroll_widget)
        weight_scroll_layout.setSpacing(6)
        weight_scroll_layout.setContentsMargins(5, 5, 5, 5)
        
        # Initialize default weights
        for param in PARAMETER_OPTIONS:
            if param not in state.parameter_weights:
                state.parameter_weights[param] = 1.0  # Default 100%
        
        # Create slider for each parameter
        for param in PARAMETER_OPTIONS:
            param_layout = QHBoxLayout()
            param_layout.setSpacing(10)
            
            param_label = QLabel(param)
            param_label.setMinimumWidth(220)
            param_label.setMaximumWidth(220)
            param_label.setWordWrap(True)
            param_layout.addWidget(param_label)
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(300)
            slider.setValue(100)  # Default 100% (1.0)
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(50)
            slider.setMinimumWidth(300)
            slider.valueChanged.connect(lambda value, p=param: self._update_weight_display(p, value))
            self.parameter_weight_sliders[param] = slider
            param_layout.addWidget(slider)
            
            weight_display = QLabel("1.00 (100%)")
            weight_display.setMinimumWidth(120)
            weight_display.setMaximumWidth(120)
            weight_display.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            weight_display.setStyleSheet("font-weight: bold; color: #0066cc;")
            self.parameter_weight_labels[param] = weight_display
            param_layout.addWidget(weight_display)
            
            param_layout.addStretch()
            weight_scroll_layout.addLayout(param_layout)
        
        weight_scroll_layout.addStretch()
        weight_scroll.setWidget(weight_scroll_widget)
        weighting_layout.addWidget(weight_scroll)
        
        main_layout.addWidget(weighting_group)
        
        # Action buttons - only Show Results and Executive Summary
        action_layout = QHBoxLayout()

        self.show_results_btn = QPushButton("Show Results")
        self.show_results_btn.setFixedWidth(150)
        self.show_results_btn.setEnabled(False)
        self.show_results_btn.clicked.connect(self._on_show_results)
        action_layout.addWidget(self.show_results_btn)

        # Removed buttons: Normalize, Calculate metric averages, Calculate Participant Rank
        # Functionality kept in code but not exposed in UI

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

        # Initialize weighting UI - removed (old system hidden)
        # self._reset_weighting_ui()  # No longer needed - using "Parameter Weighting" sliders

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

        # Reset normalized data (must be recomputed for new file)
        state.normalized_df = None
        state.metric_averages_df = None
        state.participant_rank_df = None
        self._participant_rank_signature = None

        # Reset weighting UI/weights (parameters list may change later)
        # Old system removed - weights are managed by "Parameter Weighting" sliders
        # Reset weights to defaults
        for param in PARAMETER_OPTIONS:
            state.parameter_weights[param] = 1.0
            if param in self.parameter_weight_sliders:
                self.parameter_weight_sliders[param].setValue(100)  # Reset slider to 100%
                self.parameter_weight_labels[param].setText("1.00 (100%)")

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

        # Old weighting system removed - weights always enabled via "Parameter Weighting" sliders

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
        # Removed buttons: normalize_btn, calc_metric_averages_btn, calc_participant_rank_btn

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
            # Tasks 0a and 0b should be unchecked by default
            if tid in ("0a", "0b"):
                cb.setChecked(False)
            else:
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

    def _reset_weighting_ui(self) -> None:
        """Clear all weighting rows and internal weights; keep toggle state."""
        state.parameter_weights.clear()
        self.weight_rows.clear()

        # Clear UI rows
        if hasattr(self, "weighting_rows_layout"):
            while self.weighting_rows_layout.count():
                item = self.weighting_rows_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

        # If enabled and we have parameters, create first row
        if self.weighting_enabled_cb is not None and self.weighting_enabled_cb.isChecked():
            self._ensure_next_weight_row()

    def _on_weighting_toggled(self, state_val: int) -> None:
        enabled = state_val == Qt.CheckState.Checked.value
        if not enabled:
            self._reset_weighting_ui()
            return
        self._reset_weighting_ui()

    def _selected_weight_params(self) -> set[str]:
        selected: set[str] = set()
        for combo, _slider, _lbl in self.weight_rows:
            val = combo.currentData()
            if isinstance(val, str) and val:
                selected.add(val)
        return selected

    def _current_ui_parameter_weights(self) -> Dict[str, float]:
        """
        Liefert die aktuell im Weighting-UI gewählten Gewichte als Dict[param] = weight.
        Quelle ist ausschließlich die UI (Combos + Slider), nicht self.parameter_weights.
        """
        weights: Dict[str, float] = {}
        if self.weighting_enabled_cb is None or not self.weighting_enabled_cb.isChecked():
            return weights

        for combo, slider, _lbl in self.weight_rows:
            param = combo.currentData()
            if isinstance(param, str) and param:
                weights[param] = float(self._weight_slider_to_float(slider.value()))
        return weights

    def _available_weight_params(self) -> list[str]:
        # PARAMETER_OPTIONS sind die “Parameter” im Tool
        selected = self._selected_weight_params()
        return [p for p in PARAMETER_OPTIONS if p not in selected]

    def _create_weight_row(self) -> None:
        available = self._available_weight_params()
        if not available:
            return

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        combo = QComboBox()
        combo.setFixedWidth(260)
        combo.addItem("Select parameter...", None)
        for p in available:
            combo.addItem(p, p)

        slider = QSlider(Qt.Orientation.Horizontal)
        # 1.0..5.0 in 0.5 steps => 2..10
        slider.setMinimum(2)
        slider.setMaximum(10)
        slider.setValue(self._weight_float_to_slider(1.0))
        slider.setSingleStep(1)
        slider.setPageStep(1)
        slider.setFixedWidth(220)
        slider.setEnabled(False)  # erst aktivieren, wenn Parameter gewählt

        value_lbl = QLabel(self._format_weight(1.0))
        value_lbl.setFixedWidth(44)
        value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value_lbl.setEnabled(False)

        # Signals
        combo.currentIndexChanged.connect(
            lambda _i, c=combo, s=slider, l=value_lbl: self._on_weight_row_changed(c, s, l)
        )
        slider.valueChanged.connect(
            lambda _v, c=combo, s=slider, l=value_lbl: self._on_weight_row_changed(c, s, l)
        )

        row_layout.addWidget(combo)
        row_layout.addWidget(slider)
        row_layout.addWidget(value_lbl)
        row_layout.addStretch()

        self.weighting_rows_layout.addWidget(row)
        self.weight_rows.append((combo, slider, value_lbl))

    def _on_weight_row_changed(self, combo: QComboBox, slider: QSlider, value_lbl: QLabel) -> None:
        # Wenn Toggle aus: ignorieren
        if self.weighting_enabled_cb is None or not self.weighting_enabled_cb.isChecked():
            return

        param = combo.currentData()
        if not isinstance(param, str) or not param:
            slider.setEnabled(False)
            value_lbl.setEnabled(False)
            return

        slider.setEnabled(True)
        value_lbl.setEnabled(True)

        weight = self._weight_slider_to_float(slider.value())
        value_lbl.setText(self._format_weight(weight))

        # Gewicht speichern (1.0..5.0 in 0.5er Schritten)
        state.parameter_weights[param] = float(weight)

        # Alle Combos neu aufbauen, damit keine Duplikate möglich sind
        self._rebuild_weight_combos(keep_current=True)

        # Wenn diese Zeile “aktiv” ist, nächste Zeile sicherstellen
        self._ensure_next_weight_row()

    def _rebuild_weight_combos(self, keep_current: bool = True) -> None:
        selected = self._selected_weight_params()

        for combo, slider, value_lbl in self.weight_rows:
            current = combo.currentData() if keep_current else None

            # Block signals während rebuild
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("Select parameter...", None)

            # In dieser Combo darf current weiterhin drin sein
            allowed: list[str] = []
            for p in PARAMETER_OPTIONS:
                if p == current:
                    allowed.append(p)
                elif p not in selected:
                    allowed.append(p)

            for p in allowed:
                combo.addItem(p, p)

            # Restore selection
            if isinstance(current, str) and current:
                idx = combo.findData(current)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
                slider.setEnabled(True)
                value_lbl.setEnabled(True)

                # Label/weight sync
                weight = self._weight_slider_to_float(slider.value())
                value_lbl.setText(self._format_weight(weight))
                state.parameter_weights[current] = float(weight)
            else:
                combo.setCurrentIndex(0)
                slider.setEnabled(False)
                value_lbl.setEnabled(False)

                # Falls vorher ein Param gesetzt war, sicherstellen, dass er nicht als Weight hängen bleibt
                if isinstance(current, str) and current:
                    state.parameter_weights.pop(current, None)

            combo.blockSignals(False)

    def _ensure_next_weight_row(self) -> None:
        if self.weighting_enabled_cb is None or not self.weighting_enabled_cb.isChecked():
            return

        available = self._available_weight_params()
        if not available:
            return

        if not self.weight_rows:
            self._create_weight_row()
            return

        last_combo, _last_slider, _last_lbl = self.weight_rows[-1]
        last_param = last_combo.currentData()
        if isinstance(last_param, str) and last_param:
            self._create_weight_row()

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
    
    def _update_weight_display(self, parameter: str, slider_value: int) -> None:
        """Update the weight display label when slider changes."""
        weight = slider_value / 100.0  # Convert 0-300 to 0.0-3.0
        state.parameter_weights[parameter] = weight
        percentage = slider_value
        self.parameter_weight_labels[parameter].setText(f"{weight:.2f} ({percentage}%)")
    
    def _reset_all_weights(self) -> None:
        """Reset all parameter weights to default (100%)."""
        for param in PARAMETER_OPTIONS:
            state.parameter_weights[param] = 1.0
            if param in self.parameter_weight_sliders:
                self.parameter_weight_sliders[param].setValue(100)  # Reset slider to 100%
            if param in self.parameter_weight_labels:
                self.parameter_weight_labels[param].setText("1.00 (100%)")
    
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

    def _on_normalize(self) -> None:
        """Normalize numeric data per participant by baseline task 0a (fallback 0b)."""
        if state.df is None:
            QMessageBox.information(self, "No data", "Load a TSV file first.")
            return

        try:
            state.normalized_df = normalize_by_participant_baseline(
                state.df,
                participant_col="Participant",
                task_col="TOI",
                baseline_primary="0a",
                baseline_fallback="0b",
            )
        except Exception as e:
            QMessageBox.critical(self, "Normalize failed", str(e))
            return

        # Normalisierung macht vorhandene Rank-Ergebnisse potentiell stale
        self._participant_rank_signature = None
        state.participant_rank_df = None

        QMessageBox.information(
            self,
            "Normalization complete",
            "Normalized data stored in state.normalized_df (baseline: 0a, fallback: 0b).",
        )

    def _on_calculate_metric_averages(self) -> None:
        """
        Berechnet Mittelwert und Standardabweichung je Participant und Task aus state.normalized_df.
        Zusätzlich: TCT (Mean/Std) je Participant und Task als zusätzliche Spalten (ohne Überschreiben).
        """
        if state.normalized_df is None:
            QMessageBox.information(
                self,
                "No normalized data",
                "Bitte zuerst 'Normalize' ausführen (state.normalized_df ist leer).",
            )
            return

        df = state.normalized_df

        # Spalten prüfen
        if "Participant" not in df.columns or "TOI" not in df.columns:
            QMessageBox.critical(self, "Missing columns", "normalized_df benötigt Spalten 'Participant' und 'TOI'.")
            return

        # Task-ID wie beim Task-Scan: Suffix nach letztem '_' (oder kompletter String)
        toi = df["TOI"].astype(str).str.strip()
        task_id = toi.str.rsplit("_", n=1).str[-1].str.strip()

        # Nur numerische Metriken aggregieren
        metric_cols = df.select_dtypes(include="number").columns.tolist()
        metric_cols = [c for c in metric_cols if c not in {"Participant", "TOI"}]

        if not metric_cols:
            QMessageBox.information(self, "No numeric metrics", "Keine numerischen Spalten zum Aggregieren gefunden.")
            return

        tmp = df.copy()
        tmp["_task_id"] = task_id

        # Aggregation: mean + std je Participant x Task (für alle numerischen Spalten)
        grouped = tmp.groupby(["Participant", "_task_id"], dropna=False)[metric_cols].agg(["mean", "std"])

        # Spaltennamen flach machen: "<metric>_mean", "<metric>_std"
        grouped.columns = [f"{metric}_{stat}" for (metric, stat) in grouped.columns.to_list()]

        # Index zurück in Spalten
        result = grouped.reset_index().rename(columns={"_task_id": "Task"})

        # --- Zusätzlich: TCT mean/std berechnen und als zusätzliche Spalten mergen ---
        tct_col: Optional[str] = None
        if "Duration" in df.columns:
            tct_col = "Duration"
        else:
            # heuristische Suche (case-insensitive)
            candidates = [
                "tct",
                "task completion time",
                "task_completion_time",
                "taskcompletiontime",
                "completion_time",
                "task time",
                "task_time",
            ]
            lower_map = {c.lower(): c for c in df.columns}
            for key in candidates:
                if key in lower_map:
                    tct_col = lower_map[key]
                    break

        tct_note = ""
        if tct_col is None:
            tct_note = "Keine TCT-Spalte gefunden (erwartet z.B. 'Duration')."
        elif not pd.api.types.is_numeric_dtype(df[tct_col]):
            tct_note = f"TCT-Spalte '{tct_col}' ist nicht numerisch."
        else:
            tct_tmp = tmp[["Participant", "_task_id", tct_col]].copy()
            tct_grouped = (
                tct_tmp.groupby(["Participant", "_task_id"], dropna=False)[tct_col]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"_task_id": "Task", "mean": "tct_mean", "std": "tct_std"})
            )

            # Nicht überschreiben: falls Spalten schon existieren, eindeutige Namen wählen
            if "tct_mean" in result.columns or "tct_std" in result.columns:
                tct_grouped = tct_grouped.rename(columns={"tct_mean": "tct_mean_tct", "tct_std": "tct_std_tct"})

            result = result.merge(tct_grouped, on=["Participant", "Task"], how="left")
            tct_note = f"TCT berechnet aus Spalte '{tct_col}'."

        state.metric_averages_df = result

        extra = f"\n{tct_note}" if tct_note else ""
        QMessageBox.information(
            self,
            "Metric averages calculated",
            f"Fertig. Ergebnis in state.metric_averages_df gespeichert.\n"
            f"Zeilen: {len(result)} | Spalten: {len(result.columns)}"
            f"{extra}",
        )

    def _current_rank_signature(self) -> dict:
        """Build a signature of inputs that affect participant_rank_df."""
        dataset_used = "normalized" if state.normalized_df is not None else "raw"

        selected_group_ids = [gid for gid, cb in self.result_group_checkboxes.items() if cb.isChecked()]
        selected_task_ids = [tid for tid, cb in self.result_task_checkboxes.items() if cb.isChecked()]

        excluded = []
        if self.deselect_enabled_cb.isChecked():
            excluded = sorted([name for name, action in self.deselect_param_checkboxes.items() if action.isChecked()])

        # Use weights from "Parameter Weighting" sliders (always active)
        weights = {k: float(v) for k, v in sorted(state.parameter_weights.items())}

        return {
            "dataset_used": dataset_used,
            "selected_group_ids": sorted(selected_group_ids),
            "selected_task_ids": sorted(selected_task_ids),
            "excluded_metrics": excluded,
            "weights_enabled": True,  # Weights are always enabled via Parameter Weighting sliders
            "weights": weights,
        }

    def _on_calculate_participant_rank(self) -> None:
        """
        Berechnet pro Participant ein Task-Ranking (hardest -> easiest) basierend auf kognitiver Last.

        Vorgehen:
        1) Metrikwerte je Participant×Task berechnen (mean über Zeilen).
        2) Pro Participant je Metrik Tasks ranken (Rank 1 = hardest; ties = average).
        3) Gewichtete Rank-Summe über Metriken bilden und Tasks nach kleinster Summe sortieren.
        """
        if state.df is None:
            QMessageBox.information(self, "No data", "Load a TSV file first.")
            return

        df = state.normalized_df if state.normalized_df is not None else state.df

        if "Participant" not in df.columns or "TOI" not in df.columns:
            QMessageBox.critical(self, "Missing columns", "DataFrame benötigt Spalten 'Participant' und 'TOI'.")
            return

        # --- Auswahl: Tasks ---
        selected_tasks = [tid for tid, cb in self.result_task_checkboxes.items() if cb.isChecked()]
        if not selected_tasks:
            selected_tasks = state.tasks_cache.copy()

        # --- Auswahl: Participants (über Gruppen) ---
        selected_group_ids = [gid for gid, cb in self.result_group_checkboxes.items() if cb.isChecked()]
        effective_groups = state.get_effective_participant_groups()

        selected_participants: list[str] = []
        if selected_group_ids:
            s = set()
            for gid in selected_group_ids:
                for p in effective_groups.get(gid, []):
                    s.add(p)
            selected_participants = sorted(s)
        else:
            selected_participants = state.participants_cache.copy()

        if not selected_participants:
            QMessageBox.information(self, "No participants", "Keine Teilnehmer ausgewählt/gefunden.")
            return
        if not selected_tasks:
            QMessageBox.information(self, "No tasks", "Keine Tasks ausgewählt/gefunden.")
            return

        # --- Deselect Parameters ---
        deselected: set[str] = set()
        if self.deselect_enabled_cb.isChecked():
            deselected = {name for name, action in self.deselect_param_checkboxes.items() if action.isChecked()}

        metrics = [m for m in PARAMETER_OPTIONS if m not in deselected]
        if not metrics:
            QMessageBox.information(self, "No metrics", "Alle Metriken sind abgewählt (Deselect Parameters).")
            return

        # --- Weights (from "Parameter Weighting" sliders) ---
        weights: Dict[str, float] = {}
        for m in metrics:
            # Use weight from state (set by Parameter Weighting sliders), default to 1.0
            weights[m] = float(state.parameter_weights.get(m, 1.0))

        # --- Task-ID aus TOI ableiten (Suffix nach letztem '_') ---
        tmp = df.copy()
        tmp["_task_id"] = tmp["TOI"].astype(str).str.strip().str.rsplit("_", n=1).str[-1].str.strip()

        # Filter auf Auswahl
        tmp = tmp[tmp["Participant"].isin(selected_participants)]
        tmp = tmp[tmp["_task_id"].isin(selected_tasks)]

        if tmp.empty:
            QMessageBox.information(
                self,
                "No data after filtering",
                "Nach Filterung (Participants/Tasks) sind keine Datenzeilen übrig.",
            )
            return

        # --- Metrik -> Spaltenmapping + Richtung (True: höher = härter) ---
        # Referenz: TCT↑, SD(TCT)↑, Pupil Diameter↑, Saccade Velocity↓, Peak Saccade Velocity↓, Saccade Amplitude↓.
        def _find_col_case_insensitive(candidates: list[str]) -> Optional[str]:
            lower_map = {c.lower(): c for c in tmp.columns}
            for cand in candidates:
                if cand.lower() in lower_map:
                    return lower_map[cand.lower()]
            return None

        metric_specs: Dict[str, Dict[str, object]] = {}

        # TCT
        tct_col = "Duration" if "Duration" in tmp.columns else _find_col_case_insensitive(
            ["tct", "task_completion_time", "task completion time", "taskcompletiontime", "completion_time", "task_time"]
        )
        if tct_col is not None:
            metric_specs["Task Completion Time (TCT)"] = {"col": tct_col, "agg": "mean", "harder_high": True}
            metric_specs["Standard Deviation of TCT"] = {"col": tct_col, "agg": "std", "harder_high": True}

        # Pupil
        pupil_col = _find_col_case_insensitive(["Average_pupil_diameter", "pupil", "pupil_diameter", "average pupil diameter"])
        if pupil_col is not None:
            metric_specs["Pupil Diameter"] = {"col": pupil_col, "agg": "mean", "harder_high": True}

        # Saccade velocity
        sacc_vel_col = _find_col_case_insensitive(["Average_velocity", "saccade_velocity", "average velocity"])
        if sacc_vel_col is not None:
            metric_specs["Saccade Velocity"] = {"col": sacc_vel_col, "agg": "mean", "harder_high": False}

        # Peak saccade velocity
        peak_vel_col = _find_col_case_insensitive(["Peak_velocity", "peak_velocity", "peak saccade velocity"])
        if peak_vel_col is not None:
            metric_specs["Peak Saccade Velocity"] = {"col": peak_vel_col, "agg": "mean", "harder_high": False}

        # Saccade amplitude
        sacc_amp_col = _find_col_case_insensitive(["Saccade_amplitude", "saccade_amplitude", "saccade amplitude"])
        if sacc_amp_col is not None:
            metric_specs["Saccade Amplitude"] = {"col": sacc_amp_col, "agg": "mean", "harder_high": False}

        # Nur Metriken verwenden, die wir auch wirklich berechnen können
        usable_metrics: list[str] = []
        missing_metrics: list[str] = []
        for m in metrics:
            if m in metric_specs:
                col = metric_specs[m]["col"]
                if isinstance(col, str) and col in tmp.columns and pd.api.types.is_numeric_dtype(tmp[col]):
                    usable_metrics.append(m)
                else:
                    missing_metrics.append(m)
            else:
                missing_metrics.append(m)

        if not usable_metrics:
            QMessageBox.critical(
                self,
                "No usable metrics",
                "Keine der ausgewählten Metriken konnte auf numerische Spalten gemappt werden.\n"
                f"Ausgewählt: {metrics}",
            )
            return

        # --- Schritt 1: Werte je Participant×Task berechnen ---
        base_index = tmp.groupby(["Participant", "_task_id"], dropna=False).size().reset_index()[["Participant", "_task_id"]]
        base_index = base_index.rename(columns={"_task_id": "Task"}).drop_duplicates()

        values_df = base_index.copy()

        for m in usable_metrics:
            spec = metric_specs[m]
            col = spec["col"]  # type: ignore[assignment]
            agg = spec["agg"]  # type: ignore[assignment]
            if not isinstance(col, str) or not isinstance(agg, str):
                continue

            g = tmp.groupby(["Participant", "_task_id"], dropna=False)[col]
            if agg == "mean":
                s = g.mean()
            elif agg == "std":
                s = g.std(ddof=1)
            else:
                raise ValueError(f"Unbekannte Aggregation: {agg}")

            s = s.reset_index().rename(columns={"_task_id": "Task", col: m})
            values_df = values_df.merge(s, on=["Participant", "Task"], how="left")

        # --- Schritt 2: Pro Participant je Metrik ranken (Rank 1 = hardest) ---
        ranks_df = values_df[["Participant", "Task"]].copy()

        for m in usable_metrics:
            harder_high = bool(metric_specs[m]["harder_high"])
            ascending = not harder_high
            ranks_df[m] = (
                values_df.groupby("Participant", dropna=False)[m]
                .rank(method="average", ascending=ascending, na_option="keep")
            )

        # --- Schritt 3: Gewichtete Rank-Summe + finaler Task-Rank ---
        weighted_sum = pd.Series(0.0, index=ranks_df.index)
        weight_sum = pd.Series(0.0, index=ranks_df.index)

        for m in usable_metrics:
            w = float(weights.get(m, 1.0))
            r = ranks_df[m]
            mask = r.notna()
            weighted_sum = weighted_sum + (r.fillna(0.0) * w)
            weight_sum = weight_sum + (mask.astype(float) * w)

        final_score = weighted_sum.where(weight_sum > 0, other=pd.NA)

        out = ranks_df[["Participant", "Task"]].copy()
        out["rank_sum"] = final_score

        out["final_rank"] = out.groupby("Participant", dropna=False)["rank_sum"].rank(
            method="average", ascending=True, na_option="keep"
        )

        for m in usable_metrics:
            out[f"{m}__rank"] = ranks_df[m]

        out = out.sort_values(["Participant", "final_rank", "Task"], ascending=[True, True, True]).reset_index(drop=True)

        state.participant_rank_df = out
        self._participant_rank_signature = self._current_rank_signature()

        note = ""
        if missing_metrics:
            note = "\nNicht gemappte/fehlende Metriken: " + ", ".join(missing_metrics)

        QMessageBox.information(
            self,
            "Participant rank calculated",
            "Fertig. Ergebnis in state.participant_rank_df gespeichert.\n"
            f"Participants: {out['Participant'].nunique()} | Tasks: {out['Task'].nunique()} | Rows: {len(out)}\n"
            f"Verwendete Metriken: {', '.join(usable_metrics)}"
            f"{note}",
        )

    def _show_rank_results_popup(
        self,
        title: str,
        header: str,
        sections: list[tuple[str, pd.DataFrame]],
        warnings: list[str],
    ) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setModal(True)
        dlg.setMinimumSize(900, 700)

        layout = QVBoxLayout(dlg)

        text = QTextEdit()
        text.setReadOnly(True)

        parts: list[str] = []
        parts.append(header.strip())
        parts.append("")

        for section_title, df in sections:
            parts.append(section_title)
            parts.append("-" * len(section_title))
            if df.empty:
                parts.append("(keine Daten)")
            else:
                # df ist bereits auf Rank/Task/Score reduziert
                parts.append(df.to_string(index=False))
            parts.append("")

        if warnings:
            parts.append("Warnings")
            parts.append("--------")
            for w in warnings:
                parts.append(f"- {w}")
            parts.append("")

        text.setPlainText("\n".join(parts))
        layout.addWidget(text)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        dlg.exec()

    def _on_show_results(self) -> None:
        """Handle Show Results button click."""
        if state.df is None:
            QMessageBox.information(self, "No data", "Load a TSV file first.")
            return

        selected_groups = [gid for gid, cb in self.result_group_checkboxes.items() if cb.isChecked()]
        selected_tasks = [tid for tid, cb in self.result_task_checkboxes.items() if cb.isChecked()]
        domains = [name for name, cb in self.result_domain_vars.items() if cb.isChecked()]
        mode = self.mode_combo.currentText()
        
        if not selected_groups:
            QMessageBox.warning(self, "No Groups Selected", "Please select at least one group.")
            return
        
        if not selected_tasks:
            QMessageBox.warning(self, "No Tasks Selected", "Please select at least one task.")
            return
        
        if not domains:
            QMessageBox.warning(self, "No Domains Selected", "Please select at least one result domain.")
            return
        
        deselected = []
        if self.deselect_enabled_cb.isChecked():
            deselected = [name for name, action in self.deselect_param_checkboxes.items() if action.isChecked()]
        
        # Get active parameters (exclude deselected)
        active_parameters = [p for p in PARAMETER_OPTIONS if p not in deselected]
        
        if not active_parameters:
            QMessageBox.warning(self, "No Parameters", "All parameters are deselected. Please enable at least one parameter.")
            return
        
        # Get parameter weights
        parameter_weights = state.parameter_weights.copy()
        
        # Check if statistics tab should be shown
        show_statistics = self.result_domain_vars.get("Statistics", QCheckBox()).isChecked()
        
        # Show loading dialog
        loading = LoadingDialog(self, "Processing data...")
        loading.show()
        QApplication.processEvents()  # Update UI
        
        try:
            aggregated_data = aggregate_by_groups(
                state.df,
                selected_groups,
                selected_tasks,
                deselected,
                mode,
                parameter_weights
            )
            
            loading.close()
            QApplication.processEvents()
            
            if not aggregated_data:
                QMessageBox.warning(self, "No Data", "No data available for the selected groups and tasks.")
                return
            
            # Create and show results window
            results_window = ResultsWindow(
                aggregated_data,
                selected_groups,
                selected_tasks,
                active_parameters,
                mode,
                domains,
                show_statistics,
                self
            )
            results_window.show()
        except Exception as e:
            loading.close()
            QMessageBox.critical(self, "Analysis Error", f"Failed to generate results: {str(e)}")
    
    def _on_print_exec_summary(self) -> None:
        """Handle Print Executive Summary button click."""
        if state.df is None:
            QMessageBox.information(self, "No data", "Load a TSV file first.")
            return
        
        # Check if MikTeX is installed - REQUIRED for PDF generation
        pdflatex_path = find_pdflatex()
        if not pdflatex_path:
            QMessageBox.critical(
                self,
                "MikTeX Not Found",
                "MikTeX is required to generate the executive summary PDF.\n\n"
                "Please install MikTeX from:\n"
                "https://miktex.org/download\n\n"
                "After installation:\n"
                "1. Restart this application\n"
                "2. MikTeX will automatically install required packages on first use\n\n"
                "For detailed installation instructions, see:\n"
                "INSTALL_MIKTEX.md in the project directory."
            )
            return
        
        # Get ALL groups (regardless of checkbox selection)
        effective_groups = state.get_effective_participant_groups()
        all_groups = list(effective_groups.keys()) if effective_groups else []
        
        # Get ALL tasks (regardless of checkbox selection)
        # Include all tasks including baseline tasks 0a/0b for complete summary
        all_tasks = state.tasks_cache.copy() if state.tasks_cache else []
        
        if not all_groups:
            QMessageBox.warning(self, "No Groups Available", "No participant groups found in the data.")
            return
        
        if not all_tasks:
            QMessageBox.warning(self, "No Tasks Available", "No tasks found in the data.")
            return
        
        mode = self.mode_combo.currentText()
        
        deselected = []
        if self.deselect_enabled_cb.isChecked():
            deselected = [name for name, action in self.deselect_param_checkboxes.items() if action.isChecked()]
        
        active_parameters = [p for p in PARAMETER_OPTIONS if p not in deselected]
        
        if not active_parameters:
            QMessageBox.warning(self, "No Parameters", "All parameters are deselected. Please enable at least one parameter.")
            return
        
        # Get parameter weights
        parameter_weights = state.parameter_weights.copy()
        
        # Show loading dialog
        loading = LoadingDialog(self, "Generating LaTeX PDF...")
        loading.show()
        QApplication.processEvents()
        
        try:
            # Aggregate data using ALL groups and tasks
            aggregated_data = aggregate_by_groups(
                state.df,
                all_groups,
                all_tasks,
                deselected,
                mode,
                parameter_weights
            )
            
            if not aggregated_data:
                loading.close()
                QMessageBox.warning(self, "No Data", "No data available for the groups and tasks.")
                return
            
            # Generate LaTeX PDF directly (MikTeX is already verified above)
            # Get TSV file path if available
            df_path = getattr(state, 'loaded_file_path', None)
            
            # Generate LaTeX PDF directly
            pdf_path = generate_latex_summary(
                aggregated_data,
                all_groups,
                all_tasks,
                active_parameters,
                mode,
                parameter_weights,
                df_path
            )
            
            loading.close()
            QApplication.processEvents()
            
            # Ask user if they want to open the PDF
            reply = QMessageBox.question(
                self,
                "PDF Generated Successfully",
                f"LaTeX PDF generated at:\n{pdf_path}\n\nWould you like to open it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                import os
                import platform
                if platform.system() == 'Windows':
                    os.startfile(pdf_path)
                elif platform.system() == 'Darwin':  # macOS
                    os.system(f'open "{pdf_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{pdf_path}"')
            
        except Exception as e:
            loading.close()
            QMessageBox.critical(self, "Summary Error", f"Failed to generate executive summary: {str(e)}")
    
    def _export_summary(self, export_func, summary_text: str, statistics_text: str, extension: str) -> None:
        """Helper method to export summary."""
        from pathlib import Path
        from datetime import datetime
        
        # Create date-time stamped output folder in ETT repository root
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Ensure we're using the ETT repository root directory
        ett_root = Path(__file__).parent
        output_dir = ett_root / "output" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        default_path = str(output_dir / f"executive_summary{extension}")
        
        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Executive Summary",
            default_path,
            f"{extension.upper().lstrip('.')} files (*{extension});;All files (*.*)"
        )
        
        if not filename:
            return
        
        try:
            export_func(summary_text, statistics_text, filename)
            QMessageBox.information(self, "Export Success", f"Executive summary exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")
    
    def _export_latex_summary(
        self,
        aggregated_data: Dict,
        selected_groups: List[str],
        selected_tasks: List[str],
        active_parameters: List[str],
        mode: str,
        parameter_weights: Dict[str, float]
    ) -> None:
        """Export executive summary using LaTeX/MikTeX."""
        # Show loading dialog
        loading = LoadingDialog(self, "Generating PDF...")
        loading.show()
        QApplication.processEvents()
        
        try:
            # Get TSV file path if available
            df_path = getattr(state, 'tsv_file_path', None)
            
            # Generate LaTeX PDF
            pdf_path = generate_latex_summary(
                aggregated_data,
                selected_groups,
                selected_tasks,
                active_parameters,
                mode,
                parameter_weights,
                df_path
            )
            
            loading.close()
            QApplication.processEvents()
            
            # Ask user if they want to open the PDF
            reply = QMessageBox.question(
                self,
                "PDF Generated Successfully",
                f"LaTeX PDF generated at:\n{pdf_path}\n\nWould you like to open it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                import os
                import platform
                if platform.system() == 'Windows':
                    os.startfile(pdf_path)
                elif platform.system() == 'Darwin':  # macOS
                    os.system(f'open "{pdf_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{pdf_path}"')
        
        except Exception as e:
            loading.close()
            QMessageBox.critical(
                self,
                "LaTeX Export Error",
                f"Failed to generate LaTeX PDF:\n{str(e)}\n\n"
                "Please ensure MikTeX is installed and pdflatex is in your PATH.\n"
                "Download from: https://miktex.org/download"
            )