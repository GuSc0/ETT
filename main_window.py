"""
Main window for the Eye Tracking Tool using PyQt6.
"""
from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QLineEdit, QComboBox, QCheckBox, QGroupBox, QGridLayout,
    QFrame, QMessageBox, QFileDialog, QMenu, QMenuBar, QSplitter, QSlider,
    QDialog, QTextEdit
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction

from state import state
from models import EXPECTED_COLUMNS, PARAMETER_OPTIONS
from data_processor import (
    validate_tsv_format,
    extract_participants,
    extract_tasks_from_toi,
    normalize_by_participant_baseline,
)
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

        # Weighting parameters
        self.weighting_enabled_cb: Optional[QCheckBox] = None
        self.weight_rows: list[tuple[QComboBox, QSlider, QLabel]] = []
        self.parameter_weights: Dict[str, float] = {}

        # Snapshot of the last rank computation inputs (to detect stale results)
        self._participant_rank_signature: Optional[dict] = None

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

        # Weighting parameters
        weighting_group = QGroupBox("Weighting Parameters")
        weighting_outer = QVBoxLayout(weighting_group)

        # Toggle row: checkbox + label rechts
        weighting_toggle_row = QHBoxLayout()
        self.weighting_enabled_cb = QCheckBox()
        self.weighting_enabled_cb.setChecked(False)
        self.weighting_enabled_cb.setEnabled(False)
        self.weighting_enabled_cb.stateChanged.connect(self._on_weighting_toggled)
        weighting_toggle_row.addWidget(self.weighting_enabled_cb)
        weighting_toggle_row.addWidget(QLabel("Enable custom parameter weights"))
        weighting_toggle_row.addStretch()
        weighting_outer.addLayout(weighting_toggle_row)

        # Container für dynamische Zeilen
        self.weighting_rows_container = QWidget()
        self.weighting_rows_layout = QVBoxLayout(self.weighting_rows_container)
        self.weighting_rows_layout.setContentsMargins(0, 0, 0, 0)
        self.weighting_rows_layout.setSpacing(6)
        weighting_outer.addWidget(self.weighting_rows_container)

        bottom_layout.addWidget(weighting_group)

        main_layout.addLayout(bottom_layout)

        # Action buttons
        action_layout = QHBoxLayout()

        self.show_results_btn = QPushButton("Show Results")
        self.show_results_btn.setFixedWidth(150)
        self.show_results_btn.setEnabled(False)
        self.show_results_btn.clicked.connect(self._on_show_results)
        action_layout.addWidget(self.show_results_btn)

        self.normalize_btn = QPushButton("Normalize")
        self.normalize_btn.setFixedWidth(150)
        self.normalize_btn.setEnabled(False)
        self.normalize_btn.clicked.connect(self._on_normalize)
        action_layout.addWidget(self.normalize_btn)

        self.calc_metric_averages_btn = QPushButton("Calculate metric averages")
        self.calc_metric_averages_btn.setFixedWidth(200)
        self.calc_metric_averages_btn.setEnabled(False)
        self.calc_metric_averages_btn.clicked.connect(self._on_calculate_metric_averages)
        action_layout.addWidget(self.calc_metric_averages_btn)

        self.exec_summary_btn = QPushButton("Print Executive Summary")
        self.exec_summary_btn.setFixedWidth(180)
        self.exec_summary_btn.setEnabled(False)
        self.exec_summary_btn.clicked.connect(self._on_print_exec_summary)
        action_layout.addWidget(self.exec_summary_btn)

        self.calc_participant_rank_btn = QPushButton("Calculate Participant Rank")
        self.calc_participant_rank_btn.setFixedWidth(220)
        self.calc_participant_rank_btn.setEnabled(False)
        self.calc_participant_rank_btn.clicked.connect(self._on_calculate_participant_rank)
        action_layout.addWidget(self.calc_participant_rank_btn)

        action_layout.addStretch()
        main_layout.addLayout(action_layout)

        main_layout.addStretch()

        # Initialize deselect menu
        self._build_deselect_menu()

        # Initialize weighting UI (disabled by default)
        self._reset_weighting_ui()

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
        if hasattr(self, "calc_metric_averages_btn"):
            self.calc_metric_averages_btn.setEnabled(False)

        # Reset weighting UI/weights (parameters list may change later)
        if self.weighting_enabled_cb is not None:
            self.weighting_enabled_cb.setChecked(False)
        self._reset_weighting_ui()

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

        if self.weighting_enabled_cb is not None:
            self.weighting_enabled_cb.setEnabled(True)

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
        self.normalize_btn.setEnabled(enabled)
        self.calc_metric_averages_btn.setEnabled(enabled)
        self.exec_summary_btn.setEnabled(enabled)
        self.calc_participant_rank_btn.setEnabled(enabled)

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

    def _reset_weighting_ui(self) -> None:
        """Clear all weighting rows and internal weights; keep toggle state."""
        self.parameter_weights.clear()
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
        self.parameter_weights[param] = float(weight)

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
                self.parameter_weights[current] = float(weight)
            else:
                combo.setCurrentIndex(0)
                slider.setEnabled(False)
                value_lbl.setEnabled(False)

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

        if hasattr(self, "calc_metric_averages_btn"):
            self.calc_metric_averages_btn.setEnabled(True)

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

        weights_enabled = bool(self.weighting_enabled_cb is not None and self.weighting_enabled_cb.isChecked())
        weights = {}
        if weights_enabled:
            # nur stabile, relevante Gewichte (sortiert)
            weights = {k: float(v) for k, v in sorted(self.parameter_weights.items())}

        return {
            "dataset_used": dataset_used,
            "selected_group_ids": sorted(selected_group_ids),
            "selected_task_ids": sorted(selected_task_ids),
            "excluded_metrics": excluded,
            "weights_enabled": weights_enabled,
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

        # --- Weights ---
        weights: Dict[str, float] = {m: 1.0 for m in metrics}
        if self.weighting_enabled_cb is not None and self.weighting_enabled_cb.isChecked():
            for m in metrics:
                if m in self.parameter_weights:
                    weights[m] = float(self.parameter_weights[m])

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

        deselected = []
        if self.deselect_enabled_cb.isChecked():
            deselected = [name for name, action in self.deselect_param_checkboxes.items() if action.isChecked()]

        # Parameter weights (nur anzeigen, wenn aktiviert)
        weights_text = "(disabled)"
        if self.weighting_enabled_cb is not None and self.weighting_enabled_cb.isChecked():
            if not self.parameter_weights:
                weights_text = "(none)"
            else:
                parts = [f"{k} = {self._format_weight(v)}" for k, v in sorted(self.parameter_weights.items())]
                weights_text = ", ".join(parts)

        # --- Rank domain: show computed participant rank results ---
        if "Rank" in domains:
            if state.participant_rank_df is None or state.participant_rank_df.empty:
                QMessageBox.information(
                    self,
                    "Rank Results not available",
                    "Keine Rank-Ergebnisse vorhanden.\nBitte zuerst 'Calculate Participant Rank' ausführen.",
                )
                return

            current_sig = self._current_rank_signature()
            if self._participant_rank_signature != current_sig:
                QMessageBox.information(
                    self,
                    "Rank Results stale",
                    "Die Rank-Ergebnisse passen nicht zu den aktuellen UI-Auswahlen (Groups/Tasks/Deselect/Weights oder Datenbasis).\n"
                    "Bitte 'Calculate Participant Rank' erneut ausführen.",
                )
                return

            rank_df = state.participant_rank_df.copy()

            # Scope: Tasks
            task_scope = selected_tasks if selected_tasks else state.tasks_cache.copy()
            # Scope: Participants via groups
            effective_groups = state.get_effective_participant_groups()
            effective_names = state.get_effective_group_names()

            group_ids = selected_groups if selected_groups else list(effective_groups.keys())
            if not group_ids:
                group_ids = ["ALL"]

            warnings: list[str] = []
            sections: list[tuple[str, pd.DataFrame]] = []

            dataset_used = "normalized" if state.normalized_df is not None else "raw"
            excluded_metrics = sorted(deselected) if deselected else []
            weights_list = []
            if self.weighting_enabled_cb is not None and self.weighting_enabled_cb.isChecked():
                weights_list = [f"{k}={self._format_weight(v)}" for k, v in sorted(self.parameter_weights.items())]

            # Helper: format a table
            def _table_from_scores(df_scores: pd.DataFrame) -> pd.DataFrame:
                # df_scores: columns Task, rank_sum
                t = df_scores.copy()
                t = t.sort_values(["rank_sum", "Task"], ascending=[True, True]).reset_index(drop=True)
                t.insert(0, "Rank", range(1, len(t) + 1))
                t = t.rename(columns={"rank_sum": "Score"})
                t = t[["Rank", "Task", "Score"]]
                return t

            # Build participant list per group
            group_to_participants: dict[str, list[str]] = {}
            for gid in group_ids:
                members = effective_groups.get(gid, [])
                if not members:
                    warnings.append(f"Gruppe '{effective_names.get(gid, gid)}' hat 0 Mitglieder (übersprungen).")
                    continue
                group_to_participants[gid] = members

            # If no groups had members, fallback to all participants
            if not group_to_participants:
                all_p = state.participants_cache.copy()
                if not all_p:
                    QMessageBox.information(self, "No participants", "Keine Teilnehmer gefunden.")
                    return
                group_to_participants = {"ALL": all_p}

            # Filter rank_df to task scope early
            rank_df = rank_df[rank_df["Task"].isin(task_scope)]

            # Determine which participants have which tasks (for warnings)
            # We'll warn if a participant is missing any selected task in the rank_df.
            for gid, members in group_to_participants.items():
                for p in members:
                    p_tasks = set(rank_df.loc[rank_df["Participant"] == p, "Task"].astype(str).tolist())
                    for t in task_scope:
                        if t not in p_tasks:
                            warnings.append(f"{p} hatte keine Zeilen für Task {t}")

            # Mode handling
            def _add_group_mean_sections() -> None:
                for gid, members in group_to_participants.items():
                    gname = effective_names.get(gid, gid)
                    sub = rank_df[rank_df["Participant"].isin(members)]
                    if sub.empty:
                        warnings.append(f"Gruppe '{gname}' hat nach Filterung keine Daten (übersprungen).")
                        continue
                    # mean rank_sum per task
                    g_scores = (
                        sub.groupby("Task", dropna=False)["rank_sum"]
                        .mean()
                        .reset_index()
                    )
                    g_table = _table_from_scores(g_scores)
                    sections.append((f"Group mean: {gname}", g_table))

            def _add_individual_sections() -> None:
                # participants inside selected groups (or all fallback)
                seen: set[str] = set()
                for _gid, members in group_to_participants.items():
                    for p in members:
                        if p in seen:
                            continue
                        seen.add(p)
                        sub = rank_df[rank_df["Participant"] == p][["Task", "rank_sum"]].copy()
                        if sub.empty:
                            warnings.append(f"{p} hat nach Filterung keine Daten (übersprungen).")
                            continue
                        p_table = _table_from_scores(sub)
                        sections.append((f"Participant: {p}", p_table))

            if mode == "Only group mean":
                _add_group_mean_sections()
            elif mode == "Each participant for selected groups":
                _add_individual_sections()
            elif mode == "Group mean and individual participants":
                _add_group_mean_sections()
                _add_individual_sections()
            else:
                warnings.append(f"Unbekannter Mode '{mode}' (zeige Individual-Rankings).")
                _add_individual_sections()

            # Header paragraph
            # #participants/#tasks should reflect the popup scope (after group/task selection)
            participants_in_scope: set[str] = set()
            for members in group_to_participants.values():
                participants_in_scope.update(members)
            participants_in_scope = {p for p in participants_in_scope if p in set(rank_df["Participant"].unique())}

            tasks_in_scope = sorted(set(task_scope) & set(rank_df["Task"].unique()))

            header = (
                f"Dataset: {dataset_used}; "
                f"Participants: {len(participants_in_scope)}; "
                f"Tasks: {len(tasks_in_scope)}; "
                f"Mode: {mode}; "
                f"Excluded metrics: {', '.join(excluded_metrics) if excluded_metrics else '(none)'}; "
                f"Custom weights: {', '.join(weights_list) if weights_list else '(none)'}"
            )

            self._show_rank_results_popup(
                title="Rank Results",
                header=header,
                sections=sections,
                warnings=warnings,
            )
            return

        # Fallback: existing placeholder message
        msg = (
            f"Mode: {mode}\n"
            f"Groups selected: {len(selected_groups)}\n"
            f"Tasks selected: {len(selected_tasks)}\n"
            f"Result domains: {', '.join(domains) if domains else '(none)'}\n"
            f"Deselected parameters: {', '.join(deselected) if deselected else '(none)'}\n"
            f"Parameter weights: {weights_text}"
        )
        QMessageBox.information(self, "Show Results (placeholder)", msg)

    def _on_print_exec_summary(self) -> None:
        """Handle Print Executive Summary button click."""
        if state.df is None:
            QMessageBox.information(self, "No data", "Load a TSV file first.")
            return
        QMessageBox.information(
            self,
            "Executive Summary (placeholder)",
            "This will generate the executive summary later.",
        )
