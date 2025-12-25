"""
Dialog windows and custom widgets using PyQt6.
"""
from __future__ import annotations

from typing import Optional, List, Callable, Set

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QCheckBox, QLineEdit, QComboBox,
    QFrame, QScrollArea, QGroupBox, QGridLayout, QAbstractItemView,
    QMessageBox
)
from PyQt6.QtCore import Qt

from state import state


class MultiSelectDialog(QDialog):
    """
    Modal dialog for multi-selection:
    - Select all checkbox
    - Multi-select list
    - OK / Cancel
    """
    
    def __init__(
        self,
        parent: QWidget,
        title: str,
        items: List[str],
        selected: Optional[List[str]] = None,
        gray_hint_items: Optional[List[str]] = None,
        display_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(parent)
        self.items = items
        self._selected_set = set(selected or [])
        self._gray_hint = set(gray_hint_items or [])
        self._display_func = display_func or (lambda x: x)
        self.result: Optional[List[str]] = None
        
        self.setWindowTitle(title)
        self.setMinimumSize(420, 460)
        self.setModal(True)
        
        layout = QVBoxLayout(self)
        
        # Select all checkbox
        self.select_all_cb = QCheckBox("Select all")
        self.select_all_cb.stateChanged.connect(self._toggle_select_all)
        layout.addWidget(self.select_all_cb)
        
        # List widget
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        
        for item in self.items:
            list_item = QListWidgetItem(self._display_func(item))
            list_item.setData(Qt.ItemDataRole.UserRole, item)
            if item in self._gray_hint:
                list_item.setForeground(Qt.GlobalColor.gray)
            if item in self._selected_set:
                list_item.setSelected(True)
            self.list_widget.addItem(list_item)
        
        layout.addWidget(self.list_widget)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self._on_ok)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel)
        
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
    
    def _toggle_select_all(self, state: int) -> None:
        for i in range(self.list_widget.count()):
            self.list_widget.item(i).setSelected(state == Qt.CheckState.Checked.value)
    
    def _on_ok(self) -> None:
        self.result = [
            self.list_widget.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.list_widget.count())
            if self.list_widget.item(i).isSelected()
        ]
        self.accept()
    
    def _on_cancel(self) -> None:
        self.result = None
        self.reject()


class MultiSelectPicker(QWidget):
    """
    A button that opens a multi-select popup dialog.
    """
    
    def __init__(
        self,
        parent: QWidget,
        options: List[str],
        get_global_selected: Callable[[], Set[str]],
        on_change: Optional[Callable[[], None]] = None,
        placeholder: str = "Select items...",
        popup_title: str = "Select items",
        display_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        super().__init__(parent)
        self._options = options
        self._get_global_selected = get_global_selected
        self._on_change = on_change
        self._placeholder = placeholder
        self._popup_title = popup_title
        self._display_func = display_func or (lambda x: x)
        
        self._selected: Set[str] = set()
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self._btn = QPushButton(placeholder)
        self._btn.clicked.connect(self._open_popup)
        layout.addWidget(self._btn)
        
        self._refresh_summary()
    
    def get_selected(self) -> List[str]:
        return [o for o in self._options if o in self._selected]
    
    def set_selected(self, selected: List[str]) -> None:
        self._selected = set(selected)
        self._refresh_summary()
    
    def refresh_ui(self) -> None:
        self._refresh_summary()
    
    def _refresh_summary(self) -> None:
        selected = self.get_selected()
        if not selected:
            self._btn.setText(self._placeholder)
        elif len(selected) <= 3:
            self._btn.setText(", ".join(self._display_func(s) for s in selected))
        else:
            self._btn.setText(f"{len(selected)} selected")
    
    def _open_popup(self) -> None:
        global_selected = self._get_global_selected()
        gray_hint = [o for o in self._options if o in global_selected and o not in self._selected]
        
        dialog = MultiSelectDialog(
            self,
            title=self._popup_title,
            items=self._options,
            selected=list(self._selected),
            gray_hint_items=gray_hint,
            display_func=self._display_func,
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result is not None:
            self._selected = set(dialog.result)
            self._refresh_summary()
            if self._on_change is not None:
                self._on_change()


class GroupParticipantsDialog(QDialog):
    """Dialog for grouping participants."""
    
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Group Participants")
        self.setMinimumSize(900, 600)
        self.setModal(True)
        
        self.max_groups = 15
        
        main_layout = QVBoxLayout(self)
        
        # Scrollable area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.container_layout = QVBoxLayout(container)
        
        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("Group name"), stretch=1)
        header.addWidget(QLabel("Participants"), stretch=2)
        self.container_layout.addLayout(header)
        
        # Group rows
        self.group_name_edits: List[QLineEdit] = []
        self.group_members: List[List[str]] = []
        self.select_btns: List[QPushButton] = []
        self.row_widgets: List[QWidget] = []
        
        existing_ids = sorted(state.participant_groups.keys())
        
        for i in range(self.max_groups):
            gid = existing_ids[i] if i < len(existing_ids) else f"G{i+1}"
            
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 4, 0, 4)
            
            name_edit = QLineEdit(state.group_names.get(gid, f"Group {i+1}"))
            name_edit.setFixedWidth(200)
            row_layout.addWidget(name_edit)
            
            members = state.participant_groups.get(gid, []).copy()
            
            btn = QPushButton("Select participants...")
            btn.clicked.connect(lambda checked, idx=i: self._select_participants(idx))
            row_layout.addWidget(btn, stretch=1)
            
            self.group_name_edits.append(name_edit)
            self.group_members.append(members)
            self.select_btns.append(btn)
            self.row_widgets.append(row_widget)
            
            self.container_layout.addWidget(row_widget)
            self._update_btn_text(i)
        
        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save & Apply")
        save_btn.clicked.connect(self._save_apply)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)
        
        self._update_row_visibility()
    
    def _already_selected_elsewhere(self, current_row: int) -> List[str]:
        s: Set[str] = set()
        for r, members in enumerate(self.group_members):
            if r != current_row:
                s.update(members)
        return sorted(s)
    
    def _update_btn_text(self, row_idx: int) -> None:
        sel = self.group_members[row_idx]
        if not sel:
            self.select_btns[row_idx].setText("Select participants...")
        elif len(sel) <= 4:
            self.select_btns[row_idx].setText(", ".join(sel))
        else:
            self.select_btns[row_idx].setText(f"{len(sel)} selected")
    
    def _update_row_visibility(self) -> None:
        for i in range(self.max_groups):
            if i == 0:
                self.row_widgets[i].show()
            elif self.group_members[i - 1]:
                self.row_widgets[i].show()
            else:
                self.row_widgets[i].hide()
    
    def _select_participants(self, row_idx: int) -> None:
        dialog = MultiSelectDialog(
            self,
            title=f"Select participants for {self.group_name_edits[row_idx].text() or f'Group {row_idx+1}'}",
            items=state.participants_cache,
            selected=self.group_members[row_idx],
            gray_hint_items=self._already_selected_elsewhere(row_idx),
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result is not None:
            self.group_members[row_idx] = dialog.result
            self._update_btn_text(row_idx)
            self._update_row_visibility()
    
    def _save_apply(self) -> None:
        new_groups: dict = {}
        new_names: dict = {}
        
        for i in range(self.max_groups):
            gid = f"G{i+1}"
            name = self.group_name_edits[i].text().strip() or f"Group {i+1}"
            members = self.group_members[i]
            
            if not members and i > 0:
                continue
            
            new_groups[gid] = members
            new_names[gid] = name
        
        if all(len(v) == 0 for v in new_groups.values()):
            state.participant_groups.clear()
            state.group_names.clear()
        else:
            state.participant_groups.clear()
            state.participant_groups.update(new_groups)
            state.group_names.clear()
            state.group_names.update(new_names)
        
        self.accept()


class GroupTasksDialog(QDialog):
    """Dialog for grouping tasks with task labeling."""
    
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setWindowTitle("Group Tasks")
        self.setMinimumSize(980, 640)
        self.setModal(True)
        
        self.max_groups = 15
        
        main_layout = QVBoxLayout(self)
        
        # Task naming section
        naming_group = QGroupBox("Task labels")
        naming_layout = QHBoxLayout(naming_group)
        
        naming_layout.addWidget(QLabel("Task:"))
        self.task_combo = QComboBox()
        self.task_combo.addItems([state.format_task(t) for t in state.tasks_cache])
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        naming_layout.addWidget(self.task_combo)
        
        naming_layout.addWidget(QLabel("Label:"))
        self.label_edit = QLineEdit()
        self.label_edit.returnPressed.connect(self._apply_label)
        self.label_edit.textChanged.connect(self._update_preview)
        naming_layout.addWidget(self.label_edit)
        
        self.preview_label = QLabel("Preview: ")
        naming_layout.addWidget(self.preview_label)
        
        naming_layout.addWidget(QLabel("Press Enter to apply."))
        naming_layout.addStretch()
        
        main_layout.addWidget(naming_group)
        
        # Scrollable groups area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.container_layout = QVBoxLayout(container)
        
        # Header
        header = QHBoxLayout()
        header.addWidget(QLabel("Group name"), stretch=1)
        header.addWidget(QLabel("Tasks"), stretch=2)
        self.container_layout.addLayout(header)
        
        # Group rows
        self.group_name_edits: List[QLineEdit] = []
        self.group_items: List[List[str]] = []
        self.select_btns: List[QPushButton] = []
        self.row_widgets: List[QWidget] = []
        
        existing_ids = sorted(state.task_groups.keys())
        
        for i in range(self.max_groups):
            gid = existing_ids[i] if i < len(existing_ids) else f"T{i+1}"
            
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 4, 0, 4)
            
            name_edit = QLineEdit(state.task_group_names.get(gid, f"Group {i+1}"))
            name_edit.setFixedWidth(200)
            row_layout.addWidget(name_edit)
            
            items = state.task_groups.get(gid, []).copy()
            
            btn = QPushButton("Select tasks...")
            btn.clicked.connect(lambda checked, idx=i: self._select_tasks(idx))
            row_layout.addWidget(btn, stretch=1)
            
            self.group_name_edits.append(name_edit)
            self.group_items.append(items)
            self.select_btns.append(btn)
            self.row_widgets.append(row_widget)
            
            self.container_layout.addWidget(row_widget)
            self._update_btn_text(i)
        
        scroll.setWidget(container)
        main_layout.addWidget(scroll)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        save_btn = QPushButton("Save & Apply")
        save_btn.clicked.connect(self._save_apply)
        
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        main_layout.addLayout(btn_layout)
        
        self._update_row_visibility()
        self._on_task_changed()
    
    def _on_task_changed(self) -> None:
        idx = self.task_combo.currentIndex()
        if idx < 0 or idx >= len(state.tasks_cache):
            return
        task_id = state.tasks_cache[idx]
        self.label_edit.setText(state.task_labels.get(task_id, ""))
        self._update_preview()
    
    def _update_preview(self) -> None:
        idx = self.task_combo.currentIndex()
        if idx < 0 or idx >= len(state.tasks_cache):
            return
        task_id = state.tasks_cache[idx]
        label = self.label_edit.text().strip()
        preview = f"{task_id} {label}".strip() if label else task_id
        self.preview_label.setText(f"Preview: {preview}")
    
    def _apply_label(self) -> None:
        idx = self.task_combo.currentIndex()
        if idx < 0 or idx >= len(state.tasks_cache):
            return
        task_id = state.tasks_cache[idx]
        label = self.label_edit.text().strip()
        
        if label:
            state.task_labels[task_id] = label
        else:
            state.task_labels.pop(task_id, None)
        
        # Refresh combobox
        current_idx = self.task_combo.currentIndex()
        self.task_combo.clear()
        self.task_combo.addItems([state.format_task(t) for t in state.tasks_cache])
        self.task_combo.setCurrentIndex(current_idx)
        
        # Refresh button texts
        for i in range(self.max_groups):
            self._update_btn_text(i)
        
        self._update_preview()
    
    def _already_selected_elsewhere(self, current_row: int) -> List[str]:
        s: Set[str] = set()
        for r, items in enumerate(self.group_items):
            if r != current_row:
                s.update(items)
        return sorted(s)
    
    def _update_btn_text(self, row_idx: int) -> None:
        sel = self.group_items[row_idx]
        if not sel:
            self.select_btns[row_idx].setText("Select tasks...")
        elif len(sel) <= 6:
            self.select_btns[row_idx].setText(", ".join(state.format_task(t) for t in sel))
        else:
            self.select_btns[row_idx].setText(f"{len(sel)} selected")
    
    def _update_row_visibility(self) -> None:
        for i in range(self.max_groups):
            if i == 0:
                self.row_widgets[i].show()
            elif self.group_items[i - 1]:
                self.row_widgets[i].show()
            else:
                self.row_widgets[i].hide()
    
    def _select_tasks(self, row_idx: int) -> None:
        dialog = MultiSelectDialog(
            self,
            title=f"Select tasks for {self.group_name_edits[row_idx].text() or f'Group {row_idx+1}'}",
            items=state.tasks_cache,
            selected=self.group_items[row_idx],
            gray_hint_items=self._already_selected_elsewhere(row_idx),
            display_func=state.format_task,
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result is not None:
            self.group_items[row_idx] = dialog.result
            self._update_btn_text(row_idx)
            self._update_row_visibility()
    
    def _save_apply(self) -> None:
        new_groups: dict = {}
        new_names: dict = {}
        
        for i in range(self.max_groups):
            gid = f"T{i+1}"
            name = self.group_name_edits[i].text().strip() or f"Group {i+1}"
            items = self.group_items[i]
            
            if not items and i > 0:
                continue
            
            new_groups[gid] = items
            new_names[gid] = name
        
        if all(len(v) == 0 for v in new_groups.values()):
            state.task_groups.clear()
            state.task_group_names.clear()
        else:
            state.task_groups.clear()
            state.task_groups.update(new_groups)
            state.task_group_names.clear()
            state.task_group_names.update(new_names)
        
        self.accept()
