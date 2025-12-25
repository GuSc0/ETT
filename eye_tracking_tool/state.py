"""
Global application state manager.
This class centralizes all state management to avoid global variable issues.
"""
from __future__ import annotations

from typing import Optional, List, Dict
import pandas as pd


class AppState:
    """Singleton class to manage application state."""
    
    _instance: Optional['AppState'] = None
    
    def __new__(cls) -> 'AppState':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.reset()
    
    def reset(self) -> None:
        """Reset all state to initial values."""
        # Data
        self.df: Optional[pd.DataFrame] = None
        self.loaded_file_path: str = ""
        
        # Caches
        self.participants_cache: List[str] = []
        self.tasks_cache: List[str] = []
        
        # Grouping (participants)
        self.participant_groups: Dict[str, List[str]] = {}  # group_id -> participants
        self.group_names: Dict[str, str] = {}  # group_id -> group name
        
        # Grouping (tasks)
        self.task_groups: Dict[str, List[str]] = {}  # group_id -> task_ids
        self.task_group_names: Dict[str, str] = {}  # group_id -> group name
        
        # Task labels
        self.task_labels: Dict[str, str] = {}  # task_id -> human label
    
    def format_task(self, task_id: str) -> str:
        """Format task ID with optional label."""
        label = (self.task_labels.get(task_id, "") or "").strip()
        return f"{task_id} {label}".strip() if label else task_id
    
    def get_effective_participant_groups(self) -> Dict[str, List[str]]:
        """
        If no participant groups are defined, treat all participants as one implicit group.
        """
        if self.participant_groups:
            return self.participant_groups
        if self.participants_cache:
            return {"ALL": self.participants_cache.copy()}
        return {}
    
    def get_effective_group_names(self) -> Dict[str, str]:
        """Get group names, defaulting to 'All participants' if no groups defined."""
        if self.group_names:
            return self.group_names
        return {"ALL": "All participants"}


# Global instance
state = AppState()
