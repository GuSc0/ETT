"""
Data processing functions for TSV validation and extraction.
"""
from __future__ import annotations

from typing import List
import pandas as pd
import re

from models import ValidationResult


def _natural_sort_key(text: str) -> tuple:
    """
    Generate a sort key for natural/alphanumeric sorting.
    Splits the string into text and number parts for proper numeric ordering.
    Example: "10" < "2" becomes False, "1" < "2" < "10" becomes True
    """
    def convert(text_part: str) -> list:
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text_part)]
    
    return convert(text)


def validate_tsv_format(file_path: str, expected_columns: List[str]) -> ValidationResult:
    """
    Validate TSV format by checking:
      1) file extension
      2) readable header
      3) column count
      4) exact column names + order
    """
    # 1) Extension check
    if not file_path.lower().endswith(".tsv"):
        return ValidationResult(False, "Datei ist keine TSV-Datei (Erwartet: .tsv).")

    # 2) Read header only
    try:
        head = pd.read_csv(file_path, sep="\t", nrows=0)
        cols = list(head.columns)
    except Exception as e:
        return ValidationResult(False, f"Fehler beim Einlesen der Datei: {e}")

    # 3) Column count check
    if len(cols) != len(expected_columns):
        return ValidationResult(
            False,
            f"Spaltenanzahl stimmt nicht. Erwartet {len(expected_columns)}, gefunden {len(cols)}.",
            found_columns=cols,
        )

    # 4) Exact names + order check
    if cols != expected_columns:
        return ValidationResult(
            False,
            "Spaltennamen/Reihenfolge stimmen nicht.\n"
            f"Erwartet: {expected_columns}\n"
            f"Gefunden: {cols}",
            found_columns=cols,
        )

    return ValidationResult(True, None, found_columns=cols)


def extract_participants(df: pd.DataFrame, column: str = "Participant") -> List[str]:
    """
    Returns a sorted list of unique participants from the given column.
    - Strips whitespace
    - Drops NaN/empty values
    - Casts to string
    
    Raises KeyError if column not found.
    Raises ValueError if no valid participants found.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty.")
    
    if column not in df.columns:
        raise KeyError(f"Spalte '{column}' nicht gefunden. Verfügbare Spalten: {list(df.columns)}")

    s = df[column].dropna().astype(str).str.strip()
    s = s[s != ""]
    participants = sorted(s.unique().tolist())
    
    if not participants:
        raise ValueError(f"No valid participants found in column '{column}'.")
    
    return participants


def extract_tasks_from_toi(df: pd.DataFrame, column: str = "TOI") -> List[str]:
    """
    Extracts unique task identifiers from the TOI column by taking the suffix after the last underscore.
    Filters out non-task entries: 'entire recording' and 'full' (case-insensitive).
    
    Raises KeyError if column not found.
    Raises ValueError if no valid tasks found.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty.")
    
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found. Available columns: {list(df.columns)}")

    s = df[column].dropna().astype(str).str.strip()
    s = s[s != ""]

    # Take the last segment after the last underscore; if no underscore, keep full string
    suffix = s.apply(lambda x: x.rsplit("_", 1)[-1].strip())

    # Filter out unwanted entries (case-insensitive)
    blocked = {"entire recording", "full"}
    suffix_lower = suffix.str.lower()
    suffix = suffix[~suffix_lower.isin(blocked)]

    suffix = suffix.dropna().astype(str).str.strip()
    suffix = suffix[suffix != ""]
    
    # Use natural sort for proper numeric ordering (1, 2, 3, 10, 11, 12 instead of 1, 10, 11, 12, 2, 3)
    tasks = sorted(suffix.unique().tolist(), key=_natural_sort_key)
    
    if not tasks:
        raise ValueError(f"No valid tasks found in column '{column}'.")
    
    return tasks
