"""
Analysis functions for eye tracking data processing.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from models import PARAMETER_COLUMN_MAP, PARAMETER_OPTIONS
from state import state


def _convert_numeric_column(series: pd.Series) -> pd.Series:
    """
    Convert a series to numeric, handling comma as decimal separator.
    """
    # First try converting directly
    result = pd.to_numeric(series, errors='coerce')
    
    # If that fails (many NaN), try replacing comma with dot
    if result.isna().sum() > len(series) * 0.5:
        result = pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')
    
    return result


def calculate_tct(df: pd.DataFrame, participant: str, task: str) -> Optional[float]:
    """
    Calculate Task Completion Time (TCT) for a participant-task combination.
    TCT is the duration from first event Start to last event Stop.
    
    Returns None if no data found.
    """
    task_data = df[(df['Participant'] == participant) & (df['TOI'].str.endswith(f'_{task}', na=False))]
    
    if task_data.empty:
        return None
    
    # Filter to valid events (non-empty Start/Stop)
    valid_data = task_data[task_data['Start'].notna() & task_data['Stop'].notna()]
    
    if valid_data.empty:
        return None
    
    # Convert to numeric (handle comma decimals)
    starts = _convert_numeric_column(valid_data['Start'])
    stops = _convert_numeric_column(valid_data['Stop'])
    
    min_start = starts.min()
    max_stop = stops.max()
    
    if pd.isna(min_start) or pd.isna(max_stop):
        return None
    
    return float(max_stop - min_start)


def calculate_parameter_metrics(
    df: pd.DataFrame,
    participant: str,
    task: str,
    parameter: str
) -> Dict[str, float]:
    """
    Calculate full statistics for a parameter for a participant-task combination.
    
    Returns a dictionary with: mean, std, median, min, max, q1, q3
    Returns empty dict if no data found.
    """
    task_data = df[(df['Participant'] == participant) & (df['TOI'].str.endswith(f'_{task}', na=False))]
    
    if task_data.empty:
        return {}
    
    # Get the column name for this parameter
    column_name = PARAMETER_COLUMN_MAP.get(parameter)
    
    if column_name == "calculated":
        # Special handling for TCT
        if parameter == "Task Completion Time (TCT)":
            tct = calculate_tct(df, participant, task)
            if tct is None:
                return {}
            return {
                'mean': tct,
                'std': 0.0,  # Single value, no std
                'median': tct,
                'min': tct,
                'max': tct,
                'q1': tct,
                'q3': tct,
            }
        elif parameter == "Standard Deviation of TCT":
            # This is calculated across participants, not per participant-task
            return {}
        else:
            return {}
    
    if column_name not in task_data.columns:
        return {}
    
    # Get values and convert to numeric
    values = _convert_numeric_column(task_data[column_name])
    values = values.dropna()
    
    if values.empty:
        return {}
    
    values_array = values.values
    
    return {
        'mean': float(np.mean(values_array)),
        'std': float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
        'median': float(np.median(values_array)),
        'min': float(np.min(values_array)),
        'max': float(np.max(values_array)),
        'q1': float(np.percentile(values_array, 25)),
        'q3': float(np.percentile(values_array, 75)),
    }


def aggregate_by_groups(
    df: pd.DataFrame,
    selected_groups: List[str],
    selected_tasks: List[str],
    deselected_params: List[str],
    mode: str
) -> Dict[str, Dict]:
    """
    Aggregate metrics by participant groups and tasks.
    
    Returns a nested dictionary:
    {
        'group_id': {
            'task_id': {
                'participant': {
                    'parameter': {stats_dict}
                }
            }
        }
    }
    Or aggregated by group if mode is "Only group mean"
    
    Raises ValueError if input data is invalid.
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty.")
    
    if not selected_groups:
        raise ValueError("No groups selected.")
    
    if not selected_tasks:
        raise ValueError("No tasks selected.")
    
    effective_groups = state.get_effective_participant_groups()
    
    if not effective_groups:
        raise ValueError("No participant groups available.")
    
    # Filter to selected groups
    filtered_groups = {gid: participants for gid, participants in effective_groups.items() 
                       if gid in selected_groups}
    
    if not filtered_groups:
        raise ValueError("None of the selected groups exist in the data.")
    
    # Get active parameters (exclude deselected)
    active_parameters = [p for p in PARAMETER_OPTIONS if p not in deselected_params]
    
    if not active_parameters:
        raise ValueError("All parameters are deselected.")
    
    result: Dict[str, Dict] = {}
    data_found = False
    
    for group_id, participants in filtered_groups.items():
        if not participants:
            continue
        
        result[group_id] = {}
        
        for task_id in selected_tasks:
            result[group_id][task_id] = {}
            
            for participant in participants:
                participant_data = {}
                
                for parameter in active_parameters:
                    if parameter == "Standard Deviation of TCT":
                        # Skip - this is calculated at group level
                        continue
                    
                    try:
                        metrics = calculate_parameter_metrics(df, participant, task_id, parameter)
                        if metrics:
                            participant_data[parameter] = metrics
                            data_found = True
                    except Exception as e:
                        # Log but continue with other parameters
                        print(f"Warning: Failed to calculate {parameter} for {participant}/{task_id}: {e}")
                        continue
                
                if participant_data:
                    result[group_id][task_id][participant] = participant_data
    
    if not data_found:
        raise ValueError("No data found for the selected groups, tasks, and parameters.")
    
    # If mode is "Only group mean", aggregate now
    if mode == "Only group mean":
        aggregated = _aggregate_to_group_means(result)
        if not aggregated:
            raise ValueError("Failed to aggregate data to group means.")
        return aggregated
    
    return result


def _aggregate_to_group_means(
    data: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    Aggregate participant-level data to group means.
    """
    aggregated = {}
    
    for group_id, group_data in data.items():
        aggregated[group_id] = {}
        
        for task_id, task_data in group_data.items():
            aggregated[group_id][task_id] = {}
            
            # Collect all participant values for each parameter
            param_values: Dict[str, List[float]] = {}
            
            for participant, participant_data in task_data.items():
                for parameter, metrics in participant_data.items():
                    if parameter not in param_values:
                        param_values[parameter] = []
                    param_values[parameter].append(metrics.get('mean', 0))
            
            # Calculate group statistics
            for parameter, values in param_values.items():
                if values:
                    values_array = np.array(values)
                    aggregated[group_id][task_id][parameter] = {
                        'mean': float(np.mean(values_array)),
                        'std': float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
                        'median': float(np.median(values_array)),
                        'min': float(np.min(values_array)),
                        'max': float(np.max(values_array)),
                        'q1': float(np.percentile(values_array, 25)),
                        'q3': float(np.percentile(values_array, 75)),
                    }
    
    return aggregated


def calculate_rankings(
    aggregated_data: Dict[str, Dict],
    parameter: str
) -> List[Tuple[str, str, float, int]]:
    """
    Calculate rankings for a specific parameter.
    
    Returns list of (group_id, task_id, mean_value, rank) tuples, sorted by rank.
    """
    rankings = []
    
    for group_id, group_data in aggregated_data.items():
        for task_id, task_data in group_data.items():
            if isinstance(task_data, dict) and parameter in task_data:
                mean_value = task_data[parameter].get('mean', 0)
                rankings.append((group_id, task_id, mean_value))
    
    # Sort by mean value (descending - higher values = higher cognitive load = higher rank)
    rankings.sort(key=lambda x: x[2], reverse=True)
    
    # Assign ranks (handle ties)
    result = []
    current_rank = 1
    prev_value = None
    
    for i, (group_id, task_id, value) in enumerate(rankings):
        if prev_value is not None and abs(value - prev_value) > 1e-10:
            current_rank = i + 1
        result.append((group_id, task_id, value, current_rank))
        prev_value = value
    
    return result


def normalize_for_radar(
    aggregated_data: Dict[str, Dict],
    parameters: List[str]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Normalize parameter values to 0-1 scale using min-max normalization.
    
    Returns nested dict: {group_id: {task_id: {parameter: normalized_value}}}
    """
    # First, collect all values to find min/max for each parameter
    param_min_max: Dict[str, Tuple[float, float]] = {}
    
    for parameter in parameters:
        all_values = []
        for group_data in aggregated_data.values():
            for task_data in group_data.values():
                if isinstance(task_data, dict) and parameter in task_data:
                    all_values.append(task_data[parameter].get('mean', 0))
        
        if all_values:
            param_min_max[parameter] = (min(all_values), max(all_values))
        else:
            param_min_max[parameter] = (0.0, 1.0)  # Default to avoid division by zero
    
    # Normalize
    normalized = {}
    
    for group_id, group_data in aggregated_data.items():
        normalized[group_id] = {}
        
        for task_id, task_data in group_data.items():
            normalized[group_id][task_id] = {}
            
            for parameter in parameters:
                if isinstance(task_data, dict) and parameter in task_data:
                    value = task_data[parameter].get('mean', 0)
                    min_val, max_val = param_min_max[parameter]
                    
                    if max_val > min_val:
                        normalized_value = (value - min_val) / (max_val - min_val)
                    else:
                        normalized_value = 0.5  # All values are the same
                    
                    normalized[group_id][task_id][parameter] = normalized_value
    
    return normalized


def generate_statistics_table(
    aggregated_data: Dict[str, Dict],
    selected_tasks: List[str],
    selected_groups: List[str],
    mode: str
) -> pd.DataFrame:
    """
    Generate a statistics table with full stats for all parameters.
    
    Returns a pandas DataFrame.
    """
    rows = []
    
    for group_id in selected_groups:
        if group_id not in aggregated_data:
            continue
        
        group_data = aggregated_data[group_id]
        group_name = state.get_effective_group_names().get(group_id, group_id)
        
        for task_id in selected_tasks:
            if task_id not in group_data:
                continue
            
            task_data = group_data[task_id]
            task_label = state.format_task(task_id)
            
            # Handle different data structures based on mode
            if mode == "Only group mean":
                # task_data is {parameter: {stats}}
                for parameter, stats in task_data.items():
                    rows.append({
                        'Group': group_name,
                        'Task': task_label,
                        'Parameter': parameter,
                        'Mean': stats.get('mean', 0),
                        'Std Dev': stats.get('std', 0),
                        'Median': stats.get('median', 0),
                        'Min': stats.get('min', 0),
                        'Max': stats.get('max', 0),
                        'Q1': stats.get('q1', 0),
                        'Q3': stats.get('q3', 0),
                    })
            else:
                # task_data is {participant: {parameter: {stats}}}
                for participant, participant_data in task_data.items():
                    for parameter, stats in participant_data.items():
                        rows.append({
                            'Group': group_name,
                            'Task': task_label,
                            'Participant': participant,
                            'Parameter': parameter,
                            'Mean': stats.get('mean', 0),
                            'Std Dev': stats.get('std', 0),
                            'Median': stats.get('median', 0),
                            'Min': stats.get('min', 0),
                            'Max': stats.get('max', 0),
                            'Q1': stats.get('q1', 0),
                            'Q3': stats.get('q3', 0),
                        })
    
    if not rows:
        return pd.DataFrame()
    
    return pd.DataFrame(rows)
