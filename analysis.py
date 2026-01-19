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
    Matches notebook approach: Sum of Bin_duration for the task.
    The notebook groups by Task_Number and TOI, then sums Bin_duration and converts to seconds.
    We sum all Bin_duration for this participant-task combination.
    
    Returns duration in milliseconds (will be converted to seconds in display).
    Returns None if no data found.
    """
    task_data = df[(df['Participant'] == participant) & (df['TOI'].str.endswith(f'_{task}', na=False))]
    
    if task_data.empty:
        return None
    
    # Use Bin_duration approach (matches notebook: calculate_task_durations function)
    if 'Bin_duration' not in task_data.columns:
        # Fallback to Start/Stop if Bin_duration not available
        valid_data = task_data[task_data['Start'].notna() & task_data['Stop'].notna()]
        if valid_data.empty:
            return None
        starts = _convert_numeric_column(valid_data['Start'])
        stops = _convert_numeric_column(valid_data['Stop'])
        min_start = starts.min()
        max_stop = stops.max()
        if pd.isna(min_start) or pd.isna(max_stop):
            return None
        # Return in milliseconds (will be converted to seconds later in display)
        # Note: Start/Stop might be in different units, but we'll treat as milliseconds for consistency
        duration_ms = float(max_stop - min_start)
        # If the difference seems too small (< 1), assume it's already in seconds and convert
        if duration_ms < 1.0:
            duration_ms = duration_ms * 1000.0
        return duration_ms
    
    # Convert Bin_duration to numeric (handle comma decimals)
    bin_durations = _convert_numeric_column(task_data['Bin_duration'])
    bin_durations = bin_durations.dropna()
    
    if bin_durations.empty:
        return None
    
    # Sum all Bin_duration values for this participant-task combination
    # Bin_duration is in milliseconds (as per notebook comment)
    # Group by TOI if needed (matching notebook: groupby(['Task_Number', 'TOI']))
    # But since we're already filtered to one task, we can just sum all
    total_duration_ms = float(bin_durations.sum())
    
    return total_duration_ms


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
    mode: str,
    parameter_weights: Optional[Dict[str, float]] = None
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
                            # Apply weight if provided
                            weight = (parameter_weights or {}).get(parameter, 1.0)
                            if weight != 1.0:
                                # Apply weight to mean value (primary metric used in analysis)
                                weighted_metrics = metrics.copy()
                                weighted_metrics['mean'] = metrics['mean'] * weight
                                # Also weight min/max for consistency
                                weighted_metrics['min'] = metrics['min'] * weight
                                weighted_metrics['max'] = metrics['max'] * weight
                                weighted_metrics['median'] = metrics['median'] * weight
                                weighted_metrics['q1'] = metrics['q1'] * weight
                                weighted_metrics['q3'] = metrics['q3'] * weight
                                # Std should be scaled too
                                weighted_metrics['std'] = metrics['std'] * weight
                                participant_data[parameter] = weighted_metrics
                            else:
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
    
    # Calculate Standard Deviation of TCT for each group-task combination
    _add_tct_std_to_result(result, df, selected_groups, selected_tasks, parameter_weights)
    
    # If mode is "Only group mean", aggregate now
    if mode == "Only group mean":
        aggregated = _aggregate_to_group_means(result, parameter_weights)
        if not aggregated:
            raise ValueError("Failed to aggregate data to group means.")
        return aggregated
    
    return result


def _add_tct_std_to_result(
    result: Dict[str, Dict],
    df: pd.DataFrame,
    selected_groups: List[str],
    selected_tasks: List[str],
    parameter_weights: Optional[Dict[str, float]] = None
) -> None:
    """
    Calculate Standard Deviation of TCT for each group-task combination
    and add it to the result structure.
    Only uses participants that are actually in the result structure (correctly filtered).
    """
    for group_id in selected_groups:
        if group_id not in result:
            continue
        
        for task_id in selected_tasks:
            if task_id not in result[group_id]:
                continue
            
            task_data = result[group_id][task_id]
            
            # Get participants from the actual result structure, not from state
            # This ensures we only use participants that were correctly filtered
            participants_in_result = []
            if isinstance(task_data, dict):
                # Check if it's individual participant mode (has participant keys)
                is_individual_mode = any(
                    isinstance(k, str) and isinstance(v, dict) and 
                    any(isinstance(vv, dict) for vv in v.values() if isinstance(vv, dict))
                    for k, v in task_data.items() if k != "_group_stats"
                )
                
                if is_individual_mode:
                    # Individual participant mode: get participants from task_data keys
                    participants_in_result = [p for p in task_data.keys() if p != "_group_stats"]
                else:
                    # Group mean mode: need to get participants from effective groups
                    # but only those that would have been processed
                    effective_groups = state.get_effective_participant_groups()
                    participants_in_result = effective_groups.get(group_id, [])
            
            if not participants_in_result:
                continue
            
            # Collect TCT values from participants that are actually in the result
            tct_values = []
            for participant in participants_in_result:
                tct = calculate_tct(df, participant, task_id)
                if tct is not None:
                    tct_values.append(tct)
            
            if len(tct_values) > 1:
                std_tct = float(np.std(tct_values, ddof=1))
                mean_tct = float(np.mean(tct_values))
                
                # Apply weight if provided
                weight = (parameter_weights or {}).get("Standard Deviation of TCT", 1.0)
                std_tct = std_tct * weight
                mean_tct = mean_tct * weight
                
                # Check if this is individual participant mode (has participant keys that are strings)
                is_individual_mode = isinstance(task_data, dict) and any(
                    isinstance(k, str) and isinstance(v, dict) and 
                    any(isinstance(vv, dict) for vv in v.values() if isinstance(vv, dict))
                    for k, v in task_data.items() if k != "_group_stats"
                )
                
                if is_individual_mode:
                    # Individual participant mode - add std at group-task level
                    if "_group_stats" not in task_data:
                        task_data["_group_stats"] = {}
                    task_data["_group_stats"]["Standard Deviation of TCT"] = {
                        'mean': std_tct,
                        'std': 0.0,
                        'median': std_tct,
                        'min': std_tct,
                        'max': std_tct,
                        'q1': std_tct,
                        'q3': std_tct,
                    }
                else:
                    # Group mean mode - add directly
                    task_data["Standard Deviation of TCT"] = {
                        'mean': std_tct,
                        'std': 0.0,
                        'median': std_tct,
                        'min': std_tct,
                        'max': std_tct,
                        'q1': std_tct,
                        'q3': std_tct,
                    }


def _aggregate_to_group_means(
    data: Dict[str, Dict],
    parameter_weights: Optional[Dict[str, float]] = None
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
            
            # Handle Standard Deviation of TCT if present in group stats
            if isinstance(task_data, dict) and "_group_stats" in task_data:
                for param, stats in task_data["_group_stats"].items():
                    aggregated[group_id][task_id][param] = stats
    
    return aggregated


def calculate_normalized_rankings_per_group(
    aggregated_data: Dict[str, Dict],
    selected_groups: List[str],
    selected_tasks: List[str],
    active_parameters: List[str],
    parameter_weights: Dict[str, float]
) -> Dict[str, pd.DataFrame]:
    """
    Calculate normalized rankings per group with overall ranking.
    
    Returns a dictionary mapping group_id to a DataFrame with columns:
    - Task_Number (task_id)
    - Individual parameter ranks (Rank_Peak_Velocity, etc.)
    - Overall_Rank
    - Sum_of_Ranks
    - indications
    - contraindications
    - neither
    """
    from collections import defaultdict
    
    group_rankings = {}
    
    for group_id in selected_groups:
        if group_id not in aggregated_data:
            continue
        
        group_data = aggregated_data[group_id]
        
        # Step 1: Collect and normalize values for each parameter
        param_normalized_values = {}  # parameter -> {task_id: normalized_value}
        
        for parameter in active_parameters:
            if parameter == "Standard Deviation of TCT":
                continue  # Skip for now, can be added later
            
            task_values = {}  # task_id -> mean_value
            
            for task_id in selected_tasks:
                if task_id not in group_data:
                    continue
                
                task_data = group_data[task_id]
                
                # Get mean value for this parameter
                mean_value = None
                if isinstance(task_data, dict):
                    if parameter in task_data:
                        mean_value = task_data[parameter].get('mean', 0)
                    elif "_group_stats" in task_data and parameter in task_data["_group_stats"]:
                        mean_value = task_data["_group_stats"][parameter].get('mean', 0)
                    else:
                        # Individual participant mode - calculate mean
                        participant_values = []
                        for participant_data in task_data.values():
                            if isinstance(participant_data, dict) and parameter in participant_data:
                                participant_values.append(participant_data[parameter].get('mean', 0))
                        if participant_values:
                            mean_value = float(np.mean(participant_values))
                
                if mean_value is not None:
                    task_values[task_id] = mean_value
            
            # Normalize values for this parameter (min-max normalization)
            if task_values:
                values_list = list(task_values.values())
                min_val = min(values_list)
                max_val = max(values_list)
                
                if max_val > min_val:
                    normalized = {task_id: (val - min_val) / (max_val - min_val) 
                                 for task_id, val in task_values.items()}
                else:
                    normalized = {task_id: 0.5 for task_id in task_values.keys()}
                
                param_normalized_values[parameter] = normalized
        
        # Step 2: Rank tasks for each parameter
        param_rankings = {}  # parameter -> DataFrame with Task_Number and Rank
        
        for parameter, normalized_values in param_normalized_values.items():
            # Determine ranking direction
            # Most parameters: higher = more challenging (ascending=False)
            # Saccade Amplitude: lower = more challenging (ascending=True)
            ascending = (parameter == "Saccade Amplitude")
            
            # Create DataFrame for ranking
            rank_df = pd.DataFrame([
                {'Task_Number': task_id, 'Normalized_Value': norm_val}
                for task_id, norm_val in normalized_values.items()
            ])
            
            # Sort and rank
            rank_df = rank_df.sort_values(by='Normalized_Value', ascending=ascending).reset_index(drop=True)
            
            # Create rank column name
            rank_col_name = f"Rank_{parameter.replace(' ', '_').replace('(', '').replace(')', '')}"
            rank_df[rank_col_name] = rank_df['Normalized_Value'].rank(ascending=ascending, method='min').astype(int)
            
            # Keep only Task_Number and Rank
            param_rankings[parameter] = rank_df[['Task_Number', rank_col_name]]
        
        # Step 3: Combine all rankings
        if not param_rankings:
            continue
        
        # Start with first parameter
        param_items = list(param_rankings.items())
        overall_ranking = param_items[0][1].copy()  # Get the DataFrame from the first item
        
        # Merge with remaining parameters
        for parameter, rank_df in param_items[1:]:
            overall_ranking = pd.merge(overall_ranking, rank_df, on='Task_Number', how='outer')
        
        # Step 4: Calculate weighted sum of ranks
        rank_columns = [col for col in overall_ranking.columns if col.startswith('Rank_')]
        
        # Fill NaN with max rank + 1
        if rank_columns:
            max_rank = overall_ranking[rank_columns].max().max()
            overall_ranking[rank_columns] = overall_ranking[rank_columns].fillna(max_rank + 1)
        
        # Calculate weighted sum
        sum_of_ranks = pd.Series(0.0, index=overall_ranking.index)
        
        for col in rank_columns:
            # Extract parameter name from column
            col_clean = col.replace('Rank_', '').replace('_', '').lower()
            # Map to actual parameter name
            for param in active_parameters:
                param_clean = param.replace(' ', '').replace('(', '').replace(')', '').replace('_', '').lower()
                if param_clean == col_clean:
                    weight = parameter_weights.get(param, 1.0)
                    sum_of_ranks += overall_ranking[col] * weight
                    break
        
        overall_ranking['Sum_of_Ranks'] = sum_of_ranks
        
        # Step 5: Calculate overall rank (lower sum = rank 1)
        overall_ranking['Overall_Rank'] = overall_ranking['Sum_of_Ranks'].rank(ascending=True, method='min').astype(int)
        
        # Step 6: Calculate indications/contraindications
        overall_ranking['indications'] = [[] for _ in range(len(overall_ranking))]
        overall_ranking['contraindications'] = [[] for _ in range(len(overall_ranking))]
        overall_ranking['neither'] = [[] for _ in range(len(overall_ranking))]
        
        metric_cols = {}
        for param in active_parameters:
            if param == "Standard Deviation of TCT":
                continue
            col_name = f"Rank_{param.replace(' ', '_').replace('(', '').replace(')', '')}"
            if col_name in overall_ranking.columns:
                metric_cols[param] = col_name
        
        for index, row in overall_ranking.iterrows():
            overall_task_rank = row['Overall_Rank']
            for metric_name, rank_col in metric_cols.items():
                if pd.isna(row[rank_col]):
                    continue
                metric_rank = int(row[rank_col])
                rank_difference = abs(overall_task_rank - metric_rank)
                
                if rank_difference <= 2:
                    overall_ranking.at[index, 'indications'].append(metric_name)
                elif rank_difference > 3:
                    overall_ranking.at[index, 'contraindications'].append(metric_name)
                else:
                    overall_ranking.at[index, 'neither'].append(metric_name)
        
        # Convert lists to strings
        overall_ranking['indications'] = overall_ranking['indications'].apply(
            lambda x: ', '.join(x) if x else 'None'
        )
        overall_ranking['contraindications'] = overall_ranking['contraindications'].apply(
            lambda x: ', '.join(x) if x else 'None'
        )
        overall_ranking['neither'] = overall_ranking['neither'].apply(
            lambda x: ', '.join(x) if x else 'None'
        )
        
        # Sort by overall rank
        overall_ranking = overall_ranking.sort_values(by='Overall_Rank').reset_index(drop=True)
        
        group_rankings[group_id] = overall_ranking
    
    return group_rankings


def calculate_normalized_rankings_per_participant(
    aggregated_data: Dict[str, Dict],
    selected_groups: List[str],
    selected_tasks: List[str],
    active_parameters: List[str],
    parameter_weights: Dict[str, float]
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Calculate normalized rankings per participant within each group.
    
    Returns a dictionary mapping group_id -> participant_id -> DataFrame with columns:
    - Task_Number (task_id)
    - Individual parameter ranks (Rank_Peak_Velocity, etc.)
    - Overall_Rank
    - Sum_of_Ranks
    - indications
    - contraindications
    - neither
    """
    from collections import defaultdict
    
    participant_rankings = {}  # {group_id: {participant_id: DataFrame}}
    
    for group_id in selected_groups:
        if group_id not in aggregated_data:
            continue
        
        group_data = aggregated_data[group_id]
        participant_rankings[group_id] = {}
        
        # Get all participants for this group from the data structure
        all_participants = set()
        for task_data in group_data.values():
            if isinstance(task_data, dict):
                for key in task_data.keys():
                    if key != "_group_stats" and isinstance(task_data[key], dict):
                        all_participants.add(key)
        
        # Calculate rankings for each participant
        for participant_id in all_participants:
            # Step 1: Collect and normalize values for each parameter for this participant
            param_normalized_values = {}  # parameter -> {task_id: normalized_value}
            
            for parameter in active_parameters:
                if parameter == "Standard Deviation of TCT":
                    continue  # Skip - this is group-level
                
                task_values = {}  # task_id -> mean_value
                
                for task_id in selected_tasks:
                    if task_id not in group_data:
                        continue
                    
                    task_data = group_data[task_id]
                    
                    # Get value for this participant and parameter
                    if isinstance(task_data, dict) and participant_id in task_data:
                        participant_data = task_data[participant_id]
                        if isinstance(participant_data, dict) and parameter in participant_data:
                            task_values[task_id] = participant_data[parameter].get('mean', 0)
                
                # Normalize values for this parameter (min-max normalization)
                if task_values:
                    values_list = list(task_values.values())
                    min_val = min(values_list)
                    max_val = max(values_list)
                    
                    if max_val > min_val:
                        normalized = {task_id: (val - min_val) / (max_val - min_val) 
                                     for task_id, val in task_values.items()}
                    else:
                        normalized = {task_id: 0.5 for task_id in task_values.keys()}
                    
                    param_normalized_values[parameter] = normalized
            
            # Step 2: Rank tasks for each parameter
            param_rankings = {}  # parameter -> DataFrame with Task_Number and Rank
            
            for parameter, normalized_values in param_normalized_values.items():
                # Determine ranking direction
                ascending = (parameter == "Saccade Amplitude")
                
                # Create DataFrame for ranking
                rank_df = pd.DataFrame([
                    {'Task_Number': task_id, 'Normalized_Value': norm_val}
                    for task_id, norm_val in normalized_values.items()
                ])
                
                # Sort and rank
                rank_df = rank_df.sort_values(by='Normalized_Value', ascending=ascending).reset_index(drop=True)
                
                # Create rank column name
                rank_col_name = f"Rank_{parameter.replace(' ', '_').replace('(', '').replace(')', '')}"
                rank_df[rank_col_name] = rank_df['Normalized_Value'].rank(ascending=ascending, method='min').astype(int)
                
                # Keep only Task_Number and Rank
                param_rankings[parameter] = rank_df[['Task_Number', rank_col_name]]
            
            # Step 3: Combine all rankings
            if not param_rankings:
                continue
            
            # Start with first parameter
            param_items = list(param_rankings.items())
            overall_ranking = param_items[0][1].copy()
            
            # Merge with remaining parameters
            for parameter, rank_df in param_items[1:]:
                overall_ranking = pd.merge(overall_ranking, rank_df, on='Task_Number', how='outer')
            
            # Step 4: Calculate weighted sum of ranks
            rank_columns = [col for col in overall_ranking.columns if col.startswith('Rank_')]
            
            # Fill NaN with max rank + 1
            if rank_columns:
                max_rank = overall_ranking[rank_columns].max().max()
                overall_ranking[rank_columns] = overall_ranking[rank_columns].fillna(max_rank + 1)
            
            # Calculate weighted sum
            sum_of_ranks = pd.Series(0.0, index=overall_ranking.index)
            
            for col in rank_columns:
                # Extract parameter name from column
                col_clean = col.replace('Rank_', '').replace('_', '').lower()
                # Map to actual parameter name
                for param in active_parameters:
                    param_clean = param.replace(' ', '').replace('(', '').replace(')', '').replace('_', '').lower()
                    if param_clean == col_clean:
                        weight = parameter_weights.get(param, 1.0)
                        sum_of_ranks += overall_ranking[col] * weight
                        break
            
            overall_ranking['Sum_of_Ranks'] = sum_of_ranks
            
            # Step 5: Calculate overall rank (lower sum = rank 1)
            overall_ranking['Overall_Rank'] = overall_ranking['Sum_of_Ranks'].rank(ascending=True, method='min').astype(int)
            
            # Step 6: Calculate indications/contraindications
            overall_ranking['indications'] = [[] for _ in range(len(overall_ranking))]
            overall_ranking['contraindications'] = [[] for _ in range(len(overall_ranking))]
            overall_ranking['neither'] = [[] for _ in range(len(overall_ranking))]
            
            metric_cols = {}
            for param in active_parameters:
                if param == "Standard Deviation of TCT":
                    continue
                col_name = f"Rank_{param.replace(' ', '_').replace('(', '').replace(')', '')}"
                if col_name in overall_ranking.columns:
                    metric_cols[param] = col_name
            
            for index, row in overall_ranking.iterrows():
                overall_task_rank = row['Overall_Rank']
                for metric_name, rank_col in metric_cols.items():
                    if pd.isna(row[rank_col]):
                        continue
                    metric_rank = int(row[rank_col])
                    rank_difference = abs(overall_task_rank - metric_rank)
                    
                    if rank_difference <= 2:
                        overall_ranking.at[index, 'indications'].append(metric_name)
                    elif rank_difference > 3:
                        overall_ranking.at[index, 'contraindications'].append(metric_name)
                    else:
                        overall_ranking.at[index, 'neither'].append(metric_name)
            
            # Convert lists to strings
            overall_ranking['indications'] = overall_ranking['indications'].apply(
                lambda x: ', '.join(x) if x else 'None'
            )
            overall_ranking['contraindications'] = overall_ranking['contraindications'].apply(
                lambda x: ', '.join(x) if x else 'None'
            )
            overall_ranking['neither'] = overall_ranking['neither'].apply(
                lambda x: ', '.join(x) if x else 'None'
            )
            
            # Sort by overall rank
            overall_ranking = overall_ranking.sort_values(by='Overall_Rank').reset_index(drop=True)
            
            participant_rankings[group_id][participant_id] = overall_ranking
    
    return participant_rankings


def calculate_rankings(
    aggregated_data: Dict[str, Dict],
    parameter: str
) -> List[Tuple[str, str, float, int]]:
    """
    Calculate rankings for a specific parameter (legacy function, kept for compatibility).
    
    Returns list of (group_id, task_id, mean_value, rank) tuples, sorted by rank.
    Handles both group mean mode and individual participant mode.
    """
    rankings = []
    
    for group_id, group_data in aggregated_data.items():
        for task_id, task_data in group_data.items():
            if isinstance(task_data, dict):
                # Check if it's group mean mode (parameter directly in task_data)
                if parameter in task_data:
                    mean_value = task_data[parameter].get('mean', 0)
                    rankings.append((group_id, task_id, mean_value))
                # Check if it's individual participant mode or has _group_stats
                elif "_group_stats" in task_data and parameter in task_data["_group_stats"]:
                    mean_value = task_data["_group_stats"][parameter].get('mean', 0)
                    rankings.append((group_id, task_id, mean_value))
                # Individual participant mode - calculate mean across participants
                elif any(isinstance(v, dict) for v in task_data.values() if isinstance(v, dict)):
                    participant_values = []
                    for participant_data in task_data.values():
                        if isinstance(participant_data, dict) and parameter in participant_data:
                            participant_values.append(participant_data[parameter].get('mean', 0))
                    if participant_values:
                        mean_value = float(np.mean(participant_values))
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
                # Handle both group mean mode and individual participant mode
                if isinstance(task_data, dict):
                    if parameter in task_data:
                        # Group mean mode: task_data is {parameter: {stats}}
                        all_values.append(task_data[parameter].get('mean', 0))
                    elif "_group_stats" in task_data and parameter in task_data["_group_stats"]:
                        # Standard Deviation of TCT or other group-level stats
                        all_values.append(task_data["_group_stats"][parameter].get('mean', 0))
                    else:
                        # Individual participant mode: task_data is {participant: {parameter: {stats}}}
                        # Also check _group_stats for group-level parameters like Standard Deviation of TCT
                        if "_group_stats" in task_data and parameter in task_data["_group_stats"]:
                            all_values.append(task_data["_group_stats"][parameter].get('mean', 0))
                        else:
                            for participant_data in task_data.values():
                                if isinstance(participant_data, dict) and parameter in participant_data:
                                    all_values.append(participant_data[parameter].get('mean', 0))
        
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
            
            # Initialize all parameters to 0.0 (will be overwritten if data exists)
            for parameter in parameters:
                normalized[group_id][task_id][parameter] = 0.0
            
            # Handle both group mean mode and individual participant mode
            if isinstance(task_data, dict):
                # Check if it's individual participant mode
                is_individual = any(
                    isinstance(k, str) and isinstance(v, dict) and 
                    any(isinstance(vv, dict) for vv in v.values() if isinstance(vv, dict))
                    for k, v in task_data.items() if k != "_group_stats"
                )
                
                if is_individual:
                    # Individual participant mode: task_data is {participant: {parameter: {stats}}}
                    # For radar chart, we'll use group means
                    participant_values = {}
                    for participant, participant_data in task_data.items():
                        if participant == "_group_stats":
                            continue
                        if isinstance(participant_data, dict):
                            for parameter in parameters:
                                if parameter in participant_data:
                                    if parameter not in participant_values:
                                        participant_values[parameter] = []
                                    participant_values[parameter].append(participant_data[parameter].get('mean', 0))
                    
                    # Calculate group mean for each parameter
                    for parameter in parameters:
                        value = None
                        # Check _group_stats first (for Standard Deviation of TCT)
                        if "_group_stats" in task_data and parameter in task_data["_group_stats"]:
                            value = task_data["_group_stats"][parameter].get('mean', 0)
                        elif parameter in participant_values and participant_values[parameter]:
                            value = float(np.mean(participant_values[parameter]))
                        
                        if value is not None:
                            min_val, max_val = param_min_max[parameter]
                            if max_val > min_val:
                                normalized_value = (value - min_val) / (max_val - min_val)
                            else:
                                normalized_value = 0.5 if value > 0 else 0.0
                            normalized[group_id][task_id][parameter] = normalized_value
                else:
                    # Group mean mode: task_data is {parameter: {stats}}
                    for parameter in parameters:
                        value = None
                        if parameter in task_data:
                            value = task_data[parameter].get('mean', 0)
                        elif "_group_stats" in task_data and parameter in task_data["_group_stats"]:
                            # Handle Standard Deviation of TCT in group mean mode
                            value = task_data["_group_stats"][parameter].get('mean', 0)
                        
                        if value is not None:
                            min_val, max_val = param_min_max[parameter]
                            if max_val > min_val:
                                normalized_value = (value - min_val) / (max_val - min_val)
                            else:
                                normalized_value = 0.5 if value > 0 else 0.0
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
                    if parameter == "_group_stats":
                        continue  # Skip internal metadata
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
            elif mode == "Each participant for selected groups":
                # task_data is {participant: {parameter: {stats}}}
                for participant, participant_data in task_data.items():
                    if participant == "_group_stats":
                        continue  # Skip internal metadata
                    if isinstance(participant_data, dict):
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
            else:  # "Group mean and individual participants"
                # Show both group means and individual participants
                # First, calculate and show group means
                participant_values_by_param = {}
                for participant, participant_data in task_data.items():
                    if participant == "_group_stats":
                        continue
                    if isinstance(participant_data, dict):
                        for parameter, stats in participant_data.items():
                            if parameter not in participant_values_by_param:
                                participant_values_by_param[parameter] = []
                            participant_values_by_param[parameter].append(stats.get('mean', 0))
                
                # Add group mean rows
                for parameter, values in participant_values_by_param.items():
                    if values:
                        values_array = np.array(values)
                        rows.append({
                            'Group': group_name,
                            'Task': task_label,
                            'Participant': 'Group Mean',
                            'Parameter': parameter,
                            'Mean': float(np.mean(values_array)),
                            'Std Dev': float(np.std(values_array, ddof=1)) if len(values_array) > 1 else 0.0,
                            'Median': float(np.median(values_array)),
                            'Min': float(np.min(values_array)),
                            'Max': float(np.max(values_array)),
                            'Q1': float(np.percentile(values_array, 25)),
                            'Q3': float(np.percentile(values_array, 75)),
                        })
                
                # Then add individual participant rows
                for participant, participant_data in task_data.items():
                    if participant == "_group_stats":
                        continue
                    if isinstance(participant_data, dict):
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
