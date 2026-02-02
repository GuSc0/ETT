"""
Quick test script to verify the application functionality.
"""
import sys
import pandas as pd
from pathlib import Path

# Test imports
print("Testing imports...")
try:
    from state import state
    from models import EXPECTED_COLUMNS, PARAMETER_OPTIONS, PARAMETER_COLUMN_MAP
    from data_processor import validate_tsv_format, extract_participants, extract_tasks_from_toi
    from analysis import calculate_tct, calculate_parameter_metrics, aggregate_by_groups
    print("[OK] All imports successful")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)

# Test data file loading
print("\nTesting data file loading...")
data_file = Path("Input/BHS_Setup_Metrics_update.tsv")
if not data_file.exists():
    print(f"✗ Data file not found: {data_file}")
    sys.exit(1)

print(f"[OK] Data file found: {data_file}")

# Test validation
print("\nTesting TSV validation...")
result = validate_tsv_format(str(data_file), EXPECTED_COLUMNS)
if result.ok:
    print("[OK] TSV format is valid")
else:
    print(f"[ERROR] TSV validation failed: {result.message}")
    sys.exit(1)

# Test data loading
print("\nTesting data loading...")
try:
    df = pd.read_csv(data_file, sep="\t")
    print(f"[OK] Loaded {len(df)} rows")
except Exception as e:
    print(f"[ERROR] Failed to load data: {e}")
    sys.exit(1)

# Test participant extraction
print("\nTesting participant extraction...")
try:
    participants = extract_participants(df)
    print(f"[OK] Found {len(participants)} participants: {participants[:5]}{'...' if len(participants) > 5 else ''}")
except Exception as e:
    print(f"[ERROR] Failed to extract participants: {e}")
    sys.exit(1)

# Test task extraction
print("\nTesting task extraction...")
try:
    tasks = extract_tasks_from_toi(df)
    print(f"[OK] Found {len(tasks)} tasks: {tasks[:5]}{'...' if len(tasks) > 5 else ''}")
except Exception as e:
    print(f"[ERROR] Failed to extract tasks: {e}")
    sys.exit(1)

# Test TCT calculation
print("\nTesting TCT calculation...")
if participants and tasks:
    try:
        tct = calculate_tct(df, participants[0], tasks[0])
        if tct is not None:
            print(f"[OK] TCT calculated for {participants[0]}/{tasks[0]}: {tct:.2f} ms")
        else:
            print(f"[WARNING] No TCT data for {participants[0]}/{tasks[0]}")
    except Exception as e:
        print(f"[ERROR] TCT calculation failed: {e}")

# Test parameter metrics
print("\nTesting parameter metrics calculation...")
if participants and tasks:
    try:
        metrics = calculate_parameter_metrics(df, participants[0], tasks[0], "Pupil Diameter")
        if metrics:
            print(f"[OK] Pupil Diameter metrics: mean={metrics.get('mean', 0):.4f}, std={metrics.get('std', 0):.4f}")
        else:
            print(f"[WARNING] No metrics for Pupil Diameter")
    except Exception as e:
        print(f"[ERROR] Parameter metrics calculation failed: {e}")

# Test aggregation (with sample data)
print("\nTesting data aggregation...")
try:
    # Set up state
    state.df = df
    state.participants_cache = participants
    state.tasks_cache = tasks
    
    # Create a simple group
    state.participant_groups = {"G1": participants[:min(2, len(participants))]}
    state.group_names = {"G1": "Test Group"}
    
    # Test aggregation
    aggregated = aggregate_by_groups(
        df,
        ["G1"],
        tasks[:min(2, len(tasks))],
        [],
        "Only group mean"
    )
    
    if aggregated:
        print(f"[OK] Aggregation successful: {len(aggregated)} groups")
    else:
        print("[WARNING] No aggregated data returned")
except Exception as e:
    print(f"[ERROR] Aggregation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("All basic functionality tests completed!")
print("="*50)
print("\nThis script tests core logic only (imports, validation, aggregation).")
print("To run the application with GUI: python main.py")
