# Eye Tracking Analysis Tool - Software Architecture

## Overview

The Eye Tracking Analysis Tool is a PyQt6-based desktop application for analyzing eye tracking data to compare cognitive load across different tasks. The software follows a modular architecture with clear separation of concerns.

## Architecture Diagram

See `software_architecture_diagram.png` for a visual representation of the system architecture.

## Component Overview

### 1. Entry Point (`main.py`)
- **Purpose**: Application entry point
- **Responsibilities**:
  - Initialize PyQt6 application
  - Create and display main window
  - Handle PyQt6 environment errors

### 2. Main Window (`main_window.py`)
- **Purpose**: Primary user interface
- **Responsibilities**:
  - File loading and validation UI
  - Participant and task grouping dialogs
  - Configuration UI (parameter selection, weighting, deselection)
  - Results display trigger
  - Executive summary generation trigger
- **Key Features**:
  - TSV file loading with validation
  - Participant grouping interface
  - Task naming/labeling interface
  - Parameter weighting (0-300% via sliders)
  - Parameter deselection
  - Result domain selection (Rank, Radar Chart, TCT, Statistics)
  - Mode selection (group mean, individual participants, both)

### 3. State Management (`state.py`)
- **Purpose**: Centralized application state
- **Responsibilities**:
  - Singleton pattern for global state
  - Data storage (raw DataFrame, normalized DataFrame)
  - Participant and task caches
  - Group definitions (participants and tasks)
  - Task labels
  - Parameter weights
- **Key Data Structures**:
  - `df`: Raw TSV data
  - `normalized_df`: Baseline-normalized data
  - `participant_groups`: Group ID → participant list mapping
  - `task_labels`: Task ID → human-readable label mapping
  - `parameter_weights`: Parameter name → weight (0.0-3.0) mapping

### 4. Data Processing (`data_processor.py`)
- **Purpose**: TSV validation and data extraction
- **Responsibilities**:
  - TSV format validation (columns, structure)
  - Participant extraction from "Participant" column
  - Task extraction from "TOI" column (suffix after last underscore)
  - Baseline normalization (per participant, using task 0a/0b)
- **Key Functions**:
  - `validate_tsv_format()`: Validates TSV structure
  - `extract_participants()`: Extracts unique participants
  - `extract_tasks_from_toi()`: Extracts task IDs from TOI column
  - `normalize_by_participant_baseline()`: Normalizes data by baseline task

### 5. Analysis (`analysis.py`)
- **Purpose**: Data aggregation and statistical analysis
- **Responsibilities**:
  - Parameter metric calculation (mean, std, median, min, max, Q1, Q3)
  - Task Completion Time (TCT) calculation
  - Data aggregation by groups and tasks
  - Ranking calculations (normalized rankings per group/participant)
  - Radar chart data normalization
  - Statistics table generation
- **Key Functions**:
  - `calculate_tct()`: Calculates task completion time
  - `calculate_parameter_metrics()`: Calculates full statistics for a parameter
  - `aggregate_by_groups()`: Aggregates data by participant groups
  - `aggregate_all_participants()`: Aggregates all participants (for exec summary)
  - `calculate_normalized_rankings_per_group()`: Calculates task rankings per group
  - `calculate_normalized_rankings_per_participant()`: Calculates rankings per participant
  - `normalize_for_radar()`: Normalizes values for radar chart display
  - `generate_statistics_table()`: Creates comprehensive statistics DataFrame

### 6. Results Window (`results_window.py`)
- **Purpose**: Visualization and results display
- **Responsibilities**:
  - Multi-tab interface (Rankings, Radar Charts, TCT Chart, Statistics)
  - Chart generation (radar charts, TCT bar charts)
  - Table display (rankings, statistics)
  - Export functionality (CSV, PNG)
- **Key Features**:
  - Rankings table with indications/contraindications
  - Radar charts (individual task comparisons, aggregated views)
  - Task Completion Time bar charts
  - Statistics table with full parameter statistics
  - Export dialogs for CSV and PNG files

### 7. Executive Summary (`executive_summary.py`, `executive_summary_latex.py`)
- **Purpose**: Generate executive summary reports
- **Responsibilities**:
  - Text summary generation
  - LaTeX PDF generation
  - Statistics formatting for reports
- **Key Functions**:
  - `generate_executive_summary()`: Creates text summary
  - `generate_latex_summary()`: Creates LaTeX PDF via MikTeX
  - `format_statistics_table_for_summary()`: Formats statistics for inclusion

### 8. Dialogs (`dialogs.py`)
- **Purpose**: User interaction dialogs
- **Responsibilities**:
  - Participant grouping dialog
  - Task naming/labeling dialog
  - Loading progress dialog
- **Key Dialogs**:
  - `GroupParticipantsDialog`: Multi-select dialog for grouping participants
  - `GroupTasksDialog`: Task naming and labeling interface
  - `LoadingDialog`: Progress indicator for long operations

### 9. Models (`models.py`)
- **Purpose**: Data models and constants
- **Responsibilities**:
  - Expected TSV column definitions
  - Parameter options and mappings
  - Validation result data class
- **Key Constants**:
  - `EXPECTED_COLUMNS`: Required TSV column names
  - `PARAMETER_OPTIONS`: Available analysis parameters
  - `PARAMETER_COLUMN_MAP`: Parameter name → data column mapping

## Data Flow

### 1. File Loading Flow
```
User selects TSV file
  → validate_tsv_format() [data_processor]
  → Load into pandas DataFrame
  → Store in state.df
  → extract_participants() [data_processor]
  → extract_tasks_from_toi() [data_processor]
  → Update UI with participants and tasks
```

### 2. Grouping Flow
```
User clicks "Group Participants"
  → GroupParticipantsDialog [dialogs]
  → User selects participants and creates groups
  → Store in state.participant_groups
  → Refresh UI checkboxes

User clicks "Name Tasks"
  → GroupTasksDialog [dialogs]
  → User assigns labels to tasks
  → Store in state.task_labels
  → Refresh UI task display
```

### 3. Analysis Flow
```
User clicks "Show Results"
  → Collect UI selections (groups, tasks, parameters, weights, mode)
  → aggregate_by_groups() [analysis]
    → For each group/task/participant:
      → calculate_parameter_metrics() [analysis]
      → Apply parameter weights
      → Calculate TCT and TCT std
  → Create ResultsWindow [results_window]
    → Calculate rankings
    → Generate radar charts
    → Generate TCT charts
    → Generate statistics table
    → Display in tabs
```

### 4. Executive Summary Flow
```
User clicks "Print Executive Summary"
  → Check for MikTeX installation
  → aggregate_all_participants() [analysis] (ignores groups)
  → generate_latex_summary() [executive_summary_latex]
    → Generate LaTeX template
    → Run pdflatex
    → Save PDF to output directory
  → Optionally open PDF
```

## Key Design Patterns

1. **Singleton Pattern**: `AppState` class ensures single instance for global state
2. **Separation of Concerns**: UI, processing, and analysis are clearly separated
3. **Data-Driven**: Configuration stored in state, not hardcoded
4. **Modular Design**: Each module has a single, well-defined responsibility

## Parameter System

The application analyzes the following parameters:
- Task Completion Time (TCT) - calculated from Bin_duration
- Standard Deviation of TCT - group-level statistic
- Pupil Diameter - from Average_pupil_diameter column
- Saccade Velocity - from Average_velocity column
- Peak Saccade Velocity - from Peak_velocity column
- Saccade Amplitude - from Saccade_amplitude column

Each parameter can be:
- Weighted (0-300% via sliders)
- Deselected (excluded from analysis)
- Used in rankings and visualizations

## Output Structure

Results are saved to timestamped directories:
```
output/
  YYYY-MM-DD_HH-MM-SS/
    rankings.csv
    statistics.csv
    radar_chart_*.png
    tct_chart.png
    all_charts_combined.png
```

Executive summaries are saved to:
```
output/
  exec summary - YYYY-MM-DD_HH-MM-SS/
    executive_summary.pdf
    executive_summary.tex
    tct_chart.pdf
    radar_chart.pdf (if applicable)
```
