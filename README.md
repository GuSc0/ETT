# Eye Tracking Analysis Tool

A software tool to analyze eye tracking data to compare different tasks with each other in regards to their cognitive load.

## Project Structure

```
ETT/
├── main.py                    # Entry point - run this to start the application
├── main_window.py            # Main application window (PyQt6)
├── results_window.py          # Results display window with charts and tables
├── dialogs.py                 # Dialog windows (MultiSelect, Group Participants, Group Tasks)
├── state.py                   # Global application state management
├── data_processor.py          # TSV validation and data extraction functions
├── models.py                  # Data models and constants
├── analysis.py                # Data analysis and aggregation functions
├── executive_summary.py       # Executive summary text generation
├── executive_summary_latex.py # LaTeX PDF generation for executive summaries
├── executive_summary_template.tex # LaTeX template for PDF generation
├── requirements.txt           # Python dependencies
├── Input/                     # Input data files (TSV format)
└── output/                    # Export output directory (timestamped subfolders)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Features

- Load and validate TSV eye tracking data files
- View detected participants and tasks
- Group participants into custom groups
- Group tasks with custom labels
- Configure result display options
- Deselect specific parameters from analysis

## Changes from Original

1. **Converted from Jupyter Notebook to modular Python project**
   - Separated concerns into logical modules
   - Cleaner code organization

2. **Fixed TSV loading issue**
   - The original code had issues with global variable references not being properly updated
   - Implemented a centralized `AppState` singleton class that properly manages all state
   - UI components now reference the state object directly instead of relying on global variables

3. **Migrated from tkinter to PyQt6**
   - Modern, cross-platform UI framework
   - Better widget support and styling capabilities
   - Cleaner event handling

## Requirements

- Python 3.8+
- PyQt6
- pandas
