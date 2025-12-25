#!/usr/bin/env python3
"""
Eye Tracking Analysis Tool

A software tool to analyze eye tracking data to compare different tasks
with each other in regards to their cognitive load.

Main entry point.
"""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from main_window import MainWindow


def main() -> int:
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Eye Tracking Analysis Tool")
    
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
