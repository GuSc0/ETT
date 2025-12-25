#!/usr/bin/env python3
"""
Eye Tracking Analysis Tool

A software tool to analyze eye tracking data to compare different tasks
with each other in regards to their cognitive load.

Main entry point.
"""
from __future__ import annotations

import sys


def _print_pyqt6_env_help(exc: BaseException) -> None:
    msg = f"""
Fehler beim Laden von PyQt6/Qt6 (häufig ein DLL-/Umgebungsproblem unter Windows).

Ursache (typisch):
- Inkompatible PyQt6/Qt6 Wheels
- 32-bit Python auf 64-bit Windows (oder umgekehrt)
- Fehlende Microsoft Visual C++ Runtime (2015–2022, x64)
- Konflikte durch andere Qt-DLLs im PATH (Anaconda/QGIS/alte Qt-Installationen)

Original-Fehler:
{exc!r}

Konkrete Schritte (im aktiven venv ausführen):

1) Architektur prüfen:
   python -c "import platform,sys; print(sys.version); print(platform.architecture()); print(platform.platform())"

2) PyQt6 sauber neu installieren:
   pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
   pip cache purge
   pip install --upgrade pip
   pip install PyQt6

   Falls weiterhin Fehler:
   pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
   pip install "PyQt6==6.7.1"

3) MSVC Runtime installieren/repair:
   Microsoft Visual C++ Redistributable 2015–2022 (x64)

4) PATH-Konflikte prüfen:
   where Qt6Core.dll
   where qwindows.dll

5) Diagnose-Ausgabe:
   python -c "import PyQt6; import PyQt6.QtCore as qc; print(PyQt6.__file__); print(qc.QLibraryInfo.path(qc.QLibraryInfo.LibraryPath.PluginsPath))"
"""
    print(msg.strip(), file=sys.stderr)


try:
    from PyQt6.QtWidgets import QApplication
except Exception as exc:  # noqa: BLE001
    _print_pyqt6_env_help(exc)
    raise

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
