# Installing MikTeX for LaTeX PDF Generation

The executive summary feature can generate professional PDFs using LaTeX/MikTeX. Follow these steps to install and configure MikTeX.

## Installation Steps

### Windows

1. **Download MikTeX:**
   - Visit https://miktex.org/download
   - Download the Windows installer (Basic or Complete)
   - Run the installer

2. **Installation Options:**
   - Choose "Install for all users" or "Install for current user"
   - Select installation directory (default is recommended)
   - Choose "Install missing packages on-the-fly" (recommended)

3. **Add to PATH (if not automatic):**
   - MikTeX usually adds itself to PATH automatically
   - If `pdflatex` is not found, manually add:
     - `C:\Program Files\MiKTeX\miktex\bin\x64\` (for all users)
     - `C:\Users\<YourUsername>\AppData\Local\Programs\MiKTeX\miktex\bin\x64\` (for current user)

4. **Verify Installation:**
   ```powershell
   pdflatex --version
   ```
   Should show version information.

5. **Restart the Application:**
   - Close and reopen the Eye Tracking Tool
   - The "Export to PDF (LaTeX/MikTeX)" button should now appear

## Troubleshooting

### pdflatex not found

If you get an error that pdflatex is not found:

1. **Check if MikTeX is installed:**
   ```powershell
   Test-Path "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe"
   ```

2. **Add to PATH manually:**
   - Open System Properties → Environment Variables
   - Add MikTeX bin directory to PATH
   - Restart terminal/application

3. **Use full path:**
   The application will try to find pdflatex automatically in common locations.

### LaTeX compilation errors

If PDF generation fails:

1. Check the error message in the dialog
2. Ensure all required LaTeX packages are installed:
   - MikTeX should install packages automatically
   - If not, run: `miktex packages install <package-name>`

3. Check the log file:
   - In the executive summary output folder (e.g. `output/exec summary - YYYY-MM-DD_HH-MM-SS/`), check the `.log` file produced by pdflatex.

## Alternative: Use matplotlib PDF export

If MikTeX installation is problematic, you can use the matplotlib-based PDF export (button: "Export to PDF (matplotlib)") which doesn't require LaTeX.
