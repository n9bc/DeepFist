@echo off
REM Scheduled W1AW 20m code-practice capture (2100 UTC / 4pm CDT).
REM Stop the band collector so the radio is free, then capture.
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'band_collect|rbn_harvest' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
cd /d C:\dev\DeepFist
.venv\Scripts\python.exe tools\capture_w1aw.py >> "C:\dev\DeepFist\runs\w1aw\capture.log" 2>&1
