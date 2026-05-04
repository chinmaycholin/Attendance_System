@echo off
echo Starting Smart Attendance Dashboard...
if exist "packages\Scripts\python.exe" (
    packages\Scripts\python.exe app.py
) else (
    echo [ERROR] The 'packages' folder was not found. Please run 'python setup.py' first!
    pause
)
