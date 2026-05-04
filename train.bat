@echo off
echo Starting Training Pipeline...
if exist "packages\Scripts\python.exe" (
    packages\Scripts\python.exe train.py
    echo.
    echo Training process finished! Press any key to exit.
    pause >nul
) else (
    echo [ERROR] The 'packages' folder was not found. Please run 'python setup.py' first!
    pause
)
