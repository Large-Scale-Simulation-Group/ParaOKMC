@echo off
REM Build script for Windows
%PYTHON% -m pip install . -vv
if errorlevel 1 exit 1
