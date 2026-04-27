@echo off
REM ============================================================
REM PRED-BiPN Server Launcher
REM Works on any Windows machine - no pre-installed software needed
REM ============================================================

REM Force UTF-8 code page so string parsing works on any language Windows
chcp 65001 >nul 2>&1

setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ============================================
echo  PRED-BiPN Bipartite Network Analysis Tool
echo ============================================
echo.

REM Store root dir for absolute references (cd /d may change later)
set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
set "VENV_PYTHON=%ROOT_DIR%\.venv\Scripts\python.exe"

REM --- Validate venv is functional (not just copied from another machine) ---
set "VENV_OK=0"
if exist "%VENV_PYTHON%" (
    "%VENV_PYTHON%" --version >nul 2>&1
    if !errorlevel! equ 0 set "VENV_OK=1"
)
if "%VENV_OK%"=="0" (
    if exist ".venv" (
        echo [INFO] Existing virtual environment is invalid or from another machine. Rebuilding...
        echo.
        rmdir /s /q .venv
    )
)
if "%VENV_OK%"=="1" goto :launch

REM ============================================================
REM STEP 1 - Find a working Python 3 interpreter
REM ============================================================
set "PYTHON_EXE="
call :find_python
if defined PYTHON_EXE goto :have_python

REM ============================================================
REM STEP 2 - No Python found; install it automatically
REM ============================================================
echo [INFO] Python 3 not found on this machine.
echo [INFO] Installing Python automatically (no admin required)...
echo.

REM --- 2a. Try winget (ships with Windows 10 1709+ / Windows 11) ---
where winget >nul 2>&1
if !errorlevel! equ 0 (
    echo [INFO] Installing via winget ...
    winget install Python.Python.3.11 --accept-package-agreements --accept-source-agreements --silent
    if !errorlevel! equ 0 (
        call :refresh_path
        call :find_python
        if defined PYTHON_EXE goto :have_python
    )
    echo [WARN] winget install did not succeed, falling back to direct download.
    echo.
)

REM --- 2b. Download installer from python.org using PowerShell (built into Win 7+) ---
echo [INFO] Downloading Python 3.11 from python.org ...
set "INSTALLER=%TEMP%\python-3.11.9-amd64.exe"
powershell -NoProfile -Command ^
  "[Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%INSTALLER%' -UseBasicParsing"
if not exist "%INSTALLER%" (
    echo.
    echo [ERROR] Download failed. Please install Python 3.11+ manually:
    echo         https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo [INFO] Running Python installer (this may take a minute) ...
"%INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_doc=0
del "%INSTALLER%" >nul 2>&1

call :refresh_path
call :find_python
if defined PYTHON_EXE goto :have_python

echo.
echo [ERROR] Python was installed but cannot be found in this session.
echo         Please close this window, open a NEW terminal, and run this script again.
pause
exit /b 1

REM ============================================================
REM STEP 3 - Create venv and install dependencies
REM ============================================================
:have_python
echo [OK]   Using Python: %PYTHON_EXE%
echo.

echo [INFO] Creating virtual environment ...
"%PYTHON_EXE%" -m venv .venv
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo [INFO] Installing dependencies (this may take a few minutes on first run) ...
"%VENV_PYTHON%" -m pip install --upgrade pip >nul 2>&1
"%VENV_PYTHON%" -m pip install -r backend\requirements.txt
if !errorlevel! neq 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
echo [OK]   Dependencies installed.
echo.

REM ============================================================
REM STEP 4 - Launch the application
REM ============================================================
:launch
if not exist "backend\tmp" mkdir "backend\tmp"

echo [OK]   Starting server ...
echo.
echo   URL:  http://localhost:5050
echo   Press Ctrl+C to stop the server
echo.

REM Open default browser after a short delay
start "" /b cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:5050"

REM Use venv python directly - avoids PATH/Store stub issues
REM PYTHONUTF8=1 ensures correct encoding on non-English Windows (e.g. Greek)
set "PYTHONUTF8=1"
cd "%ROOT_DIR%\backend"
"%VENV_PYTHON%" app.py
pause
exit /b 0

REM ============================================================
REM Helper: Search for a real Python 3 interpreter
REM ============================================================
:find_python
set "PYTHON_EXE="

REM -- Check common install locations first (avoids the Microsoft Store stub) --
for /d %%d in ("%LOCALAPPDATA%\Programs\Python\Python3*") do (
    if exist "%%d\python.exe" (
        set "PYTHON_EXE=%%d\python.exe"
        exit /b 0
    )
)
for /d %%d in ("%ProgramFiles%\Python3*") do (
    if exist "%%d\python.exe" (
        set "PYTHON_EXE=%%d\python.exe"
        exit /b 0
    )
)

REM -- Try python on PATH, but verify it is real (not the Store stub) --
where python >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%p in ('where python 2^>nul') do (
        REM Skip the WindowsApps stub
        echo %%p | findstr /i "WindowsApps" >nul 2>&1
        if !errorlevel! neq 0 (
            "%%p" --version >nul 2>&1
            if !errorlevel! equ 0 (
                for /f "tokens=2 delims= " %%v in ('"%%p" --version 2^>^&1') do (
                    echo %%v | findstr /b "3." >nul 2>&1
                    if !errorlevel! equ 0 (
                        set "PYTHON_EXE=%%p"
                        exit /b 0
                    )
                )
            )
        )
    )
)

REM -- Also try python3 on PATH --
where python3 >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%p in ('where python3 2^>nul') do (
        echo %%p | findstr /i "WindowsApps" >nul 2>&1
        if !errorlevel! neq 0 (
            "%%p" --version >nul 2>&1
            if !errorlevel! equ 0 (
                set "PYTHON_EXE=%%p"
                exit /b 0
            )
        )
    )
)
exit /b 1

REM ============================================================
REM Helper: Reload PATH from the registry (picks up new installs)
REM ============================================================
:refresh_path
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%b"
set "PATH=!SYS_PATH!;!USER_PATH!"
exit /b 0

