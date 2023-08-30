@echo off
setlocal enabledelayedexpansion

@echo "Hi there, what you are running is meant for the developers of this project, not for users." & echo.
@echo "If you only want to use Easy Diffusion, you've downloaded the wrong file."
@echo "Please download and follow the instructions at https://github.com/easydiffusion/easydiffusion#installation" & echo.
@echo "If you are actually a developer of this project, please type Y and press enter" & echo.

set /p answer=Are you a developer of this project (Y/N)?
if /i "%answer:~,1%" NEQ "Y" exit /b

@rem verify dependencies
call makensis /VERSION >.tmp1 2>.tmp2
if "!ERRORLEVEL!" NEQ "0" (
    echo makensis.exe not found! Download it from https://sourceforge.net/projects/nsisbi/files/ and set it on the PATH variable.
    pause
    exit
)

set /p OUT_DIR=Output folder path (will create the installer files inside this, e.g. F:\EasyDiffusion): 

mkdir "%OUT_DIR%\scripts"
mkdir "%OUT_DIR%\installer_files"

set BASE_DIR=%cd%

@rem STEP 1: copy the installer files for Windows

copy "%BASE_DIR%\scripts\on_env_start.bat" "%OUT_DIR%\scripts\"
copy "%BASE_DIR%\scripts\config.yaml.sample" "%OUT_DIR%\scripts\config.yaml.sample"
copy "%BASE_DIR%\scripts\Start Stable Diffusion UI.cmd" "%OUT_DIR%\"
copy "%BASE_DIR%\LICENSE" "%OUT_DIR%\"
copy "%BASE_DIR%\CreativeML Open RAIL-M License" "%OUT_DIR%\"
copy "%BASE_DIR%\How to install and run.txt" "%OUT_DIR%\"
copy "%BASE_DIR%\NSIS\cyborg_flower_girl.ico" "%OUT_DIR%\installer_files\"
copy "%BASE_DIR%\NSIS\cyborg_flower_girl.bmp" "%OUT_DIR%\installer_files\"
echo. > "%OUT_DIR%\scripts\install_status.txt"

echo ----
echo Basic files ready. Verify the files in %OUT_DIR%, then press Enter to initialize the environment, or close to quit.
echo ----
pause

@rem STEP 2: Initialize the environment with git, python and conda

cd /d "%OUT_DIR%\"
call "%BASE_DIR%\scripts\bootstrap.bat"

echo ----
echo Environment ready. Verify the environment, then press Enter to download the necessary packages, or close to quit.
echo ----
pause

@rem STEP 3: Download the packages and create a working installation

cd /d "%OUT_DIR%\"
start "Install Easy Diffusion" /D "%OUT_DIR%" "Start Stable Diffusion UI.cmd"

echo ----
echo Installation in progress (in a new window). Once complete, verify the installation, then press Enter to create an installer from these files, or close to quit.
echo ----
pause

@rem STEP 4: Build the installer from a working installation

cd /d "%OUT_DIR%\"

echo ^^!define EXISTING_INSTALLATION_DIR "%OUT_DIR%" > nsisconf.nsh
call makensis /NOCD /V4 "%BASE_DIR%\NSIS\sdui.nsi"

echo ----
if "!ERRORLEVEL!" EQU "0" (
    echo Installer built successfully at %OUT_DIR%
) else (
    echo Installer failed to build at %OUT_DIR%
)
echo ----
pause