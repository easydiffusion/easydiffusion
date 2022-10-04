@echo off

echo. & echo "Stable Diffusion UI - v2.5" & echo.

set PATH=C:\Windows\System32;%PATH%

set START_CMD_FILENAME=Start Stable Diffusion UI.cmd
set SD_BASE_DIR=%cd%

@rem Confirm or change the installation dir
call installer\bootstrap\check-install-dir.bat

@rem set the vars again, if the installer dir has changed
set SD_BASE_DIR=%cd%

echo Working in %SD_BASE_DIR%

@rem Setup the packages required for the installer
call installer\bootstrap\bootstrap.bat

@rem Test the bootstrap
call git --version
call python --version

@rem Download the rest of the installer and UI
call python installer\installer\main.py

pause