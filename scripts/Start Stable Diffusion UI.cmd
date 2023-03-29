@echo off

cd /d %~dp0
echo Install dir: %~dp0

set PATH=C:\Windows\System32;%PATH%

if exist "on_sd_start.bat" (
    echo ================================================================================
    echo.
    echo !!!! WARNING !!!!
    echo.
    echo It looks like you're trying to run the installation script from a source code 
    echo download. This will not work.
    echo.
    echo Recommended: Please close this window and download the installer from
    echo https://stable-diffusion-ui.github.io/docs/installation/
    echo.
    echo ================================================================================
    echo.
    pause
    exit /b
) 

@rem set legacy installer's PATH, if it exists
if exist "installer" set PATH=%cd%\installer;%cd%\installer\Library\bin;%cd%\installer\Scripts;%cd%\installer\Library\usr\bin;%PATH%

@rem set new installer's PATH, if it downloaded any packages
if exist "installer_files\env" set PATH=%cd%\installer_files\env;%cd%\installer_files\env\Library\bin;%cd%\installer_files\env\Scripts;%cd%\installer_files\Library\usr\bin;%PATH%

set PYTHONPATH=%cd%\installer;%cd%\installer_files\env

@rem Test the core requirements
call where git
call git --version

call where conda
call conda --version

@rem Download the rest of the installer and UI
call scripts\on_env_start.bat

@pause
