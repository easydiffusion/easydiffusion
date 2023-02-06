@echo off

echo "Opening Stable Diffusion UI - Developer Console.." & echo.

set PATH=C:\Windows\System32;%PATH%

@rem set legacy and new installer's PATH, if they exist
if exist "installer" set PATH=%cd%\installer;%cd%\installer\Library\bin;%cd%\installer\Scripts;%cd%\installer\Library\usr\bin;%PATH%
if exist "installer_files\env" set PATH=%cd%\installer_files\env;%cd%\installer_files\env\Library\bin;%cd%\installer_files\env\Scripts;%cd%\installer_files\Library\usr\bin;%PATH%

set PYTHONPATH=%cd%\installer;%cd%\installer_files\env

@rem activate the installer env
call conda activate

@rem Test the environment
echo "Environment Info:"
call where git
call git --version

call where conda
call conda --version

echo.

@rem activate the legacy environment (if present) and set PYTHONPATH
if exist "installer_files\env" (
    set PYTHONPATH=%cd%\installer_files\env\lib\site-packages
)
if exist "stable-diffusion\env" (
    call conda activate .\stable-diffusion\env
    set PYTHONPATH=%cd%\stable-diffusion\env\lib\site-packages
)

call where python
call python --version

echo PYTHONPATH=%PYTHONPATH%

@rem done
echo.

cmd /k
