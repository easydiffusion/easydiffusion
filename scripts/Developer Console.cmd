@echo off

echo "Opening Stable Diffusion UI - Developer Console.." & echo.

cd /d %~dp0

set PATH=C:\Windows\System32;C:\Windows\System32\WindowsPowerShell\v1.0;%PATH%

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
echo COMSPEC=%COMSPEC%
echo.
powershell -Command "(Get-WmiObject Win32_VideoController | Select-Object Name, AdapterRAM, DriverDate, DriverVersion)"

@rem activate the legacy environment (if present) and set PYTHONPATH
if exist "installer_files\env" (
    set PYTHONPATH=%cd%\installer_files\env\lib\site-packages
    set PYTHON=%cd%\installer_files\env\python.exe
    echo PYTHON=%PYTHON%
)
if exist "stable-diffusion\env" (
    call conda activate .\stable-diffusion\env
    set PYTHONPATH=%cd%\stable-diffusion\env\lib\site-packages
    set PYTHON=%cd%\stable-diffusion\env\python.exe
    echo PYTHON=%PYTHON%
)

@REM call where python
call "%PYTHON%" --version

echo PYTHONPATH=%PYTHONPATH%

if exist "%cd%\profile" (
    set HF_HOME=%cd%\profile\.cache\huggingface
)

@rem done
echo.

cmd /k
