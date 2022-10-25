@echo off

echo "Opening Stable Diffusion UI - Developer Console.." & echo.

set PATH=C:\Windows\System32;%PATH%

@rem set legacy and new installer's PATH, if they exist
if exist "installer" set PATH=%cd%\installer;%cd%\installer\Library\bin;%cd%\installer\Scripts;%cd%\installer\Library\usr\bin;%PATH%
if exist "installer_files\env" set PATH=%cd%\installer_files\env;%cd%\installer_files\env\Library\bin;%cd%\installer_files\env\Scripts;%cd%\installer_files\Library\usr\bin;%PATH%

@rem activate the installer env
call conda activate

@rem Test the environment
echo "Environment Info:"
call where git
call git --version

call where python
call python --version

call where conda
call conda --version

echo.

@rem activate the environment
call conda activate .\stable-diffusion\env

cmd /k
