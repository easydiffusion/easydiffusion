@echo off

set PATH=C:\Windows\System32;%PATH%

@rem set legacy installer's PATH, if it exists
if exist "installer" set PATH=%cd%\installer;%cd%\installer\Library\bin;%cd%\installer\Scripts;%PATH%

@rem Setup the packages required for the installer
call scripts\bootstrap.bat

@rem set new installer's PATH, if it downloaded any packages
if exist "installer_files\env" set PATH=%cd%\installer_files\env;%cd%\installer_files\env\Library\bin;%cd%\installer_files\env\Scripts;%PATH%

@rem Test the bootstrap
call where git
call git --version

call where python
call python --version

call where conda
call conda --version

@rem Download the rest of the installer and UI
call scripts\on_env_start.bat

@pause
