@echo off

set SD_BASE_DIR=%cd%
set MAMBA_ROOT_PREFIX=%SD_BASE_DIR%\env\mamba
set INSTALL_ENV_DIR=%SD_BASE_DIR%\env\installer_env

call "%MAMBA_ROOT_PREFIX%\condabin\mamba_hook.bat"

call micromamba activate "%INSTALL_ENV_DIR%"

cmd /k