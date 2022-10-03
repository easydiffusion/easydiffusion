@echo off

set suggested_dir=%~d0\stable-diffusion-ui

echo "Please install Stable Diffusion UI at the root of your drive. This avoids problems with path length limits in Windows." & echo.
set /p answer="Press Enter to install at %suggested_dir%, or type 'c' (without quotes) to install at the current location (press enter or type 'c'): "

if /i "%answer:~,1%" NEQ "c" (
    if exist "%suggested_dir%" (
        echo. & echo "Sorry, %suggested_dir% already exists! Cannot overwrite that folder!" & echo.
        pause
        exit
    )

    xcopy "%SD_BASE_DIR%" "%suggested_dir%" /s /i /Y /Q
    echo Please run the %START_CMD_FILENAME% file inside %suggested_dir% . Do not use this folder anymore > "%SD_BASE_DIR%/READ_ME - DO_NOT_USE_THIS_FOLDER.txt"

    cd %suggested_dir%
)