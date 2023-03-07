@echo off
setlocal enabledelayedexpansion

@rem This script will install git and conda (if not found on the PATH variable)
@rem  using micromamba (an 8mb static-linked single-file binary, conda replacement).
@rem For users who already have git and conda, this step will be skipped.

@rem This enables a user to install this project without manually installing conda and git.

@rem config
set MAMBA_ROOT_PREFIX=%cd%\installer_files\mamba
set INSTALL_ENV_DIR=%cd%\installer_files\env
set LEGACY_INSTALL_ENV_DIR=%cd%\installer
set MICROMAMBA_DOWNLOAD_URL=https://github.com/cmdr2/stable-diffusion-ui/releases/download/v1.1/micromamba.exe
set umamba_exists=F

set OLD_APPDATA=%APPDATA%
set OLD_USERPROFILE=%USERPROFILE%
set APPDATA=%cd%\installer_files\appdata
set USERPROFILE=%cd%\profile

@rem figure out whether git and conda needs to be installed
if exist "%INSTALL_ENV_DIR%" set PATH=%INSTALL_ENV_DIR%;%INSTALL_ENV_DIR%\Library\bin;%INSTALL_ENV_DIR%\Scripts;%INSTALL_ENV_DIR%\Library\usr\bin;%PATH%

set PACKAGES_TO_INSTALL=

if not exist "%LEGACY_INSTALL_ENV_DIR%\etc\profile.d\conda.sh" (
    if not exist "%INSTALL_ENV_DIR%\etc\profile.d\conda.sh" set PACKAGES_TO_INSTALL=%PACKAGES_TO_INSTALL% conda python=3.8.5
)

call git --version >.tmp1 2>.tmp2
if "!ERRORLEVEL!" NEQ "0" set PACKAGES_TO_INSTALL=%PACKAGES_TO_INSTALL% git

call "%MAMBA_ROOT_PREFIX%\micromamba.exe" --version >.tmp1 2>.tmp2
if "!ERRORLEVEL!" EQU "0" set umamba_exists=T

@rem (if necessary) install git and conda into a contained environment
if "%PACKAGES_TO_INSTALL%" NEQ "" (
    @rem download micromamba
    if "%umamba_exists%" == "F" (
        echo "Downloading micromamba from %MICROMAMBA_DOWNLOAD_URL% to %MAMBA_ROOT_PREFIX%\micromamba.exe"

        mkdir "%MAMBA_ROOT_PREFIX%"
        call curl -Lk "%MICROMAMBA_DOWNLOAD_URL%" > "%MAMBA_ROOT_PREFIX%\micromamba.exe"

        if "!ERRORLEVEL!" NEQ "0" (
            echo "There was a problem downloading micromamba. Cannot continue."
            pause
            exit /b
        )

        mkdir "%APPDATA%"
        mkdir "%USERPROFILE%"

        @rem test the mamba binary
        echo Micromamba version:
        call "%MAMBA_ROOT_PREFIX%\micromamba.exe" --version
    )

    @rem create the installer env
    if not exist "%INSTALL_ENV_DIR%" (
        call "%MAMBA_ROOT_PREFIX%\micromamba.exe" create -y --prefix "%INSTALL_ENV_DIR%"
    )

    echo "Packages to install:%PACKAGES_TO_INSTALL%"

    call "%MAMBA_ROOT_PREFIX%\micromamba.exe" install -y --prefix "%INSTALL_ENV_DIR%" -c conda-forge %PACKAGES_TO_INSTALL%

    if not exist "%INSTALL_ENV_DIR%" (
        echo "There was a problem while installing%PACKAGES_TO_INSTALL% using micromamba. Cannot continue."
        pause
        exit /b
    )
)

@rem revert to the old APPDATA. only needed it for bypassing a bug in micromamba (with special characters)
set APPDATA=%OLD_APPDATA%
set USERPROFILE=%OLD_USERPROFILE%
