@echo. & echo "Stable Diffusion UI - v2" & echo.

@cd ..

@>nul grep -c "sd_ui_git_cloned" scripts\install_status.txt
@if "%ERRORLEVEL%" EQU "0" (
    @echo "Stable Diffusion UI's git repository was already installed. Updating.."

    @cd sd-ui-files

    @call git reset --hard
    @call git pull

    @cd ..
) else (
    @echo. & echo "Downloading Stable Diffusion UI.." & echo.

    @call git clone https://github.com/cmdr2/stable-diffusion-ui.git sd-ui-files && (
        @echo sd_ui_git_cloned >> scripts\install_status.txt
    ) || (
        @echo "Error downloading Stable Diffusion UI. Please try re-running this installer. If it doesn't work, please copy the messages in this window, and ask the community at https://discord.com/invite/u9yhsFmEkB or file an issue at https://github.com/cmdr2/stable-diffusion-ui/issues"
        pause
        @exit /b
    )
)

@xcopy sd-ui-files\ui ui /s /i /Y
@xcopy sd-ui-files\scripts scripts /s /i /Y

@call scripts\on_sd_start.bat
