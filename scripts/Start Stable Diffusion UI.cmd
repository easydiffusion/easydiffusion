@echo off

@REM Delete the post-activate hook from the old installer
@if exist "installer\etc\conda\activate.d\post_activate.bat" (
    del "installer\etc\conda\activate.d\post_activate.bat"
)

@call installer\Scripts\activate.bat

@call conda-unpack

@call conda --version
@call git --version

@cd installer

@call scripts\on_env_start.bat

@pause