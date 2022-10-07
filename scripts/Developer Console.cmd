@echo off

echo "Opening Stable Diffusion UI - Developer Console.." & echo.

@call installer\Scripts\activate.bat

@call conda-unpack

@call conda --version
@call git --version

@call conda activate .\stable-diffusion\env

cmd /k