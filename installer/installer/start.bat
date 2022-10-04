@echo off
rem Never edit this file. If you really, really have to, beware that a script doesn't like
rem being overwritten while it is running (the auto-updater will do that).
rem The trick is to update this file while another script is running, and vice versa.

call python %SD_BASE_DIR%\installer\installer\main.py

pause