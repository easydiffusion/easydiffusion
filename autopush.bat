@echo off
set /p commit_message="Enter a commit message: "

echo.
echo Committing with message: "%commit_message%"

git pull
git add .
git commit -m "%commit_message%"
git push

echo.
echo Done!
pause
