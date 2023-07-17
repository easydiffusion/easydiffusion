@echo off
set /p commit_message="Enter a commit message: "

echo.
echo Committing with message: "%commit_message%"

git add .
git commit -m "%commit_message%"

echo.
echo Pushing to remote...

echo.
echo Done!
pause
