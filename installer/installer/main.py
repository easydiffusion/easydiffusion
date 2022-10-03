import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from installer.tasks import (
    fetch_project_repo,
    apply_project_update,
)

tasks = [
    fetch_project_repo,
    apply_project_update,
]

def run_tasks():
    for task in tasks:
        task.run()

run_tasks()
