import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from installer import helpers
from installer.tasks import (
    fetch_project_repo,
    apply_project_update,
    fetch_stable_diffusion_repo,
)

tasks = [
    fetch_project_repo,
    apply_project_update,
    fetch_stable_diffusion_repo,
]

helpers.log(f'Starting Stable Diffusion UI at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')

def run_tasks():
    for task in tasks:
        task.run()

run_tasks()
