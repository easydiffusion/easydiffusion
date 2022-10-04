from os import path

from installer import app, helpers

project_repo_git_path = path.join(app.project_repo_dir_path, '.git')

def run():
    branch_name = app.config.get('update_branch', app.DEFAULT_PROJECT_BRANCH)

    if path.exists(project_repo_git_path):
        helpers.log(f"Stable Diffusion UI's git repository was already installed. Updating from {branch_name}..")

        helpers.run("git reset --hard", run_in_folder=app.project_repo_dir_path)
        helpers.run(f'git -c advice.detachedHead=false checkout "{branch_name}"', run_in_folder=app.project_repo_dir_path)
        helpers.run("git pull", run_in_folder=app.project_repo_dir_path)
    else:
        helpers.log("\nDownloading Stable Diffusion UI..\n")
        helpers.log(f"Using the {branch_name} channel\n")

        if helpers.run(f'git clone {app.PROJECT_REPO_URL} "{app.project_repo_dir_path}"'):
            helpers.log("Downloaded Stable Diffusion UI")
        else:
            helpers.show_install_error(error_msg="Could not download Stable Diffusion UI")
            exit(1)

        helpers.run(f'git -c advice.detachedHead=false checkout "{branch_name}"', run_in_folder=app.project_repo_dir_path)
