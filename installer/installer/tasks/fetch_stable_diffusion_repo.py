from os import path

from installer import app, helpers

patch_file_names = [
    'sd_custom.patch',
]

stable_diffusion_repo_git_path = path.join(app.stable_diffusion_repo_dir_path, '.git')
patches_dir_path = path.join(app.installer_dir_path, 'patches')

is_developer_mode = app.config.get('is_developer_mode', False)

def run():
    fetch_repo()
    apply_patches()

def fetch_repo():
    commit_id = app.config.get('stable_diffusion_commit', app.DEFAULT_STABLE_DIFFUSION_COMMIT)

    if path.exists(stable_diffusion_repo_git_path):
        helpers.log(f"Stable Diffusion's git repository was already installed. Using commit: {commit_id}..")

        if not is_developer_mode:
            helpers.run("git reset --hard", run_in_folder=app.stable_diffusion_repo_dir_path)
            helpers.run("git fetch origin", run_in_folder=app.stable_diffusion_repo_dir_path)
            helpers.run(f'git -c advice.detachedHead=false checkout "{commit_id}"', run_in_folder=app.stable_diffusion_repo_dir_path)
    else:
        helpers.log("\nDownloading Stable Diffusion..\n")
        helpers.log(f"Using commit: {commit_id}\n")

        if helpers.run(f'git clone {app.STABLE_DIFFUSION_REPO_URL} "{app.stable_diffusion_repo_dir_path}"')

        if path.exists(stable_diffusion_repo_git_path):
            helpers.log("Downloaded Stable Diffusion")
        else:
            helpers.fail_with_install_error(error_msg="Could not download Stable Diffusion")

        helpers.run(f'git -c advice.detachedHead=false checkout "{commit_id}"', run_in_folder=app.stable_diffusion_repo_dir_path)

def apply_patches():
    if is_developer_mode:
        return

    for patch_file_name in patch_file_names:
        patch_file_path = path.join(patches_dir_path, patch_file_name)
        helpers.run(f"git apply {patch_file_path}", run_in_folder=app.stable_diffusion_repo_dir_path)
