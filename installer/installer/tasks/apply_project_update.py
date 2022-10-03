from os import path
import shutil

from installer import app, helpers

def run():
    is_developer_mode = app.config.get('is_developer_mode', False)
    if not is_developer_mode:
        # @xcopy sd-ui-files\ui ui /s /i /Y
        # @copy sd-ui-files\scripts\on_sd_start.bat scripts\ /Y
        # @copy "sd-ui-files\scripts\Start Stable Diffusion UI.cmd" . /Y

        installer_src_path = path.join(app.project_repo_dir_path, 'installer')
        ui_src_path = path.join(app.project_repo_dir_path, 'ui')
        engine_src_path = path.join(app.project_repo_dir_path, 'engine')

        shutil.copytree(ui_src_path, app.ui_dir_path, dirs_exist_ok=True)
