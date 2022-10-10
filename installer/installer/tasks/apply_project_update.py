from os import path
import shutil

from installer import app

def run():
    is_developer_mode = app.config.get('is_developer_mode', False)
    if is_developer_mode:
        return

    installer_src_path = path.join(app.project_repo_dir_path, 'installer')
    ui_src_path = path.join(app.project_repo_dir_path, 'ui')
    engine_src_path = path.join(app.project_repo_dir_path, 'engine')

    start_cmd_src_path = path.join(app.project_repo_dir_path, app.START_CMD_FILE_NAME)
    start_cmd_dst_path = path.join(app.SD_BASE_DIR, app.START_CMD_FILE_NAME)

    dev_console_cmd_src_path = path.join(app.project_repo_dir_path, app.DEV_CONSOLE_CMD_FILE_NAME)
    dev_console_cmd_dst_path = path.join(app.SD_BASE_DIR, app.DEV_CONSOLE_CMD_FILE_NAME)

    shutil.rmtree(app.installer_dir_path, ignore_errors=True)
    shutil.rmtree(app.ui_dir_path, ignore_errors=True)
    shutil.rmtree(app.engine_dir_path, ignore_errors=True)

    shutil.copytree(installer_src_path, app.installer_dir_path, dirs_exist_ok=True)
    shutil.copytree(ui_src_path, app.ui_dir_path, dirs_exist_ok=True)
    shutil.copytree(engine_src_path, app.engine_dir_path, dirs_exist_ok=True)

    shutil.copy(start_cmd_src_path, start_cmd_dst_path)
    shutil.copy(dev_console_cmd_src_path, dev_console_cmd_dst_path)
