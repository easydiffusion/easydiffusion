import os

from installer import app, helpers

def run():
    helpers.log("\nStable Diffusion is ready!\n")

    env = os.environ.copy()
    env['SD_DIR'] = app.stable_diffusion_repo_dir_path
    env['PYTHONPATH'] = app.stable_diffusion_repo_dir_path + ';' + os.path.join(app.project_env_dir_path, 'lib', 'site-packages')
    env['SD_UI_PATH'] = app.ui_dir_path

    helpers.log(f'PYTHONPATH={env["PYTHONPATH"]}')
    helpers.run('python --version', log_the_cmd=True)

    host = app.config.get('host', 'localhost')
    port = app.config.get('port', '9000')

    ui_server_cmd = f'uvicorn server:app --app-dir "{app.ui_dir_path}" --port {port} --host {host}'

    helpers.run(ui_server_cmd, run_in_folder=app.stable_diffusion_repo_dir_path, log_the_cmd=True, env=env)
