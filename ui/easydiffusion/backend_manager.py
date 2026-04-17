import os
import ast
import sys
import importlib.util
import traceback

from easydiffusion.utils import log

backend = None
curr_backend_name = None


def is_valid_backend(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        node = ast.parse(file.read())

    # Check for presence of a dictionary named 'ed_info'
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "ed_info":
                    return True
    return False


def find_valid_backends(root_dir) -> dict:
    backends_path = os.path.join(root_dir, "backends")
    valid_backends = {}

    if not os.path.exists(backends_path):
        return valid_backends

    for item in os.listdir(backends_path):
        item_path = os.path.join(backends_path, item)

        if os.path.isdir(item_path):
            init_file = os.path.join(item_path, "__init__.py")
            if os.path.exists(init_file) and is_valid_backend(init_file):
                valid_backends[item] = item_path
        elif item.endswith(".py"):
            if is_valid_backend(item_path):
                backend_name = os.path.splitext(item)[0]  # strip the .py extension
                valid_backends[backend_name] = item_path

    return valid_backends


def load_backend_module(backend_name, backend_dict):
    if backend_name not in backend_dict:
        raise ValueError(f"Backend '{backend_name}' not found.")

    module_path = backend_dict[backend_name]

    mod_dir = os.path.dirname(module_path)

    sys.path.insert(0, mod_dir)

    # If it's a package (directory), add its parent directory to sys.path
    if os.path.isdir(module_path):
        module_path = os.path.join(module_path, "__init__.py")

    spec = importlib.util.spec_from_file_location(backend_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if mod_dir in sys.path:
        sys.path.remove(mod_dir)

    log.info(f"Loaded backend: {module}")

    return module


def start_backend():
    global backend, curr_backend_name

    from easydiffusion.app import getConfig, ROOT_DIR

    curr_dir = os.path.dirname(__file__)

    backends = find_valid_backends(curr_dir)
    plugin_backends = find_valid_backends(ROOT_DIR)
    backends.update(plugin_backends)

    config = getConfig()
    backend_name = config["backend"]

    if backend_name not in backends:
        raise RuntimeError(
            f"Couldn't find the backend configured in config.yaml: {backend_name}. Please check the name!"
        )

    if backend is not None and backend_name != curr_backend_name:
        try:
            backend.stop_backend()
        except:
            log.exception(traceback.format_exc())

    log.info(f"Loading backend: {backend_name}")
    backend = load_backend_module(backend_name, backends)

    try:
        backend.start_backend()
    except:
        log.exception(traceback.format_exc())
