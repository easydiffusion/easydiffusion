import importlib.util
import pathlib
import unittest


def _load_check_modules():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    module_path = repo_root / "scripts" / "check_modules.py"

    spec = importlib.util.spec_from_file_location("check_modules", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestBackendConfigEnvOverrides(unittest.TestCase):
    def test_commandline_args_string(self):
        check_modules = _load_check_modules()
        env = {}

        check_modules.apply_backend_config_env_overrides({"COMMANDLINE_ARGS": "--opt-split-attention"}, env=env)

        self.assertEqual(env.get("COMMANDLINE_ARGS"), "--opt-split-attention")

    def test_commandline_args_list(self):
        check_modules = _load_check_modules()
        env = {}

        check_modules.apply_backend_config_env_overrides(
            {"COMMANDLINE_ARGS": ["--opt-split-attention", "--foo=bar"]}, env=env
        )

        self.assertEqual(env.get("COMMANDLINE_ARGS"), "--opt-split-attention --foo=bar")


if __name__ == "__main__":
    unittest.main()

