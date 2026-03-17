import importlib.util
import pathlib
import sys
import types
import unittest


_MOCKED_MODULES = (
    "sdkit",
    "sdkit.utils",
    "torchruntime",
    "torchruntime.utils",
    "easydiffusion",
    "easydiffusion.app",
    "easydiffusion.model_manager",
    "common",
)


def _install_mock_dependencies():
    original_modules = {name: sys.modules.get(name) for name in _MOCKED_MODULES}

    sdkit = types.ModuleType("sdkit")
    sdkit_utils = types.ModuleType("sdkit.utils")
    sdkit_utils.base64_str_to_img = lambda img: img
    sdkit_utils.img_to_base64_str = lambda img: img
    sdkit_utils.log = types.SimpleNamespace(info=lambda *args, **kwargs: None)
    sdkit.utils = sdkit_utils

    torchruntime = types.ModuleType("torchruntime")
    torchruntime_utils = types.ModuleType("torchruntime.utils")
    torchruntime_utils.get_device = lambda *args, **kwargs: "cpu"
    torchruntime.utils = torchruntime_utils

    easydiffusion = types.ModuleType("easydiffusion")
    easydiffusion_app = types.ModuleType("easydiffusion.app")
    easydiffusion_app.getConfig = lambda: {}
    easydiffusion_model_manager = types.ModuleType("easydiffusion.model_manager")
    easydiffusion_model_manager.get_model_dirs = lambda: {}
    easydiffusion.app = easydiffusion_app
    easydiffusion.model_manager = easydiffusion_model_manager

    common = types.ModuleType("common")
    common.kill = lambda *args, **kwargs: None

    sys.modules.update(
        {
            "sdkit": sdkit,
            "sdkit.utils": sdkit_utils,
            "torchruntime": torchruntime,
            "torchruntime.utils": torchruntime_utils,
            "easydiffusion": easydiffusion,
            "easydiffusion.app": easydiffusion_app,
            "easydiffusion.model_manager": easydiffusion_model_manager,
            "common": common,
        }
    )
    return original_modules


def _restore_mock_dependencies(original_modules):
    for name, module in original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_webui_common():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    module_path = repo_root / "ui" / "easydiffusion" / "backends" / "webui_common.py"

    spec = importlib.util.spec_from_file_location("issue1992_webui_common", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.webui_opts = {}
    return module


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = repr(payload)

    def json(self):
        return self._payload


class TestWebuiFaceFilterPayloads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._original_modules = _install_mock_dependencies()
        cls.webui_common = _load_webui_common()

    @classmethod
    def tearDownClass(cls):
        _restore_mock_dependencies(cls._original_modules)
        sys.modules.pop("issue1992_webui_common", None)

    def setUp(self):
        self.webui_common.webui_opts = {}

    def test_face_only_filters_do_not_trigger_implicit_upscale(self):
        test_cases = (
            ("gfpgan", ["gfpgan"], {"gfpgan": {}}, {"gfpgan_visibility": 1}),
            (
                "codeformer",
                ["codeformer"],
                {"codeformer": {"codeformer_fidelity": 0.35}},
                {"codeformer_visibility": 1, "codeformer_weight": 0.35},
            ),
        )

        for name, filters, filter_params, expected_payload in test_cases:
            with self.subTest(filter_name=name):
                payloads = []

                def fake_post(uri, json=None, **kwargs):
                    self.assertEqual(uri, "/sdapi/v1/extra-batch-images")
                    payloads.append(json)
                    if json["upscaler_1"] == "None" and json["upscaling_resize"] > 1:
                        return _FakeResponse(500, {"detail": "None.pth"})
                    return _FakeResponse(200, {"images": ["filtered-face"]})

                self.webui_common.webui_post = fake_post

                images = self.webui_common.filter_images(
                    None,
                    ["input-image"],
                    filters,
                    filter_params,
                    input_type="base64",
                )

                self.assertEqual(images, ["data:image/jpeg;base64,filtered-face"])
                self.assertEqual(len(payloads), 1)
                self.assertEqual(payloads[0]["upscaling_resize"], 1)
                self.assertEqual(payloads[0]["upscaler_1"], "None")
                for key, value in expected_payload.items():
                    self.assertEqual(payloads[0][key], value)

    def test_explicit_upscale_settings_are_still_forwarded(self):
        payloads = []

        def fake_post(uri, json=None, **kwargs):
            self.assertEqual(uri, "/sdapi/v1/extra-batch-images")
            payloads.append(json)
            return _FakeResponse(200, {"images": ["upscaled-image"]})

        self.webui_common.webui_post = fake_post

        images = self.webui_common.filter_images(
            None,
            ["input-image"],
            ["gfpgan", "realesrgan"],
            {"gfpgan": {}, "realesrgan": {"upscaler": "RealESRGAN_x4plus", "scale": 4}},
            input_type="base64",
        )

        self.assertEqual(images, ["data:image/jpeg;base64,upscaled-image"])
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["gfpgan_visibility"], 1)
        self.assertEqual(payloads[0]["upscaler_1"], "R-ESRGAN 4x+")
        self.assertEqual(payloads[0]["upscaling_resize"], 4)


if __name__ == "__main__":
    unittest.main()
