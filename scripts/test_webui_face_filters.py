import importlib.util
import pathlib
import sys
import types
import unittest


def _load_webui_common():
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    module_path = repo_root / "ui" / "easydiffusion" / "backends" / "webui_common.py"

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
    def test_gfpgan_only_filter_does_not_trigger_implicit_upscale(self):
        webui_common = _load_webui_common()
        payloads = []

        def fake_post(uri, json=None, **kwargs):
            self.assertEqual(uri, "/sdapi/v1/extra-batch-images")
            payloads.append(json)
            if json["upscaler_1"] == "None" and json["upscaling_resize"] > 1:
                return _FakeResponse(500, {"detail": "None.pth"})
            return _FakeResponse(200, {"images": ["filtered-face"]})

        webui_common.webui_post = fake_post

        images = webui_common.filter_images(
            None, ["input-image"], ["gfpgan"], {"gfpgan": {}}, input_type="base64"
        )

        self.assertEqual(images, ["data:image/jpeg;base64,filtered-face"])
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["gfpgan_visibility"], 1)
        self.assertEqual(payloads[0]["upscaling_resize"], 1)
        self.assertEqual(payloads[0]["upscaler_1"], "None")

    def test_codeformer_only_filter_does_not_trigger_implicit_upscale(self):
        webui_common = _load_webui_common()
        payloads = []

        def fake_post(uri, json=None, **kwargs):
            self.assertEqual(uri, "/sdapi/v1/extra-batch-images")
            payloads.append(json)
            if json["upscaler_1"] == "None" and json["upscaling_resize"] > 1:
                return _FakeResponse(500, {"detail": "None.pth"})
            return _FakeResponse(200, {"images": ["filtered-face"]})

        webui_common.webui_post = fake_post

        images = webui_common.filter_images(
            None,
            ["input-image"],
            ["codeformer"],
            {"codeformer": {"codeformer_fidelity": 0.35}},
            input_type="base64",
        )

        self.assertEqual(images, ["data:image/jpeg;base64,filtered-face"])
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["codeformer_visibility"], 1)
        self.assertEqual(payloads[0]["codeformer_weight"], 0.35)
        self.assertEqual(payloads[0]["upscaling_resize"], 1)
        self.assertEqual(payloads[0]["upscaler_1"], "None")

    def test_explicit_upscale_settings_are_still_forwarded(self):
        webui_common = _load_webui_common()
        payloads = []

        def fake_post(uri, json=None, **kwargs):
            self.assertEqual(uri, "/sdapi/v1/extra-batch-images")
            payloads.append(json)
            return _FakeResponse(200, {"images": ["upscaled-image"]})

        webui_common.webui_post = fake_post

        images = webui_common.filter_images(
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
