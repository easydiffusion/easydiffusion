import os
import tempfile
import pytest
import shutil

from easydiffusion.local_models import (
    MODEL_EXTENSIONS,
    get_model_dirs,
    list_models,
    enumerate_all_models,
    resolve_model_path,
)


@pytest.fixture
def temp_dir():
    """Fixture to provide a temporary directory for tests."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir)


def test_get_model_dirs_standard(temp_dir):
    dirs = get_model_dirs("stable-diffusion", temp_dir)
    assert dirs == [os.path.join(temp_dir, "stable-diffusion")]


def test_get_model_dirs_with_alternate(temp_dir):
    # Create alternate dir
    alt_dir = os.path.join(temp_dir, "Stable-diffusion")
    os.makedirs(alt_dir, exist_ok=True)
    dirs = get_model_dirs("stable-diffusion", temp_dir)
    expected = [os.path.join(temp_dir, "stable-diffusion")]
    # On case-insensitive filesystems, alternate may not be added if same as standard
    if os.path.exists(os.path.join(temp_dir, "stable-diffusion")):
        try:
            if not os.path.samefile(os.path.join(temp_dir, "stable-diffusion"), alt_dir):
                expected.append(alt_dir)
        except OSError:
            expected.append(alt_dir)
    assert dirs == expected


def test_get_model_dirs_no_alternate(temp_dir):
    # No alternate dir exists
    dirs = get_model_dirs("stable-diffusion", temp_dir)
    assert dirs == [os.path.join(temp_dir, "stable-diffusion")]


def test_list_models_empty_dir(temp_dir):
    result = list_models("stable-diffusion", temp_dir)
    assert result == []


def test_list_models_with_files(temp_dir):
    model_dir = os.path.join(temp_dir, "stable-diffusion")
    os.makedirs(model_dir)
    # Create test files
    with open(os.path.join(model_dir, "model1.ckpt"), "w") as f:
        f.write("dummy")
    with open(os.path.join(model_dir, "model2.safetensors"), "w") as f:
        f.write("dummy")
    with open(os.path.join(model_dir, "notamodel.txt"), "w") as f:
        f.write("dummy")  # Should not be included

    result = list_models("stable-diffusion", temp_dir)
    assert len(result) == 2
    models = {m["model"]: m for m in result}
    assert "model1" in models
    assert "model2" in models
    assert models["model1"]["tags"] == ["stable-diffusion"]
    assert models["model2"]["tags"] == ["stable-diffusion"]


def test_list_models_with_alternate(temp_dir):
    # Create standard dir with one file
    std_dir = os.path.join(temp_dir, "stable-diffusion")
    os.makedirs(std_dir, exist_ok=True)
    with open(os.path.join(std_dir, "std_model.ckpt"), "w") as f:
        f.write("dummy")

    # Create alternate dir with another file (on Windows, this may be the same dir)
    alt_dir = os.path.join(temp_dir, "Stable-diffusion")
    os.makedirs(alt_dir, exist_ok=True)
    with open(os.path.join(alt_dir, "alt_model.safetensors"), "w") as f:
        f.write("dummy")

    result = list_models("stable-diffusion", temp_dir)
    # On case-insensitive filesystems, may have duplicates or merged
    models = {m["model"]: m for m in result}
    assert "std_model" in models
    assert "alt_model" in models


def test_enumerate_all_models(temp_dir):
    # Create dirs and files for two types
    for model_type in ["stable-diffusion", "vae"]:
        model_dir = os.path.join(temp_dir, model_type)
        os.makedirs(model_dir)
        ext = MODEL_EXTENSIONS[model_type][0]
        with open(os.path.join(model_dir, f"test{ext}"), "w") as f:
            f.write("dummy")

    result = enumerate_all_models(temp_dir)
    assert len(result) == 2
    tags = {m["tags"][0] for m in result}
    assert tags == {"stable-diffusion", "vae"}


def test_resolve_model_path_with_extension(temp_dir):
    model_dir = os.path.join(temp_dir, "stable-diffusion")
    os.makedirs(model_dir)
    file_path = os.path.join(model_dir, "model.ckpt")
    with open(file_path, "w") as f:
        f.write("dummy")

    result = resolve_model_path("model.ckpt", "stable-diffusion", temp_dir)
    assert result == file_path


def test_resolve_model_path_without_extension(temp_dir):
    model_dir = os.path.join(temp_dir, "stable-diffusion")
    os.makedirs(model_dir)
    file_path = os.path.join(model_dir, "model.safetensors")
    with open(file_path, "w") as f:
        f.write("dummy")

    result = resolve_model_path("model", "stable-diffusion", temp_dir)
    assert result == file_path


def test_resolve_model_path_in_alternate(temp_dir):
    # Create alternate dir
    alt_dir = os.path.join(temp_dir, "Stable-diffusion")
    os.makedirs(alt_dir, exist_ok=True)
    file_path = os.path.join(alt_dir, "alt_model.ckpt")
    with open(file_path, "w") as f:
        f.write("dummy")

    result = resolve_model_path("alt_model", "stable-diffusion", temp_dir)
    assert os.path.normcase(result) == os.path.normcase(file_path)


def test_resolve_model_path_not_found(temp_dir):
    result = resolve_model_path("nonexistent", "stable-diffusion", temp_dir)
    assert result is None
