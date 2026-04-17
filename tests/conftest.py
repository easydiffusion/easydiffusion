import pytest

from easydiffusion.backends.test_backend import TestBackend
from support import register_dummy_backend, unregister_dummy_backend


@pytest.fixture
def dummy_backend_registry(request):
    node_path = str(request.node.path).replace("\\", "/")
    previous_delay = TestBackend.GENERATE_STEP_DELAY_SECONDS
    previous_callback = TestBackend.GENERATE_STEP_CALLBACK

    TestBackend.GENERATE_STEP_CALLBACK = None
    if "/tests/integration/" in f"/{node_path}":
        TestBackend.GENERATE_STEP_DELAY_SECONDS = TestBackend.DEFAULT_GENERATE_STEP_DELAY_SECONDS
    else:
        TestBackend.GENERATE_STEP_DELAY_SECONDS = 0

    backend_name, backend_class, previous = register_dummy_backend()
    try:
        yield backend_name, backend_class
    finally:
        unregister_dummy_backend(backend_name, previous)
        TestBackend.GENERATE_STEP_DELAY_SECONDS = previous_delay
        TestBackend.GENERATE_STEP_CALLBACK = previous_callback
