import pytest

from support import register_dummy_backend, unregister_dummy_backend


@pytest.fixture
def dummy_backend_registry():
    backend_name, backend_class, previous = register_dummy_backend()
    try:
        yield backend_name, backend_class
    finally:
        unregister_dummy_backend(backend_name, previous)
