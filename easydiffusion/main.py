"""
Main entry point for EasyDiffusion server.

Sets up configuration, task queue, and starts the FastAPI server.
"""

from pathlib import Path
import logging

from easydiffusion.config import ConfigManager, create_default_config
from easydiffusion.backends import get_backend_class
from easydiffusion.server import server_api  # required for uvicorn
from easydiffusion.workers import Workers


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def init():
    """
    Initialize the application.
    """
    # Determine config file path
    config_path = Path("config.yaml")

    # Create default config if it doesn't exist
    if not config_path.exists():
        logger.info(f"Creating default configuration at {config_path}")
        create_default_config(config_path)

    # Initialize configuration manager
    logger.info(f"Loading configuration from {config_path}")
    config_manager = ConfigManager(config_path)
    config_manager.load()

    # Check that users are configured
    if not config_manager.get_users():
        raise ValueError("No users configured in config.yaml. Please add at least one user to the 'users' list.")

    # Log current configuration
    config = config_manager.get_all()
    logger.info(f"Configuration loaded: {config}")

    logger.info("Initializing workers")
    backend_name = config.get("backend", {}).get("backend_name", "sdkit3")
    workers = Workers(get_backend_class(backend_name), backend_name=backend_name)

    # Start workers for configured devices
    devices = config.get("backend", {}).get("devices", "auto")
    logger.info(f"Starting workers for devices: {devices}")
    workers.update_devices(devices)

    # Store in app state (will be set when app is created)
    server_api.state.config_manager = config_manager
    server_api.state.workers = workers

    logger.info("Application initialized")


# Initialize on import
init()

if __name__ == "__main__":
    import uvicorn

    config = server_api.state.config_manager.get_all()
    port = config.get("network", {}).get("port", 9000)
    external_access = config.get("network", {}).get("external_access", False)
    host = "0.0.0.0" if external_access else "127.0.0.1"

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("easydiffusion.server:server_api", host=host, port=port)
