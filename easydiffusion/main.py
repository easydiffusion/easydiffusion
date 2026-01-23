"""
Main entry point for EasyDiffusion server.

Sets up configuration, task queue, and starts the FastAPI server.
"""

from pathlib import Path
import logging

from easydiffusion.config import ConfigManager, create_default_config
from easydiffusion.server import server_api  # required for uvicorn
from easydiffusion.task_queue import TaskQueue
from easydiffusion.worker_manager import WorkerManager


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

    # Initialize task queue
    logger.info("Initializing task queue")
    task_queue = TaskQueue()

    # Initialize worker manager
    logger.info("Initializing worker manager")
    backend_name = config.get("rendering", {}).get("backend", "sdkit3")
    worker_manager = WorkerManager(task_queue, backend_name)

    # Start workers for configured devices
    render_devices = config.get("rendering", {}).get("devices", "auto")
    logger.info(f"Starting workers for devices: {render_devices}")
    worker_manager.update_workers(render_devices)

    # Store in app state (will be set when app is created)
    server_api.state.config_manager = config_manager
    server_api.state.task_queue = task_queue
    server_api.state.worker_manager = worker_manager

    logger.info("Application initialized")


# Initialize on import
init()
