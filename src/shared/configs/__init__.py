import os
import torch
import logging
import logging.config
from pathlib import Path
from ._settings import Settings
from ._log import get_project_logger
from ._mode import mode

PROJECT_SETTINGS = Settings()

ROOT_PATH = Path(__file__).parent.parent.parent


logger = get_project_logger()


def _print_base_settings():
    logger.info(f"ROOT_PATH: {ROOT_PATH}")


_print_base_settings()
