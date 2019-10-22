import logging
import os
import numpy as np
import torch
from pathlib import Path


def get_project_root() -> Path:
        """Returns project root folder."""
            return Path(__file__).parent.parent


def get_data_dir() -> Path:
    return get_project_root() / 'data'


def make_logger(file_: str = "NO_FILE") -> logging.Logger:
    log_level = getattr(logging, os.getenv("LOG_LEVEL", "INFO"))
    fmt = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
    logging.basicConfig(level=log_level, format=fmt)
    return logging.getLogger(file_.split("/")[-1])


def set_seeds():
    np.random.seed(3778)
    torch.manual_seed(3778)
