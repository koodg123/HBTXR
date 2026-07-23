from typing import Iterable  # Import Iterable from typing for richer type annotations
from lightning.pytorch.loggers import (
    Logger,
    TensorBoardLogger,
)  # Import the Logger base class and TensorBoardLogger from lightning


# Define make_logger to build loggers from a config dictionary or a list of dictionaries
def make_logger(logger_cfgs: Iterable[dict] | dict) -> list[Logger]:
    loggers = list()  # Initialize an empty list to store loggers
    # If the config is a list, convert each item to a logger
    if isinstance(logger_cfgs, list):
        loggers = [make_single_logger(logger_cfg) for logger_cfg in logger_cfgs]
    # If the config is a single dictionary, convert it to a one-item logger list
    elif isinstance(logger_cfgs, dict):
        loggers = [make_single_logger(logger_cfgs)]
    return loggers  # Return the created logger list


# Define make_single_logger to build a logger from one config dictionary
def make_single_logger(logger_cfg: dict) -> Logger:
    # Check whether the config type is "tensorboard"
    if logger_cfg["type"] == "tensorboard":
        # Create a TensorBoardLogger from the config, using defaults for missing fields
        logger = TensorBoardLogger(
            save_dir=logger_cfg.get("save_dir", "logs"),  # Log save directory, defaulting to "logs"
            name=logger_cfg.get("name", "temp_exp"),  # Experiment name, defaulting to "temp_exp"
            version=logger_cfg.get("version"),  # Log version, with no default when omitted
        )
        return logger  # Return the created TensorBoardLogger instance
    else:
        # If the type is not "tensorboard", create a default TensorBoardLogger saved under "logs"
        return TensorBoardLogger("logs")
