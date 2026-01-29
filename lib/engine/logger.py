from __future__ import annotations

import os
from loguru import logger


def init_logger(output_dir: str, name: str = "monolite"):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{name}.log")

    logger.remove()
    logger.add(log_path, level="INFO", enqueue=True, backtrace=True, diagnose=False)
    logger.add(lambda msg: print(msg, end=""), level="INFO")
    return logger
