from loguru import logger
from lib.engine.logger import init_logger


def get_logger(output_dir: str = "./runs/exp", name: str = "monolite"):
    return init_logger(output_dir, name)