import sys
from loguru import logger
logger.add(sys.stderr,colorize=True,format="{time} {level} {message}", filter="monolite", level="INFO")

