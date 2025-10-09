import sys

from loguru import logger

logger.remove()

logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> <level>{level}</level> <level>{message}</level>",
    colorize=True,
)

logger.add(
    "training.log",
    rotation="10 MB",
    format="<green>{time:HH:mm:ss}</green> <level>{level}</level> <level>{message}</level>",
    colorize=False,
)
