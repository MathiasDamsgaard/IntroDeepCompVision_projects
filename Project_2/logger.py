import sys
from pathlib import Path

from loguru import logger

logger.remove()

logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> <level>{level}</level> <level>{message}</level>",
    colorize=True,
)

log_path = Path(__file__).parent / "training.log"

logger.add(
    log_path,
    rotation="10 MB",
    format="<green>{time:HH:mm:ss}</green> <level>{level}</level> <level>{message}</level>",
    colorize=False,
)
