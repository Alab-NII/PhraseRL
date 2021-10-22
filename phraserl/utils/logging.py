import os

frmt = (
    "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> | "
    "<c>{name}:{function}:{line}</c> - "
    "<level>{message}</level>"
)

os.environ["LOGURU_FORMAT"] = frmt
os.environ["LOGURU_LEVEL"] = "INFO"

from loguru import logger  # noqa: F401, E402
