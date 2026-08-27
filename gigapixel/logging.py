from enum import Enum
from typing import Optional, Tuple

from loguru import logger


class Level(Enum):
    TRACE = 5
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


def log(
    start: Optional[str] = None,
    end: Optional[str] = None,
    format: Optional[Tuple[int, ...]] = None,
    level: Level = Level.INFO,
):
    """
    Decorator that logs function entry and/or exit via loguru.

    :param start: Message template printed before the function runs.
    :param end: Message template printed after the function returns.
    :param format: Tuple of positional argument indices to inject into templates.
                   Use -1 to reference the return value.
    :param level: Loguru severity level.
    """
    def outer_wrapper(function):
        def wrapper(*args, **kwargs):
            if start:
                indices = format if format else ()
                format_args = [args[i] for i in indices if i != -1]
                logger.log(level.name, start.format(*format_args))

            result = function(*args, **kwargs)

            if end:
                indices = format if format else ()
                format_args = [
                    result if i == -1 else args[i] for i in indices
                ]
                logger.log(level.name, end.format(*format_args))

            return result
        return wrapper
    return outer_wrapper
