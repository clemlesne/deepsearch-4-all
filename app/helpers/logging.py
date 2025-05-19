from logging import DEBUG, INFO, Logger
from os import getenv

from structlog import (
    configure,
    configure_once,
    get_logger as structlog_get_logger,
    make_filtering_bound_logger,
)
from structlog.contextvars import merge_contextvars
from structlog.dev import ConsoleRenderer
from structlog.processors import (
    StackInfoRenderer,
    TimeStamper,
    UnicodeDecoder,
    add_log_level,
)
from structlog.stdlib import PositionalArgumentsFormatter

from app.helpers import IS_CI

VERSION = getenv("VERSION", "0.0.0-unknown")


def enable_debug_logging() -> None:
    configure(
        wrapper_class=make_filtering_bound_logger(DEBUG),
    )


configure_once(
    cache_logger_on_first_use=True,
    context_class=dict,
    wrapper_class=make_filtering_bound_logger(INFO),
    processors=[
        # Add contextvars support
        merge_contextvars,
        # Add log level
        add_log_level,
        # Enable %s-style formatting
        PositionalArgumentsFormatter(),
        # Add timestamp
        TimeStamper(fmt="iso", utc=True),
        # Add exceptions info
        StackInfoRenderer(),
        # Decode Unicode to str
        UnicodeDecoder(),
        # Pretty printing in a terminal session
        ConsoleRenderer(),
    ],
)

# Framework does not exactly expose Logger, but that's easier to work with
logger: Logger = structlog_get_logger("deepsearch-4-all")

# Enable debug logging on CI
if IS_CI:
    enable_debug_logging()
    logger.warning("CI environment detected, be aware configuration may differ")
