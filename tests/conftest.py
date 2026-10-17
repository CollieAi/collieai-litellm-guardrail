"""The adapter's tests. They sit beside the package, as in its public repository: the package's directory goes on the
path, as `pip install` would put the package there, and so does this one, for the shared `litellm_contract` helpers."""

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
ADAPTER_DIR = TESTS_DIR.parent
for directory in (ADAPTER_DIR, TESTS_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


@pytest.fixture(autouse=True)
def _adapter_logger_given_back() -> Iterator[None]:
    """A built guardrail sets up its logger (`guardrail._make_visible`); each test gets it back as it was."""
    log = logging.getLogger("collieai_guardrail")
    level, handlers = log.level, list(log.handlers)
    yield
    log.handlers[:] = handlers
    log.setLevel(level)
