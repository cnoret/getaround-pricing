"""
Shared test settings.

The dashboard opens its data with paths relative to the repo root
(data/...), like `streamlit run dashboard/app.py`, so tests run from there.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Maximum time for one dashboard run (the raw Excel export takes about a second to load)
TIMEOUT = 60


@pytest.fixture(autouse=True)
def repo_root(monkeypatch):
    monkeypatch.chdir(ROOT)


def no_exception(at):
    """Readable message if the app raised an exception."""
    assert not at.exception, "\n\n".join(e.value for e in at.exception)
