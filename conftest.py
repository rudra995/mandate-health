"""Pytest bootstrap: put the repo root on ``sys.path``.

Without this, ``tests/`` is what lands on the path and ``import simulator``
fails. Kept at the root rather than inside ``tests/`` so a plain ``pytest``
from anywhere in the repo behaves the same way.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
