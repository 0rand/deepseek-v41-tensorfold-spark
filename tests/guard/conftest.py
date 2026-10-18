"""CPU-only tests of the thermal / memory guard (scripts/guard): durable policy, admission and loopback transport.

    python -m pytest -q tests/guard

No GPU, Docker, SSH or systemd is touched: the backend's host actions are replaced by local fakes.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GUARD = HERE.parents[1] / "scripts" / "guard"
for path in (GUARD, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
