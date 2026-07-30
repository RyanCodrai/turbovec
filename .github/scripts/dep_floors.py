#!/usr/bin/env python3
"""Print the declared dependency floors of turbovec's extras as exact pins.

`pyproject.toml` promises `haystack-ai>=2.23.0` and `agno>=2.5.4`, but CI has
only ever installed `>=2.0` for both and pip resolves those to the newest
release — so the floors users are told to trust have never been executed
(#306). This turns the declaration into the thing CI installs: every `>=X`
becomes `==X`, so the floors job runs against exactly the oldest supported
release of each integration.

Because the pins are *derived* from pyproject rather than copied into a
workflow, raising a floor automatically moves what CI tests, and a floor that
was never a real release fails loudly at install time instead of silently
resolving upward.

Usage:
    dep_floors.py                 # every extra
    dep_floors.py haystack agno   # named extras only
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "turbovec-python" / "pyproject.toml"


def main(argv: list[str]) -> int:
    data = tomllib.loads(PYPROJECT.read_text())
    extras = data["project"]["optional-dependencies"]

    wanted = argv or sorted(extras)
    missing = [e for e in wanted if e not in extras]
    if missing:
        print(f"unknown extra(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    for extra in wanted:
        for req in extras[extra]:
            if ">=" not in req:
                print(
                    f"{extra}: cannot derive a floor from '{req}' "
                    "(expected a '>=' constraint)",
                    file=sys.stderr,
                )
                return 1
            name, _, floor = req.partition(">=")
            print(f"{name.strip()}=={floor.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
