#!/usr/bin/env python3
"""Decide whether a PR has explicitly opted out of a gate.

Shared by the changelog gate and the mutation gate so the two ask exactly the
same question. They used to implement this separately — Python in one, `grep`
in the other — and drifted: the Python side split on U+2028/U+0085 and the
bash side did not, and NBSP padding was handled by two different code paths.
Two predicates that are supposed to be one predicate will always diverge, so
now there is one.

A gate is skipped when either:

  * the PR carries the gate's label, or
  * the marker appears **alone on its own line** in the PR body, outside any
    code block.

The line-anchoring is not cosmetic. The first version of both gates used a
plain substring test, and the PR that introduced them documented the markers
in its own body — so both gates silently disabled themselves and reported
green having tested nothing.

Code blocks are excluded for the same reason one step further out: a fenced
block is the natural way to show a literal marker when documenting the hatch,
so a body that *explains* the hatch must not trip it. Fenced blocks (``` or
~~~, with or without an info string) and four-space indented blocks are both
stripped before the scan.

Usage:
    escape_hatch.py --marker '[skip changelog]' --label skip-changelog

Exit 0 if the hatch is present (and print why), 1 if it is not. Reads PR_BODY
and PR_LABELS from the environment.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# Only these are treated as line terminators. `str.splitlines()` also splits
# on U+2028, U+2029, U+0085 and form feed, which `grep -x` does not — that was
# one of the two ways the Python and bash hatches disagreed. A marker is only
# "on its own line" if it is on its own line by the same rule everyone else
# uses.
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def body_lines(body: str) -> list[str]:
    return body.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def strip_code_blocks(lines: list[str]) -> list[str]:
    """Blank out fenced and indented code blocks.

    Lines are replaced rather than removed so that any future caller wanting
    line numbers still gets the right ones.
    """
    out: list[str] = []
    fence: str | None = None
    for line in lines:
        m = _FENCE.match(line)
        if fence is None:
            if m:
                # Opening fence: remember which character opened it, so a ```
                # block containing ~~~ (or vice versa) is not closed early.
                fence = m.group(1)[0]
                out.append("")
                continue
            # An indented code block. Markdown requires a preceding blank
            # line, but treating every 4-space/tab-indented line as code is
            # the safe direction here: it can only ever *arm* the gate.
            if line.startswith(("    ", "\t")):
                out.append("")
                continue
            out.append(line)
        else:
            # Inside a fence: a closing fence is one of the same character.
            if m and m.group(1)[0] == fence:
                fence = None
            out.append("")
    return out


def hatch_reason(marker: str, label: str, body: str, labels: str) -> str | None:
    """Why the gate should be skipped, or None if it should run."""
    present = {s.strip() for s in re.split(r"[\n,]", labels) if s.strip()}
    if label in present:
        return f"'{label}' label present"

    marker = marker.lower()
    for line in strip_code_blocks(body_lines(body)):
        if line.strip().lower() == marker:
            return f"'{marker}' found alone on its own line in the PR body"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    why = hatch_reason(
        args.marker,
        args.label,
        os.environ.get("PR_BODY", ""),
        os.environ.get("PR_LABELS", ""),
    )
    if why:
        print(why)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
