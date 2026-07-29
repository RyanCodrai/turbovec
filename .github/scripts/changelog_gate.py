#!/usr/bin/env python3
"""Fail a pull request that changes public surface without touching CHANGELOG.md.

Four of the six fix commits audited in #368 changed user-visible behaviour and
recorded nothing: a new cargo feature, two new public API items, a new load
error class, a removed importable module. Nothing enforced it, so it depended
on whoever wrote the PR remembering. This is the enforcement.

The gate is deliberately narrow. It only looks at files that ship to users:

  * ``turbovec/src/**.rs``            - the Rust crate's compiled surface
  * ``turbovec-python/src/**.rs``     - the binding crate
  * ``turbovec-python/python/turbovec/**.py`` - the shipped Python package,
    including the four framework integrations

and within those it only counts *substantive* lines. A diff that adds or
rewrites nothing but comments and blank lines does not trip the gate, so
documenting a function is not an excuse to invent a changelog entry.

Escape hatch, for changes that genuinely are not user-visible (an internal
refactor, a test-only helper, a comment sweep the heuristic misreads):

  * add the ``skip-changelog`` label to the PR, or
  * put ``[skip changelog]`` anywhere in the PR body.

Both are recorded on the PR, so "we decided this needs no entry" stays
visible to review rather than being invisible by default.

Usage:
    changelog_gate.py --base <sha-or-ref> --head <sha-or-ref>

Reads PR_BODY and PR_LABELS (newline- or comma-separated) from the
environment for the escape hatch; both are optional so the script can be run
against any two commits locally.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

CHANGELOG = "CHANGELOG.md"

# Shipped surface. Anything not matched here is out of scope by construction:
# tests, benchmarks, examples, docs, workflows, manifests.
SURFACE_PATTERNS = (
    re.compile(r"^turbovec/src/.*\.rs$"),
    re.compile(r"^turbovec-python/src/.*\.rs$"),
    re.compile(r"^turbovec-python/python/turbovec/.*\.py$"),
)

# Files that match a surface pattern but contain only test code. Keeping this
# list explicit (rather than trying to parse `#[cfg(test)]` regions) means the
# exemption is auditable: if a file stops being test-only, it must be removed
# from here by hand.
SURFACE_EXCLUDES = frozenset(
    {
        "turbovec/src/kernel_tests.rs",
    }
)

ESCAPE_LABEL = "skip-changelog"
ESCAPE_PHRASE = "[skip changelog]"

# Line-level noise filter. A changed line matching one of these is not, on its
# own, evidence of a user-visible change.
_RUST_COMMENT = re.compile(r"^\s*(//.*|/\*.*|\*.*|\*/)?\s*$")
_PY_COMMENT = re.compile(r"^\s*(#.*)?$")


def run(*args: str) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True
    ).stdout


def is_surface(path: str) -> bool:
    if path in SURFACE_EXCLUDES:
        return False
    return any(p.match(path) for p in SURFACE_PATTERNS)


def substantive(path: str, base: str, head: str) -> bool:
    """True if the diff for `path` touches anything but comments and blanks."""
    matcher = _PY_COMMENT if path.endswith(".py") else _RUST_COMMENT
    diff = run("git", "diff", "--unified=0", f"{base}...{head}", "--", path)
    for line in diff.splitlines():
        if line.startswith(("+++", "---", "@@", "diff ", "index ", "new file", "deleted file", "similarity", "rename ")):
            continue
        if not line.startswith(("+", "-")):
            continue
        if not matcher.match(line[1:]):
            return True
    # A pure add/delete of a whole file with no substantive lines is still a
    # mode/rename-only change; treat it as non-substantive.
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", default="HEAD")
    args = ap.parse_args()

    body = os.environ.get("PR_BODY", "")
    labels = re.split(r"[\n,]", os.environ.get("PR_LABELS", ""))
    labels = {label.strip() for label in labels if label.strip()}

    if ESCAPE_LABEL in labels:
        print(f"'{ESCAPE_LABEL}' label present - changelog gate skipped.")
        return 0
    if ESCAPE_PHRASE in body.lower():
        print(f"'{ESCAPE_PHRASE}' found in the PR body - changelog gate skipped.")
        return 0

    changed = [
        p for p in run(
            "git", "diff", "--name-only", f"{args.base}...{args.head}"
        ).splitlines() if p
    ]

    if CHANGELOG in changed:
        print(f"{CHANGELOG} was updated.")
        return 0

    offenders = [p for p in changed if is_surface(p) and substantive(p, args.base, args.head)]

    if not offenders:
        print("No substantive changes to shipped surface; no changelog entry required.")
        return 0

    print(f"::error::public surface changed without a {CHANGELOG} entry")
    print()
    print("These shipped files changed by more than comments and blank lines:")
    for p in offenders:
        print(f"  {p}")
    print()
    print(f"Add an entry under '## [Unreleased]' in {CHANGELOG}, under the")
    print("surface it affects (Rust crate / Python package), describing the")
    print("change from a user's point of view and referencing the issue.")
    print()
    print("If the change genuinely is not user-visible, say so explicitly:")
    print(f"  * add the '{ESCAPE_LABEL}' label to this PR, or")
    print(f"  * put '{ESCAPE_PHRASE}' in the PR body.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
