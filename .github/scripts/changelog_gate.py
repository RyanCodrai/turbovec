#!/usr/bin/env python3
"""Fail a pull request that changes public surface without recording it.

Four of the six fix commits audited in #368 changed user-visible behaviour and
recorded nothing: a new cargo feature, two new public API items, a new load
error class, a removed importable module. Nothing enforced it, so it depended
on whoever wrote the PR remembering. This is the enforcement.

Scope. Only files that ship, or that describe what ships:

  * ``turbovec/src/**.rs``                      the Rust crate's surface
  * ``turbovec-python/src/**.rs``               the binding crate
  * ``turbovec-python/python/turbovec/**.py``   the shipped Python package,
    including the four framework integrations
  * ``turbovec/Cargo.toml``, ``turbovec-python/Cargo.toml``,
    ``turbovec-python/pyproject.toml`` — features, MSRV, ``requires-python``,
    extras and dependency floors are all user-visible packaging surface, and
    "a new cargo feature" was one of the four misses this gate exists for.

Within those it ignores two kinds of change:

  * comments and blank lines, so documenting a function is not an excuse to
    invent a changelog entry;
  * anything inside a ``#[cfg(test)]`` region, so landing a regression test
    beside a fix — exactly the workflow #367 wants — is not taxed.

And it requires the changelog to have actually gained something: an added,
non-blank line under ``## [Unreleased]``. Touching the file is not enough.

Escape hatch, for changes that genuinely are not user-visible:

  * add the ``skip-changelog`` label to the PR, or
  * put ``[skip changelog]`` **alone on its own line** in the PR body.

The line must be exactly the marker. A body that merely mentions the marker in
prose — quoting CONTRIBUTING.md, or a note saying the hatch was deliberately
*not* used — does not disarm the gate. The first version of this script used a
substring test and silently disabled itself on the very PR that introduced it.

Usage:
    changelog_gate.py --base <sha-or-ref> --head <sha-or-ref>

Reads PR_BODY and PR_LABELS (newline- or comma-separated) from the
environment; both are optional so the script runs against any two commits.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

CHANGELOG = "CHANGELOG.md"
UNRELEASED = re.compile(r"^##\s*\[Unreleased\]", re.IGNORECASE)
HEADING = re.compile(r"^##\s")

SURFACE_PATTERNS = (
    re.compile(r"^turbovec/src/.*\.rs$"),
    re.compile(r"^turbovec-python/src/.*\.rs$"),
    re.compile(r"^turbovec-python/python/turbovec/.*\.py$"),
    re.compile(r"^turbovec/Cargo\.toml$"),
    re.compile(r"^turbovec-python/Cargo\.toml$"),
    re.compile(r"^turbovec-python/pyproject\.toml$"),
)

# Files that match a surface pattern but hold only test code. Kept explicit so
# the exemption is auditable: if such a file stops being test-only it has to be
# removed from here by hand. Every other file gets per-region handling below.
SURFACE_EXCLUDES = frozenset({"turbovec/src/kernel_tests.rs"})

ESCAPE_LABEL = "skip-changelog"
ESCAPE_PHRASE = "[skip changelog]"

# Blank, or a line comment, or a block-comment body line. The last alternative
# is deliberately `\*` followed by whitespace, `/`, or end-of-line: an earlier
# version used `\*.*`, which classified every Rust deref assignment
# (`*norm = n_val;`) as a comment. There are 51 such lines in shipped src, so
# that was a live false negative in exactly what this gate exists to catch.
_RUST_COMMENT = re.compile(r"^\s*(?://.*|/\*.*|\*/.*|\*(?:[\s/].*)?)?\s*$")
_HASH_COMMENT = re.compile(r"^\s*(?:#.*)?$")

# `#[cfg(test)]`, `#[cfg(any(test, ...))]`, `#[cfg(all(test, ...))]`.
_CFG_TEST = re.compile(r"^(\s*)#\[cfg\((?:any\(|all\()?\s*test\b")


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def blob(rev: str, path: str) -> list[str]:
    """File contents at `rev`, or [] if it does not exist there."""
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True)
    return r.stdout.splitlines() if r.returncode == 0 else []


def comment_matcher(path: str):
    return _HASH_COMMENT if path.endswith((".py", ".toml")) else _RUST_COMMENT


def cfg_test_regions(lines: list[str]) -> set[int]:
    """1-based line numbers covered by a `#[cfg(test)]` item.

    The item's extent is found by indentation, not by counting braces: a brace
    counter has to understand strings, chars and comments to be correct, and
    getting it wrong would silently exempt a whole file. The attribute's own
    indentation is its closing brace's indentation in any rustfmt-shaped code,
    which is what this tree is.

    An attributed item with no block (`#[cfg(test)] static X: T = ...;`) ends
    at its first statement terminator, so a runaway search cannot swallow the
    rest of the enclosing scope.
    """
    covered: set[int] = set()
    i = 0
    while i < len(lines):
        m = _CFG_TEST.match(lines[i])
        if not m:
            i += 1
            continue
        closing = m.group(1) + "}"
        seen_brace = "{" in lines[i]
        end = None
        j = i + 1
        while j < len(lines):
            line = lines[j].rstrip()
            if not seen_brace:
                if "{" in line:
                    seen_brace = True
                elif line.endswith(";"):
                    end = j
                    break
            if seen_brace and line == closing:
                end = j
                break
            j += 1
        if end is None:
            end = len(lines) - 1
        covered.update(range(i + 1, end + 2))
        i = end + 1
    return covered


def parse_hunks(diff: str):
    """Yield (kind, lineno, text) for each changed line.

    `lineno` is the new-file line for '+' and the old-file line for '-', which
    is the side whose cfg(test) map that line must be looked up in.
    """
    old = new = 0
    for line in diff.splitlines():
        h = re.match(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if h:
            old, new = int(h.group(1)), int(h.group(2))
            continue
        if line.startswith(("+++", "---")):
            continue
        if line.startswith("+"):
            yield "+", new, line[1:]
            new += 1
        elif line.startswith("-"):
            yield "-", old, line[1:]
            old += 1


def substantive(path: str, base: str, head: str) -> bool:
    """True if `path`'s diff touches shipped, non-comment, non-test code."""
    matcher = comment_matcher(path)
    if path.endswith(".rs"):
        head_tests = cfg_test_regions(blob(head, path))
        base_tests = cfg_test_regions(blob(base, path))
    else:
        head_tests = base_tests = set()

    diff = run("git", "diff", "--unified=0", f"{base}...{head}", "--", path)
    for kind, lineno, text in parse_hunks(diff):
        if matcher.match(text):
            continue
        if lineno in (head_tests if kind == "+" else base_tests):
            continue
        return True
    return False


def changelog_gained_an_entry(base: str, head: str) -> tuple[bool, str]:
    """An added non-blank line under `## [Unreleased]` in the head file.

    Merely touching CHANGELOG.md used to satisfy the gate, so a new public
    function plus one appended blank line walked straight through.
    """
    diff = run("git", "diff", "--unified=0", f"{base}...{head}", "--", CHANGELOG)
    added = [n for k, n, t in parse_hunks(diff) if k == "+" and t.strip()]
    if not added:
        return False, f"{CHANGELOG} changed, but added no non-blank line."

    lines = blob(head, CHANGELOG)
    start = end = None
    for i, line in enumerate(lines, 1):
        if start is None:
            if UNRELEASED.match(line):
                start = i
        elif HEADING.match(line):
            end = i
            break
    if start is None:
        return True, f"{CHANGELOG} gained content (no '## [Unreleased]' heading found)."
    end = end or len(lines) + 1

    inside = [n for n in added if start < n < end]
    if not inside:
        return False, (
            f"{CHANGELOG} changed, but nothing was added under "
            f"'## [Unreleased]' (lines {start}-{end - 1})."
        )
    return True, f"{CHANGELOG} gained {len(inside)} line(s) under '## [Unreleased]'."


def escape_hatch() -> str | None:
    labels = {
        s.strip()
        for s in re.split(r"[\n,]", os.environ.get("PR_LABELS", ""))
        if s.strip()
    }
    if ESCAPE_LABEL in labels:
        return f"'{ESCAPE_LABEL}' label present"
    for line in os.environ.get("PR_BODY", "").splitlines():
        if line.strip().lower() == ESCAPE_PHRASE:
            return f"'{ESCAPE_PHRASE}' found alone on its own line in the PR body"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--head", default="HEAD")
    args = ap.parse_args()

    hatch = escape_hatch()
    if hatch:
        print(f"{hatch} - changelog gate skipped.")
        return 0

    changed = [
        p
        for p in run(
            "git", "diff", "--name-only", f"{args.base}...{args.head}"
        ).splitlines()
        if p
    ]
    offenders = [
        p
        for p in changed
        if p not in SURFACE_EXCLUDES
        and any(pat.match(p) for pat in SURFACE_PATTERNS)
        and substantive(p, args.base, args.head)
    ]

    if not offenders:
        print("No substantive changes to shipped surface; no changelog entry required.")
        return 0

    if CHANGELOG in changed:
        ok, why = changelog_gained_an_entry(args.base, args.head)
        print(why)
        if ok:
            return 0
    else:
        print(f"{CHANGELOG} was not touched.")

    print(f"::error::public surface changed without a {CHANGELOG} entry")
    print()
    print("These shipped files changed outside comments and #[cfg(test)] regions:")
    for p in offenders:
        print(f"  {p}")
    print()
    print("Add an entry under '## [Unreleased]', under the surface it affects")
    print("(Rust crate / Python package), describing the change from a user's")
    print("point of view and referencing the issue.")
    print()
    print("If the change genuinely is not user-visible, say so explicitly:")
    print(f"  * add the '{ESCAPE_LABEL}' label to this PR, or")
    print(f"  * put '{ESCAPE_PHRASE}' alone on its own line in the PR body.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
