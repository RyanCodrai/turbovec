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

The line must be exactly the marker, and must not be inside a code block. A
body that merely mentions the marker in prose — quoting CONTRIBUTING.md, or a
note saying the hatch was deliberately *not* used, or showing it in a fenced
block — does not disarm the gate. The first version of this script used a
substring test and silently disabled itself on the very PR that introduced it.
The predicate is shared with the mutation gate; see ``escape_hatch.py``.

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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from escape_hatch import hatch_reason  # noqa: E402

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

# A char literal: `'x'`, `'\n'`, `'\x7f'`, `'\u{7f}'`. Matched so a `'` that is
# a lifetime (`'a`) is not mistaken for the start of one — and so the brace in
# `'\u{7f}'` is not counted.
_CHAR_LIT = re.compile(r"'(?:\\(?:x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]{1,6}\}|.)|[^\\'])'")

# The opening of a raw string: `r"`, `r#"`, `br##"`, `cr#"`. The hashes must be
# followed by the quote, so the raw *identifier* `r#type` is not a match.
_RAW_OPEN = re.compile(r"[bc]?r(#*)\"")
_IDENT_CHAR = re.compile(r"[A-Za-z0-9_]")


def strip_literals(lines: list[str]) -> list[str]:
    """Remove comments, strings and char literals, keeping line numbering.

    Brace depth is what delimits a `#[cfg(test)]` item, and a brace inside a
    comment or a string is not a brace. Counting them raw is not merely
    imprecise, it is unsound in the dangerous direction: an *extra* `{` inflates
    the depth, and the surplus `}` that eventually balances it is borrowed from
    the enclosing block, so the region closes LATE and swallows shipped code
    that follows it (#395).

    Handled: line comments, nested block comments (Rust's do nest), strings and
    byte/C strings, raw strings of any hash count, char literals and lifetimes.
    All of these can span lines, so the scan carries state across them and must
    start at the top of the file rather than at the attribute.

    Literal spans are deleted, not padded, so line numbering survives but
    column positions do not: code following a literal on the same line shifts
    left by the literal's width. Only brace depth and line indices are read
    downstream, so that is enough — pad the spans if a caller ever needs
    columns. Text outside a literal, whitespace included, is untouched, so
    leading indentation always survives.
    """
    out: list[str] = []
    kind: str | None = None  # None | "block" | "str" | "raw"
    extra = 0  # block-comment nesting depth, or raw-string hash count
    for line in lines:
        code: list[str] = []
        i, n = 0, len(line)
        while i < n:
            if kind == "block":
                if line.startswith("*/", i):
                    extra -= 1
                    i += 2
                    if extra == 0:
                        kind = None
                elif line.startswith("/*", i):
                    extra += 1
                    i += 2
                else:
                    i += 1
            elif kind == "str":
                if line[i] == "\\":
                    i += 2
                elif line[i] == '"':
                    kind = None
                    i += 1
                else:
                    i += 1
            elif kind == "raw":
                if line[i] == '"' and line[i + 1 : i + 1 + extra] == "#" * extra:
                    kind = None
                    i += 1 + extra
                else:
                    i += 1
            elif line.startswith("//", i):
                break
            elif line.startswith("/*", i):
                kind, extra = "block", 1
                i += 2
            elif line[i] == '"':
                kind = "str"
                i += 1
            elif line[i] == "'":
                m = _CHAR_LIT.match(line, i)
                # No match means a lifetime tick; step over just the quote.
                i = m.end() if m else i + 1
            else:
                m = _RAW_OPEN.match(line, i)
                if m and not (i and _IDENT_CHAR.match(line[i - 1])):
                    kind, extra = "raw", len(m.group(1))
                    i = m.end()
                else:
                    code.append(line[i])
                    i += 1
        out.append("".join(code))
    return out


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def blob(rev: str, path: str) -> list[str]:
    """File contents at `rev`, or [] if it does not exist there."""
    r = subprocess.run(["git", "show", f"{rev}:{path}"], capture_output=True, text=True)
    return r.stdout.splitlines() if r.returncode == 0 else []


def comment_matcher(path: str):
    return _HASH_COMMENT if path.endswith((".py", ".toml")) else _RUST_COMMENT


def cfg_test_regions(lines: list[str]) -> tuple[set[int], str | None]:
    """1-based line numbers covered by a `#[cfg(test)]` item.

    Returns `(covered, failure)`. If `failure` is non-None the caller must
    treat the whole file as non-exempt: **this heuristic fails closed.** An
    earlier version searched for a line exactly equal to `indent + "}"` and,
    when it could not find one, exempted everything to end of file. Three
    one-line shapes triggered that — a single-line `#[cfg(test)] fn` inside an
    `impl`, a closing brace with a trailing comment (`} // end tests`), and a
    one-line `#[cfg(test)] use super::*;` — and any of them landing in the
    tree would have silently exempted the rest of the file from the gate
    forever. For a gate, "I got confused" must mean *check more*, not *check
    less*.

    Extent is tracked by brace depth from the attribute onwards, over lines
    that `strip_literals` has cleared of comments, strings and char literals
    first. Depth used to be counted on the raw text, and a single `{` in an
    ordinary comment inside a *nested* `#[cfg(test)]` item was enough to make
    the region close late and swallow the shipped code after it (#395). See
    `strip_literals` for why that direction of miscount is the dangerous one.

    The attribute is matched on the stripped text too, so a `#[cfg(test)]`
    written inside a comment or a string does not open a region.
    """
    lines = strip_literals(lines)
    covered: set[int] = set()
    i = 0
    while i < len(lines):
        m = _CFG_TEST.match(lines[i])
        if not m:
            i += 1
            continue

        depth = 0
        opened = False
        end = None
        j = i
        while j < len(lines):
            # Strip the attribute itself so its brackets can't confuse the
            # `;` test below; `#[...]` contains no braces either way.
            text = lines[j]
            if j == i:
                text = _CFG_TEST.sub("", text, count=1)
            code = text.rstrip()

            depth += code.count("{") - code.count("}")
            if "{" in code:
                opened = True

            if opened and depth <= 0:
                end = j
                break
            # A blockless item (`use super::*;`, `static X: T = 1;`) ends at
            # its first terminator. Checked on line `i` too, which is how the
            # one-line `#[cfg(test)] use super::*;` shape used to escape.
            if not opened and code.endswith((";", ",")):
                end = j
                break
            j += 1

        if end is None:
            return set(), (
                f"could not find the end of the #[cfg(test)] item starting at "
                f"line {i + 1}"
            )

        covered.update(range(i + 1, end + 2))
        i = end + 1
    return covered, None


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
    head_tests: set[int] = set()
    base_tests: set[int] = set()
    if path.endswith(".rs"):
        for rev, into in ((head, "head"), (base, "base")):
            covered, failure = cfg_test_regions(blob(rev, path))
            if failure:
                # Fail closed: no exemption at all for this file, and say so,
                # because the alternative reads as "the gate passed".
                print(
                    f"note: {path} ({into}): {failure}; "
                    "treating the whole file as shipped code."
                )
                head_tests = base_tests = set()
                break
            if into == "head":
                head_tests = covered
            else:
                base_tests = covered

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
    return hatch_reason(
        ESCAPE_PHRASE,
        ESCAPE_LABEL,
        os.environ.get("PR_BODY", ""),
        os.environ.get("PR_LABELS", ""),
    )


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
