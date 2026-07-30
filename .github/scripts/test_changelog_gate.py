#!/usr/bin/env python3
"""Tests for the `#[cfg(test)]` extent finder in `changelog_gate.py`.

The gate exempts everything inside a `#[cfg(test)]` item, and finds that
item's extent by counting brace depth. Braces that are not braces — inside a
comment, a string, a char literal — used to be counted, and an odd number of
them made a *nested* region close late and swallow the shipped code that
followed it, which is a gate silently not gating (#395).

So each test here is written to fail against the pre-fix script: every one
either puts a stray brace somewhere a naive count would see it, or checks a
shape the fix must not have broken.

Usage:
    python3 -m unittest discover -s .github/scripts -p 'test_*.py'
"""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import changelog_gate as gate

GATE = str(Path(__file__).resolve().parent / "changelog_gate.py")


def regions(src: str) -> set[int]:
    """The covered 1-based line numbers, asserting the finder did not bail."""
    covered, failure = gate.cfg_test_regions(src.splitlines())
    assert failure is None, failure
    return covered


# A nested `#[cfg(test)]` hook inside a closure, with a shipped call after it —
# the shape from #395. `HOOK` is substituted with the line under test; the hook
# runs from line 4 to line 7, and line 8 is shipped code that must stay outside.
NESTED = """\
impl Index {
    fn fit(&mut self) {
        let fitted = catch(|| {
            #[cfg(test)]
            if FORCE_FIT_PANIC.with(|f| f.replace(false)) {
%s
            }
            encode::fit_calibration(fit_src, 4096, dim, rotation, &mut scratch)
        });
    }
}
"""


class NestedRegionExtent(unittest.TestCase):
    """A stray `{` in a non-code position must not extend the region."""

    def assert_hook_only(self, hook: str) -> None:
        covered = regions(NESTED % hook)
        self.assertEqual(covered, {4, 5, 6, 7}, f"hook line: {hook!r}")

    def test_plain_body_is_the_baseline(self):
        self.assert_hook_only('                panic!("forced fit panic");')

    def test_line_comment(self):
        self.assert_hook_only("                // test-only, see #373 {")

    def test_block_comment(self):
        self.assert_hook_only("                /* test-only { */")

    def test_string_literal(self):
        self.assert_hook_only('                panic!("forced fit panic {");')

    def test_char_literal(self):
        self.assert_hook_only("                assert_eq!(c, '{');")

    def test_raw_string(self):
        self.assert_hook_only('                assert_eq!(s, r#"a { b"#);')

    def test_escaped_unicode_char_literal(self):
        self.assert_hook_only(r"                assert_eq!(c, '\u{7b}');")

    def test_two_stray_braces_do_not_fail_closed(self):
        # Even parity used to balance out into a *shorter* region rather than a
        # longer one; both directions are wrong, and neither happens now.
        self.assert_hook_only("                // {{ test-only")


class PrefixedLiterals(unittest.TestCase):
    """A `b`/`c`/`r` prefix is consumed with the literal it opens.

    Brace counts do not notice a leftover prefix, so these assert on
    `strip_literals` directly: the guarantee is that a literal span is deleted
    *whole*, and a stray `b` left where `b"xy"` was is that guarantee failing.
    """

    def assert_stripped(self, line: str, expected: str) -> None:
        self.assertEqual(gate.strip_literals([line]), [expected])

    def test_byte_string(self):
        self.assert_stripped('let a = b"xy"; Z', "let a = ; Z")

    def test_c_string(self):
        self.assert_stripped('let a = c"xy"; Z', "let a = ; Z")

    def test_byte_char(self):
        self.assert_stripped("let a = b'x'; Z", "let a = ; Z")

    def test_raw_prefixes(self):
        self.assert_stripped('let a = br"xy"; Z', "let a = ; Z")
        self.assert_stripped('let a = cr#"xy"#; Z', "let a = ; Z")

    def test_raw_identifier_is_not_a_literal(self):
        # `r#type` is a raw *identifier*; it must survive untouched.
        self.assert_stripped("match r#type { }", "match r#type { }")

    def test_prefix_letter_inside_an_identifier(self):
        # The `b` is the tail of a name, not a prefix, so it stays; the string
        # that follows is still stripped.
        self.assert_stripped("foo.b;", "foo.b;")
        self.assert_stripped('let x = sub"a";', "let x = sub;")


class MultiLineLiterals(unittest.TestCase):
    """Comments and strings that span lines carry state across them."""

    def test_multi_line_block_comment(self):
        self.assert_covered(
            """\
mod outer {
    #[cfg(test)]
    fn t() {
        /* a {
           still in the comment
        */
    }
    fn shipped() {}
}
""",
            {2, 3, 4, 5, 6, 7},
        )

    def test_multi_line_string(self):
        self.assert_covered(
            """\
mod outer {
    #[cfg(test)]
    fn t() {
        let s = "a {
            still in the string";
    }
    fn shipped() {}
}
""",
            {2, 3, 4, 5, 6},
        )

    def assert_covered(self, src: str, expected: set[int]) -> None:
        self.assertEqual(regions(src), expected)


class PreservedBehaviour(unittest.TestCase):
    """Shapes the fix must not have changed."""

    def test_top_level_mod(self):
        self.assertEqual(
            regions(
                """\
pub fn shipped() {}

#[cfg(test)]
mod tests {
    #[test]
    fn t() {}
}
"""
            ),
            {3, 4, 5, 6, 7},
        )

    def test_blockless_item(self):
        self.assertEqual(
            regions(
                """\
#[cfg(test)]
use super::*;

pub fn shipped() {}
"""
            ),
            {1, 2},
        )

    def test_cfg_test_inside_a_comment_opens_nothing(self):
        # Already true before the fix — the attribute pattern is anchored — and
        # still true now that it is matched against stripped text.
        covered, failure = gate.cfg_test_regions(
            """\
mod outer {
    // #[cfg(test)]
    fn shipped() {}
}
""".splitlines()
        )
        self.assertIsNone(failure)
        self.assertEqual(covered, set())

    def test_unterminated_region_still_fails_closed(self):
        covered, failure = gate.cfg_test_regions(
            """\
mod outer {
    #[cfg(test)]
    fn t() {
""".splitlines()
        )
        self.assertEqual(covered, set())
        self.assertIn("could not find the end", failure or "")


class EndToEnd(unittest.TestCase):
    """The whole gate, over a real two-commit git history."""

    def run_gate(self, base_src: str, head_src: str) -> tuple[int, str]:
        with tempfile.TemporaryDirectory() as d:
            def git(*args: str) -> None:
                subprocess.run(["git", "-C", d, *args], check=True, capture_output=True)

            src = Path(d) / "turbovec" / "src"
            src.mkdir(parents=True)
            git("init", "-q", ".")
            git("config", "user.email", "t@t.t")
            git("config", "user.name", "t")
            (src / "lib.rs").write_text(base_src)
            git("add", "-A")
            git("commit", "-qm", "base")
            base = subprocess.run(
                ["git", "-C", d, "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            (src / "lib.rs").write_text(head_src)
            git("add", "-A")
            git("commit", "-qm", "head")
            r = subprocess.run(
                ["python3", GATE, "--base", base, "--head", "HEAD"],
                cwd=d, capture_output=True, text=True,
                env={"PATH": "/usr/bin:/bin", "PR_BODY": "", "PR_LABELS": ""},
            )
            return r.returncode, r.stdout + r.stderr

    def test_shipped_change_after_a_brace_comment_is_caught(self):
        # The comment with the stray `{` is already in `base`; the PR under
        # test only halves the calibration sample count on the line after the
        # hook. That line is shipped code, so the gate must fail.
        base = NESTED % "                // test-only, see #373 {"
        rc, out = self.run_gate(base, base.replace("4096", "2048"))
        self.assertEqual(rc, 1, out)
        self.assertIn("turbovec/src/lib.rs", out)

    def test_test_only_change_is_still_exempt(self):
        # The other direction: changing a line genuinely inside the hook must
        # still pass, or the fix would have turned the gate into noise.
        base = NESTED % '                panic!("forced fit panic");'
        rc, out = self.run_gate(base, base.replace("forced fit", "forced calibration"))
        self.assertEqual(rc, 0, out)
        self.assertIn("No substantive changes", out)


if __name__ == "__main__":
    unittest.main()
