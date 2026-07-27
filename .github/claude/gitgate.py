"""Deterministic, non-LLM git/gh helpers for the goal harness.

Everything here is factual — thin subprocess wrappers around `git` and `gh`.
These are the checks the model cannot hallucinate past: they establish that
work was actually committed, sits on the right branch, changes something, and
(post-push) that required CI is green. Delivery requires BOTH this gate AND an
independent verifier verdict — either alone is insufficient.
"""
from __future__ import annotations

import os
import subprocess
import time

REPO = os.environ.get("GITHUB_REPOSITORY", "")


def _run(args, check=False):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed:\n{r.stderr or r.stdout}")
    return r


# --- read-only facts -------------------------------------------------------

def current_branch() -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def tree_dirty() -> bool:
    return bool(_run(["git", "status", "--porcelain"]).stdout.strip())


def changed_files(base: str) -> list[str]:
    out = _run(["git", "diff", "--name-only", f"origin/{base}...HEAD"]).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def diff(base: str, limit: int = 200_000) -> str:
    return _run(["git", "diff", f"origin/{base}...HEAD"]).stdout[:limit]


def _is_guardrail(path: str) -> bool:
    return path.startswith(".github/workflows/") or os.path.basename(path) == "CODEOWNERS"


# --- mutations -------------------------------------------------------------

def auto_commit(message: str) -> None:
    """Safety net: never let a run end with uncommitted work in the runner."""
    _run(["git", "add", "-A"], check=True)
    _run(["git", "commit", "-m", message], check=True)


def push(branch: str) -> None:
    _run(["git", "push", "origin", f"HEAD:{branch}"], check=True)


def remote_matches_head(branch: str) -> bool:
    """Confirm the push actually landed (kills the 'claimed push vanished' bug)."""
    _run(["git", "fetch", "origin", branch])
    local = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    remote = _run(["git", "rev-parse", f"origin/{branch}"]).stdout.strip()
    return bool(local) and local == remote


def add_label(number: str, label: str) -> None:
    _run(["gh", "pr", "edit", str(number), "-R", REPO, "--add-label", label], check=True)


def comment(number: str, body: str, is_pr: bool = True) -> None:
    kind = "pr" if is_pr else "issue"
    _run(["gh", kind, "comment", str(number), "-R", REPO, "--body", body])


def open_pr(branch: str, base: str, goal: str) -> str:
    title = (goal.strip().splitlines() or ["Harness change"])[0][:70]
    r = _run(
        ["gh", "pr", "create", "-R", REPO, "--head", branch, "--base", base,
         "--title", title, "--body", f"Opened by the goal harness.\n\n## Goal\n{goal}"],
        check=True,
    )
    url = r.stdout.strip().splitlines()[-1]
    return url.rstrip("/").split("/")[-1]


# --- the gate --------------------------------------------------------------

def precheck(branch: str, base: str) -> tuple[bool, str]:
    """Non-hallucinable delivery gate. Returns (ok, findings)."""
    problems: list[str] = []
    br = current_branch()
    if br != branch:
        problems.append(f"HEAD is on `{br}`, not the target branch `{branch}`.")
    if br == "main":
        problems.append("Refusing to deliver from `main`.")
    if tree_dirty():
        problems.append("Working tree still has uncommitted changes.")
    files = changed_files(base)
    if not files:
        problems.append(f"No changes vs `{base}` — nothing was implemented.")
    guard = [f for f in files if _is_guardrail(f)]
    if guard:
        problems.append(
            f"Changes touch protected guardrails ({', '.join(guard)}); not auto-deliverable — a human must merge."
        )
    return (not problems, "\n".join(f"- {p}" for p in problems))


def poll_ci(pr: str, timeout: int) -> tuple[str, str]:
    """Poll required checks until settled or timeout. Returns (state, detail).

    state ∈ {"green", "red", "timeout"}.
    """
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        r = _run(["gh", "pr", "checks", str(pr), "-R", REPO, "--required"])
        last = r.stdout.strip()
        low = last.lower()
        if not any(s in low for s in ("pending", "in_progress", "queued")):
            return ("green" if r.returncode == 0 else "red", last or "(no required checks)")
        time.sleep(20)
    return ("timeout", last)
