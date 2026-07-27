"""Deterministic, non-LLM git/gh helpers for the goal harness.

Everything here is factual — thin subprocess wrappers around `git` and `gh`.
These are the checks the model cannot hallucinate past: they establish that
work was actually committed, sits on the right branch, changes something, and
(post-push) that required CI is green. Delivery requires BOTH this gate AND an
independent verifier verdict — either alone is insufficient.

Write credentials (GH_TOKEN) live ONLY in this orchestrator's process, never in
the LLM agent subprocesses (see harness.agent_env). Push injects the token
itself, because the checkout persists no credentials.
"""
from __future__ import annotations

import json
import os
import subprocess
import time

REPO = os.environ.get("GITHUB_REPOSITORY", "")


def _redact(s: str) -> str:
    tok = os.environ.get("GH_TOKEN", "")
    return s.replace(tok, "***") if tok else s


def _run(args, check=False):
    r = subprocess.run(args, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(_redact(f"{args[0]} failed:\n{r.stderr or r.stdout}"))
    return r


# --- read-only facts -------------------------------------------------------

def current_branch() -> str:
    return _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()


def head_sha() -> str:
    return _run(["git", "rev-parse", "HEAD"]).stdout.strip()


def tree_dirty() -> bool:
    return bool(_run(["git", "status", "--porcelain"]).stdout.strip())


def changed_files(base: str) -> list[str]:
    out = _run(["git", "diff", "--name-only", f"origin/{base}...HEAD"]).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def diff(base: str, limit: int = 200_000) -> str:
    return _run(["git", "diff", f"origin/{base}...HEAD"]).stdout[:limit]


def _is_guardrail(path: str) -> bool:
    # Protect the controls AND the harness's own code — a change to
    # .github/claude/*.py would be executed with secrets on the next run.
    return (
        path.startswith(".github/workflows/")
        or path.startswith(".github/claude/")
        or os.path.basename(path) == "CODEOWNERS"
    )


# --- mutations (orchestrator-only; hold GH_TOKEN) --------------------------

def auto_commit(message: str) -> None:
    """Safety net: never let a run end with uncommitted work in the runner.
    `git add -A` respects .gitignore, so ignored junk is not swept in."""
    _run(["git", "add", "-A"], check=True)
    _run(["git", "commit", "-m", message], check=True)


def _push_url() -> str | None:
    tok = os.environ.get("GH_TOKEN", "")
    return f"https://x-access-token:{tok}@github.com/{REPO}.git" if tok and REPO else None


def push(branch: str) -> None:
    url = _push_url() or "origin"
    _run(["git", "push", url, f"HEAD:{branch}"], check=True)


def remote_matches_head(branch: str) -> bool:
    """Confirm the push actually landed (kills 'claimed push vanished')."""
    _run(["git", "fetch", "origin", branch])
    local = head_sha()
    remote = _run(["git", "rev-parse", f"origin/{branch}"]).stdout.strip()
    return bool(local) and local == remote


def add_label(number: str, label: str) -> None:
    _run(["gh", "pr", "edit", str(number), "-R", REPO, "--add-label", label], check=True)


def comment(number: str, body: str, is_pr: bool = True) -> None:
    _run(["gh", "pr" if is_pr else "issue", "comment", str(number), "-R", REPO, "--body", body])


def open_pr(branch: str, base: str, goal: str) -> str:
    title = (goal.strip().splitlines() or ["Harness change"])[0][:70]
    r = _run(
        ["gh", "pr", "create", "-R", REPO, "--head", branch, "--base", base,
         "--title", title, "--body", f"Opened by the goal harness.\n\n## Goal\n{goal}"],
        check=True,
    )
    return r.stdout.strip().splitlines()[-1].rstrip("/").split("/")[-1]


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
            f"Changes touch protected paths ({', '.join(guard)}); not auto-deliverable — a human must merge."
        )
    return (not problems, "\n".join(f"- {p}" for p in problems))


def poll_ci(pr: str, timeout: int, grace: int = 90) -> tuple[str, str]:
    """Poll required checks by machine-readable bucket until settled or timeout.

    Returns (state, detail); state ∈ {"green", "red", "timeout"}.
    Note: the #261 executor independently re-checks CI before merging, so a
    lenient "no checks yet" grace here can never cause a red PR to merge.
    """
    deadline = time.time() + timeout
    start = time.time()
    detail = ""
    while time.time() < deadline:
        r = _run(["gh", "pr", "checks", str(pr), "-R", REPO, "--required", "--json", "bucket,name"])
        try:
            checks = json.loads(r.stdout) if r.stdout.strip() else []
        except json.JSONDecodeError:
            checks = []
        if not checks:  # none registered yet (or genuinely none required)
            if time.time() - start < grace:
                time.sleep(15)
                continue
            return ("green", "(no required checks)")
        buckets = [c.get("bucket") for c in checks]
        # gh emits: pass, fail, pending, skipping, cancel. A cancelled required
        # check is NOT a pass — treat fail+cancel as red.
        if "fail" in buckets or "cancel" in buckets:
            bad = [c["name"] for c in checks if c.get("bucket") in ("fail", "cancel")]
            return ("red", "failing/cancelled: " + ", ".join(bad))
        if "pending" in buckets:
            detail = "pending: " + ", ".join(c["name"] for c in checks if c.get("bucket") == "pending")
            time.sleep(20)
            continue
        # remaining are pass / skipping (skipping = deliberately not blocking)
        return ("green", "all required checks passed")
    return ("timeout", detail or "still pending at timeout")
