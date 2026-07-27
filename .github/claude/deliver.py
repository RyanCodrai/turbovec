#!/usr/bin/env python3
"""Agent-free delivery half of the @harness goal harness.

The credential boundary is a JOB boundary: the agent job runs the LLM loop
with no write token anywhere in its process tree (nothing to recover from
/proc/<pid>/environ), and this job holds the Write-level bot token but runs no
LLM. The two communicate through exactly one artifact: a manifest + git bundle.

Nothing from the agent job is trusted without re-verification here:
- the bundle's HEAD must equal the manifest's verified SHA;
- the deterministic gate (right branch, clean tree, non-empty diff, no
  guardrail paths: .github/workflows/, .github/claude/, CODEOWNERS) is re-run
  in THIS job, on trusted code, before any push;
- required CI must be green before the `claude-merge` label is applied.

HARNESS_MODE=shadow reports what would be delivered and pushes/labels nothing.

Auth: GH_TOKEN (CLAUDE_BOT_TOKEN, Write-level). The merge-bypass App token is
NOT here — it stays isolated in claude-merge.yml.
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gitgate as g  # noqa: E402

MODE = os.environ.get("HARNESS_MODE", "shadow")  # "shadow" | "live"
CI_TIMEOUT = int(os.environ.get("HARNESS_CI_TIMEOUT", "600"))
ART = Path(os.environ.get("HARNESS_ARTIFACT_DIR", "/tmp/harness-out"))


def _load_manifest() -> dict | None:
    p = ART / "manifest.json"
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text())
        return m if isinstance(m, dict) else None
    except json.JSONDecodeError:
        return None


def _post(m: dict, body: str) -> None:
    """Comment on the run's target — the PR when one exists (including one this
    job just opened, see run()), else the originating issue."""
    if m.get("pr"):
        num, is_pr = str(m["pr"]), True
    else:
        num, is_pr = str(m.get("issue") or ""), False
    if not num:
        print(f"no comment target; report was:\n{body}", file=sys.stderr)
        return
    try:
        g.comment(num, body, is_pr=is_pr)
    except Exception as e:  # never let reporting itself crash the run silently
        print(f"comment failed: {e}", file=sys.stderr)


def run() -> int:
    m = _load_manifest()
    if m is None:
        # The workflow's fallback step comments when the artifact never
        # arrived; an unreadable manifest inside one is a harness bug.
        print("no readable manifest in the delivery artifact", file=sys.stderr)
        return 1

    if m.get("status") != "verified":
        _post(m, m.get("report") or "⚠️ Harness did not produce a delivery report — see the agent job log.")
        return 1

    branch = str(m.get("branch") or "")
    base = str(m.get("base") or "main")
    sha = str(m.get("sha") or "")
    bundle = ART / "delivery.bundle"
    if not (branch and sha and bundle.exists()):
        _post(m, "⚠️ Harness delivery artifact is incomplete (missing branch, SHA, or bundle) — no label applied.")
        return 1

    got = g.fetch_bundle(str(bundle))
    if got != sha:
        _post(m, f"⚠️ Delivery bundle HEAD `{got[:12]}` does not match the verified SHA `{sha[:12]}` — refusing to deliver.")
        return 1
    g.checkout_sha(branch, sha)

    # Re-run the non-LLM gate here, on code from the trusted checkout — the
    # agent job's copy of this gate ran next to an LLM and is not relied on.
    ok, findings = g.precheck(branch, base)
    if not ok:
        _post(m, f"⚠️ Deterministic gate failed in the deliver job — no push, no label:\n\n{findings}")
        return 1

    files = ", ".join(f"`{f}`" for f in (m.get("files") or g.changed_files(base))) or "(none)"
    if MODE != "live":
        _post(
            m,
            "🔎 **Harness (shadow mode)** — verified and would deliver now.\n\n"
            f"- Verifier: PASS\n- Head: `{sha[:12]}`\n- Files: {files}\n\n"
            "Set `HARNESS_MODE: live` to enable push + `claude-merge` labelling.",
        )
        print("SHADOW: would push + label; stopping.")
        return 0

    g.push(branch)
    if not g.remote_matches_head(branch):
        g.push(branch)  # one retry; this is an infra issue, not the model's fault
        if not g.remote_matches_head(branch):
            _post(m, "⚠️ Harness pushed but the remote branch did not advance — aborting, no label applied.")
            return 1

    pr_num = str(m.get("pr") or "")
    if not pr_num:
        pr_num = g.open_pr(branch, base, str(m.get("goal") or "Harness change"))
        m["pr"] = pr_num  # later reports target the PR we just opened

    state, detail = g.poll_ci(pr_num, CI_TIMEOUT)
    if state != "green":
        _post(
            m,
            f"⚠️ Pushed the verified commits but required CI is not green ({state}) — "
            f"no `claude-merge` label applied. Re-trigger `@harness` to iterate on the failure.\n\n{detail}",
        )
        return 1

    g.add_label(pr_num, "claude-merge")
    _post(m, "✅ Delivered and verified — applied `claude-merge`.")
    print("DELIVERED")
    return 0


def main() -> None:
    try:
        code = run()
    except Exception:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        _post(_load_manifest() or {}, "⚠️ Harness deliver job crashed — no label applied. See the job log.")
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()
