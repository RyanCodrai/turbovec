# Contributing

Thanks for your interest in turbovec.

## Workflow

1. **Open an issue** describing what you've spotted — a bug, a missing feature, a documentation gap, a performance question. Include enough context that the conversation can start without back-and-forth on what you mean.
2. **Discuss.** If you want to suggest an implementation approach, do that in the issue. This is where the design conversation lives.
3. **Request contributor access** if you want to land code yourself. In your issue or in a follow-up, say so explicitly — e.g. "happy to take this; can I get contributor access?" I'll review the engagement so far and decide. If yes, I'll add you as a collaborator and you can open a PR for the issue.

If you don't request contributor access, that's fine — the issue itself is a valuable contribution, and I (or another contributor) may pick it up.

Only I merge to `main`.

## Why this is the workflow

The contributions that move turbovec forward are **good ideas, clearly articulated** — a sharp framing of a real problem, the right question to ask, an insight about how something should work, a benchmark observation that points at a real gap. That's the work I most need help with, and the work hardest to delegate. A well-written issue is more valuable than a PR.

The reason I've moved to invitation-only PRs is that the cost of reviewing a PR I have to mentally reconstruct from scratch — figure out what it's trying to do, why, whether it's correct, whether it fits the project's direction — is higher than just writing the change myself. This has become particularly acute with AI-assisted PRs that are technically clean but arrive without the design context or reasoning that makes review tractable. When the cognitive load of review exceeds the cognitive load of writing the change, the PR is a net loss for the project.

The "by invitation" gate isn't about credentials — it's about making sure the issue-side work has happened first, so when a PR arrives, review can be about *the code* rather than reconstructing *the why*. Contributors who've done that work via issue engagement are a joy to review. PRs that arrive cold without context aren't.

## For invited contributors: commit and PR conventions

- **One logical change per PR.** Refactors get their own PR, separate from feature work.
- **Commit messages:** short imperative title, body explaining *why* (the *what* is in the diff). Multi-line bodies should preserve formatting — use a HEREDOC if writing from the shell.
- **PRs reference their issue** with `Closes #N` and include a test plan.
- **`Co-Authored-By:` trailers** are fine on commits where Claude or another tool collaborated — leave them in place.

## The changelog gate

CI fails a PR that changes shipped code without touching `CHANGELOG.md`. It
exists because four consecutive fix commits landed a new cargo feature, two new
public API items, a new load-rejection class and a *removed importable module*
with no changelog line between them — nothing enforced it, so it depended on
whoever wrote the PR remembering.

The gate is narrow on purpose. It only looks at:

- `turbovec/src/**.rs` (excluding the test-only `kernel_tests.rs`)
- `turbovec-python/src/**.rs`
- `turbovec-python/python/turbovec/**.py`, which includes the four framework
  integrations

and within those it only counts lines that are not comments or blank. Tests,
benchmarks, examples, docs and workflows are out of scope entirely, and a
comment-only sweep of a shipped file does not trip it.

Write the entry under `## [Unreleased]`, under the surface it affects — the
Rust crate and the Python distribution version independently and each has its
own subsection. Describe the change as a user experiences it, and reference the
issue.

### Escape hatch

For a change that genuinely is not user-visible — an internal refactor, a
private helper, a docstring rewrite the comment heuristic can't see through —
say so explicitly, either way:

- add the **`skip-changelog`** label to the PR, or
- put **`[skip changelog]`** anywhere in the PR body.

Both re-trigger the gate when you add them, and both leave the decision
recorded on the PR, so "this needs no entry" is a visible claim someone can
disagree with rather than a silent omission.

## The mutation gate

A separate check mutates the code your PR touched and fails if the test suite
doesn't notice. It exists because an audit of fifteen fix commits found six
that shipped a test which also passes on the *unfixed* code — reverting the fix
left the suite green. A test that cannot fail isn't coverage.

It runs only against `turbovec/src` lines in your diff, and only up to a
per-PR cap; above that it samples across the diff rather than exhausting the
first file. A green tick on a large PR therefore means "the sample was clean",
not "every mutant was caught".

Each `MISSED` line names an edit to your change that nothing caught. Usually
the answer is an assertion that discriminates — if the fix is a perf change,
that means asserting the property the fast path is supposed to preserve, or
bounding the work done, not just re-checking the result.

### Escape hatch

Some mutants are genuinely equivalent, and some lines have no observable
semantics to assert on. Say so explicitly:

- add the **`skip-mutants`** label to the PR, or
- put **`[skip mutants]`** in the PR body.

## What CI checks

Beyond the release-profile test matrix, `ci.yml` runs:

- **Rust (debug profile).** The release matrix elides `debug_assert!` and
  integer-overflow checks, so the block-alignment and buffer-length invariants
  guarding the SIMD kernels never executed. This leg runs the unit tests plus
  the suites that drive those paths, in debug. The `io_v6` suite is excluded:
  unoptimized it is dominated by the per-load codebook solve and it carries no
  `debug_assert!` coverage of its own.
- **Clippy**, with an explicit allow-list of the lints the tree already
  triggers. New lint classes fail; the allow-list is a debt list, and burning
  entries off it is a welcome standalone PR.
- **Integration extras at their declared floors.** `pyproject.toml`'s `>=`
  constraints are turned into `==` pins and the four integration suites run
  against them, so the oldest supported release of each framework is actually
  executed rather than resolved past.

`cargo fmt --check` is deliberately *not* run — see the note in `ci.yml`.

## Integration contributions

If you're adding or modifying an integration (LangChain, LlamaIndex, Haystack, Agno, or a new framework), structurally compare against the canonical in-tree reference store (`InMemoryVectorStore`, `SimpleVectorStore`, `InMemoryDocumentStore`, etc.) for that framework. The wrappers should match the reference's surface and idioms — that's the bar for a drop-in replacement.

## Build, test, bench

See the [Building](README.md#building) and [Running benchmarks](README.md#running-benchmarks) sections of the README. To run the integration test suites (LangChain, LlamaIndex, Haystack, Agno), install the corresponding extras — otherwise they're skipped:

```bash
pip install -e ".[langchain,llama-index,haystack,agno]"
```
