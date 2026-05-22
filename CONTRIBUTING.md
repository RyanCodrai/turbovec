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

## Integration contributions

If you're adding or modifying an integration (LangChain, LlamaIndex, Haystack, Agno, or a new framework), structurally compare against the canonical in-tree reference store (`InMemoryVectorStore`, `SimpleVectorStore`, `InMemoryDocumentStore`, etc.) for that framework. The wrappers should match the reference's surface and idioms — that's the bar for a drop-in replacement.

## Build, test, bench

See the [Building](README.md#building) and [Running benchmarks](README.md#running-benchmarks) sections of the README. To run the integration test suites (LangChain, LlamaIndex, Haystack, Agno), install the corresponding extras — otherwise they're skipped:

```bash
pip install -e ".[langchain,llama-index,haystack,agno]"
```
