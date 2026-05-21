# Contributing

Thanks for your interest in turbovec.

## Workflow

1. **Open an issue** describing the change and your proposed approach.
2. **Discuss** — leave space for a 👍 or design back-and-forth before writing code. The issue is where the design conversation lives.
3. **Open a PR** referencing the issue with `Closes #N`.

This applies to features, refactors, behaviour changes, and anything touching public API, on-disk format, or recall. The narrow exceptions that can skip the issue step:

- Typo / wording fixes
- One-line obvious bug fixes
- Documentation-only PRs

Everything else — including "I think this is small" — wants an issue first. The cost of writing one is low; the cost of building something that doesn't fit the project is high.

## Commit and PR conventions

- **One logical change per PR.** Refactors get their own PR, separate from feature work.
- **Commit messages:** short imperative title, body explaining *why* (the *what* is in the diff). Multi-line bodies should preserve formatting — use a HEREDOC if writing from the shell.
- **PRs reference their issue** with `Closes #N` and include a test plan.
- **`Co-Authored-By:` trailers** are fine on commits where Claude or another tool collaborated — leave them in place.

## Integration contributions

If you're adding or modifying an integration (LangChain, LlamaIndex, Haystack, Agno, or a new framework), structurally compare against the canonical in-tree reference store (`InMemoryVectorStore`, `SimpleVectorStore`, `InMemoryDocumentStore` etc.) for that framework. The wrappers should match the reference's surface and idioms — that's the bar for a drop-in replacement.

## Build, test, bench

See the [Building](README.md#building) and [Running benchmarks](README.md#running-benchmarks) sections of the README. To run the integration test suites (LangChain, LlamaIndex, Haystack, Agno), install the corresponding extras — otherwise they're skipped:

```bash
pip install -e ".[langchain,llama-index,haystack,agno]"
```
