# Contributing

Thanks for your interest in turbovec.

## Workflow

The default contribution flow for everyone:

1. **Open an issue** describing what you've spotted — a bug, a missing feature, a documentation gap, a performance question. Include enough context that the conversation can start without back-and-forth on what you mean.
2. **Discuss.** If you want to suggest an implementation approach, do that in the issue. The issue is where the design conversation lives — and where most of the value of a contribution actually lives.

That's it for most people. **Pull requests on this repo are by invitation.** I add contributors as collaborators when their engagement on issues makes it clear we're aligned on direction and that a PR from them will be worth reviewing. There's no checklist for that; it's a judgment call based on whether the issue discussion is insightful and the technical understanding is solid.

Once you're a collaborator and have an agreed-on approach on an issue, you can open a PR. Only I merge to `main`.

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
