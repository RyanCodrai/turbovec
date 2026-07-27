---
name: intake
description: First-pass triage of a new issue — read it closely, dig through the repo for the code and prior work it touches, assess what kind of issue it is and roughly what it would take, then finish with a plain-English summary. Use for triaging an issue.
---

# Intake (issue triage)

Triage this issue. Work through the steps in order, then post one comment
with your findings.

## 1. Understand the issue

- Read the issue and every comment. Work out what the reporter is actually
  asking for, and whether it's a bug, a feature request, a question, or a
  design discussion.
- Pull out the concrete specifics: versions, file names, error messages,
  reproduction steps, and any linked issues, PRs, or branches.

## 2. Dig into the repo

- Find the code the issue touches. Search the tree and read the relevant
  files — the modules, functions, and on-disk formats involved.
- Check the concern against the *current* code. Things move; part of it may
  already be handled, or the described behavior may have changed.
- Look for related prior work you can reach: earlier issues or PRs on the
  same area, existing tests, TODOs, or design notes.

## 3. Assess

- Say what kind of issue it is and how confident you are that it's valid.
- Point to the specific files and functions that would change, and give a
  rough sense of scope — small and localized vs broad and cross-cutting.
  A direction, not a precise estimate.
- Flag anything risky, ambiguous, or that needs a maintainer decision.
- Be explicit about what you could not verify. In this environment you can
  read and search the code but not build, run, or test it, so don't claim a
  bug reproduces or that anything compiles — reason from the code and say
  when you're inferring rather than confirming.

## 4. Summarize

Finish with a plain-English wrap-up for a non-expert, following the
`summarize` skill: two or three short paragraphs, direct, no hype, no
headers or bullet points in this closing part. Say what the issue is about
and why it matters, then where it lands in the code and roughly what
addressing it involves.
