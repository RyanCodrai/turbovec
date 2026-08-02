---
name: s
description: Write a short, plain-English summary of the previous message, or with `all` of the whole conversation, issue, pull request, or diff. One or two paragraphs in everyday language, aimed at a non-expert. Use when asked to summarize, simplify, or explain something simply.
---

# Summary (plain English)

Summarize the previous message. Use this to restate something that was too
long or too technical.

Arguments (combinable):

- `all` — summarize everything in view: the whole conversation so far, or the
  whole issue, pull request, or diff this is attached to.
- `w N` — cap the summary at roughly N words.
- `l N` — write the summary as N lines.
- `p N` — write the summary as N paragraphs.

Style:

- Write flowing prose, one or two short paragraphs by default.
- Use plain English a non-expert can follow. Cover what the thing does and why
  it matters.
- Stay direct and matter-of-fact.
- Describe turbovec on its own terms.
- Keep every fact from the source, including its caveats, numbers, and
  uncertainty. Simplify the language and preserve the substance.
