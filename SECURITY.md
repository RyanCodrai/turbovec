# Security Policy

Thank you for helping keep turbovec and its users safe. Security reports are
genuinely appreciated.

## Reporting a vulnerability

**Please do not open a public issue for a security vulnerability.** A public
report tips off attackers before a fix is available.

Instead, report it privately through GitHub:

1. Go to the [**Security** tab](https://github.com/RyanCodrai/turbovec/security)
   of this repository.
2. Click **"Report a vulnerability"** to open a private advisory visible only
   to the maintainers.

This routes the report through GitHub's private vulnerability reporting, where
we can discuss, develop, and review a fix — and, if appropriate, request a CVE
— without disclosing the issue until a patch is ready.

### What to include

A good report makes a fix faster. Where you can, please include:

- the affected component (core Rust crate, Python bindings, or a specific
  framework integration) and the version,
- a description of the issue and its impact,
- a minimal reproduction — for a malformed-input bug, the crafted `.tv` /
  `.tvim` bytes or the API call sequence that triggers it,
- the platform/architecture if relevant.

## What to expect

- We aim to acknowledge a report within a few days.
- We will confirm the issue, keep you updated on the fix, and coordinate a
  disclosure timeline with you.
- With your permission, we will credit you in the published advisory.

Once a fix is released, we publish a GitHub Security Advisory so that the
vulnerability is recorded in the GitHub Advisory Database and downstream users
on crates.io and PyPI are alerted (e.g. via Dependabot) to upgrade.

## Supported versions

turbovec is pre-1.0 and ships frequently. Security fixes target the **latest
released version** of each surface:

- the `turbovec` crate on [crates.io](https://crates.io/crates/turbovec), and
- the `turbovec` distribution on [PyPI](https://pypi.org/project/turbovec/).

Please reproduce on the latest release before reporting where possible.

## Scope

In scope — for example:

- memory-unsafety or out-of-bounds access reachable from the public API,
- a panic, crash, or unbounded allocation triggered by loading an untrusted
  `.tv` / `.tvim` index file,
- a panic or silently-wrong result reachable from normal Python/Rust API use
  (e.g. malformed query input),
- data-integrity defects in the framework integration wrappers.

Generally out of scope:

- resource exhaustion driven only by a caller's own legitimately-large data on
  a supported (64-bit) target,
- behavior on unsupported targets (turbovec requires a 64-bit platform and
  refuses to compile elsewhere).

When in doubt, report it — we would rather triage an out-of-scope report than
miss a real one.
