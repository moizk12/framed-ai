---
name: verify-change
description: Verify an implementation before claiming completion. Use when finishing a code change, checking a diff, running tests/lint, classifying evidence as proven/reported/not-tested, or when the user asks to verify, validate, or confirm a patch is done.
disable-model-invocation: true
---

# Verify change

Confirm the intended change is complete, in-scope, and evidenced. Do not claim success from code inspection alone.

## Workflow

1. State the exact intended change in one or two sentences.
2. Inspect the actual diff (`git status`, `git diff`, and staged changes). Confirm every changed path belongs to the intended change.
3. Stop and split the work if unrelated files, refactors, or formatting-only churn appeared.
4. Discover this repository's test and check commands. Look for CI config, `package.json` scripts, `pyproject.toml`/`pytest.ini`, Makefiles, README test sections, and existing test folders. Do not assume one universal command.
5. Run the smallest targeted tests that cover the changed behaviour.
6. Run broader tests proportional to risk: adjacent suite for local changes; full relevant suite when contracts, persistence, routing, or shared UI behaviour changed.
7. Run syntax, lint, or static checks the repo already uses. Do not invent a new toolchain.
8. Run `git diff --check`.
9. Inspect final `git status`. Note leftover files, generated artefacts, and whether they should be committed.
10. Classify each important claim:

- **proven**: observed by a command, test, or runtime check in this session
- **reported**: taken from existing evidence or another agent, not re-run
- **not tested**: in scope but unverified

## Return

A short evidence block:

```text
Change:
Scope: clean / drifted
Commands:
Proven:
Reported:
Not tested:
Blockers:
```

If the repo's normal test environment is broken for an unrelated environment reason, record that failure instead of inventing a passing result.
