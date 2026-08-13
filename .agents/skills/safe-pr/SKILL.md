---
name: safe-pr
description: Prepare a branch or pull request safely. Use when committing, pushing, opening or updating a PR, checking base/head SHAs, or producing a review handoff. Never force-push, merge, tag, or deploy unless the task explicitly authorizes that action.
disable-model-invocation: true
---

# Safe PR

Move only the intended branch. Preserve history. Default to non-destructive Git.

## Workflow

1. Verify base and head (`git fetch`, `git rev-parse`, `git status -sb`, `git log` against the intended base).
2. Require a clean tree for the files you intend to ship, except for the change being committed.
3. Fetch before push. Do not overwrite a remote head that has moved.
4. Inspect changed paths. Commit only the intended files.
5. Verify tests and evidence from `verify-change` or existing reports. Do not claim unrun checks.
6. Commit narrowly. Follow the repository's recent commit-message style. Prefer why over what.
7. Push the intended branch only. Use `-u` when creating upstream. Do not push unrelated local branches.
8. Never force-push by default. Never `--no-verify` unless the user explicitly asked.
9. Never merge, tag, or deploy unless the current task explicitly authorizes that action. Draft PRs stay draft until a human authorizes ready/merge.
10. Return a concise handoff.

## Return

```text
Branch:
Base:
Head SHA:
Pushed: yes/no
PR:
Tests:
Known limitations:
Merge/tag/deploy: no
```
