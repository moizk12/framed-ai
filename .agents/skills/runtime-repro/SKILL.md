---
name: runtime-repro
description: Reproduce a runtime failure before patching it. Use when a live test, server, gate, flake, crash, or integration failure must be reproduced, classified, and fixed from evidence rather than from a prompt description.
disable-model-invocation: true
---

# Runtime reproduction

Reproduce first. Patch second. Do not infer a fix from a report when the failure can be re-run.

## Workflow

1. Record the exact repository SHA (`git rev-parse HEAD`) and branch.
2. Create isolated evidence/runtime state. Do not reuse a failed database, cache, or working directory unless the task explicitly requires that artefact.
3. Reproduce with the same entrypoint, environment, and fixtures the failure used. Capture command, exit status, and timestamp.
4. Save useful logs outside tracked source when appropriate. Redact secrets, tokens, cookies, and personal data.
5. Classify the failure before editing:

- **A — environment/setup**
- **B — fixture/data**
- **C — harness/tooling**
- **D — implementation**

6. Identify the smallest causal mechanism. Quote the failing check, stack frame, or invariant.
7. Patch only in-scope implementation or harness defects. For A/B, fix the environment or fixture and continue. Do not broaden into unrelated hardening.
8. Rerun from fresh state after the patch. Require the original failing assertion to pass.
9. If a new independent defect appears, stop broadening. Report reproduction evidence and the smallest next action.

## Return

```text
SHA:
Evidence path:
Reproduced: yes/no
Category: A/B/C/D
Cause:
Patch:
Rerun: pass/fail
New independent defect: yes/no
```
