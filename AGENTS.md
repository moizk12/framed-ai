# FRAMED agent instructions

Stable repository truth. Load this file. Load a Skill only when the task needs that procedure. Do not restate this document in prompts or handoffs.

## Mission

FRAMED explores a customizable artificial mind through visual intelligence. Photography is the first bounded world.

The durable cognitive programme is:

experience → provenance-preserving storage → later retrieval → changed deliberation → feedback/outcome → proposed update → replay/evaluation → controlled promotion → measurably improved later behaviour

Research ambition may be broad. Operational authority must stay explicit, auditable, versioned, and reversible.

Do not claim AGI, consciousness, sentience, human-equivalent understanding, or autonomous self-improvement beyond what evidence proves.

## Two-track boundary

Track A is the public beta. Track B is isolated cognition research.

- Track A: upload → Balanced Mentor critique → evidence → limitations → feedback.
- Track A does not expose cognition, ECHO, memory controls, research identities, ledgers, run IDs, Slice A controls, TestDaemon history, mentor-mode switching, or automatic long-term continuity.
- Track B uses its own persistence and identity. Public databases and research cognition databases must never silently become the same authority.
- Track A does not automatically deploy Track B. Track B does not wait for Track A unless a genuinely shared contract requires it.

## Public product (Track A)

- Flask, server-rendered HTML, modular CSS, progressive vanilla JavaScript.
- Do not migrate the public beta to React/Next/Vue/Svelte without an explicit future decision.
- Public persistent state will eventually use PostgreSQL.
- Production browser code calls the versioned public API only (`/api/v1/analyses`, `/api/v1/feedback`). No production fallback to legacy `/analyze`.
- Render untrusted text with safe DOM APIs. Do not introduce HTML-injection sinks.
- Public beta keeps Balanced Mentor as the only public critique mode.

## Research cognition (Track B)

- Separate experience from belief. Memory is provisional until explicitly promoted.
- Require deterministic replay, explicit baselines, and rollback.
- Do not automatically promote research state into production behaviour.
- Explicit research/live-gate experiments must not treat HTTP 200 plus empty cognition as success. A required cognitive run must exist and reach a terminal state.
- Do not begin Slice B, expose cognition in the public UI, merge, tag, or promote research state without explicit human authorization.

## Git boundaries

Canonical repository: `https://github.com/moizk12/framed-ai.git`

- GitHub is canonical versioned history.
- Do not develop in OneDrive or Music copies of this repository.
- New work belongs in clean/disposable clones under `C:\Dev\FRAMED\`.
- Keep feature branches isolated. Do not merge frontend and cognition branches into each other.
- Do not merge to `main`, tag a release, deploy production, or force-push shared/preserved branches unless a later task explicitly authorizes that action.
- Never use `git reset --hard` or `git clean -fd` against preserved or shared branches unless explicitly authorized.
- Normal feature-branch commits and non-force pushes are allowed when the task requests them.
- Do not modify preserved Research Starter material or reopen closed preservation exceptions.

## Engineering style

- Prefer implementation over speculative planning once requirements are clear.
- Inspect before changing. Reuse existing architecture before adding abstractions.
- Ship the smallest complete solution. Do not broaden a task into unrelated cleanup.
- Investigate deeply when evidence requires it. Do not manufacture work.
- Report blockers with reproduction evidence and the smallest next action.
- Minimize repeated context: targeted search, batched reads, no full-repo rereads, no restating loaded Skills.

Autonomy for normal implementation: inspect → implement → test → fix → verify → commit when authorized. Stop only for genuine ambiguity, destructive actions, an authority boundary above, or evidence of a substantially different problem.

## Verification

Discover repository-specific commands. Do not invent a universal test command.

- Completing an implementation: `.agents/skills/verify-change/SKILL.md`
- Reproducing a runtime failure before patching: `.agents/skills/runtime-repro/SKILL.md`
- Verifying real UI: `.agents/skills/visual-qa/SKILL.md`
- Preparing a branch or pull request: `.agents/skills/safe-pr/SKILL.md`

Handoffs contain state and evidence, not essays. Distinguish proven, reported, and not tested.
