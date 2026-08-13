---
name: visual-qa
description: Verify a real user interface by running it and inspecting layout, states, keyboard, focus, reduced motion, and overflow. Use when checking UI, CSS, screenshots, responsive behaviour, or visual polish, not when unit tests or snapshots are the only evidence.
disable-model-invocation: true
---

# Visual QA

Verify the running UI. Code review and screenshot snapshots are supporting evidence, not a substitute for looking at the product.

## Workflow

1. Run the actual UI with the repository's normal preview or test-server command.
2. Inspect at 320, 375, 768, and 1280 CSS pixels.
3. Inspect landing, loading, success, error, and empty states that exist in the current flow.
4. Check keyboard reachability for primary actions, skip links, and dialogs.
5. Check visible focus and that interactive controls are usable.
6. Check reduced-motion behaviour where the UI has motion.
7. Check overflow: no accidental horizontal scroll, clipped text, or untappable controls.
8. Capture and visually inspect screenshots. Do not trust filenames.
9. Iterate only on concrete observed issues. Record each issue as viewport + state + what is wrong.
10. Do not turn a polish pass into a redesign. Do not switch design systems or add decorative visual language that is not already in the product.

## Return

```text
Command:
URL:
Viewports: 320/375/768/1280
States inspected:
Issues found:
Changes made:
Remaining visual risk:
```
