# Work Report — September 7, 2026

**Project:** Robot pick-and-place (Isaac Sim + trained robot policies)

## Summary

Picked up where the Sept 3 session left off: tested the pickup + place policies together live again, got feedback that they've improved but still fail sometimes, traced that to a real gap in the training data (every recorded box was always the same size), and added a way to fix it — with the new capability tested live before handing it back over.

## What I did today

- **Reviewed everything left in progress from the last session** — confirmed the pickup/place dual-policy switching feature (M/N keys) and its three bug fixes were all still intact and consistent with each other, and checked nothing else had silently broken.
- **Re-ran the live two-policy test.** Started both trained models as servers (pickup — the newest, most-trained version — and place) and reconnected the simulation to both.
- **Got real feedback from watching it live:** pickup and place are noticeably better than before, but still fail sometimes.
- **Investigated why.** Confirmed a real gap: every recording session so far used one fixed box size for the entire session — only the box's position and rotation were ever randomized between attempts, never its size. So the policy has never actually seen size variation, which likely explains some of the remaining failures.
- **Decided on a fix, weighing the options first.** Could have (a) just re-run recording sessions manually with a few different box sizes, (b) added real randomization into the existing recording tool, or (c) built a whole new tool. Went with (b) — most useful, least duplicated effort — after confirming that with you.
- **Built it:** the recording tool can now randomize the box's size on every single recording, not just its position — opt-in via new flags, so it doesn't change anything for anyone not using it.
- **Tested the launch before trusting it.** First attempt showed no visible difference (a logging issue hid the actual sampled value) — traced and fixed that, then confirmed on a clean run that box size really is being randomized each time (an 18%-smaller box on the test run, logged and visually different) with no new errors introduced.
- **Handed back to live two-policy testing**, currently in progress with both models connected.

## Known open items

- The new box-size randomization needs a few more live resets watched (in progress) to confirm the robot can still successfully grab differently-sized boxes, not just that the size changes visibly.
- A recurring (so far harmless-looking) error message shows up in the logs after every simulation reset — logged for follow-up, not fixed today.
- One old data folder from an earlier test isn't currently protected from being accidentally committed — flagged, not yet resolved.

## Next up

- Watch a full pick → place cycle succeed live with the new size-randomized box.
- Once confirmed reliable, record a new, larger dataset that actually includes size variation.
- Retrain both policies on that improved dataset.
