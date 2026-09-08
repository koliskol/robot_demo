# Work Report — September 3, 2026

**Project:** Robot pick-and-place (Isaac Sim + trained robot policies)

## Summary

The robot needs two skills to move a box between two tables: a **pickup policy** (grab the box) and a **place policy** (set it down). Pickup was already trained. Today I finished training the place policy, checked that both are learning correctly, and built a way to test both policies together in one live simulation session — finding and fixing three real bugs along the way.

## What I did today

1. **Checked training status of both policies**
   Pickup was fully trained. Place policy's earlier training attempt had been cut off almost immediately (only 170 steps in, nothing saved) — it needed to be redone.

2. **Trained the place policy**
   Ran the full training job: 30,000 steps on 50 recorded demonstrations, about 1 hour 16 minutes. Finished cleanly with a low final training loss (0.049).

3. **Checked both models against real recordings**
   Replayed real recorded episodes through each trained model and compared its predictions to what the human operator actually did. Both came back low and consistent (pickup: 0.0097 rad average error, place: 0.0133 rad) — a good sign training worked. This confirms the models learned the data, not yet that they can control the robot live.

4. **Built one-session testing for both policies**
   Previously, testing pickup vs. place meant restarting the whole simulation each time. Added a shortcut so both work in the same session: press **M** to hand control to the pickup policy, **N** for the place policy, backed by two model servers running side by side.

5. **Found and fixed three bugs during live testing**
   - The simulation crashed right after a scene reset because a safety check ran one beat too early — fixed by waiting for the physics engine to catch up.
   - Closing the simulation window triggered the same crash — fixed the same way.
   - My first attempt at fixing the above introduced a worse problem: the simulation could freeze and flood 10 million+ log lines in minutes before needing a force-stop. Replaced it with a version that keeps the simulation moving while it waits, and gives up cleanly after 5 seconds instead of spinning forever.

6. **Live-tested the pickup/place switch**
   Connected to both models and switched back and forth repeatedly over about 4.5 minutes without issue — confirmed the switch itself works correctly.

7. **Scoped automatic driving between tables (postponed)**
   Discussed adding automatic navigation so the robot can drive itself between tables. Laid out two options — a full mapping/localization system, or a simpler "drive to a known spot" approach since both table locations are already fixed. Deliberately deferred to a later session.

## Next up

- Resume live pick-and-place testing: box starts on the first table, pickup policy picks it up, place policy sets it down on the second — matching how the real training data was collected.
- Revisit automatic driving between tables, starting with the simpler "known-location" approach.
