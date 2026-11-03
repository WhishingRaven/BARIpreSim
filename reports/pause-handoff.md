# User-requested pause

Historical pause handoff. The user has resumed the goal; see the current
[retention and coherence report](retention-and-coherence.md). Original 0.5 m
continuations remain stopped and preserved.

The current default arena radius is 1.0 m (diameter 2.0 m). Defaults, JSON
configurations, AGENTS.md, documentation, camera bounds, ground rings, HUD, and
print generator were updated. Robot dimensions, target spacing, sensor range,
timing, and success gates are unchanged. The complete suite passed 87 tests in
24.65 s with RuntimeWarning treated as an error. Wall sensing/collision and camera
containment are checked at both radii. A wall-contact test now allows enough
physical travel to reach the enlarged boundary.

The native 1 m demo was visually inspected in both camera views. Pending task
selection while paused, continuous square/line/follow switches, speed adjustment,
sensor/occlusion toggles, recording save, and restart were exercised. Its continuous
recording is `runs/native-demo-radius-1m/demo.npz` (1,228 frames, 122.8 s). The window
is left paused at time zero. The displayed actor was trained at radius 0.5 m and
remains unvalidated; this inspection makes no behavioral-success claim.

All three long training processes and their validation observer were stopped.
[paused-training.json](paused-training.json) records verified atomic checkpoints,
hashes, completed updates, and commands for resuming the historical experiments
into new directories. The 1,000-update eight-world evaluations failed: 0/60 isolated
and 0/180 directed-switch successes. No model is accepted.

On resume, honor the revised 1 m target. Decide whether historical training remains
useful before spending more compute. Larger-arena training needs a separate frozen
configuration and results. Checkpoints retain their saved radius unless an explicit
compatible configuration overrides it for evaluation/demo.

The resized PDFs pass geometry checks and all pages are rendered, but final visual
review remains pending. Do not deliver them as print-ready. Former 0.5 m PDFs and
evidence are archived. Remaining goal work includes converged GRU behavior, all
switching and controlled memory/occlusion comparisons, a successful learned
recording/export, no-PSRAM ESP32-S3 peak RAM and latency measurements, calibration,
and a measured physical arena.

The resumed user subsequently selected anonymous following. See
[the current task-definition and migration report](anonymous-following.md) before
resuming or interpreting any historical fixed-ID checkpoint. New version-10/11
training uses the revised goal; previous runs retain their recorded meanings.
