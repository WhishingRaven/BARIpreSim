# Command-derived motion history: rejected feature candidate

Adding nominal motion features does not pass the prospective information gate.
On the primary whole-episode linear readout, mean body-center error changes from
135.2 to
137.8 mm, a
-1.93% reduction (negative means
worse). The required reduction was at least 15%, without increasing p90 error.
**No new actor was trained, no deployed input contract changed, and no default
policy was promoted.** The [evidence bundle](command-model-history.json) preserves
the candidate, prospective rule, complete readouts and verification.

The motivation was a verified architectural difference: the constrained actor's
encoder ignores both previous-command fields exactly, whereas the successful
local circle teacher integrates its own command history. Changing only those
fields leaves both actor actions and hidden output bit-exact for fixed sensor
input and hidden state. That proves the restriction, not that it caused failed
closed-loop formation. The earlier ordinary command-input actors also failed.

The experimental preprocessor uses only the previous saturated requests, reset
masks and provisional motor/geometry calibration. It integrates acceleration-
limited wheel speeds and a nominal differential-drive pose relative to controller
reset. The six outputs are normalized relative x/y, sine/cosine of nominal yaw,
and normalized nominal left/right speed. They are derived history, not measured
velocity, hardware odometry, a true global pose, an arena-center estimate or a
per-robot target. There is no environment object, object ID, collision indicator,
visibility mask or evaluator input in the feature implementation.

The feature state persists through task switches and resets independently for
each robot only at an episode/controller reset. Serialization continuation and
robot permutation are exact. Its arithmetic matches the original local teacher's
nominal pose/wheels exactly; all 128 student-world wheel sidecars match exactly.
In a separate 200-control, collision-free physical check including two switches,
the relative pose error is below 2.2e-14. True poses are read only by that verifier.
Under a deliberately blocked wall approach, nominal displacement drifts by
273.3 mm. The model never receives
the collision or repairs that error. Consequently these values cannot be treated
as actual self-state.

The readout reuses the same 100/28 whole-episode split and local-teacher-derived
center labels as the [earlier readout](local-circle-state-readout.json). It fits
ridge regression on standardized training features, comparing current fields,
frozen hidden state, and their combinations with the six new features. The 28
episodes are held out for the readout only: the frozen actors saw this student
dataset during earlier collection/training studies. The labels themselves may
drift after contact, as documented in the
[student-label diagnostic](local-circle-student-label-diagnostic.json).
Readout targets, weights and readiness masks never control a robot.

This negative linear result does not prove that the history is useless to a
nonlinear recurrent policy, that the current sensors are sufficient or
insufficient, or that a larger actor would succeed. It fails the declared cheap
screen for this proposed follow-up and therefore prevents the conditional actor
pilot. No behavioral, occlusion or memory-benefit claim follows.

The prototype holds five float64 values per robot (40 bytes of numeric state),
plus six float32 output values. Those partial host counts exclude Python object
overhead, scratch, inference buffers and firmware. They are not an ESP32 memory
or timing measurement. Production Python/tests remain the unchanged 180-test
source. All requirements in the [completion plan](../docs/completion-plan.md),
including reliable four-task GRU-RL and measured no-PSRAM deployment, remain open.
