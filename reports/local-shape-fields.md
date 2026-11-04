# Local square and line feasibility references

No teacher passes every frozen maintenance/random-start gate. No larger data collection or learner is triggered; preserve all failures and unchanged numerical thresholds. The [evidence bundle](local-shape-fields.json) retains every
prospectively frozen protocol, source, local input, physical recording, failure,
geometry fixture and permutation replay. These are **unlearned host teachers**,
separate from every GRU checkpoint and deployed actor. Training updates are zero.

All use the radius-1 m world, five differential-drive disks and six nearest-range
observations under hypothetical planar FOV60. Body/motor dimensions, range/noise,
latency and control timing remain unmeasured assumptions. All-five error35mm,
hold3s, timeout30s, no contact/wall attempts and larger-cohort >=90% gates are
unchanged. Each teacher must also hold four already formed layouts at different
orientations before qualifying on randomized starts. Reused development sets are
separate from the final held-out seeds, which remain reserved.

The square controller uses the existing own leader-role bit as a proposed center
role for this task. It does not identify anyone else's role. That robot moves
toward a local wall-history center estimate, or an anonymous inferred-body
centroid when the wall estimate is unready. The other four use radius
square_side/sqrt(2), with local tangential spacing and closing constraints.
Evaluators still match every robot without a privileged target assignment.
This role/center choice is a documented simulation design, not a confirmed
hardware formation specification. A local hypothesis count is not a true count.

Squarev1 holds all four maintenance fixtures but acquires/holds only2/5 random
starts, with zero safety attempts. Seed2003's 35.351mm final error stays failed
against35mm. Last50-command reconstruction matches all five cases exactly.
In one failure, every outer robot estimates only the central body and issues
zero commands; all physical poses are unchanged for the last five seconds.

The [scalar-cone proof](local-square-cone-suppression.json) distinguishes this
from geometric centerline visibility. Each outer robot has direct centerline
visibility to at least two other corners, while its six nearest scalar outputs
contain only the central robot and walls. At the recorded headings, all four
outer robots have zero outer scalar hits. Static counterfactual headings20–40deg
reveal two outer robots in every case. These fixtures are observer-only raycasts,
not teleports, actual motion success or inputs supplied to a controller.
Closest-witness-ray occlusion statistics do not exhaust scalar aggregation loss.

Squarev2 changes one local sensing behavior: an outer-role robot with a ready
estimated center, estimated radius within40mm of the ring and zero inferred
outer candidates requests ordinary [-0.25,+0.25] wheel commands. This is a
physical in-place rotation under the same speed/acceleration limits. It neither
uses true visibility nor modifies sensors, bodies, world state or learned actor
commands. The nine common square initial poses and control time arrays match
exactly between versions. Histories remain independent and episode resets clear
them; no newly privileged sensor flag is introduced.

Squarev2's fixed20-case gate finishes at13/20 official successes and final holds,
with zero contact/wall attempts. The larger sample contradicts treating its5/5
initial result as reliable acquisition. A separately planned squarev3 retains
the same scan/history and adds temporary release only for an outer-role robot
seeing a locally inferred body40–160mm from its estimated arena center and
within400mm of itself. It uses the circle teacher's0.05m/s release gain and the
same70mm/s total cap; the centered-body hypothesis is exempt, and actual body
identity/role/count/position never selects release. Closing margins and all goal
thresholds remain unchanged. Every common initial pose/control-time array
matches v2; its complete gate outcomes are retained below, whether passed or failed.

The old line teacher has nonzero detour at the exact170mm spacing. A pure local
geometry fixture produces endpoint wheels[+0.615,-0.615] and[-0.615,+0.615] at a
correct line. Its lateral detour vector is9.783mm/s. The updated line teacher
uses accumulated wall/range history and reduces repulsion/detour activation to
158/165mm, so both vanish at170mm. Inferred closing clearance changes30 to15mm;
this is a controller parameter, not a reduced collision footprint or acceptance
threshold. This combined repair does not isolate each change's causal effect,
and narrower margins are not a formal physical safety guarantee. Exact-line
geometry now commands zero; exact-square and blind-center centroid fixtures,
own-role permutation and full episode-history reset checks also pass.

| Teacher | Condition | Official success | Final hold | Contact attempts | Walls | Mean final max error (mm) |
| --- | --- | --- | --- | --- | --- | --- |
| square_v1 | maintenance | 4/4 | 4/4 | 0 | 0 | 19.8 |
| square_v1 | development | 2/5 | 2/5 | 0 | 0 | 36.5 |
| square_scan_v2 | maintenance | 4/4 | 4/4 | 0 | 0 | 22.7 |
| square_scan_v2 | development | 5/5 | 5/5 | 0 | 0 | 9.2 |
| square_scan_v2 | larger | 13/20 | 13/20 | 0 | 0 | 47.1 |
| square_release_v3 | maintenance | 4/4 | 2/4 | 0 | 0 | 33.4 |
| square_release_v3 | development | 3/5 | 3/5 | 0 | 0 | 31.4 |
| line_history_v2 | maintenance | 4/4 | 4/4 | 0 | 0 | 1.3 |
| line_history_v2 | development | 3/4 | 3/4 | 0 | 0 | 29.0 |
| line_gain_v3 | maintenance | 4/4 | 4/4 | 0 | 0 | 2.2 |
| line_gain_v3 | development | 4/4 | 4/4 | 0 | 0 | 5.1 |
| line_gain_v3 | larger | 14/20 | 14/20 | 0 | 0 | 47.2 |

The line repair holds4/4 formed layouts and acquires/holds3/4 random starts with
zero safety attempts, versus2/4 for the historical detour arm on exactly the same
initial worlds and control times. Seed3011 still fails at94.3mm. Mean final error
below35mm is not all-case success. All initial-stage failures remain retained;
no model/data subset is filtered by outcomes. Larger cohorts are executed only
when their prospectively declared first gates pass.

A separately planned linev3 changes only the perpendicular PCA-alignment gain
from1 to3. The remaining v2 mechanism, 40mm/s translation cap, yaw bounds,
30s timeout and numerical gates stay fixed. It finishes its first random cohort
with4/4 official
successes and4/4
final holds. All eight common maintenance/development initial pose and time
arrays match v2. Its further gate results, if triggered, appear in the table;
this is a new preplanned parameter comparison, not an extension of a failed run.

Every new reference reproduces all300 commands after permuting all five local
observation/history streams, including the own-role field. No fixed robot index
drives a follower command. Teacher actions receive only local9, retained own
history and calibration; ground-truth coordinates, actual wheels/odometry,
classes/identities/counts, evaluator assignments and true visibility are absent.
Geometry truth in fixtures and error/visibility overlays is evaluator-only.
Large hypothesis tables remain host-teacher-only; these programs establish
neither ESP32 memory compliance nor a learned controller.

The [square scan clip](../output/gif/local-square-scan-unlearned-v2.gif) and
[line alignment clip](../output/gif/local-line-gain-unlearned-v2.gif) replay one
development success each, explicitly labeled UNLEARNED HOST TEACHER. They show
the three-deck/two-wheel meshes, arena, actual range arcs, targets and HUD.
All61 draws per clip preserve the complete replay environment/RNG and source
recording hashes; they execute no policy or physics steps. Initial/final PNGs
were visually inspected. These selected illustrative successes do not replace
the complete failed cohort results or establish a continuous learned four-task
demonstration. A first render's generic recurrent-control subtitle was corrected
in fresh v2 artifacts; the earlier files remain preserved.

Production Python/tests remain identical to the verified180-test source. No
default or top-level TODO changes. The full [completion plan](../docs/completion-plan.md)
still requires reliable learned four-task GRU-RL acquisition/maintenance,
unfinished/repeated switching, controlled occlusion and memory comparisons,
multiple training seeds/fresh held-out acceptance, continuous3D success,
final actor export and no-PSRAM ESP32-S3 full-firmware peak RAM/latency/deadline
measurements, physical calibration and a measured arena.
