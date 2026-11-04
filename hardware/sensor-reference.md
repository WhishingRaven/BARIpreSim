# VL53L7CX reference and the six-reading interface

Manufacturer reference checked on 2026-11-03: ST DS13865 Rev 8, September 2024.
The chip's nominal detection volume is 60 degrees horizontally and vertically
(90 degrees diagonally), with 4x4 or 8x8 zone output. ST qualifies detection volume
by scene, configuration, distance, reflectance and ambient conditions. These are
manufacturer specifications, not measurements of the assembled robots.
[ST datasheet, sections 1 and 2.2](https://www.st.com/resource/en/datasheet/vl53l7cx.pdf).

The default simulator deliberately reduces sensing to six planar center rays
with zero angular width. Its 800 mm maximum, 15 mm minimum, noise, sampling and
latency remain provisional simulation settings. Nominal chip FoV does not define
which hardware zones produce the six actor distances. Optional nonzero-FoV
configurations use exact horizontal cone minima as an explicitly unmeasured
sensitivity model. Hardware transfer needs a corresponding measured aggregation rule;
the existing ray checkpoints are not validated for that different observation
process.

The cone model has no vertical zones or floor returns and applies noise after
reducing to one scalar per module. Its width is not a measured effective FoV.
Object/bearing witnesses remain observer-only; the actor input stays six anonymous
distances. See the [sensing contract](../docs/contracts.md) and separate hypothetical
configuration files before comparing these results with physical zone data.

Zone selection also determines whether the floor participates. As a geometric
illustration, using the assumed 78 mm sensor height, a horizontal optical axis
and a downward ray 30 degrees from it gives a floor intersection at 156 mm
slant range, or 135 mm horizontal distance. This is a ray calculation, not a
prediction of the sensor's zone response or proof that a particular zone reports
the floor. It shows why taking a minimum across all zones cannot be assumed to
match the simulator's horizontal nearest-wall/obstacle/robot surface contract.

Hardware integration must record sensor poses and orientation, selected resolution
and zone indices, target-selection settings, the delivered distance/status fields,
and the rule producing one distance per direction. Capture floor, wall, robot,
partial-zone and occluded scenes to establish the effective angular/range response.
Treat unavailable readings consistently with the existing maximum-range fallback;
an actor validity mask still requires hardware support and a new contract.

The actor continues to receive six anonymous distances. Zone IDs, detected-target
counts, target status, object classes and observer ground truth are not new actor
features. Neither this reference nor a future aggregation prototype establishes
physical calibration, behavioral transfer, ESP32 peak SRAM or loop timing.
