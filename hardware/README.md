# Hardware integration and measurement

All geometry and timing values in `configs/config.json` are simulation assumptions.
The user confirmed an **ESP32-S3 with 512 KB internal SRAM and no PSRAM**. Flash
capacity, board variant, and physical control period remain unspecified. The actor,
preprocessing, hidden state, inference buffers, and required firmware runtime must
all fit internal SRAM. No target board has been measured. Copy `board.template.json` to the
experiment output and fill it from the actual board, build map, and instrumented
firmware. A generated header or small weight file is not evidence of RAM compliance.

The FP32 export contains a deterministic one-step actor and preprocessing, with no
critic, optimizer, training loss, or global-state code. Include `actor.hpp`, allocate
one `float hidden[bari::H] = {}` per physical robot, and preserve it between control
steps, during occlusion, and during task changes. Call `bari::preprocess` with six
distances in **meters**, the previous normalized requested wheel commands, and the
preassigned leader role. Pass mission task ID 0/1/2/3 for circle/square/line/follow.
Call `bari::step`; then saturate and acceleration-limit the output with `bari::decode`.
The resulting wheel values are rim speeds in m/s; conversion to motor control or
wheel rad/s needs calibrated wheel radius and the hardware team's motor API.

At controller reset, zero hidden state, previous commands, wheel state, and sensor
history. During task switches, change only mission context at the next control
boundary. Keep the previous command, sensor history, and hidden state. On a missing
reading, pass NaN or infinity; all unrecognized readings map to maximum range in the
same way as training. There is no object-class, neighbor-count, or validity-mask input.

The scripted simulation leader uses global state. It is **not** firmware for a
physical leader. Hardware must supply an observable leader route (for example a
preprogrammed calibrated motor trajectory) and validate it separately. Followers
still receive only their local ranges, mission context, and previous commands.

Before deployment, confirm the PDF envelope axes, wheel radius/separation, collision
footprint, motor signs/dead zones/speed curves, acceleration and stopping behavior,
six mounting directions, FOV/range, VL53L7CX zone aggregation, sampling/control timing,
latency, wall/robot IR response, odometry availability, and motor/sensor API ownership.
Do not add odometry or a validity mask before hardware support is confirmed. Update
the versioned contract and rerun training, evaluation, and export after material changes.

For the target build, measure linked flash and weight placement, peak internal RAM
and PSRAM, task stacks, heap high-water use, I/O buffers, preprocessing/history,
activations, hidden state, compiler scratch/spills, and all other firmware. Measure
inference and complete control-loop latency over repeated task switches and worst
sensor traffic. Record p50/p99/worst case, missed deadlines, test duration, and free
headroom against the agreed budget. The host report intentionally leaves these
fields unmeasured. Quantization is not implemented; any future quantization needs
new sequence parity and behavioral/occlusion/switching evaluation.

The [ESP32-S3 benchmark project](../firmware/README.md) provides PSRAM rejection,
golden-sequence parity, internal heap/stack watermarks, and inference timing. It
cross-compiles with ESP-IDF 5.5.1 and no PSRAM; it has not run on hardware here. Its instrumentation memory is
separate from the actor; use linked static memory plus dynamic runtime measurements
to account for the whole firmware without double-counting stacks allocated on heap.

After receiving robots, calibrate one robot safely before a five-robot run.

The [full-size arena sheet](../output/pdf/bari-arena-full-size.pdf) has a nominal
2,000 mm inner diameter on a 2,100 mm square page. The
[A4 assembly bundle](../output/pdf/bari-arena-a4-tiles.pdf) contains one guide plus
96 tiles (twelve columns, eight rows), with 10 mm overlap and matching registration
crosses. Both use vector geometry and request no viewer print scaling. Regenerate:

```sh
python -m pip install -e '.[hardware]'
python hardware/arena_template.py --output output/pdf
python -m pytest -q tests/test_arena_template.py
```

Print at **100% / Actual size** with Fit, Shrink and Poster disabled. Check a 100 mm
ruler before printing all tiles. Trim white margins to the dotted content frame,
overlap the artwork by 10 mm, align crosses and tape from the back. Put the wall's
inner face on the solid boundary centerline. Wall thickness and height are not
specified by the template. Axes and registration marks are assembly aids.

Digital checks verify PDF page sizes, unchanged physical point units, 100 mm rulers,
the 1,000 mm boundary radius, tile translations and complete boundary coverage. All
97 A4 pages were rendered and visually inspected, including all tile geometry,
headers, footers, scale labels and page counters.
The earlier radius-0.5 m files and visual evidence are archived separately.
Digital checks do not verify a printer, the
assembled diameter, wall sensing, or the physical arena. Copy
`arena.template.json` to the experiment output and record actual printer/scale,
horizontal and vertical inner diameters, wall height/material, and IR response.
Generated manifests retain null physical measurements and can be regenerated;
keep the actual build record separately. Validate boundary sensing and stopping
before closed-loop formation demonstrations.
