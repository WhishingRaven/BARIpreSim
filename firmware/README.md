# ESP32-S3 actor benchmark

This ESP-IDF project benchmarks generated FP32 actor inference on an ESP32-S3
without PSRAM. It is not a motor/sensor firmware implementation. The target board
variant, flash capacity, physical control period, and hardware APIs remain open.
The current H64 near-chain following actor cross-compiles with **ESP-IDF 5.5.1**,
Xtensa GCC 14.2.0, CMake 3.30.2 and Ninja 1.12.1. All eight FP32 weight arrays
(103,944 bytes) reside in flash-mapped `.flash.rodata`. The benchmark application
binary is 348,944 bytes; linked allocations are 16,384 bytes IRAM and 71,259 bytes
D/IRAM, including 15,360 bytes of latency-sample instrumentation. See the
[current H64 build report](../reports/follow-rl50-esp32s3-build.md) for section
placement, effective configuration and provenance. This is a partial following
policy, not the final four-task model.

A separate [local steering operator probe](../reports/local-steering-esp32s3-build.md)
cross-builds the full local preprocessing/geometry/GRU/decoder pipeline with a
verification-only actor. Its image is 1,220,480 B, linked IRAM is 16,384 B and
D/IRAM is 262,923 B. Actor weights (108,044 B) and lookup arrays (853,260 B)
are in flash-mapped rodata. One robot's observer/hidden state is 168,696 B;
scratch is 35,100 B and benchmark timing storage is 7,680 B. No device was run.
This probe is separate from the near-chain learned policy above and does not
establish complete-firmware peak SRAM, inference latency or deadline compliance.
The [preparation report](../reports/local-steering.md) records training as pending.

PSRAM is disabled in the effective build. No identified target device is connected
and no on-device measurements have been performed. Static linker allocations do
not establish peak RAM compliance. The SDK and tools are isolated under
`/tmp/bari-esp-idf-5.5.1` and `/tmp/bari-idf-tools`; they are not repository
dependencies or board evidence.

[Historical H32 build provenance](../reports/esp32s3-cross-build.json),
[linker sizes](../reports/esp32s3-link-size.json), and the
[effective SDK configuration](../reports/esp32s3-sdkconfig.txt) are retained.
That older benchmark binary is `firmware/build/bari_actor_benchmark.bin` (260,112 bytes).
Static allocations are 16,384 bytes IRAM and 71,131 bytes D/IRAM; these figures do
not include all dynamic heaps/stacks and do not establish peak RAM compliance.
The 2 MB build flash setting and 100 ms benchmark deadline are provisional, not
confirmed physical board specifications. The compiled actor is an unvalidated
seed-11 version-2 checkpoint, used for operator/build feasibility only.

With the pinned [official ESP-IDF setup](https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/get-started/linux-macos-setup.html),
from the repository root, create a fresh project to preserve archived evidence:

```sh
python - <<'PY'
import shutil
shutil.copytree("firmware", "runs/firmware-follow-device",
                ignore=shutil.ignore_patterns("build", "sdkconfig", "sdkconfig.old", "export_data"))
PY
python -m bari export --checkpoint runs/constrained-follow-rl-terminal-23/checkpoint.pt --output runs/firmware-follow-device/main/export_data
cd runs/firmware-follow-device
idf.py set-target esp32s3
idf.py build
idf.py size
idf.py size-components
```

Choose the actual board's flash size in `idf.py menuconfig` before flashing, and
retain `sdkconfig`, toolchain/IDF versions, build logs, linker map, and size reports
with the experiment. The default control period of 100 ms is a benchmark assumption,
not a confirmed hardware requirement. The compile-time guard rejects PSRAM-enabled
builds. The SDK defaults request 240 MHz and an 8192-byte main task stack; record
the effective build settings, not just this defaults file.

After connecting the identified board, run `idf.py -p PORT flash monitor`. The
benchmark repeats the exported golden sensor/action/task-switch sequences 20 times.
It checks finite outputs and absolute action/hidden error against FP32 golden data,
and prints machine-readable `BARI_BENCH` metrics for inference p50/p99/worst latency,
benchmark-loop worst time, missed assumed deadlines, internal 8-bit heap total/minimum
free/peak used, main task minimum free stack, and parity. It never starts motors.

The golden inputs are fixed recorded inputs, including previous reference commands;
the C++ hidden state accumulates independently. Each golden reset also resets the
benchmark wheel-state decoder. Include physical sensor acquisition, task scheduling,
motor control, I/O, communications, all task stacks/heaps, and worst traffic in a
later complete-firmware measurement. Timing of this actor benchmark does not prove
that the final physical control loop meets its eventual deadline.

The latency sample array uses 15,360 bytes of static RAM at the default 96 golden
steps; golden fixtures consume flash. These are benchmark instrumentation, separate
from the deployed actor. Keep both instrumented and production build measurements.
The heap watermark excludes statically allocated DRAM/IRAM; combine linker memory
reports with dynamic heap use to account for the entire required firmware. Avoid
double-counting heap-allocated task stacks. Internal SRAM's nominal 512 KB is not
the amount automatically available for model buffers. Report measured total use
and headroom; do not claim compliance from weight bytes or free heap alone.

API references: [ESP-IDF ESP32-S3 RAM usage](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-guides/performance/ram-usage.html),
[heap capabilities and watermarks](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/api-reference/system/mem_alloc.html).
