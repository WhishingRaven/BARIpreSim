# ESP32-S3 actor benchmark

This ESP-IDF project benchmarks generated FP32 actor inference on an ESP32-S3
without PSRAM. It is not a motor/sensor firmware implementation. The target board
variant, flash capacity, physical control period, and hardware APIs remain open.
The benchmark cross-compiled successfully with **ESP-IDF 5.5.1**, Xtensa GCC 14.2.0,
CMake 3.30.2, and Ninja 1.12.1. PSRAM is disabled in the effective build. No target
device is connected and no on-device measurements have been performed. The SDK
and tools used here are isolated under `/tmp/bari-esp-idf-5.5.1` and
`/tmp/bari-idf-tools`; they are not repository dependencies or board evidence.

[Build provenance](../reports/esp32s3-cross-build.json),
[linker sizes](../reports/esp32s3-link-size.json), and the
[effective SDK configuration](../reports/esp32s3-sdkconfig.txt) are retained.
The benchmark binary is `firmware/build/bari_actor_benchmark.bin` (260,112 bytes).
Static allocations are 16,384 bytes IRAM and 71,131 bytes D/IRAM; these figures do
not include all dynamic heaps/stacks and do not establish peak RAM compliance.
The 2 MB build flash setting and 100 ms benchmark deadline are provisional, not
confirmed physical board specifications. The compiled actor is an unvalidated
seed-11 version-2 checkpoint, used for operator/build feasibility only.

With the pinned [official ESP-IDF setup](https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/get-started/linux-macos-setup.html),
from the repository root:

```sh
python -m bari export --checkpoint runs/smoke/checkpoint.pt --output firmware/main/export_data
cd firmware
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
