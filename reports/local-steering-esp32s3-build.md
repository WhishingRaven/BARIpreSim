# Local steering pipeline: ESP32-S3 operator build

The complete local range preprocessing, geometry lookup, GRU64 and three-field
steering decoder cross-compile for ESP32-S3 with PSRAM disabled. This uses a
**verification-only actor**, not a passed learned controller. The
[evidence bundle](local-steering-esp32s3-build.json) records the actual ELF,
map, binaries, source, effective settings, tool versions and host adaptation proof.

| Quantity | Linked/build result | Scope |
| --- | ---: | --- |
| Application image | 1,220,480 B | Benchmark/runtime and assets; separate from model weights |
| Eight actor weight arrays | 108,044 B | All in flash-mapped `.flash.rodata`; no critic/optimizer/log_std |
| Five geometry lookup arrays | 853,260 B | All in flash-mapped `.flash.rodata`, not SRAM copies |
| Linked IRAM | 16,384 B | Static allocation, not whole runtime peak |
| Linked D/IRAM | 262,923 B | Includes static observer/hidden, scratch and instrumentation |
| Observer plus GRU state | 168,696 B | Actual target struct size in `.dram0.bss`, one robot |
| Shared geometry scratch | 35,100 B | `.dram0.bss`; fully overwritten and reused |
| Latency sample array | 7,680 B | Static benchmark instrumentation |
| Golden fixtures | 39,936 B | Flash instrumentation |
| Device peak RAM / latency | Pending | No device execution or full firmware measurement |

Host full-pipeline parity covers 96 identical sensor/previous-command/task/reset
steps, with maximum feature/latent/wheel/hidden error 1.02e-6 and exact local
feature outputs. The target GCC14/libstdc++ build initially rejects the six-point
`std::sort` path with an array-bounds diagnostic. A fresh target source replaces
only that sorting implementation with bounded insertion sort and preserves the
same comparator. All 1,500 candidate-ID/coordinate cases remain bit exact, and
the full 96-step host pipeline retains its parity bound. No warning is disabled;
the failed build and unchanged training/host sources remain preserved.

The benchmark has no sensor or motor drivers. It is ready to report frontend and
actor/decoder timing separately, total inference percentiles/worst time, internal
heap high-water, stack headroom, parity and missed assumed deadlines on a board.
Heap high-water excludes static allocations and must be combined with the linker
evidence and required firmware workload. Current fields remain null because the
application has not run on a device.

ESP-IDF5.5.1 is pinned at `fcae32885b0296b32044cb99ecbdc50d98dddb83`. CPU240MHz, main stack8192B,
2MiB flash, a1.5MiB factory application slot and100ms control period are explicit
benchmark assumptions. The physical board variant, flash capacity and control
period remain unconfirmed. No serial port is accessed, board flashed or motor
command sent. The final accepted actor, sensor/motor/runtime integration and
full no-PSRAM peak SRAM/latency/deadline measurements remain required by the
[completion plan](../docs/completion-plan.md).
