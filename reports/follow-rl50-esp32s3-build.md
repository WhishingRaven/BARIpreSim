# Current H64 following actor: ESP32-S3 build

The unchanged H64 near-chain following actor now cross-compiles for ESP32-S3 with
PSRAM disabled. All eight FP32 weight arrays, totaling **103,944 bytes**,
are located in the ELF's flash-mapped `.flash.rodata`. This replaces neither the
historical H32 build nor the need for a final validated policy. The
[evidence bundle](follow-rl50-esp32s3-build.json) retains the build, section/symbol
placement, effective configuration, binaries, source hashes and tool versions.

| Quantity | Target build result | Meaning |
| --- | ---: | --- |
| Application binary | 348,944 B | Image including benchmark/runtime, distinct from model weight bytes |
| FP32 weight arrays | 103,944 B | All eight arrays in flash-mapped rodata |
| Linked IRAM | 16,384 B | Static allocation, not whole firmware peak RAM |
| Linked D/IRAM | 71,259 B | Static code/data/BSS allocation, not dynamic heap high-water |
| Independent hidden state | 256 B | One deployed robot; statically allocated in this benchmark |
| Golden input/output fixtures | 29,568 B | Flash instrumentation, not deployed model weights |
| Latency samples | 15,360 B | Static benchmark instrumentation included in linked D/IRAM |
| Device peak RAM / latency | Pending | No board execution or full firmware measurement |

Placement is verified using symbol addresses, containing ELF sections and the
linker scripts, rather than assuming an `nm` data letter means resident RAM.
No actual external-RAM allocation exists, and the effective SDK configuration
disables PSRAM. The `.ext_ram.dummy` NOLOAD placeholder skips the flash rodata
virtual range on the shared bus; it is not a PSRAM allocation. Host parity remains the previously measured maximum action/hidden
error of 6.11e-7 across 96 recurrent steps; this is not device parity.

The build uses the pinned
[official ESP-IDF 5.5.1 setup](https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/get-started/linux-macos-setup.html),
commit `fcae32885b0296b32044cb99ecbdc50d98dddb83`, Xtensa GCC 14.2.0, CMake 3.30.2 and Ninja 1.12.1.
SDK/tools are isolated under `/tmp`; durable provenance and generated build
artifacts remain in `runs/firmware-constrained-follow-rl50`. CPU 240 MHz, main
stack 8,192 B, flash 2 MB and benchmark deadline 100 ms are effective build
settings. The flash capacity and control deadline are assumptions, not board
specifications. An initial unsupported `idf.py -j` invocation failed before
compilation; the standard `idf.py build` then succeeded without changing the
model, preprocessing, benchmark source or configuration.

The benchmark preserves the ordinary local input/hidden contract, task contexts,
normalization, wheel decoding and reset sequences. The rejected command-history
feature prototype is not included. This actor has the scoped near-chain results
in the [following report](no-command-following.md); randomized acquisition and
four-task behavior remain incomplete. No actor is promoted based on compilation.

No serial device was opened, flashed or reset. There are no measured peak RAM,
inference p50/p99/worst latency, deadline, device parity or complete-firmware
headroom results. The required sensor, motor, communications and task runtime
must still fit the ESP32-S3's internal 512 KB SRAM with the actor and buffers,
without PSRAM. Static allocations and flash-mapped weights cannot prove that
requirement. Actual board measurements and the other requirements in the
[completion plan](../docs/completion-plan.md) remain open.
