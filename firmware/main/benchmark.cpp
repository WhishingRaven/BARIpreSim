// Actor benchmark only: no motor output or sensor-driver claim.
#include "sdkconfig.h"
#if !defined(CONFIG_IDF_TARGET_ESP32S3)
#error "BARI deployment target is ESP32-S3"
#endif
#if defined(CONFIG_SPIRAM) && CONFIG_SPIRAM
#error "PSRAM is prohibited by the deployment contract"
#endif
#include "actor.hpp"
#include "golden_fixture.hpp"
#include "esp_chip_info.h"
#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>

namespace {
constexpr int REPEATS=20;
constexpr int SAMPLES=bari_golden::STEPS*REPEATS;
// Provisionally matches simulation, not an agreed physical control deadline.
constexpr int CONTROL_PERIOD_US=100000;
static int64_t inference_us[SAMPLES];
static float hidden[bari::H];
static float wheels[2];
}

extern "C" void app_main() {
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    std::printf("BARI_BENCH target=ESP32-S3 cores=%d idf=%s psram_allowed=false\n",chip.cores,esp_get_idf_version());
    const uint32_t caps=MALLOC_CAP_INTERNAL|MALLOC_CAP_8BIT;
    const size_t heap_total=heap_caps_get_total_size(caps);
    int64_t worst_loop_us=0;
    int missed_deadlines=0;
    float maximum_parity_error=0;
    int sample=0;
    const int64_t started=esp_timer_get_time();
    for(int repeat=0;repeat<REPEATS;repeat++) {
        std::fill(hidden,hidden+bari::H,0.f);
        std::fill(wheels,wheels+2,0.f);
        for(int t=0;t<bari_golden::STEPS;t++) {
            const int64_t loop_start=esp_timer_get_time();
            const float* row=bari_golden::inputs[t];
            if(row[10]>0) std::fill(wheels,wheels+2,0.f);
            float local[9], action[2];
            bari::preprocess(row,row+6,row[8]>0,local);
            const int64_t inference_start=esp_timer_get_time();
            bari::step(local,static_cast<int>(row[9]),hidden,action,row[10]>0);
            inference_us[sample++]=esp_timer_get_time()-inference_start;
            for(int i=0;i<2+bari::H;i++) {
                const float actual=i<2 ? action[i] : hidden[i-2];
                const float error=std::fabs(actual-bari_golden::expected[t][i]);
                if(!std::isfinite(error)) {
                    std::printf("BARI_BENCH nonfinite_output=true\n");
                    return;
                }
                maximum_parity_error=std::max(maximum_parity_error,error);
            }
            bari::decode(action,wheels);
            const int64_t elapsed=esp_timer_get_time()-loop_start;
            worst_loop_us=std::max(worst_loop_us,elapsed);
            if(elapsed>CONTROL_PERIOD_US) missed_deadlines++;
            // Yield for watchdog/scheduler; no claim to emulate physical sensor traffic.
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }
    std::sort(inference_us,inference_us+SAMPLES);
    const size_t minimum_free=heap_caps_get_minimum_free_size(caps);
    std::printf("BARI_BENCH {\"status\":\"actor-benchmark-only\",\"samples\":%d,"
                "\"duration_us\":%lld,\"inference_p50_us\":%lld,\"inference_p99_us\":%lld,"
                "\"inference_worst_us\":%lld,\"benchmark_loop_worst_us\":%lld,"
                "\"assumed_control_period_us\":%d,\"missed_benchmark_deadlines\":%d,"
                "\"internal_8bit_heap_total_bytes\":%u,\"internal_8bit_heap_min_free_bytes\":%u,"
                "\"internal_8bit_heap_peak_used_bytes\":%u,\"main_stack_min_free_bytes\":%u,"
                "\"max_action_hidden_absolute_error\":%.9g,\"parity_pass\":%s}\n",
                SAMPLES,static_cast<long long>(esp_timer_get_time()-started),
                static_cast<long long>(inference_us[SAMPLES/2]),
                static_cast<long long>(inference_us[SAMPLES*99/100]),
                static_cast<long long>(inference_us[SAMPLES-1]),static_cast<long long>(worst_loop_us),
                CONTROL_PERIOD_US,missed_deadlines,static_cast<unsigned>(heap_total),
                static_cast<unsigned>(minimum_free),static_cast<unsigned>(heap_total-minimum_free),
                static_cast<unsigned>(uxTaskGetStackHighWaterMark(nullptr)),maximum_parity_error,
                maximum_parity_error<=2e-5f ? "true" : "false");
    // Sample arrays and golden fixtures are benchmark instrumentation.
    // Heap high-water excludes static DRAM/IRAM: combine with linker size reports.
}
