"""Actor-only deterministic FP32 C++ export with multi-step host parity.

The generated operator is explicit GRUCell math and needs no inference runtime.
Host parity and partial buffer sizes do not establish ESP32 deployability.
"""
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
import torch
from .geometry import normalize_ranges
from .policy import load_checkpoint, actor_contract, PREVIOUS_COMMAND_LIMIT


def export_actor(checkpoint, output, steps=96):
    actor, config, payload = load_checkpoint(checkpoint)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    h = config.hidden_size
    arrays = []
    parameters = actor.state_dict()
    for key in ("encoder.weight", "encoder.bias", "gru.weight_ih", "gru.weight_hh", "gru.bias_ih",
                "gru.bias_hh", "head.weight", "head.bias"):
        values = parameters[key].numpy().ravel()
        name = key.replace(".", "_")
        literals = ",".join(f"{float(v):.9e}f" for v in values)
        arrays.append(f"static const float {name}[{len(values)}] = {{{literals}}};")
    header = """// Generated actor-only FP32 weights. No critic or optimizer.
#pragma once
#include <cmath>
#include <algorithm>
namespace bari {
"""+f"constexpr int H = {h};\n"+"\n".join(arrays)+"\n"
    header += f"""
constexpr float SENSOR_MIN = {config.sensor_min:.9e}f;
constexpr float SENSOR_MAX = {config.sensor_max:.9e}f;
constexpr float MAX_SPEED = {config.max_wheel_speed:.9e}f;
constexpr float MAX_ACCELERATION = {config.max_wheel_acceleration:.9e}f;
constexpr float CONTROL_DT = {config.control_dt:.9e}f;
constexpr float ACTION_CORRELATION = {actor.action_correlation:.9e}f;
constexpr float PREVIOUS_COMMAND_LIMIT = {PREVIOUS_COMMAND_LIMIT:.9e}f;
"""
    header += """
inline float sigmoid(float x) { return 1.f/(1.f+std::exp(-x)); }
inline void preprocess(const float ranges[6], const float previous[2], bool leader, float local[9]) {
  for(int i=0;i<6;i++) {
    float x=std::isfinite(ranges[i]) ? ranges[i] : SENSOR_MAX;
    local[i]=(std::clamp(x,SENSOR_MIN,SENSOR_MAX)-SENSOR_MIN)/(SENSOR_MAX-SENSOR_MIN);
  }
  local[6]=std::clamp(previous[0],-1.f,1.f);
  local[7]=std::clamp(previous[1],-1.f,1.f);
  local[8]=leader ? 1.f : 0.f;
}
inline void step(const float local[9], int task, float hidden[H], float action[2], bool reset=false) {
  if(reset) std::fill(hidden,hidden+H,0.f);
  float input[13], encoded[H], next[H];
  std::copy(local,local+9,input);
  for(int i=0;i<4;i++) input[9+i]=(task==i) ? 1.f : 0.f;
  for(int i=0;i<H;i++) {
    float x=encoder_bias[i];
    for(int j=0;j<13;j++) x+=encoder_weight[i*13+j]*input[j];
    encoded[i]=std::tanh(x);
  }
  for(int i=0;i<H;i++) {
    float gi[3], gh[3];
    for(int g=0;g<3;g++) {
      int row=g*H+i;
      gi[g]=gru_bias_ih[row]; gh[g]=gru_bias_hh[row];
      for(int j=0;j<H;j++) {
        gi[g]+=gru_weight_ih[row*H+j]*encoded[j];
        gh[g]+=gru_weight_hh[row*H+j]*hidden[j];
      }
    }
    // PyTorch gate ordering r,z,n; reset applied after recurrent affine including bias.
    float r=sigmoid(gi[0]+gh[0]), z=sigmoid(gi[1]+gh[1]);
    float n=std::tanh(gi[2]+r*gh[2]);
    next[i]=(1.f-z)*n+z*hidden[i];
  }
  std::copy(next,next+H,hidden);
  for(int i=0;i<2;i++) {
    float x=head_bias[i];
    for(int j=0;j<H;j++) x+=head_weight[i*H+j]*hidden[j];
    if(ACTION_CORRELATION>0.f) {
      float previous=std::atanh(std::clamp(local[6+i],-PREVIOUS_COMMAND_LIMIT,PREVIOUS_COMMAND_LIMIT));
      x=(1.f-ACTION_CORRELATION)*x+ACTION_CORRELATION*previous;
    }
    action[i]=std::tanh(x);
  }
}
// Call once per control period, preserving wheel state across task switches.
inline void decode(const float action[2], float wheel_state[2]) {
  for(int i=0;i<2;i++) {
    float target=std::clamp(action[i],-1.f,1.f)*MAX_SPEED;
    wheel_state[i]+=std::clamp(target-wheel_state[i],-MAX_ACCELERATION*CONTROL_DT,MAX_ACCELERATION*CONTROL_DT);
  }
}
} // namespace bari
"""
    (output/"actor.hpp").write_text(header)
    contract = actor_contract(config, actor)
    contract["inference"] = "deterministic tanh(mean), FP32; task IDs circle=0,square=1,line=2,follow=3"
    (output/"contract.json").write_text(json.dumps(contract, indent=2)+"\n")
    rng = np.random.default_rng(814)
    hidden = torch.zeros(1, h)
    previous = np.zeros(2, dtype=np.float32)
    inputs, expected = [], []
    for t in range(steps):
        ranges = rng.uniform(-.1, 1.1, 6).astype(np.float32)
        if t % 7 == 0:
            ranges[t % 6] = np.nan
        if t % 13 == 0:
            ranges[(t+1) % 6] = np.inf
        task, reset, role = (t//11) % 4, t % 43 == 0, float(t % 3 == 0)
        if reset:
            hidden.zero_()
            previous[:] = 0
        elif actor.action_correlation and t % 17 == 0:
            # Exercise the inverse-tanh clamp, including exact saturated commands.
            previous[:] = [1, -1]
        local = np.concatenate((normalize_ranges(ranges, config), previous, [role])).astype(np.float32)
        with torch.no_grad():
            mean, hidden = actor(torch.from_numpy(local[None]), torch.eye(4)[task:task+1], hidden)
            action = torch.tanh(mean)[0].numpy()
        inputs.append(np.concatenate((ranges, previous, [role, task, float(reset)])))
        expected.append(np.concatenate((action, hidden[0].numpy())))
        previous = action.copy()
    np.asarray(inputs, dtype=np.float32).tofile(output/"golden_inputs.f32")
    np.asarray(expected, dtype=np.float32).tofile(output/"golden_expected.f32")
    def literals(values):
        return ",".join("NAN" if np.isnan(v) else "INFINITY" if np.isposinf(v)
                        else "-INFINITY" if np.isneginf(v) else f"{float(v):.9e}f" for v in values)
    fixture = "#pragma once\n#include <cmath>\nnamespace bari_golden {\n"
    fixture += f"constexpr int STEPS={steps};\n"
    fixture += "static const float inputs[STEPS][11]={\n"+",\n".join("{"+literals(row)+"}" for row in inputs)+"};\n"
    fixture += f"static const float expected[STEPS][{h+2}]={{\n"+",\n".join("{"+literals(row)+"}" for row in expected)+"};\n}\n"
    (output/"golden_fixture.hpp").write_text(fixture)
    source = """#include "actor.hpp"
#include "golden_fixture.hpp"
#include <fstream>
int main(int argc,char**argv) {
  if(argc!=3) return 2;
  std::ifstream in(argv[1],std::ios::binary);
  std::ofstream out(argv[2],std::ios::binary);
  float hidden[bari::H]={}, row[11], local[9], action[2];
  while(in.read(reinterpret_cast<char*>(row),sizeof(row))) {
    bari::preprocess(row,row+6,row[8]>0,local);
    bari::step(local,static_cast<int>(row[9]),hidden,action,row[10]>0);
    out.write(reinterpret_cast<char*>(action),sizeof(action));
    out.write(reinterpret_cast<char*>(hidden),sizeof(hidden));
  }
  return out.good() ? 0 : 3;
}
"""
    (output/"parity.cpp").write_text(source)
    compiler = shutil.which("c++")
    if not compiler:
        raise RuntimeError("c++ compiler required for host parity")
    command = [compiler, "-std=c++17", "-O2", str(output/"parity.cpp"), "-o", str(output/"parity")]
    subprocess.run(command, check=True, capture_output=True)
    subprocess.run([str((output/"parity").resolve()), str(output/"golden_inputs.f32"),
                    str(output/"golden_actual.f32")], check=True)
    expected = np.asarray(expected, dtype=np.float32)
    actual = np.fromfile(output/"golden_actual.f32", dtype=np.float32).reshape(expected.shape)
    error = float(np.max(np.abs(actual-expected)))
    if not np.allclose(actual, expected, atol=2e-5, rtol=2e-5):
        raise AssertionError(f"C++ GRU parity failed: max absolute error {error}")
    weight_bytes = sum(parameters[key].numel()*4 for key in parameters if key != "log_std")
    report = {"status": "host-validated; on-device verification pending", "precision": "FP32",
              "actor_spec": actor.specification(), "contract_version": contract["version"],
              "source_checkpoint_evidence": payload["evidence"], "steps": steps,
              "max_absolute_action_hidden_error": error, "atol": 2e-5, "rtol": 2e-5,
              "exported_weight_bytes": weight_bytes, "hidden_bytes_per_robot": h*4,
              "step_local_arrays_bytes": (13+2*h+6)*4,
              "preprocessing_local_bytes": 9*4, "wheel_state_bytes": 2*4,
              "weight_storage": "static const; actual flash/RAM placement requires board map",
              "excluded_from_estimates": ["compiler spills", "stack frames", "sensor queues", "I/O buffers",
                                          "firmware heaps", "other firmware", "runtime and library code"],
              "peak_ram_bytes": None, "flash_binary_bytes": None, "board": None,
              "target_chip": "ESP32-S3", "internal_sram_capacity_bytes": 524288, "psram_permitted": False,
              "device_latency_ms": None, "deadline_compliance": None, "compiler_command": command}
    (output/"host_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return report
