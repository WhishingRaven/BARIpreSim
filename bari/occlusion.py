"""Frozen moving-robot nearest-hit probes, separate from learned task evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import TASKS
from .env import Environment
from .evaluate import Recorder
from .geometry import ray_disks


def run_probe(config, protocol, condition, seed, record=None):
    if condition not in protocol["expected_occlusion"]:
        raise ValueError(condition)
    poses = np.asarray(protocol["initial_poses"], dtype=float).copy()
    blocker = protocol["blocker"]
    if condition == "baseline":
        poses[blocker, 1] = protocol["baseline_blocker_y_m"]
    env = Environment(config, seed)
    env.reset(seed, poses=poses)
    initial = env.poses.copy()
    recorder = Recorder(env, f"scripted-occlusion-probe/{condition}") if record else None
    observer, beam, target = protocol["observer"], protocol["beam"], protocol["target"]
    rows, physical_ranges = [], {}

    def measure():
        ranges, hits, origins, directions = env.raycast()
        hidden = bool(env.occlusion_diagnostics()["blocked"][observer, beam, target])
        target_range = float(ray_disks(origins, directions, env.poses[target:target+1, :2],
                                      np.array([config.body_radius]))[observer, beam, 0])
        physical_ranges[env.tick] = float(ranges[observer, beam])
        delay = round(config.sensor_latency/config.physics_dt)
        period = round(config.sensor_period/config.physics_dt)
        sample_tick = ((env.tick-delay)//period)*period if env.tick >= delay else None
        rows.append({"time_s": env.time, "target_hidden_behind_nearer_surface": hidden,
                     "nearest_object": int(hits[observer, beam]),
                     "nearest_range_m": float(ranges[observer, beam]), "target_surface_range_m": target_range,
                     "actor_range_m": float(env.readings[observer, beam]), "delivered_sample_tick": sample_tick,
                     "task": env.task})

    measure()
    while env.time+1e-9 < protocol["duration_s"]:
        task = TASKS[min(int((env.time+1e-9)//1), 5) % len(TASKS)]
        if task != env.task:
            env.request_task(task)
        commands = np.full((5, 2), protocol["forward_command"])
        if condition == "transient":
            if protocol["blocker_turn_start_s"]-1e-9 <= env.time < protocol["blocker_turn_end_s"]-1e-9:
                commands[blocker] = protocol["blocker_turn_command"]
            elif env.time+1e-9 >= protocol["blocker_turn_end_s"]:
                commands[blocker] = protocol["blocker_depart_command"]
        env.step(commands)
        measure()
        if recorder:
            recorder.append(env, commands)
    hidden = [row for row in rows if row["target_hidden_behind_nearer_surface"]]
    onset = hidden[0]["time_s"] if hidden else None
    clear = next((row["time_s"] for row in rows if onset is not None and row["time_s"] > onset
                  and not row["target_hidden_behind_nearer_surface"]), None)
    reacquired = next((row["time_s"] for row in rows if clear is not None
                       and row["delivered_sample_tick"] is not None
                       and row["delivered_sample_tick"]*config.physics_dt+1e-9 >= clear), None)
    expected = protocol["expected_occlusion"][condition]
    def agrees(actual, target_time):
        return actual is None if target_time is None else actual is not None and abs(actual-target_time) <= protocol["timing_tolerance_s"]+1e-9
    timing_pass = agrees(onset, expected["onset_s"]) and agrees(clear, expected["end_s"])
    errors = [abs(row["actor_range_m"]-physical_ranges[row["delivered_sample_tick"]])
              for row in rows if row["delivered_sample_tick"] in physical_ranges]
    result = {"condition": condition, "seed": seed, "purpose": protocol["purpose"],
              "initial_poses": initial.tolist(), "robot_count": len(env.poses),
              "occlusion_onset_s": onset, "visibility_return_s": clear,
              "occlusion_duration_s": None if onset is None else (clear or env.time)-onset,
              "sensor_reacquisition_s": reacquired,
              "sensor_response_latency_s": None if reacquired is None else reacquired-clear,
              "nominal_timing_pass": timing_pass, "collisions": env.collisions,
              "wall_violations": env.wall_violations,
              "displacement_per_robot_m": np.linalg.norm(env.poses[:, :2]-initial[:, :2], axis=1).tolist(),
              "compared_delivered_samples": len(errors),
              "max_delivered_range_error_from_noise_m": max(errors) if errors else None,
              "switches": env.switches, "trace": rows}
    if recorder:
        recorder.save(record, env)
    return result


def probe_suite(config, protocol_path, seeds, output, record=False):
    raw = Path(protocol_path).read_bytes()
    protocol = json.loads(raw)
    if protocol.get("version") != "bari-natural-occlusion-probe-v1":
        raise ValueError("Unsupported natural-occlusion protocol version")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/"frozen_protocol.json").write_bytes(raw)
    config.save(output/"frozen_config.json")
    rows = [run_probe(config, protocol, condition, seed,
                      output/f"{condition}-{seed}.npz" if record else None)
            for seed in seeds for condition in protocol["expected_occlusion"]]
    from .policy import revision
    report = {"status": "sensor/physics probes only; no learned robustness claim", "code_revision": revision(),
              "config_sha256": config.digest(), "protocol_sha256": hashlib.sha256(raw).hexdigest(),
              "seeds": seeds, "episodes": rows}
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"probes": len(rows), "nominal_timing_passes": sum(row["nominal_timing_pass"] for row in rows),
                      "collisions": sum(row["collisions"] for row in rows)}, indent=2))
    return report
