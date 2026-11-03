"""Fixed-seed isolated/switching evaluation and continuous trajectory recording."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .config import TASKS
from .env import Environment
from .metrics import EVALUATION_VERSION, evaluation_contract

SWITCH_TOUR = ("circle", "square", "circle", "line", "circle", "follow", "square", "line",
               "square", "follow", "line", "follow", "circle")


class Recorder:
    def __init__(self, env, model="scripted-privileged"):
        self.metadata = {"seed": env.seed, "model": model, "config": env.config.to_dict(),
                         "recording_version": 2, "follow_ordering": env.follow_ordering,
                         "evaluator_version": EVALUATION_VERSION,
                         "initial_poses": env.poses.tolist(),
                         "config_sha256": env.config.digest(), "recording": "one row per control step"}
        self.rows = []
        self.follow_trails = {}

    def append(self, env, commands):
        if env.task == "follow" and env.tracker.start_time not in self.follow_trails:
            self.follow_trails[env.tracker.start_time] = [point.tolist() for point in env.tracker.path[:5]]
        self.rows.append({"time": env.time, "poses": env.poses.copy(), "wheels": env.wheels.copy(),
                          "commands": commands.copy(), "observation": env.observation().copy(),
                          "task": TASKS.index(env.task), "collisions": env.collisions,
                          "wall_violations": env.wall_violations, "error": env.tracker.latest["max_error_m"],
                          "success": env.tracker.success, "visible": env.visibility(),
                          "hold": env.tracker.hold, "contacts": env.contact_robots.copy(),
                          "blocked_robot_beams": env.occlusion_diagnostics()["blocked"],
                          "targets": env.tracker.latest["fitted_targets"].copy()})

    def save(self, path, env):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray([row[key] for row in self.rows]) for key in self.rows[0]}
        metadata = {**self.metadata, "switches": env.switches,
                    "following_initial_trails": [{"start_s": time, "points": points} for time, points in self.follow_trails.items()],
                    "segments": env.tracker.segments, "final": env.tracker.summary()}
        np.savez_compressed(path, **arrays, metadata=json.dumps(metadata))


def checkpoint_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def run_episode(config, controller, seed, task, schedule=(), layout="random", record=None, model="scripted-privileged"):
    env = Environment(config, seed, task, layout)
    controller.reset()
    recorder = Recorder(env, model) if record else None
    pending = iter(sorted(schedule))
    upcoming = next(pending, None)
    visibility_count = []
    occlusion_counts, occlusion_events = [], []
    switch_measurements = {}
    hidden_since = None
    while env.time+1e-9 < config.timeout_seconds:
        while upcoming and env.time+1e-9 >= upcoming[0]:
            env.request_task(upcoming[1])
            upcoming = next(pending, None)
        previous_commands, previous_wheels = env.previous_commands.copy(), env.wheels.copy()
        commands = controller.actions(env)
        _, _, terminated, truncated, _ = env.step(commands)
        for index, event in enumerate(env.switches):
            if event["applied_s"] is not None and index not in switch_measurements:
                switch_measurements[index] = {
                    "response_latency_s": event["applied_s"]-event["requested_s"],
                    "max_command_change": float(np.linalg.norm(env.previous_commands-previous_commands, axis=1).max()),
                    "max_wheel_speed_change_m_s": float(np.abs(env.wheels-previous_wheels).max())}
        visible = env.visibility()
        visibility_count.append(int(visible.sum()))
        severity = env.occlusion_diagnostics()["blocked_robot_beam_pairs"]
        occlusion_counts.append(severity)
        if severity and hidden_since is None:
            hidden_since = env.time
        elif not severity and hidden_since is not None:
            occlusion_events.append({"onset_s": hidden_since, "end_s": env.time,
                                     "duration_s": env.time-hidden_since, "visibility_returned": True})
            hidden_since = None
        if recorder:
            recorder.append(env, commands)
        if terminated or truncated:
            break
    result = {"seed": seed, "initial_task": task, "layout": layout,
              "final_task": env.task,
              "switches": [{**event, **switch_measurements.get(index, {})} for index, event in enumerate(env.switches)],
              "segments": env.tracker.segments, **env.tracker.summary(),
              "min_visible_directed_pairs": min(visibility_count),
              "mean_visible_directed_pairs": float(np.mean(visibility_count)),
              "peak_blocked_robot_beam_pairs": max(occlusion_counts),
              "occlusion_events": occlusion_events}
    if hidden_since is not None:
        occlusion_events.append({"onset_s": hidden_since, "end_s": env.time,
                                 "duration_s": env.time-hidden_since, "visibility_returned": False})
    if recorder:
        recorder.save(record, env)
    return result


def evaluate(config, controller, seeds, output, model="scripted-privileged", checkpoint=None,
             transitions=False, layout="random", mode="deterministic", record=False, checkpoint_metadata=None,
             isolated_tasks=None):
    from .policy import revision
    code_revision = revision()
    checkpoint_metadata = checkpoint_metadata or {}
    selected_tasks = tuple(TASKS if isolated_tasks is None else isolated_tasks)
    if not selected_tasks or len(set(selected_tasks)) != len(selected_tasks) or any(task not in TASKS for task in selected_tasks):
        raise ValueError("Choose distinct, known isolated evaluation tasks")
    source_hash = checkpoint_metadata.get("file_sha256") or checkpoint_hash(checkpoint)
    output = Path(output)
    if (output/"report.json").exists() or (output/"evaluation_contract.json").exists():
        raise ValueError("Choose a fresh evaluation directory; existing evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    source = output/"source"
    source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        (source/path.name).write_bytes(path.read_bytes())
    config.save(output/"frozen_config.json")
    contract = evaluation_contract(config)
    contract_text = json.dumps(contract, sort_keys=True, indent=2)+"\n"
    (output/"evaluation_contract.json").write_text(contract_text)
    episodes = []
    for seed in seeds:
        for task in selected_tasks:
            path = output/f"{task}-{seed}.npz" if record else None
            row = run_episode(config, controller, seed, task, layout=layout, record=path, model=model)
            episodes.append(row)
        if transitions:
            for source in TASKS:
                for destination in TASKS:
                    if source == destination:
                        continue
                    row = run_episode(config, controller, seed, source,
                                      [(min(5.0, config.timeout_seconds/3), destination)], layout,
                                      output/f"{source}-{destination}-{seed}.npz" if record else None, model)
                    row["directed_transition"] = [source, destination]
                    row["new_task_acquisition_s"] = row["acquisition_s"]
                    row["old_task_finished_before_switch"] = row["segments"][0]["success"]
                    row["transition_collisions"] = row["collisions"]-row["segments"][0]["collisions"]
                    row["transition_wall_violations"] = row["wall_violations"]-row["segments"][0]["wall_violations"]
                    episodes.append(row)
            interval = min(2.0, config.timeout_seconds/(len(SWITCH_TOUR)+1))
            schedule = [(i*interval, task) for i, task in enumerate(SWITCH_TOUR) if i]
            episodes.append(run_episode(config, controller, seed, SWITCH_TOUR[0], schedule, layout,
                                        output/f"continuous-{seed}.npz", model))
    groups = {}
    for task in selected_tasks:
        rows = [r for r in episodes if r["initial_task"] == task and not r["switches"]]
        rate = float(np.mean([r["success"] for r in rows]))
        groups[task] = {"episodes": len(rows), "success_rate": rate,
                        "mean_final_max_error_m": float(np.mean([r["max_error_m"] for r in rows])),
                        "required_success_rate": config.required_success_rate,
                        "passes_assumed_gate": rate >= config.required_success_rate}
    report = {"model": model, "checkpoint_sha256": source_hash, "evaluation_revision": code_revision,
              "evaluator_version": EVALUATION_VERSION,
              "evaluation_contract_sha256": hashlib.sha256(contract_text.encode()).hexdigest(),
              "training_seed": checkpoint_metadata.get("training_seed"),
              "training_updates": checkpoint_metadata.get("updates"),
              "training_version": checkpoint_metadata.get("training_version", "legacy-team-reward"),
              "checkpoint_evidence": checkpoint_metadata.get("evidence"),
              "actor_initialization": checkpoint_metadata.get("actor_initialization"),
              "training_metadata": checkpoint_metadata.get("training_metadata"),
              "actor_spec": checkpoint_metadata.get("actor_spec"),
              "policy_contract_version": checkpoint_metadata.get("contract_version"),
              "action_selection": mode, "config_sha256": config.digest(), "seeds": seeds,
              "seed_split": "held-out" if all(seed in config.evaluation_seeds for seed in seeds) else "diagnostic/validation",
              "isolated_tasks": list(selected_tasks), "layout": layout,
              "scope": "full-four-task-suite" if set(selected_tasks) == set(TASKS) else "partial isolated-task diagnostic",
              "threshold_status": "provisional assumptions frozen before this run",
              "summary": groups, "episodes": episodes}
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(groups, indent=2))
    return report
