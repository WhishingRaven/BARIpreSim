"""Optional privileged-label warm start; deployed actor inputs stay local.

The teacher can use simulator ground truth. Only its wheel commands are labels;
global state/targets/masks never enter the actor. Imitation alone is not GRU RL.
"""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import torch
from .baseline import ScriptedController
from .config import Config, TASKS, observation_contract
from .curriculum import TrainingPlan
from .env import Environment
from .metrics import ANONYMOUS_ORDER, LEGACY_ORDER
from .policy import Actor, Critic, RecurrentController, load_checkpoint, revision, save_checkpoint

LEGACY_DATASET_VERSION = "bari-teacher-local-v1"
STUDENT_DATASET_VERSION = "bari-teacher-local-v2"
DATASET_VERSION = "bari-teacher-local-v3"
IMITATION_VERSION = "bari-imitation-v3"


def snapshot_source(output):
    source = output/"source"
    source.mkdir(exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        (source/path.name).write_bytes(path.read_bytes())


def collect_teacher(config, seed, episodes, output, plan=None, student_checkpoint=None, follow_ordering=ANONYMOUS_ORDER):
    if episodes < 1:
        raise ValueError("At least one teacher episode required")
    student, student_metadata = None, None
    if student_checkpoint:
        actor, saved, payload = load_checkpoint(student_checkpoint)
        if observation_contract(saved) != observation_contract(config):
            raise ValueError("Student collection must preserve its observation/action contract")
        student = RecurrentController(actor)
        student_metadata = {"path": str(student_checkpoint), "sha256": payload["file_sha256"],
                            "evidence": payload["evidence"], "training_version": payload.get("training_version"),
                            "action_selection": "deterministic mean", "teacher_intervention_probability": 0.,
                            "follow_leader": "scripted mission infrastructure; excluded from actor training"}
    output = Path(output)
    if (output/"demonstrations.npz").exists():
        raise ValueError("Choose a fresh demonstration directory")
    output.mkdir(parents=True, exist_ok=True)
    config.save(output/"config.json")
    snapshot_source(output)
    plan = (plan or TrainingPlan(curriculum_updates=0)).validate(config)
    rng = np.random.default_rng(seed)
    data = {key: [] for key in ("observations", "tasks", "expert_commands", "executed_commands", "actor_weights", "reset_masks")}
    episode_metadata = []
    for episode in range(episodes):
        task = TASKS[episode % len(TASKS)]
        world_seed = int(rng.integers(0, 2**31))
        while world_seed in plan.reserved_seeds:
            world_seed = int(rng.integers(0, 2**31))
        env = Environment(config, world_seed, task, "chain" if task == "follow" else "random", follow_ordering)
        if task == "follow":
            angle = rng.uniform(-np.pi, np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            poses = env.poses.copy()
            poses[:, :2] = poses[:, :2]@rotation.T+rng.uniform(-.025, .025, (1, 2))
            poses[:, 2] = angle
            env.reset(world_seed, task, poses=poses)
        initial = env.poses.copy()
        initial_state = env.state_dict()
        teacher = ScriptedController()
        teacher.reset()
        if student is not None:
            student.reset()
        switching = rng.random() >= plan.isolated_episode_probability
        remaining = plan.switch_steps(rng, config)
        rows = {key: [] for key in data}
        while env.time+1e-9 < config.timeout_seconds:
            if switching and remaining <= 0:
                env.request_task(str(rng.choice([name for name in TASKS if name != env.task])))
                remaining = plan.switch_steps(rng, config)
            env.begin_control()
            observation, tasks = env.observation().copy(), env.task_context().copy()
            # Run the student before computing privileged labels. Labels cannot
            # influence its action or GRU state. The allowed scripted follow
            # leader remains mission infrastructure in RecurrentController.
            commands = student.actions(env) if student is not None else None
            labels = teacher.actions(env)
            commands = labels if commands is None else commands
            weights = np.ones(5, dtype=np.float32)
            if env.task == "follow":
                weights[0] = 0
            mask = np.ones(5, dtype=np.float32) if rows["observations"] else np.zeros(5, dtype=np.float32)
            for key, value in zip(rows, (observation, tasks, labels, commands, weights, mask)):
                rows[key].append(value)
            env.step(commands)
            remaining -= 1
        for key in data:
            data[key].append(rows[key])
        episode_metadata.append({"world_seed": world_seed, "initial_task": task,
                                 "initial_poses": initial.tolist(), "initial_state": initial_state,
                                 "switching": switching,
                                 "switches": env.switches, "segments": env.tracker.segments,
                                 "final": env.tracker.summary()})
    metadata = {"version": DATASET_VERSION, "teacher": ScriptedController.version,
                "follow_ordering": follow_ordering,
                "evidence": "privileged teacher labels; not learned-policy evidence",
                "execution": "student-closed-loop" if student is not None else "teacher-closed-loop",
                "student": student_metadata,
                "dataset_seed": seed, "config": config.to_dict(), "config_sha256": config.digest(),
                "contract": observation_contract(config), "source_revision": revision(),
                "reserved_seeds": plan.reserved_seeds, "plan": plan.to_dict(), "episodes": episode_metadata,
                "initial_pose_sampling": "random formations; rotated/translated ordered follow chain; no near-target curriculum",
                "information_boundary": "Actor receives local observations and task context only; expert wheel commands are training labels. Metadata/global diagnostics never enter its encoder."}
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in data.items()}
    path = output/"demonstrations.npz"
    np.savez_compressed(path, **arrays, metadata=json.dumps(metadata))
    (output/"manifest.json").write_text(json.dumps({"dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                                   **metadata}, indent=2)+"\n")
    print(json.dumps({"dataset": str(path), "episodes": episodes, "timesteps": arrays["observations"].shape[1]}))
    return path


def load_demonstrations(path):
    raw = Path(path).read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        data = {key: archive[key].copy() for key in ("observations", "tasks", "expert_commands", "actor_weights", "reset_masks")}
        if metadata["version"] in (STUDENT_DATASET_VERSION, DATASET_VERSION):
            if "executed_commands" not in archive:
                raise ValueError("Version-2 datasets must record actual executed commands")
            data["executed_commands"] = archive["executed_commands"].copy()
        else:
            # Version 1 exclusively executed its teacher labels. Preserve those
            # immutable archives while presenting a common aggregation schema.
            data["executed_commands"] = data["expert_commands"].copy()
    if metadata["version"] not in (LEGACY_DATASET_VERSION, STUDENT_DATASET_VERSION, DATASET_VERSION):
        raise ValueError("Unsupported teacher dataset")
    if metadata["version"] == DATASET_VERSION:
        if metadata.get("follow_ordering") not in (ANONYMOUS_ORDER, LEGACY_ORDER) or any(
                row.get("initial_state", {}).get("follow_ordering") != metadata["follow_ordering"] for row in metadata["episodes"]):
            raise ValueError("Teacher dataset following-order contract mismatch")
    else:
        metadata["follow_ordering"] = LEGACY_ORDER
    config = Config(**metadata["config"]).validate()
    if config.digest() != metadata["config_sha256"] or observation_contract(config) != metadata["contract"]:
        raise ValueError("Teacher dataset contract/configuration mismatch")
    obs = data["observations"]
    if obs.ndim != 4 or obs.shape[2:] != (5, 9) or not min(obs.shape[:2]):
        raise ValueError("Expected nonempty [episodes,time,5,9] local observations")
    prefix = obs.shape[:3]
    shapes = {"tasks": (*prefix, 4), "expert_commands": (*prefix, 2), "executed_commands": (*prefix, 2),
              "actor_weights": prefix, "reset_masks": prefix}
    for key, value in data.items():
        if value.dtype != np.float32 or not np.isfinite(value).all() or (key in shapes and value.shape != shapes[key]):
            raise ValueError("Invalid teacher array: "+key)
    if (np.any((obs[..., :6] < 0) | (obs[..., :6] > 1)) or np.any(np.abs(data["expert_commands"]) > 1)
            or np.any(np.abs(data["executed_commands"]) > 1)):
        raise ValueError("Teacher data must use the deployment normalization/action bounds")
    tasks = data["tasks"]
    if not np.all((tasks == 0) | (tasks == 1)) or not np.all(tasks.sum(-1) == 1) or not np.all(tasks == tasks[:, :, :1]):
        raise ValueError("Teacher task context must be a shared four-way one-hot mission input")
    if len(metadata["episodes"]) != obs.shape[0]:
        raise ValueError("Teacher episode metadata count mismatch")
    if metadata["version"] in (STUDENT_DATASET_VERSION, DATASET_VERSION):
        if metadata.get("execution") not in ("teacher-closed-loop", "student-closed-loop"):
            raise ValueError("Version-2 datasets must identify the executed controller")
        if any(row.get("initial_state", {}).get("config_sha256") != config.digest() for row in metadata["episodes"]):
            raise ValueError("Version-2 datasets must retain exact initial environment snapshots")
    if not np.array_equal(obs[:, 1:, :, 6:8], data["executed_commands"][:, :-1]) or np.any(obs[:, 0, :, 6:8]):
        raise ValueError("Teacher previous-command history is inconsistent")
    role = np.zeros(prefix, dtype=np.float32); role[..., 0] = 1
    if not np.array_equal(obs[..., 8], role):
        raise ValueError("Teacher roles violate the local actor contract")
    expected = np.ones(prefix, dtype=np.float32); expected[..., 0] = 1-data["tasks"][..., 0, 3]
    if not np.array_equal(data["actor_weights"], expected):
        raise ValueError("Scripted following leader must be excluded from imitation")
    mask = np.ones(prefix, dtype=np.float32); mask[:, 0] = 0
    if not np.array_equal(data["reset_masks"], mask):
        raise ValueError("Teacher hidden resets belong only at episode starts")
    if any(row["world_seed"] in metadata["reserved_seeds"] for row in metadata["episodes"]):
        raise ValueError("Teacher dataset contains a reserved validation/evaluation seed")
    return data, metadata, config, hashlib.sha256(raw).hexdigest()


def imitate(path, seed, epochs, output):
    if epochs < 1:
        raise ValueError("At least one imitation epoch required")
    paths = [path] if isinstance(path, (str, Path)) else list(path)
    if not paths:
        raise ValueError("At least one demonstration dataset required")
    loaded = [load_demonstrations(item) for item in paths]
    config = loaded[0][2]
    hashes = [item[3] for item in loaded]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Each demonstration dataset may be aggregated only once")
    for arrays, metadata, saved, _ in loaded:
        if metadata["follow_ordering"] != loaded[0][1]["follow_ordering"]:
            raise ValueError("Aggregated datasets must preserve following-order semantics")
        if saved.digest() != config.digest() or arrays["observations"].shape[1:] != loaded[0][0]["observations"].shape[1:]:
            raise ValueError("Aggregated datasets must preserve configuration and episode shapes")
    data = {key: np.concatenate([item[0][key] for item in loaded], axis=0) for key in loaded[0][0]}
    datasets = [{"path": str(item), "sha256": loaded[i][3], "episodes": len(loaded[i][0]["observations"]),
                 "execution": loaded[i][1].get("execution", "teacher-closed-loop"), "teacher": loaded[i][1]["teacher"],
                 "student": loaded[i][1].get("student"), "follow_ordering": loaded[i][1]["follow_ordering"]} for i, item in enumerate(paths)]
    aggregate_hash = hashlib.sha256(json.dumps(hashes, separators=(",", ":")).encode()).hexdigest()
    output = Path(output)
    if (output/"checkpoint.pt").exists() or (output/"imitation.jsonl").exists():
        raise ValueError("Choose a fresh imitation directory")
    output.mkdir(parents=True, exist_ok=True)
    config.save(output/"config.json")
    snapshot_source(output)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor = Actor(config.hidden_size)
    critic = Critic(input_size=98+5*config.hidden_size, normalize_values=True)  # Untrained; excluded from initialization.
    optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
    plan = {"version": IMITATION_VERSION, "datasets": datasets, "aggregate_dataset_sha256": aggregate_hash,
            "epochs": epochs, "seed": seed, "batch_episodes": 8,
            "bptt_steps": 64, "learning_rate": .001, "gradient_clip": .5,
            "source_revision": revision(), "evidence": "imitation-only; zero RL updates"}
    (output/"imitation_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    tensors = {key: torch.from_numpy(value) for key, value in data.items()}
    episodes, steps = data["observations"].shape[:2]
    with (output/"imitation.jsonl").open("w") as log:
        for epoch in range(epochs):
            order = rng.permutation(episodes)
            squared_error = count = 0.
            for begin in range(0, episodes, 8):
                indices = order[begin:begin+8]
                batch = {key: value[indices].transpose(0, 1).reshape(steps, -1, *value.shape[3:])
                         for key, value in tensors.items()}
                hidden = torch.zeros(len(indices)*5, config.hidden_size)
                for start in range(0, steps, 64):
                    chunk = {key: value[start:start+64] for key, value in batch.items()}
                    means, hidden = actor.sequence(chunk["observations"], chunk["tasks"], hidden.detach(), chunk["reset_masks"])
                    weights = chunk["actor_weights"]
                    error = ((torch.tanh(means)-chunk["expert_commands"]).square().sum(-1)*weights).sum()
                    denominator = 2*weights.sum().clamp_min(1)
                    loss = error/denominator
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite imitation loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), .5, error_if_nonfinite=True)
                    optimizer.step()
                    squared_error += float(error.detach()); count += float(2*weights.sum())
            stats = {"epoch": epoch+1, "training_wheel_mse": squared_error/max(1, count),
                     "evidence": "imitation-only; not task success", "seed": seed}
            log.write(json.dumps(stats)+"\n"); log.flush()
            print(json.dumps(stats), flush=True)
            save_checkpoint(output/"checkpoint.pt", actor, critic, optimizer, config, seed, 0,
                            "imitation-only", plan["source_revision"], training_version=IMITATION_VERSION,
                            training_metadata={**plan, "completed_epochs": epoch+1})
    return output/"checkpoint.pt"
