"""Command-line entry points. Use python -m bari --help."""
import argparse
import json
from pathlib import Path
import torch
from .baseline import ScriptedController, PlannedController
from .config import Config, TASKS, observation_contract
from .env import Environment
from .policy import (RecurrentController, load_checkpoint, TRAINING_VERSION, SUPPORTED_TRAINING_VERSIONS,
                     MULTIWORLD_VERSION, MULTIWORLD_VERSIONS, PACKED_TRAINING_VERSION)


def main():
    parser = argparse.ArgumentParser(description="BARI five-robot GRU simulation")
    parser.add_argument("command", choices=("config", "train", "evaluate", "demo", "export", "replay", "occlusion-probe", "collect-teacher", "imitate"))
    parser.add_argument("--config", help="Full JSON configuration; checkpoint config used by default")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="runs/default")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--task", choices=TASKS, help="Training/demo task; packed checkpoints default to their trained task")
    parser.add_argument("--tasks", choices=TASKS, nargs="+", help="Limit isolated evaluation; default evaluates all four tasks")
    parser.add_argument("--layout", choices=("random", "chain", "occluded"), default="random")
    parser.add_argument("--baseline", choices=("reactive", "planned"), default="reactive",
                        help="Scripted diagnostic controller; planned uses privileged simulator lookahead")
    parser.add_argument("--updates", type=int, default=100,
                        help="Total update target for packed v14; additional updates for legacy trainers")
    parser.add_argument("--worlds", type=int, help="Independent training worlds; version 8 defaults to eight, exact resume inherits its count")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        help="Packed PPO learner device; simulation and rollout inference remain CPU")
    parser.add_argument("--actor-initialization", help="Fresh RL run: load actor weights only; not an exact resume")
    parser.add_argument("--demonstrations", nargs="+", help="One or more local-history teacher NPZ datasets to aggregate")
    parser.add_argument("--student-checkpoint", help="Collect teacher labels while this local GRU student drives")
    parser.add_argument("--teacher-scan-seconds", type=float, default=0.,
                        help="Teacher-only initial scan per task segment; zero preserves reactive labels")
    parser.add_argument("--teacher-intervention-probability", type=float, default=0.,
                        help="Collection only: probability that the teacher executes for the learned group; requires --student-checkpoint")
    parser.add_argument("--episodes", type=int, default=64, help="Teacher collection episode count")
    parser.add_argument("--epochs", type=int, default=10, help="Optional imitation epoch count")
    parser.add_argument("--training-version", choices=SUPPORTED_TRAINING_VERSIONS,
                        help="Select a fresh training algorithm; exact resume requires the saved version")
    parser.add_argument("--smoke", action="store_true", help="Mark checkpoint as smoke-only; run two short updates")
    parser.add_argument("--transitions", action="store_true")
    parser.add_argument("--record", action="store_true", help="Record evaluation episodes")
    parser.add_argument("--reset-hidden", action="store_true", help="Memory ablation, same actor weights")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--snapshot", help="Save a 3D PNG instead of opening interactive window")
    parser.add_argument("--steps", type=int, default=0, help="Control steps before snapshot")
    parser.add_argument("--recording", help="NPZ trajectory for replay")
    parser.add_argument("--movie", help="Export replay as GIF")
    parser.add_argument("--stride", type=int, default=1, help="Recorded control steps per replay frame")
    parser.add_argument("--fps", type=float, default=10, help="Replay frames per wall-clock second")
    parser.add_argument("--protocol", default="configs/occlusion-probes.json", help="Frozen natural-occlusion probe protocol")
    parser.add_argument("--plan", help="Training plan JSON; fresh runs default to the documented curriculum, resume inherits its saved plan")
    parser.add_argument("--paused", action="store_true", help="Open the live demonstration paused")
    args = parser.parse_args()
    requested_task = args.task
    args.task = args.task or "circle"
    if args.training_version and args.command != "train":
        parser.error("--training-version applies only to training")
    if args.device and args.command != "train":
        parser.error("--device applies only to training")
    if args.baseline != "reactive" and (args.checkpoint or args.command not in ("evaluate", "demo")):
        parser.error("--baseline planned applies only to scripted evaluation/demo; it cannot wrap a learned actor")
    if args.actor_initialization and (args.command != "train" or args.checkpoint):
        parser.error("--actor-initialization starts a fresh RL run and cannot combine with --checkpoint")
    if args.command in ("collect-teacher", "imitate") and args.checkpoint:
        parser.error("Teacher/imitation stages are fresh runs; --checkpoint is not supported")
    if args.command == "imitate" and args.config:
        parser.error("Imitation uses the frozen demonstration configuration")
    if args.student_checkpoint and args.command != "collect-teacher":
        parser.error("--student-checkpoint applies only to teacher-label collection")
    if args.teacher_scan_seconds != 0 and args.command != "collect-teacher":
        parser.error("--teacher-scan-seconds applies only to teacher-label collection")
    if args.teacher_intervention_probability != 0 and args.command != "collect-teacher":
        parser.error("--teacher-intervention-probability applies only to teacher-label collection")
    if args.tasks and args.command != "evaluate":
        parser.error("--tasks applies only to isolated evaluation")
    if args.worlds is not None and (args.command != "train" or args.worlds < 1):
        parser.error("--worlds requires training and a positive world count")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    config = Config.load(args.config)
    controller = PlannedController() if args.baseline == "planned" else ScriptedController()
    model = controller.version
    payload = None
    if args.checkpoint:
        actor, saved, payload = load_checkpoint(args.checkpoint)
        config = config if args.config else saved
        if config.hidden_size != saved.hidden_size or observation_contract(config) != observation_contract(saved):
            raise ValueError("Evaluation must preserve checkpoint observation/action contract")
        controller = RecurrentController(actor, not args.stochastic, args.reset_hidden)
        model = f"GRU-{config.hidden_size}/{payload['evidence']}"
        if payload.get("training_version") == PACKED_TRAINING_VERSION:
            trained_task = payload["trainer_state"]["plan"]["tasks"][0]
            if requested_task is None:
                args.task = trained_task
            if args.command in ("demo", "evaluate"):
                if args.transitions:
                    parser.error("Separate-model switching needs an explicit model selector and hidden-state rules")
                if args.task != trained_task or (args.tasks and args.tasks != [trained_task]):
                    parser.error("This separate model is trained only for "+trained_task)
                args.tasks = [trained_task]
    if config.profile_version == 2 and args.layout != "random":
        parser.error("Profile 2 uses random starts in training, evaluation and demo")
    if args.command == "config":
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        config.save(output/"config.json")
        (output/"contract.json").write_text(json.dumps(observation_contract(config), indent=2)+"\n")
    elif args.command == "train":
        from .train import Trainer
        if args.checkpoint:
            if args.config and config.to_dict() != saved.to_dict():
                parser.error("Exact resume must use the saved configuration")
            if payload["training_version"] == PACKED_TRAINING_VERSION:
                from .packed import PackedTrainer
                trainer = PackedTrainer.restore(args.checkpoint, args.device)
            else:
                if args.device:
                    parser.error("GPU/device resume requires a v14 checkpoint")
                trainer = Trainer.restore(args.checkpoint)
            if args.training_version and args.training_version != trainer.training_version:
                parser.error("Exact resume must preserve the saved training version")
            if args.worlds is not None and args.worlds != len(getattr(trainer, "worlds", [trainer])):
                parser.error("Exact resume must preserve the saved world count")
            if args.plan:
                from .curriculum import TrainingPlan
                plan = TrainingPlan.load(args.plan, config)
                if trainer.plan is None or plan.digest() != trainer.plan.digest():
                    parser.error("Exact resume must preserve the saved training plan")
        else:
            if args.smoke:
                config.rollout_steps, config.sequence_length, config.ppo_epochs = 32, 8, 1
            from .curriculum import TrainingPlan
            version = args.training_version or (PACKED_TRAINING_VERSION if config.profile_version == 2 or args.device else TRAINING_VERSION)
            if args.device and version != PACKED_TRAINING_VERSION:
                parser.error("--device requires packed v14 training")
            if version == PACKED_TRAINING_VERSION:
                if not args.config:
                    config = Config.load(Path(__file__).resolve().parent.parent/"configs"/f"{args.task}-v1.json")
                if args.smoke:
                    config.rollout_steps, config.sequence_length, config.ppo_epochs = 32, 8, 1
                plan = TrainingPlan.load(args.plan, config) if args.plan else TrainingPlan(
                    version="bari-training-plan-v2", tasks=[args.task], isolated_episode_probability=1.,
                    curriculum_updates=0, curriculum_start_probability=0.)
                from .packed import PackedTrainer
                trainer = PackedTrainer(config, args.seed, plan, args.worlds or (2 if args.smoke else 64), args.device or "auto")
            else:
                plan = TrainingPlan.load(args.plan, config) if args.plan else (
                    TrainingPlan(curriculum_updates=0, curriculum_start_probability=0.)
                    if config.profile_version == 2 else TrainingPlan())
            if version == PACKED_TRAINING_VERSION:
                pass
            elif version in MULTIWORLD_VERSIONS and (version != TRAINING_VERSION or (args.worlds or 1) > 1):
                from .multiworld import MultiWorldTrainer
                trainer = MultiWorldTrainer(config, args.seed, plan, args.worlds or (2 if args.smoke else 8), version)
            else:
                if args.worlds not in (None, 1):
                    parser.error("Multiple worlds require a supported multi-world training version")
                trainer = Trainer(config, args.seed, plan, version)
            if args.actor_initialization:
                trainer.initialize_actor(args.actor_initialization)
        if args.smoke and args.checkpoint:
            parser.error("Use a fresh smoke run or resume without --smoke")
        trainer.run(2 if args.smoke else args.updates, args.output, args.smoke)
    elif args.command == "evaluate":
        from .evaluate import evaluate
        evaluate(config, controller, args.seeds or config.evaluation_seeds, args.output, model,
                 args.checkpoint, args.transitions, args.layout,
                 ("stochastic" if args.stochastic else "deterministic")+("/hidden-reset" if args.reset_hidden else ""), args.record, payload,
                 isolated_tasks=args.tasks)
    elif args.command == "demo":
        from .render import demo
        demo(Environment(config, args.seed, args.task, args.layout), controller, model,
             args.snapshot, args.steps, str(Path(args.output)/"demo.npz") if args.record else None, args.paused)
    elif args.command == "export":
        from .export import export_actor
        if not args.checkpoint:
            parser.error("export requires --checkpoint")
        export_actor(args.checkpoint, args.output)
    elif args.command == "replay":
        from .replay import replay
        if not args.recording:
            parser.error("replay requires --recording")
        replay(args.recording, args.snapshot, args.movie, args.steps, args.stride, args.fps)
    elif args.command == "occlusion-probe":
        from .occlusion import probe_suite
        probe_suite(config, args.protocol, args.seeds or config.evaluation_seeds, args.output, args.record)
    elif args.command == "collect-teacher":
        from .imitation import collect_teacher
        from .curriculum import TrainingPlan
        collect_teacher(config, args.seed, args.episodes, args.output,
                        TrainingPlan.load(args.plan, config) if args.plan else None, args.student_checkpoint,
                        scan_seconds=args.teacher_scan_seconds, teacher_probability=args.teacher_intervention_probability)
    elif args.command == "imitate":
        from .imitation import imitate
        if not args.demonstrations:
            parser.error("imitate requires --demonstrations")
        imitate(args.demonstrations, args.seed, args.epochs, args.output)


if __name__ == "__main__":
    main()
