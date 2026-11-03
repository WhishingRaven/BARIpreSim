"""Training-only initial-state curriculum and isolated/switching episode mix.

No curriculum targets or assignments enter the actor. Poses are sampled only at
episode resets; task switches continue the existing physical world and GRU state.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import numpy as np
from .geometry import targets


@dataclass
class TrainingPlan:
    version: str = "bari-training-plan-v1"
    isolated_episode_probability: float = .5
    switch_min_seconds: float = 3.0
    switch_max_seconds: float = 15.0  # Exclusive upper bound.
    curriculum_updates: int = 2000
    curriculum_start_probability: float = .8
    position_jitter_start_m: float = .015
    position_jitter_end_m: float = .16
    reserved_seeds: list = field(default_factory=lambda: [2001, 2003, 2011, 2017, 2027,
                                                         10001, 10007, 10009, 10037, 10039])

    def validate(self, config):
        if self.version != "bari-training-plan-v1":
            raise ValueError("Unsupported training plan")
        if not 0 <= self.isolated_episode_probability <= 1 or not 0 <= self.curriculum_start_probability <= 1:
            raise ValueError("Training probabilities must be in [0,1]")
        if not isinstance(self.curriculum_updates, int) or self.curriculum_updates < 0:
            raise ValueError("curriculum_updates must be a nonnegative integer")
        if not 0 <= self.position_jitter_start_m <= self.position_jitter_end_m:
            raise ValueError("Invalid curriculum position jitter")
        steps = [value/config.control_dt for value in (self.switch_min_seconds, self.switch_max_seconds)]
        if not 1 <= steps[0] < steps[1] or any(abs(value-round(value)) > 1e-8 for value in steps):
            raise ValueError("Switch bounds must be ordered control-period multiples")
        if not set(config.evaluation_seeds).issubset(self.reserved_seeds):
            raise ValueError("Training plan must reserve all held-out evaluation seeds")
        return self

    def to_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path, config):
        return cls(**json.loads(Path(path).read_text())).validate(config)

    def switch_steps(self, rng, config):
        return int(rng.integers(round(self.switch_min_seconds/config.control_dt),
                                round(self.switch_max_seconds/config.control_dt)))

    def sample_poses(self, env, rng, task, update):
        progress = min(1.0, update/max(1, self.curriculum_updates))
        probability = self.curriculum_start_probability*(1-progress) if self.curriculum_updates else 0.0
        if rng.random() >= probability:
            return None
        jitter = self.position_jitter_start_m+progress*(self.position_jitter_end_m-self.position_jitter_start_m)
        for _ in range(2048):
            angle = rng.uniform(-np.pi, np.pi)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            if task == "follow":
                points = np.column_stack((.32-np.arange(5)*env.config.follow_spacing, np.zeros(5)))
                points += rng.uniform(-min(jitter*.25, .02), min(jitter*.25, .02), (5, 2))
                heading = angle+rng.normal(0, .05, 5)
            else:
                points = targets(task, env.config)[rng.permutation(5)]
                points += rng.uniform(-jitter, jitter, (5, 2))
                heading = rng.uniform(-np.pi, np.pi, 5)
            poses = np.column_stack((points@rotation.T+rng.uniform(-.025, .025, (1, 2)), heading))
            if env.valid_poses(poses):
                return poses
        # Fall back to ordinary valid random/ordered-chain sampling, not invalid geometry.
        return None
