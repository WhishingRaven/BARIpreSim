import numpy as np
import pytest
from bari.config import Config, TASKS
from bari.evaluate import run_episode


class MissionController:
    def reset(self):
        pass

    def actions(self, env):
        env.begin_control()
        command = .5 if TASKS.index(env.task) % 2 else -.5
        return np.full((5, 2), command)


def test_switch_report_exposes_latency_and_acceleration_limited_command_changes():
    config = Config(timeout_seconds=1.)
    report = run_episode(config, MissionController(), 51, "circle", [(.2, "square"), (.5, "line"), (.7, "follow")])
    assert len(report["switches"]) == 3
    assert [event["applied_s"] for event in report["switches"]] == pytest.approx([.2, .5, .7])
    for event in report["switches"]:
        assert event["response_latency_s"] == 0
        assert event["max_command_change"] == pytest.approx(np.sqrt(2))
        assert event["max_wheel_speed_change_m_s"] <= config.max_wheel_acceleration*config.control_dt+1e-8
    assert len(report["segments"]) == 3
    assert all(not segment["success"] for segment in report["segments"])


def test_evaluation_snapshots_source_and_never_overwrites_frozen_evidence(tmp_path):
    import hashlib
    from bari.evaluate import evaluate
    config = Config(timeout_seconds=.1)
    report = evaluate(config, MissionController(), [51], tmp_path)
    frozen = (tmp_path/"evaluation_contract.json").read_bytes()
    digest = hashlib.sha256()
    for path in sorted((tmp_path/"source").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    assert digest.hexdigest() == report["evaluation_revision"]["python_source_sha256"]
    with pytest.raises(ValueError, match="fresh evaluation directory"):
        evaluate(config, MissionController(), [52], tmp_path)
    assert (tmp_path/"evaluation_contract.json").read_bytes() == frozen


def test_one_task_diagnostic_does_not_report_a_complete_four_task_suite(tmp_path):
    from bari.evaluate import evaluate
    config = Config(timeout_seconds=.1)
    report = evaluate(config, MissionController(), [51], tmp_path/"follow", layout="chain", isolated_tasks=["follow"])
    assert report["scope"] == "partial isolated-task diagnostic"
    assert report["isolated_tasks"] == ["follow"]
    assert report["layout"] == "chain"
    assert set(report["summary"]) == {"follow"}
    assert len(report["episodes"]) == 1
    with pytest.raises(ValueError, match="distinct, known"):
        evaluate(config, MissionController(), [51], tmp_path/"duplicate", isolated_tasks=["line", "line"])
