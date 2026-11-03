import json
import numpy as np
import pytest
from bari.config import Config
from bari.occlusion import run_probe


@pytest.mark.parametrize("condition", ["baseline", "persistent", "transient"])
def test_moving_physical_robots_realize_frozen_nearest_hit_protocol(condition):
    protocol = json.loads(open("configs/occlusion-probes.json").read())
    result = run_probe(Config(sensor_noise_std=0), protocol, condition, 10001)
    assert result["nominal_timing_pass"]
    assert result["robot_count"] == 5
    assert min(result["displacement_per_robot_m"]) > .05
    assert result["collisions"] == result["wall_violations"] == 0
    assert result["max_delivered_range_error_from_noise_m"] < 1e-10
    if condition == "transient":
        assert result["visibility_return_s"] == pytest.approx(4.9)
        assert result["sensor_reacquisition_s"] == pytest.approx(5.0)
        assert result["sensor_response_latency_s"] == pytest.approx(.1)
        before = result["trace"][0]
        after = result["trace"][-1]
        assert before["nearest_object"] == 1
        assert after["nearest_object"] == 2
        assert after["nearest_range_m"] == pytest.approx(.246)
    assert {event["to"] for event in result["switches"]} == {"circle", "square", "line", "follow"}
