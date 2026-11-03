import json
import numpy as np
from bari.config import Config
from bari.env import Environment
from bari.export import export_actor
from bari.policy import Actor, Critic, save_checkpoint
import torch
import pytest


@pytest.mark.parametrize("hidden_size", [32, 64])
def test_cpp_multi_step_parity_and_actor_only(tmp_path, hidden_size):
    actor, critic = Actor(hidden_size), Critic()
    optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()))
    checkpoint = tmp_path/"checkpoint.pt"
    save_checkpoint(checkpoint, actor, critic, optimizer, Config(hidden_size=hidden_size), 11, 0, "smoke-only")
    report = export_actor(checkpoint, tmp_path/"export", steps=96)
    assert report["max_absolute_action_hidden_error"] < 2e-5
    assert report["peak_ram_bytes"] is None
    header = (tmp_path/"export"/"actor.hpp").read_text()
    assert "critic" not in header.lower().split("#pragma")[1]


@pytest.mark.parametrize("radius", [.5, 1.0])
def test_render_has_no_effect_on_world_sensor_rng_or_control(tmp_path, radius):
    import matplotlib
    matplotlib.use("Agg")
    from bari.render import View
    import matplotlib.pyplot as plt
    a, b = Environment(Config(arena_radius=radius), seed=10), Environment(Config(arena_radius=radius), seed=10)
    view = View(a)
    for _ in range(2):
        view.draw()
        # Both the historical and enlarged walls must fit within the camera.
        for bounds in (view.axis.get_xlim(), view.axis.get_ylim()):
            assert bounds[0] < -radius and bounds[1] > radius
            assert bounds[1]-bounds[0] < 2.5*radius
        a.step(np.full((5, 2), .3))
        b.step(np.full((5, 2), .3))
    view.save(tmp_path/"frame.png")
    plt.close(view.figure)
    np.testing.assert_array_equal(a.poses, b.poses)
    np.testing.assert_array_equal(a.readings, b.readings)
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
