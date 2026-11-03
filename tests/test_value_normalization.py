import pytest
import torch
from bari.policy import Critic


def test_popart_preserves_physical_values_and_optimizer_parameter_references():
    torch.manual_seed(59)
    critic = Critic(normalize_values=True)
    optimizer = torch.optim.Adam(critic.parameters())
    parameter_ids = [id(p) for group in optimizer.param_groups for p in group["params"]]
    state = torch.randn(11, 29)
    original = critic(state).detach()
    batches = [torch.linspace(-120, -20, 50), torch.linspace(-5, 25, 30)]
    for batch in batches:
        critic.update_value_statistics(batch)
        torch.testing.assert_close(critic(state), original, rtol=2e-4, atol=1e-5)
        assert parameter_ids == [id(p) for p in critic.parameters()]
    pooled = torch.cat(batches).double()
    assert critic.value_count == len(pooled)
    torch.testing.assert_close(critic.value_mean, pooled.mean())
    torch.testing.assert_close(critic.value_std(), pooled.std(correction=0))
    normalized = critic.normalized_targets(pooled)
    torch.testing.assert_close(normalized.mean(), torch.tensor(0., dtype=torch.float64), atol=1e-12, rtol=0)
    torch.testing.assert_close(normalized.square().mean(), torch.tensor(1., dtype=torch.float64))


def test_constant_returns_stay_finite_and_legacy_critic_has_no_normalizer_state():
    critic = Critic(normalize_values=True)
    critic.update_value_statistics(torch.full((5, 5), -4.))
    assert critic.value_std() == pytest.approx(.1)
    assert torch.count_nonzero(critic.normalized_targets(torch.full((5, 5), -4.))) == 0
    assert torch.isfinite(critic(torch.zeros(29))).all()
    for invalid in (torch.tensor([]), torch.tensor([float("nan")])):
        with pytest.raises(ValueError, match="finite, nonempty"):
            critic.update_value_statistics(invalid)
    legacy = Critic()
    assert not any(key.startswith("value_") for key in legacy.state_dict())
