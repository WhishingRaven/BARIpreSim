import numpy as np
import pytest
from bari.config import Config
from bari.env import Environment


@pytest.mark.parametrize('width',[0,30])
@pytest.mark.parametrize('missing',[np.nan,np.inf,-np.inf])
def test_rendered_missing_ranges_match_actor_maximum_fallback(monkeypatch,width,missing):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import bari.render as render
    env=Environment(Config(sensor_fov_deg=width))
    env.readings[:]=missing
    np.testing.assert_array_equal(env.observation()[:,:6],np.ones((5,6)))
    captured=[];original=render.Line3DCollection
    def capture(segments,*args,**kwargs):
        captured.append(segments)
        return original(segments,*args,**kwargs)
    monkeypatch.setattr(render,'Line3DCollection',capture)
    view=render.View(env)
    try:
        view.draw()
        radial=[np.asarray(s) for s in captured[0] if len(s)==2]
        assert radial
        lengths=[np.linalg.norm(s[1,:2]-s[0,:2]) for s in radial]
        np.testing.assert_allclose(lengths,env.config.sensor_max,atol=1e-12)
    finally:
        plt.close(view.figure)
