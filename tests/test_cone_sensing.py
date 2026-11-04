from copy import deepcopy
import numpy as np
import pytest
import torch
from bari.config import Config, observation_contract, CONE_OCCLUSION_BASIS
from bari.env import Environment
from bari.evaluate import Recorder
from bari.export import export_actor
from bari.geometry import cone_disks, ray_disks, wrap
from bari.metrics import evaluation_contract
from bari.policy import Actor, Critic, load_checkpoint, save_checkpoint


@pytest.mark.parametrize("width", [-.1, 180.1, float("nan"), float("inf")])
def test_invalid_horizontal_fov(width):
    with pytest.raises(ValueError):Config(sensor_fov_deg=width).validate()


@pytest.mark.parametrize("bearing,width", [(20,60), (40,60), (10,90)])
def test_exact_disk_center_or_sector_edge_surface(bearing, width):
    origins=np.zeros((1,1,2));angle=np.array([[0.]])
    center=.3*np.array([[np.cos(np.deg2rad(bearing)),np.sin(np.deg2rad(bearing))]])
    radii=np.array([.069])
    distance,directions=cone_disks(origins,angle,np.deg2rad(width/2),center,radii)
    chosen=np.deg2rad(min(bearing,width/2))
    expected=.3*np.cos(np.deg2rad(bearing)-chosen)-np.sqrt(.069**2-(.3*np.sin(np.deg2rad(bearing)-chosen))**2)
    assert distance[0,0,0]==pytest.approx(expected)
    np.testing.assert_allclose(directions[0,0,0],[np.cos(chosen),np.sin(chosen)],atol=1e-14)
    endpoint=origins[0,0]+distance[0,0,0]*directions[0,0,0]
    assert np.linalg.norm(endpoint-center[0])==pytest.approx(.069)


def test_wrapped_cone_and_absent_disk_do_not_invent_hits():
    origins=np.zeros((1,1,2));angles=np.array([[np.deg2rad(350)]])
    centers=np.array([[.3*np.cos(np.deg2rad(10)),.3*np.sin(np.deg2rad(10))],[-.4,0]])
    distance,directions=cone_disks(origins,angles,np.deg2rad(30),centers,np.array([.04,.04]))
    assert distance[0,0,0]==pytest.approx(.26)
    assert np.isinf(distance[0,0,1])


@pytest.mark.parametrize("yaw,width", [(40,90),(100,60)])
def test_exact_circular_wall_minimum_and_fixed_sensor_origin(yaw,width):
    config=Config(sensor_fov_deg=width,sensor_mount_radius=0,sensor_noise_std=0,sensor_latency=0)
    env=Environment(config)
    poses=np.array([[.4,0,np.deg2rad(yaw)],[-.4,-.3,0],[-.4,0,0],[-.4,.3,0],[0,-.5,0]])
    env.reset(poses=poses)
    ranges,objects,origins,directions=env.raycast()
    chosen=np.deg2rad(max(0,yaw-width/2))
    expected=-.4*np.cos(chosen)+np.sqrt(config.arena_radius**2-.4**2*np.sin(chosen)**2)
    assert ranges[0,0]==pytest.approx(expected)
    assert objects[0,0]==-1
    np.testing.assert_allclose(origins[0,0],poses[0,:2])
    np.testing.assert_allclose(directions[0,0],[np.cos(chosen),np.sin(chosen)],atol=1e-14)


def dense_reference(env,k=2001):
    c=env.config
    nominal=env.poses[:,2,None]+np.deg2rad(c.sensor_angles_deg)[None]
    origins=env.poses[:,None,:2]+c.sensor_mount_radius*np.stack((np.cos(nominal),np.sin(nominal)),axis=-1)
    angles=nominal[:,:,None]+np.deg2rad(np.linspace(-c.sensor_fov_deg/2,c.sensor_fov_deg/2,k))
    directions=np.stack((np.cos(angles),np.sin(angles)),axis=-1).reshape(5,6*k,2)
    origins=np.repeat(origins[:,:,None],k,axis=2).reshape(5,6*k,2)
    centers=env.poses[:,:2];radii=np.full(5,c.body_radius)
    if c.obstacles:
        centers=np.concatenate((centers,np.asarray(c.obstacles)[:,:2]))
        radii=np.concatenate((radii,np.asarray(c.obstacles)[:,2]))
    intersections=ray_disks(origins,directions,centers,radii)
    for robot in range(5):intersections[robot,:,robot]=np.inf
    dot=(origins*directions).sum(-1)
    wall=-dot+np.sqrt(np.maximum(dot**2+c.arena_radius**2-(origins**2).sum(-1),0))
    return np.minimum(intersections.min(-1),wall).reshape(5,6,k).min(-1)


@pytest.mark.parametrize("width", [15,60,180])
def test_analytic_cones_match_independent_dense_rays_and_surface_witnesses(width):
    for seed in [7,17,29]:
        env=Environment(Config(sensor_fov_deg=width,obstacles=[[.72,0,.08]]),seed)
        actual,objects,origins,directions=env.raycast()
        dense=dense_reference(env)
        assert np.all(actual<=dense+1e-10)
        np.testing.assert_allclose(actual,dense,atol=4e-5,rtol=0)
        assert not np.any(objects==np.arange(5)[:,None])
        nominal=env.poses[:,2,None]+np.deg2rad(env.config.sensor_angles_deg)[None]
        assert np.max(np.abs(wrap(np.arctan2(directions[...,1],directions[...,0])-nominal)))<=np.deg2rad(width/2)+1e-12
        endpoints=origins+actual[...,None]*directions
        centers=np.concatenate((env.poses[:,:2],np.asarray(env.config.obstacles)[:,:2]))
        radii=np.r_[np.full(5,env.config.body_radius),np.asarray(env.config.obstacles)[:,2]]
        for i in range(5):
            for j in range(6):
                hit=objects[i,j]
                if hit==-1:assert np.linalg.norm(endpoints[i,j])==pytest.approx(env.config.arena_radius)
                else:assert np.linalg.norm(endpoints[i,j]-centers[hit])==pytest.approx(radii[hit])


def test_cone_foreground_occludes_collidable_robot_without_actor_identity():
    c=Config(sensor_fov_deg=60,sensor_noise_std=0,sensor_latency=0)
    env=Environment(c)
    poses=np.array([[0,0,0],[.2,0,0],[.4,0,0],[-.3,.2,0],[-.3,-.2,0]])
    env.reset(poses=poses)
    before=env.observation()[0].copy()
    assert env.raycast()[1][0,0]==1
    assert env.raycast()[0][0,0]==pytest.approx(.2-c.sensor_mount_radius-c.body_radius)
    assert env.occlusion_diagnostics()['blocked'][0,0,2]
    assert not env.visibility()[0,2]
    assert env.occlusion_diagnostics()['basis']==CONE_OCCLUSION_BASIS
    poses[2,0]=.45;env.reset(poses=poses)
    np.testing.assert_array_equal(env.observation()[0],before)
    assert env.observation().shape==(5,9) and env.observation().dtype==np.float32
    action=np.zeros((5,2));action[0]=[1,1]
    for _ in range(30):env.step(action)
    distances=np.linalg.norm(env.poses[:,None,:2]-env.poses[None,:,:2],axis=-1)+np.eye(5)
    assert distances.min()>=2*c.body_radius-1e-12 and env.collisions>0


def test_minimum_suppression_is_not_reported_as_ray_occlusion():
    env=Environment(Config(sensor_fov_deg=60,sensor_noise_std=0,sensor_latency=0))
    poses=np.array([[0,0,0],[.18*np.cos(np.deg2rad(20)),.18*np.sin(np.deg2rad(20)),0],
                    [.25*np.cos(np.deg2rad(-20)),.25*np.sin(np.deg2rad(-20)),0],[-.4,.3,0],[-.4,-.3,0]])
    env.reset(poses=poses)
    assert env.raycast()[1][0,0]==1
    assert env.visibility()[0,2]
    assert not env.occlusion_diagnostics()['blocked'][0,0,2]


def test_cone_contract_is_explicit_and_rejects_incompatible_checkpoint(tmp_path):
    config=Config(sensor_fov_deg=60)
    assert 'sensor_model' not in observation_contract(Config())
    assert observation_contract(config)['local_shape']==[9]
    assert observation_contract(config)['sensor_model']['horizontal_fov_deg']==60
    assert evaluation_contract(config)['occlusion_observer_basis']==CONE_OCCLUSION_BASIS
    assert Recorder(Environment(config)).metadata['sensor_model']==observation_contract(config)['sensor_model']
    actor,critic=Actor(),Critic();checkpoint=tmp_path/'actor.pt'
    save_checkpoint(checkpoint,actor,critic,torch.optim.Adam(actor.parameters()),config,11,0,'smoke-only')
    loaded,_,payload=load_checkpoint(checkpoint)
    assert loaded.encoder.in_features==13
    payload.pop('file_sha256');payload['contract']['sensor_model']['horizontal_fov_deg']=15
    bad=tmp_path/'bad.pt';torch.save(payload,bad)
    with pytest.raises(ValueError,match='sensor contract'):load_checkpoint(bad)


def test_cone_model_cpp_sequence_parity_has_no_extra_actor_fields(tmp_path):
    config=Config(sensor_fov_deg=30)
    actor,critic=Actor(),Critic();checkpoint=tmp_path/'actor.pt'
    save_checkpoint(checkpoint,actor,critic,torch.optim.Adam(actor.parameters()),config,11,0,'smoke-only')
    report=export_actor(checkpoint,tmp_path/'export',steps=96)
    assert report['max_absolute_action_hidden_error']<2e-5
    assert report['peak_ram_bytes'] is None


def test_cone_render_and_queries_preserve_latency_rng_and_switch_state(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from bari.render import View
    config=Config(sensor_fov_deg=30,sensor_latency=.2,sensor_period=.2)
    a,b=Environment(config,17),Environment(config,17)
    assert np.all(a.observation()[:,:6]==1)
    view=View(a)
    for _ in range(3):
        old=a.state_dict();view.draw();a.raycast();a.occlusion_diagnostics()
        assert a.state_dict()==old
        a.step(np.full((5,2),.2));b.step(np.full((5,2),.2))
    assert a.state_dict()==b.state_dict()
    assert 'FOV assumption' in view.hud.get_text() and 'No measured hit bearing' in view.hud.get_text()
    before=deepcopy(a.state_dict());a.request_task('follow');a.begin_control()
    np.testing.assert_array_equal(a.poses,before['poses']);np.testing.assert_array_equal(a.readings,before['readings'])
    assert a.time==before['tick']*config.physics_dt and a.previous_commands.tolist()==before['previous_commands']
    view.save(tmp_path/'cone.png');plt.close(view.figure)
