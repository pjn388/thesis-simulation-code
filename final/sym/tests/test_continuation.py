import numpy as np
from pipe_sim.continuation import ramp_points, progress_at, energy_history, read_history, seed_deck, pressure_step, ENERGY, RESTART


def test_ramp_and_restart_progress_use_physical_time(tmp_path):
    ramp=ramp_points(2.)
    assert np.all(np.diff(ramp[:,1])>=0)
    assert np.array_equal(ramp[[0,-1]],[[0,0],[2,1]])
    state=progress_at({'kind':'dynamic','duration_seconds':2.,'target_pressure_mpa':35.},3.)
    assert state['fraction']==.5 and state['pressure']==17.5
    assert progress_at({'kind':'seed'},2.)['fraction']==1.


def test_restart_history_filters_unaccepted_and_incomplete_frames(tmp_path):
    (tmp_path/'job.sta').write_text('3 1 1 2 2.5 .5 .5\n3 2 1U 60 2.5 .5 .5\n')
    text=''
    for t in (2.5,3.):
        text+=f' displacements (vx,vy,vz) for set HISTORY and time {t}\n\n 1 0 -2 0\n 2 0 0 0\n'
        text+=f' equivalent plastic strain for set PIPE and time {t}\n'+''.join(f' 1 {i} .2\n' for i in range(1,9))
        text+=f' total internal energy for set PIPE and time {t}\n\n 1.2E+02\n'
        text+=f' total kinetic energy for set PIPE and time {t}\n\n 1.2E-01\n'
    (tmp_path/'job.dat').write_text(text)
    e=energy_history(tmp_path/'job.dat')
    assert e[2.5]['internal_energy_nmm']==120.
    rows=read_history(tmp_path,1,2,1,'dynamic',duration=1.)
    assert len(rows)==1
    assert rows[0]['pressure_mpa']==17.5
    assert rows[0]['kinetic_energy_nmm']==.12
    assert rows[0]['crown_inward_displacement_mm']==2.


def test_continuation_preserves_pressure_end_load_ratio_and_seed_history():
    before='*MATERIAL, NAME=PIPE_STEEL\n*ELASTIC\n208000,.3\n'+('*STEP\n*STATIC\n.025,1,1e-6,.05\n*END STEP\n'*2)
    parent=before+'''** Stage: pressurise
*STEP, NLGEOM, INC=1000
*STATIC, SOLVER=PARDISO
0.005, 1., 1e-06, 0.005
*CONTROLS, PARAMETERS=CONTACT
,,,120
*MODEL CHANGE, TYPE=CONTACT PAIR, REMOVE
PIPE_CONTACT, INDENTER_SURFACE
*DLOAD
1,P1,35
2,P2,-330
*END STEP
'''
    seed=seed_deck(parent)
    assert seed.replace('*DENSITY\n7.85e-09\n','').replace(ENERGY,'').replace(RESTART,'')==before
    step=pressure_step(parent,True,2.)
    assert '*DYNAMIC, ALPHA=-0.05, SOLVER=PARDISO\n0.001, 2, 2e-06, 0.005\n' in step
    assert '*DLOAD, AMPLITUDE=PRESSURE_RAMP\n1,P1, 35\n2,P2, -330\n' in step
    assert step.count('*MODEL CHANGE')==1
    assert '*CONTROLS, PARAMETERS=CONTACT\n,,,120\n' in step
