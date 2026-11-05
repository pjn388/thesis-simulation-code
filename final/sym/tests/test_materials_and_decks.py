import numpy as np
import pytest
from pipe_sim.config import Case
from pipe_sim.materials import engineering_to_plastic, ramberg_osgood, total_curve_to_plastic
from pipe_sim.geometry import build_model
from pipe_sim.decks import write_deck


def test_material_strain_conventions_are_explicit():
    true=engineering_to_plastic([100.],[.01],200000)
    assert true[0,0]==101
    assert true[0,1]==pytest.approx(np.log1p(.01)-101/200000)
    normal=ramberg_osgood([365],209000,365,.002,9.98)
    offset=ramberg_osgood([365],209000,365,.002,9.98,'offset')
    assert normal[0,1]==pytest.approx(.002*365/209000)
    assert offset[0,1]==pytest.approx(.002)
    c=Case();c.material.plastic=[[365,0],[400,.02]]
    with pytest.raises(ValueError,match='failure strain'):c.validate()


def test_material_curve_preserves_early_plasticity_without_yield_splice():
    # Two points straddle sigma=E*eps at exactly 200 MPa / .001.
    result=total_curve_to_plastic([190,210,250],[.0008,.0012,.01],200000.)
    assert result[0]==pytest.approx([200,0])
    assert result[1]==pytest.approx([210,.0012-210/200000])
    assert result[2]==pytest.approx([250,.01-250/200000])
    with pytest.raises(ValueError,match='intersection'):
        total_curve_to_plastic([200,250],[.01,.02],200000.)
    with pytest.raises(ValueError,match='increasing'):
        total_curve_to_plastic([200,190],[.001,.01],200000.)


def test_calculix_mesh_and_loading_history(tmp_path):
    c=Case();c.mesh.wall_layers=2;c.mesh.fine_size=6;m=build_model(c)
    write_deck(c,m,tmp_path)
    text=(tmp_path/'job.inp').read_text()
    assert text.count('*STEP,')==3
    assert 'ROT NODE=' in text
    assert 'PRESSURE-OVERCLOSURE=LINEAR' in text
    # Pressure starts after indentation and recovery.
    assert '*DLOAD' not in text.split('** Stage: pressurise')[0]
    assert '*MODEL CHANGE, TYPE=CONTACT PAIR, REMOVE' in text
    assert '*NODE PRINT' in text and '*EL PRINT' in text
    assert '*OUTPUT, FIELD' not in text


def test_compression_only_contact_keeps_stiffness_and_zero_opening_pressure(tmp_path):
    c=Case();c.dent.normal_law='compression_only';c.mesh.wall_layers=2;c.mesh.fine_size=6
    m=build_model(c);write_deck(c,m,tmp_path,'calculix')
    text=(tmp_path/'job.inp').read_text()
    table=text.split('*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=TABULAR\n')[1].split('*')[0]
    data=np.array([[float(x) for x in line.split(',')] for line in table.strip().splitlines()])
    assert data[0]==pytest.approx([0.,0.])
    assert data[1,0]/data[1,1]==pytest.approx(c.dent.normal_stiffness)
    assert data[1,1]==pytest.approx(c.pipe.thickness)
    c.dent.normal_law='unsupported'
    with pytest.raises(ValueError,match='normal contact law'):c.validate()


def test_pressure_refinement_preserves_denting_and_loads(tmp_path):
    c=Case();c.mesh.wall_layers=2;c.mesh.fine_size=6
    m=build_model(c)
    write_deck(c,m,tmp_path)
    before=(tmp_path/'job.inp').read_text()
    mesh=(tmp_path/'mesh.inp').read_bytes()
    c.analysis.pressure_max_increment=.005;c.validate()
    write_deck(c,m,tmp_path)
    after=(tmp_path/'job.inp').read_text()
    prefix, pressure=before.split('** Stage: pressurise')
    assert after == prefix+'** Stage: pressurise'+pressure.replace(
        '0.025, 1., 1e-06, 0.1\n', '0.005, 1., 1e-06, 0.005\n', 1)
    assert (tmp_path/'mesh.inp').read_bytes()==mesh
    c.dent.enabled=False
    write_deck(c,m,tmp_path)
    assert (tmp_path/'job.inp').read_text().count('0.005, 1., 1e-06, 0.005\n')==1


@pytest.mark.parametrize('cap', [0., 1e-7, .2, True, float('nan')])
def test_invalid_pressure_caps_are_rejected(cap):
    c=Case();c.analysis.pressure_max_increment=cap
    with pytest.raises(ValueError):c.validate()


def test_pressure_iteration_budget_preserves_other_input(tmp_path):
    c=Case();c.mesh.wall_layers=2;c.mesh.fine_size=6
    m=build_model(c);write_deck(c,m,tmp_path)
    original=(tmp_path/'job.inp').read_text()
    mesh=(tmp_path/'mesh.inp').read_bytes()
    c.analysis.pressure_contact_iterations=120;c.validate()
    write_deck(c,m,tmp_path)
    actual=(tmp_path/'job.inp').read_text()
    card='*CONTROLS, PARAMETERS=CONTACT\n,,,120\n'
    assert actual.count(card)==1
    assert card not in actual.split('** Stage: pressurise')[0]
    assert actual.replace(card,'')==original
    assert (tmp_path/'mesh.inp').read_bytes()==mesh


@pytest.mark.parametrize('cap',[0,-1,True,120.5,'120'])
def test_invalid_pressure_iteration_budgets(cap):
    c=Case();c.analysis.pressure_contact_iterations=cap
    with pytest.raises(ValueError,match='pressure_contact_iterations'):c.validate()
