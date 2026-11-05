import pandas as pd
import pytest
from pipe_sim.config import Case
from pipe_sim.results import dat_blocks, failure_assessment, final_frd_displacements, stress_invariants, stage_at, contact_history, residual_displacement


def history(values, pressures=None):
    return pd.DataFrame([dict(stage='pressurise',time=2+(i+1)/10,pressure_mpa=p,max_peeq=v,
                              max_peeq_element=42,max_peeq_ip=3) for i,(v,p) in enumerate(zip(values,pressures or [10,20,30]))])


def test_failure_is_not_solver_stop_or_target_pressure():
    c=Case();df=history([.1,.2,.3])
    assert failure_assessment(df,c,True)['burst_pressure_mpa'] is None
    assert failure_assessment(df,c,False)['status']=='incomplete_before_criterion'
    result=failure_assessment(history([.1,.7,.85]),c,False)
    assert result['pressure_bracket_mpa']==[20,30]
    assert result['failure_element']==42
    assert not result['simulation_completed']


def test_denting_failure_and_stage_boundaries():
    c=Case();df=history([.9,.95,1.]);df.loc[0,'stage']='indent'
    assert failure_assessment(df,c,True)['status']=='criterion_reached_during_denting'
    assert stage_at(c,1.)==('indent',0.)
    assert stage_at(c,2.)==('withdraw',0.)
    assert stage_at(c,2.5)==('pressurise',10.)


def test_stress_invariants():
    vm,eta,principal=stress_invariants([100,0,0,0,0,0])
    assert vm==pytest.approx(100);assert eta==pytest.approx(1/3);assert principal==100
    assert stress_invariants([100,100,100,0,0,0])[1] is None


def test_read_calculix_headers_and_fortran_exponents(tmp_path):
    p=tmp_path/'test.dat';p.write_text(' equivalent plastic strain (elem, integ.pnt.,pe)for set PIPE and time  0.2500000E+01\n\n  42 3 0.810000D+00\n\n global coordinates (elem, integ.pnt.,x,y,z) for set DEFECT and time  0.2500000E+01\n  42 3 0.0 1.0 2.0\n')
    blocks=list(dat_blocks(p));assert blocks[0][0]=='PEEQ';assert blocks[0][3][0]==[42,3,.81]
    assert blocks[1][0]=='COORD'


def test_frd_fixed_width_negative_displacements(tmp_path):
    p=tmp_path/'test.frd';p.write_text(' -4  DISP        4    1\n -1         1-1.04949E-02 2.00000E-03-3.00000E-04\n -3\n')
    assert final_frd_displacements(p)[1]==pytest.approx([-.0104949,.002,-.0003])
    with p.open('a') as f:f.write(' -4  DISP        4    1\n -1         1 9.00000E+00 9.00000E+00 9.00000E+00\n')
    assert final_frd_displacements(p)[1]==pytest.approx([-.0104949,.002,-.0003])


def test_contact_penetration_sign_and_pressure_component(tmp_path):
    p=tmp_path/'contact.frd'
    p.write_text('  100CL  101 5.00000E-01 1 0 1 1\n -4  CONTACT     6    1\n -1         1-1.20000E-03 0.00000E+00 0.00000E+00 2.00000E+02 0.00000E+00 0.00000E+00\n -3\n')
    result=contact_history(p,Case())
    assert result[0]['max_nodal_penetration_mm']==pytest.approx(.0012)
    assert result[0]['max_nodal_contact_pressure_mpa']==200
    with p.open('a') as stream:
        stream.write('  100CL  102 1.50000E+00 1 0 1 1\n -4  CONTACT     6    1\n -1         1 1.20000E-03 0.00000E+00 0.00000E+00-2.00000E+02 0.00000E+00 0.00000E+00\n -3\n')
    result=contact_history(p,Case())
    assert result[1]['max_nodal_penetration_mm']==0
    assert result[1]['min_nodal_contact_pressure_mpa']==-200


def test_partial_unloading_is_not_a_residual_dent():
    df=pd.DataFrame([dict(stage='withdraw',time=1.5,crown_inward_displacement_mm=1.9)])
    assert residual_displacement(df) is None
    df.loc[1]=['withdraw',2.,0.8]
    assert residual_displacement(df)==pytest.approx(.8)
