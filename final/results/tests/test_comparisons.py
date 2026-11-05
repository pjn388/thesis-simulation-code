import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
import pytest
from comparison_utils import conversion,canonical_curve,loop_branches,error_metrics,pressure_gauge_curve


def test_regeneration_preserves_executed_case(tmp_path,monkeypatch):
    import json
    import prepare_cases
    from pipe_sim.config import Case
    monkeypatch.setattr(prepare_cases,'RESULTS',tmp_path)
    (tmp_path/'config').mkdir();case=Case();path=tmp_path/'case.json';case.save(path)
    (tmp_path/'config/runs.json').write_text(json.dumps([{'case_id':case.name}]))
    original=path.read_bytes();prepare_cases.save_case(case,path)
    case.pipe.thickness+=.1
    with pytest.raises(ValueError,match='executed case'):prepare_cases.save_case(case,path)
    assert path.read_bytes()==original


def test_units_and_loop_order_are_preserved():
    assert conversion('bar','MPa')==.1
    assert conversion('microstrain','1')==1e-6
    assert conversion('percent','microstrain')==10000
    with pytest.raises(ValueError):conversion('mm','MPa')
    d=pd.DataFrame(dict(point_order=[1,2,3,4],x=[0,1,2,1],y=[0,1000,2000,0],x_unit='mm',y_unit='N'))
    result=canonical_curve(d,'mm','kN')
    assert result.x.tolist()==[0,1,2,1]
    branches=loop_branches(result)
    assert branches['withdraw'].x.tolist()==[2,1]
    assert branches['withdraw'].y.tolist()==[2,0]


def test_errors_do_not_extrapolate_or_mix_unloading():
    ref=pd.DataFrame({'x':[0,1,2,3],'y':[0,1,2,3]})
    sim=pd.DataFrame({'x':[1,2],'y':[2,3]})
    result,errors=error_metrics(ref,sim)
    assert result['coverage_fraction']==.5
    assert result['rmse']==1
    assert errors.x.tolist()==[1,2]


def test_post_dent_gauge_zero_and_strain_measure():
    # A 10% stretch at gauge installation and 21% total stretch under pressure
    # should read 10% additional engineering extension, not 11% or delta E.
    d=pd.DataFrame({'requested_node':[4,4], 'time':[2.,3.], 'stage':['withdraw','pressurise'],
                    'strain_measure':['green_lagrange']*2,'axial_strain':[(1.1**2-1)/2,(1.21**2-1)/2]})
    assert pressure_gauge_curve(d,4,'axial_strain').gauge_engineering_strain.iloc[0]==pytest.approx(.1)
    with pytest.raises(ValueError):pressure_gauge_curve(d.iloc[1:],4,'axial_strain')
    d.strain_measure='spatial_logarithmic'
    with pytest.raises(ValueError):pressure_gauge_curve(d,4,'axial_strain')


def test_changed_specimen_or_input_is_rejected(tmp_path,monkeypatch):
    import json
    import compare
    from pipe_sim.config import Case
    from prepare_cases import digest
    monkeypatch.setattr(compare,'FINAL',tmp_path)
    run=tmp_path/'run';run.mkdir()
    case=Case();case.save(tmp_path/'wanted.json');case.save(run/'case.json')
    (run/'job.inp').write_text('original input')
    (run/'manifest.json').write_text(json.dumps({'files':{'job.inp':digest(run/'job.inp')}}))
    source=tmp_path/'reference.csv';source.write_text('source data')
    entry={'case_file':'wanted.json','sources':{'reference.csv':digest(source)},
           'material':{'source':'reference.csv','source_sha256':digest(source)}}
    assert compare.check_run({'run':'run'},entry)==run
    (run/'job.inp').write_text('edited input')
    with pytest.raises(ValueError,match='hash mismatch'):compare.check_run({'run':'run'},entry)
    case.pipe.od=101;case.save(run/'case.json')
    with pytest.raises(ValueError,match='does not match'):compare.check_run({'run':'run'},entry)
