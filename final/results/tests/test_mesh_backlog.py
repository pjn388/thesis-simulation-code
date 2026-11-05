import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from pipe_sim.config import Case
from pipe_sim.runner import sha256
import prepare_mesh_backlog as backlog


@pytest.mark.parametrize('parameter,value',[('wall_layers',6),('fine_size',4.)])
def test_mesh_variant_changes_one_mesh_parameter_only(parameter,value):
    parent=Case();before=parent.to_dict()
    child=backlog.mesh_variant(parent,'child',parameter,value).to_dict()
    expected=copy.deepcopy(before)
    expected['mesh'][parameter]=value
    expected['name']=child['name'];expected['notes']=child['notes']
    assert child==expected
    assert parent.to_dict()==before


def test_register_retains_parents_and_rejects_duplicate_batch(tmp_path,monkeypatch):
    cases=tmp_path/'cases';cases.mkdir()
    jobs=[]
    for name in ['full_zhao2021_p1_adaptive','orientation_combined_60_adaptive']:
        case=Case();case.name=name;path=cases/(name+'.json');case.save(path)
        jobs.append(dict(id=name,kind='orientation',case_file=str(path.relative_to(tmp_path)),case_sha256=sha256(path),run='runs/'+name,entry=None))
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'jobs':jobs,'orientation_combined_0':jobs[0]['id']}))
    for key,value in [('FINAL',tmp_path),('CASES',cases),('PLAN',plan)]:monkeypatch.setattr(backlog,key,value)
    lanes=backlog.register('test')
    result=json.loads(plan.read_text())
    assert result['jobs'][:2]==jobs and len(result['jobs'])==6
    assert len(lanes['a'])==len(lanes['b'])==2
    assert all(j['kind']=='iteration' and len(j['parameter_changes'])==1 for j in result['jobs'][2:])
    assert all(sha256(tmp_path/j['case_file'])==j['case_sha256'] for j in result['jobs'])
    previous=plan.read_bytes()
    with pytest.raises(FileExistsError):backlog.register('test')
    assert plan.read_bytes()==previous
