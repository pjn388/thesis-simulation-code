"""One production C3D8 gouge-only response comparison; no fitted burst claim."""
import copy
import json
import campaign
from pipe_sim.config import Case
from pipe_sim.runner import sha256, write_json


def prepare():
    parent_id = 'orientation_gouge_00_adaptive'
    name = parent_id + '_c3d8_20261020'
    plan = json.loads(campaign.PLAN.read_text())
    assert not any(j['id'] == name for j in plan['jobs'])
    parent = next(j for j in plan['jobs'] if j['id'] == parent_id)
    source = campaign.FINAL / parent['case_file']
    assert sha256(source) == parent['case_sha256']
    case = Case.load(source)
    before = case.to_dict()
    assert not case.dent.enabled and case.mesh.order == 2
    case.name = name
    case.mesh.order = 1
    case.notes.append('20 October bounded production-solver axial gouge-only formulation sensitivity: order2to1 only. Parent mesh spacings, wall count, geometry, material, boundary conditions, pressure and PEEQ0.8 criterion unchanged. Compare common-pressure displacement/strain and criterion bracket against retained C3D20R parent. Separates formulation response without denting/contact history; cannot establish physical burst or an angle ranking. Failed experimental tangent is not used.')
    case.validate()
    for key in ('pipe','gouge','dent','material','analysis','probe_positions'):
        assert case.to_dict()[key] == before[key], key
    assert {k:v for k,v in case.to_dict()['mesh'].items() if k!='order'} == {k:v for k,v in before['mesh'].items() if k!='order'}
    path = campaign.CASES / (name + '.json')
    assert not path.exists()
    case.save(path)
    child = copy.deepcopy(parent)
    child.update(id=name, parent_attempt=parent_id, case_file=str(path.relative_to(campaign.FINAL)), case_sha256=sha256(path), run='sym/runs/campaign/'+name, changes={'mesh':{'order':1}}, description='Bounded axial gouge-only C3D8 response comparison; unchanged parent physics')
    child.pop('superseded_by', None)
    if child.get('entry'):
        child['entry'].update(id=name,case_file=child['case_file'],case_sha256=child['case_sha256'],assumptions=case.notes)
    plan['jobs'].append(child)
    write_json(campaign.PLAN, plan)
    print(name)


if __name__ == '__main__':
    prepare()
