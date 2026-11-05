"""Register isolated mesh checks without replacing any physical baseline."""
import argparse
import copy
import json
from pathlib import Path

from pipe_sim.config import Case
from pipe_sim.runner import sha256
from campaign import PLAN, FINAL, CASES


def mesh_variant(parent, name, parameter, value):
    if parameter not in ('wall_layers', 'fine_size'):
        raise ValueError('Only one wall or local surface resolution parameter may change')
    case = Case.from_dict(parent.to_dict())
    old = getattr(case.mesh, parameter)
    setattr(case.mesh, parameter, value)
    case.name = name
    case.notes.append(f'Mesh sensitivity of {parent.name}: mesh.{parameter} {old} -> {value}. '
                      'Physical geometry, material, contact, loading and criterion unchanged. '
                      'Compare common converged history, residual dent and first criterion bracket; '
                      'failure/timeout is not burst or demonstrated mesh convergence.')
    case.validate()
    return case


def register(tag):
    if not tag or any(c not in '0123456789abcdefghijklmnopqrstuvwxyz_-' for c in tag):
        raise ValueError('Invalid study tag')
    plan = json.loads(PLAN.read_text())
    specs = [('p1', 'full_zhao2021_p1_adaptive'),
             ('combined60', 'orientation_combined_60_adaptive')]
    prepared = []
    lanes = {'a': [], 'b': []}
    for lane, (label, parent_id) in zip(lanes, specs):
        job = next(j for j in plan['jobs'] if j['id'] == parent_id)
        source = FINAL/job['case_file']
        if sha256(source) != job['case_sha256']:
            raise ValueError('Parent frozen case hash changed')
        parent = Case.load(source)
        for suffix, parameter, value in [('wall6', 'wall_layers', 6), ('fine4', 'fine_size', 4.)]:
            name = f'mesh_{label}_{suffix}_{tag}'
            path = CASES/(name+'.json')
            if path.exists() or any(j['id'] == name for j in plan['jobs']):
                raise FileExistsError(name)
            case = mesh_variant(parent, name, parameter, value)
            child = copy.deepcopy(job)
            child.pop('superseded_by', None)
            child.update(id=name, kind='iteration', case_file=str(path.relative_to(FINAL)),
                         run='sym/runs/campaign/'+name, parent_attempt=parent_id,
                         estimated_elements=None, study='mesh_sensitivity_'+tag,
                         parameter_changes={f'mesh.{parameter}': {'old': getattr(parent.mesh, parameter), 'new': value}})
            if child.get('entry'):
                child['entry'].update(id=name, case_file=child['case_file'], assumptions=case.notes)
            prepared.append((path, case, child))
            lanes[lane].append(name)
    for path, case, child in prepared:
        case.save(path)
        child['case_sha256'] = sha256(path)
        if child.get('entry'):
            child['entry']['case_sha256'] = child['case_sha256']
        plan['jobs'].append(child)
    temporary = PLAN.with_suffix('.tmp')
    temporary.write_text(json.dumps(plan, indent=2)+'\n')
    temporary.replace(PLAN)
    return lanes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    print(json.dumps(register(parser.parse_args().tag), indent=2))
