"""One bounded frozen-mesh combined60 pressure-resolution control.

No angle expansion is automatic. Physical validation remains unresolved.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import pandas as pd
import campaign
from pipe_sim.config import Case
from pipe_sim.model import Model
from pipe_sim.decks import write_deck
from pipe_sim.runner import sha256, write_json

PARENT = 'orientation_combined_60_adaptive'
NAME = PARENT + '_pressure005_20261020'
OUT = campaign.FINAL / 'results/output/orientation_pressure005_20261020'


def prepare():
    plan = json.loads(campaign.PLAN.read_text())
    assert not any(j['id'] == NAME for j in plan['jobs'])
    parent = next(j for j in plan['jobs'] if j['id'] == PARENT)
    source = campaign.FINAL / parent['case_file']
    old = campaign.FINAL / parent['run']
    assert sha256(source) == parent['case_sha256']
    manifest = json.loads((old / 'manifest.json').read_text())
    for filename, expected in manifest['files'].items():
        assert sha256(old / filename) == expected, filename
    case = Case.load(source)
    before = case.to_dict()
    assert before == Case.load(old / 'case.json').to_dict()
    assert case.mesh.order == 2 and case.analysis.pressure_max_increment is None
    assert case.analysis.max_pressure == 35 and case.analysis.max_increment == .05
    assert case.analysis.failure_peeq == .8
    case.name = NAME
    case.analysis.pressure_max_increment = .005
    case.notes.append('Pressure resolution only: exact frozen combined60 C3D20R mesh and pre-pressure deck; pressure initial/maximum increment0.005. Physical inputs, material, criterion0.8 and convergence controls unchanged. Parent bracket24.0625-24.5MPa is provisional. Compare intervals, common-load PEEQ/crown response, contact and withdrawal; no physical burst or angle ranking acceptance.')
    case.validate()
    for key in ('pipe', 'gouge', 'dent', 'material', 'mesh', 'probe_positions'):
        assert case.to_dict()[key] == before[key]
    assert {k:v for k,v in case.to_dict()['analysis'].items() if k != 'pressure_max_increment'} == {k:v for k,v in before['analysis'].items() if k != 'pressure_max_increment'}
    cp = campaign.CASES / (NAME + '.json')
    directory = campaign.RUNS / NAME
    assert not cp.exists() and not directory.exists()
    case.save(cp)
    directory.mkdir()
    case.save(directory / 'case.json')
    for filename in ('model.json', 'mesh.vtu', 'geometry.png'):
        shutil.copy2(old / filename, directory / filename)
    write_deck(case, Model.load(old / 'model.json'), directory)
    assert sha256(directory / 'mesh.inp') == sha256(old / 'mesh.inp')
    prefix, pressure = (old / 'job.inp').read_text().split('** Stage: pressurise')
    prefix = prefix.replace(PARENT, NAME, 1)
    original = f'{case.analysis.initial_increment:.12g}, 1., {case.analysis.min_increment:.12g}, {case.analysis.max_increment:.12g}\n'
    assert pressure.count(original) == 1
    expected = prefix + '** Stage: pressurise' + pressure.replace(original, '0.005, 1., 1e-06, 0.005\n', 1)
    assert (directory / 'job.inp').read_text() == expected
    manifest.update(case=NAME, created_utc=datetime.now(timezone.utc).isoformat(),
        parent_run=parent['run'], parent_manifest_sha256=sha256(old / 'manifest.json'),
        preparation_script_sha256=sha256(Path(__file__)),
        mesh_provenance='Exact frozen parent model/mesh; no remeshing',
        files={n:sha256(directory / n) for n in ('case.json','model.json','mesh.inp','job.inp')},
        source_hashes={str(p.relative_to(campaign.FINAL / 'sym')):sha256(p) for p in (campaign.FINAL / 'sym/src/pipe_sim').glob('*.py')})
    write_json(directory / 'manifest.json', manifest)
    write_json(directory / 'status.json', {'state':'generated'})
    register()


def register():
    plan = json.loads(campaign.PLAN.read_text())
    assert not any(j['id'] == NAME for j in plan['jobs'])
    parent = next(j for j in plan['jobs'] if j['id'] == PARENT)
    cp = campaign.CASES / (NAME + '.json')
    directory = campaign.RUNS / NAME
    old = campaign.FINAL / parent['run']
    case = Case.load(cp)
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert json.loads((directory / 'status.json').read_text())['state'] == 'generated'
    for filename, expected_hash in manifest['files'].items():
        assert sha256(directory / filename) == expected_hash, filename
    assert case.to_dict() == Case.load(directory / 'case.json').to_dict()
    assert sha256(directory / 'mesh.inp') == sha256(old / 'mesh.inp')
    before = Case.load(old / 'case.json').to_dict()
    after = case.to_dict()
    for key in before:
        if key not in ('name','notes','analysis'):
            assert before[key] == after[key], key
    assert after['analysis']['pressure_max_increment'] == .005
    assert {k:v for k,v in before['analysis'].items() if k != 'pressure_max_increment'} == {k:v for k,v in after['analysis'].items() if k != 'pressure_max_increment'}
    parent_deck = (old / 'job.inp').read_text().replace(PARENT, NAME, 1)
    prefix, pressure = parent_deck.split('** Stage: pressurise')
    original = f'{case.analysis.initial_increment:.12g}, 1., {case.analysis.min_increment:.12g}, {case.analysis.max_increment:.12g}\n'
    assert pressure.count(original) == 1
    assert (directory / 'job.inp').read_text() == prefix + '** Stage: pressurise' + pressure.replace(original, '0.005, 1., 1e-06, 0.005\n', 1)
    child = copy.deepcopy(parent)
    child.update(id=NAME, parent_attempt=PARENT, case_file=str(cp.relative_to(campaign.FINAL)),
        case_sha256=sha256(cp), run=str(directory.relative_to(campaign.FINAL)),
        changes={'analysis':{'pressure_max_increment':.005}},
        description='Bounded combined60 frozen-mesh pressure-resolution control')
    child.pop('superseded_by', None)
    if child.get('entry') is not None:
        child['entry'].update(id=NAME, case_file=child['case_file'], case_sha256=child['case_sha256'], assumptions=case.notes)
    plan['jobs'].append(child)
    write_json(campaign.PLAN, plan)
    print(NAME, flush=True)


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, sources = [], {}
    for name in (PARENT, NAME):
        d = campaign.RUNS / name
        status = json.loads((d / 'status.json').read_text())
        row = {'case_id':name, 'solver_state':status['state'], 'validated':False}
        if (d / 'results/summary.json').exists():
            row.update(json.loads((d / 'results/summary.json').read_text()))
            h = pd.read_csv(d / 'results/history.csv')
            row['maximum_pressure_mpa'] = float(h.pressure_mpa.max())
            for f in ('results/summary.json','results/history.csv','manifest.json','job.sta'):
                sources[str((d / f).relative_to(campaign.FINAL))] = sha256(d / f)
        rows.append(row)
    write_json(OUT / 'comparison.json', {'cases':rows, 'sources':sources,
        'limitations':'Numerical pressure-resolution comparison only; no physical burst reference for combined60, no calibrated fracture law, mesh convergence or angle ranking acceptance. Preserve any valid earlier crossing despite later stop.'})
    print(OUT, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','register','report'])
    args = parser.parse_args()
    {'prepare':prepare, 'register':register, 'report':report}[args.action]()
