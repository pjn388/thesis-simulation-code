"""One user-approved pressure-only CONTACT iteration-budget trial; no auto retry."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import campaign
from pipe_sim.config import Case
from pipe_sim.model import Model
from pipe_sim.decks import write_deck
from pipe_sim.runner import sha256, write_json
from terminal_progress import solver_directories

PARENT = 'orientation_combined_90_adaptive_pressure005_20261020'
NAME = 'orientation_combined_90_pressure005_it120_20261022'
OUT = campaign.FINAL / 'results/output/orientation90_it120_20261022'
CARD = '*CONTROLS, PARAMETERS=CONTACT\n,,,120\n'
SOLVER_HASH = '0f974d82f4355a93006fcc4f4bcabcd93da3631013b6ba33934648c41f1fa72b'


def verify():
    jobs = {j['id']:j for j in json.loads(campaign.PLAN.read_text())['jobs']}
    for name in (PARENT, NAME):
        job = jobs[name]; directory = campaign.FINAL / job['run']
        assert sha256(campaign.FINAL / job['case_file']) == job['case_sha256']
        manifest = json.loads((directory / 'manifest.json').read_text())
        for filename, digest in manifest['files'].items():
            assert sha256(directory / filename) == digest, (name, filename)
        assert Case.load(directory / 'case.json').to_dict() == Case.load(campaign.FINAL / job['case_file']).to_dict()
    old, new = [campaign.FINAL / jobs[n]['run'] for n in (PARENT, NAME)]
    a, b = [Case.load(d / 'case.json').to_dict() for d in (old, new)]
    assert b['analysis']['pressure_contact_iterations'] == 120
    assert a['analysis']['pressure_contact_iterations'] is None
    assert b['analysis']['failure_peeq'] == .8
    assert b['analysis']['pressure_max_increment'] == .005
    for key in a:
        if key not in ('name', 'notes', 'analysis'): assert a[key] == b[key], key
    assert {k:v for k,v in a['analysis'].items() if k != 'pressure_contact_iterations'} == {k:v for k,v in b['analysis'].items() if k != 'pressure_contact_iterations'}
    expected = (old / 'job.inp').read_text().replace(PARENT, NAME, 1)
    actual = (new / 'job.inp').read_text()
    assert actual.count(CARD) == 1
    assert CARD not in actual.split('** Stage: pressurise')[0]
    assert actual.replace(CARD, '') == expected
    assert sha256(new / 'mesh.inp') == sha256(old / 'mesh.inp')
    assert sha256(new / 'model.json') == sha256(old / 'model.json')
    assert sha256(campaign.FINAL / 'sym/vendor/ccx_pardiso/ccx_2.23_pardiso') == SOLVER_HASH
    return {'input_identity_passed':True, 'only_solver_change':'pressure CONTACT itf2f60->120; all FIELD tolerances unchanged',
            'parent':PARENT, 'child':NAME, 'solver_sha256':SOLVER_HASH,
            'parent_manifest_sha256':sha256(old / 'manifest.json'), 'child_manifest_sha256':sha256(new / 'manifest.json')}


def prepare():
    plan = json.loads(campaign.PLAN.read_text())
    assert not any(j['id'] == NAME for j in plan['jobs'])
    parent = next(j for j in plan['jobs'] if j['id'] == PARENT)
    old = campaign.FINAL / parent['run']
    manifest = json.loads((old / 'manifest.json').read_text())
    for filename, digest in manifest['files'].items(): assert sha256(old / filename) == digest
    assert sha256(campaign.FINAL / parent['case_file']) == parent['case_sha256']
    case = Case.load(old / 'case.json')
    assert case.to_dict() == Case.load(campaign.FINAL / parent['case_file']).to_dict()
    assert case.analysis.pressure_contact_iterations is None
    case.name = NAME; case.analysis.pressure_contact_iterations = 120
    case.notes.append('User-approved bounded22October trial: pressure-stage CONTACT itf2f60->120 only; blank other CONTACT fields preserve defaults. Exact frozen parent mesh, pre-pressure stages, material, loading and FIELD tolerances. PEEQ0.8 unchanged. No physical burst or convergence guarantee; no automatic retries or angle expansion.')
    case.validate()
    directory = campaign.RUNS / NAME; cp = campaign.CASES / (NAME + '.json')
    assert not directory.exists() and not cp.exists()
    directory.mkdir(); case.save(cp); case.save(directory / 'case.json')
    for filename in ('model.json','mesh.vtu','geometry.png'): shutil.copy2(old / filename, directory / filename)
    write_deck(case, Model.load(old / 'model.json'), directory)
    manifest.update(case=NAME, created_utc=datetime.now(timezone.utc).isoformat(),
        parent_run=parent['run'], parent_manifest_sha256=sha256(old / 'manifest.json'),
        preparation_script_sha256=sha256(Path(__file__)), mesh_provenance='Exact frozen parent mesh; no remeshing',
        files={n:sha256(directory/n) for n in ('case.json','model.json','mesh.inp','job.inp')},
        source_hashes={str(p.relative_to(campaign.FINAL/'sym')):sha256(p) for p in (campaign.FINAL/'sym/src/pipe_sim').glob('*.py')})
    write_json(directory / 'manifest.json', manifest)
    write_json(directory / 'status.json', {'state':'generated'})
    child = copy.deepcopy(parent)
    child.update(id=NAME, parent_attempt=PARENT, case_file=str(cp.relative_to(campaign.FINAL)),
        case_sha256=sha256(cp), run=str(directory.relative_to(campaign.FINAL)),
        changes={'analysis':{'pressure_contact_iterations':120}}, description='Bounded pressure-only CONTACT iteration-cap120 trial')
    child.pop('superseded_by', None)
    if child.get('entry') is not None:
        child['entry'].update(id=NAME, case_file=child['case_file'], case_sha256=child['case_sha256'], assumptions=case.notes)
    plan['jobs'].append(child); write_json(campaign.PLAN, plan)
    OUT.mkdir(parents=True, exist_ok=True); write_json(OUT / 'input_verification.json', verify())
    print(NAME, flush=True)


def report():
    """Terminal accepted-history comparison; interpolation never extrapolates."""
    OUT.mkdir(parents=True, exist_ok=True)
    audit = verify(); rows = []; histories = {}; sources = {}
    for name in (PARENT, NAME):
        d = campaign.RUNS / name; status = json.loads((d/'status.json').read_text())
        row = {'case_id':name, 'solver_state':status['state'], 'validated':False}
        if status['state'] not in ('generated','running','building') and (d/'results/summary.json').exists():
            row.update(json.loads((d/'results/summary.json').read_text()))
            histories[name] = pd.read_csv(d/'results/history.csv')
            for file in ('results/history.csv','results/summary.json','job.sta'):
                sources[str((d/file).relative_to(campaign.FINAL))] = sha256(d/file)
        rows.append(row)
        for file in ('manifest.json','status.json'):
            sources[str((d/file).relative_to(campaign.FINAL))] = sha256(d/file)
    audit.update(cases=rows, sources=sources,
        limitations='Only accepted extracted histories. No physical90degree burst reference. No fitting, extrapolated comparison, or solver-stop-as-burst. A criterion crossing remains provisional; inspect contact, withdrawal, material range and mesh limitations.')
    if NAME in histories and PARENT in histories:
        parent, child = histories[PARENT], histories[NAME]
        metrics = {}; comparisons = []
        for stage in ('indent','withdraw','pressurise'):
            a, b = [h[h.stage == stage] for h in (parent,child)]
            if a.empty or b.empty: continue
            coordinate = 'pressure_mpa' if stage == 'pressurise' else 'time'
            a = a.sort_values(coordinate).drop_duplicates(coordinate,keep='last')
            b = b.sort_values(coordinate).drop_duplicates(coordinate,keep='last')
            b = b[b[coordinate].between(a[coordinate].min(), a[coordinate].max())]
            if b.empty: continue
            frame = pd.DataFrame({'stage':stage, coordinate:b[coordinate].to_numpy()})
            metrics[stage] = {'overlapping_samples':len(b), 'comparison':'parent linearly interpolated at child samples within overlap only'}
            for field in ('max_peeq','crown_inward_displacement_mm','indenter_force_n'):
                reference = np.interp(b[coordinate], a[coordinate], a[field])
                frame['parent_'+field] = reference
                frame['trial_'+field] = b[field].to_numpy()
                difference = b[field].to_numpy()-reference
                frame['difference_'+field] = difference
                metrics[stage]['max_abs_difference_'+field] = float(np.max(np.abs(difference)))
            comparisons.append(frame)
        if comparisons: pd.concat(comparisons, ignore_index=True).to_csv(OUT/'overlapping_response.csv', index=False)
        audit['overlap_metrics'] = metrics
        fig, axes = plt.subplots(1,2,figsize=(11,4),layout='constrained')
        for name, label in ((PARENT,'Parent: 60 iterations'),(NAME,'Trial: 120 iterations')):
            h = histories[name]; p = h[h.stage == 'pressurise']
            axes[0].plot(p.pressure_mpa,p.max_peeq,label=label)
            axes[1].plot(p.pressure_mpa,p.crown_inward_displacement_mm,label=label)
        axes[0].axhline(.8,color='black',linestyle=':',label='Provisional PEEQ criterion')
        for ax, ylabel in zip(axes,('Maximum PEEQ','Crown inward displacement (mm)')):
            ax.set(xlabel='Pressure (MPa)',ylabel=ylabel); ax.grid(alpha=.2); ax.legend(fontsize=8)
        fig.suptitle('90 degree iteration-budget comparison — not validated physical burst')
        fig.savefig(OUT/'pressure_comparison.png',dpi=160); fig.savefig(OUT/'pressure_comparison.pdf'); plt.close(fig)
    write_json(OUT/'comparison.json', audit)
    print(OUT,flush=True)


def run():
    verify()
    assert json.loads((campaign.FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
    native = solver_directories(set(),run_root=campaign.FINAL/'sym')
    assert native is not None and len(native)<2, 'No free native solver slot'
    assert campaign.available_memory_mb()>=13048, 'Insufficient memory admission headroom'
    assert shutil.disk_usage(campaign.FINAL).free>=40*1024**3, 'Disk admission reserve'
    progress = campaign.OUT/'queue_it120.json'
    write_json(progress,{'state':'running','current':NAME,'index':0,'total':1,'cases':[NAME]})
    try:
        campaign.execute(NAME)
        report()
    except Exception as exc:
        write_json(progress,{'state':'failed','current':NAME,'error':str(exc),'total':1,'cases':[NAME]})
        raise
    write_json(progress,{'state':'finished','index':1,'total':1,'cases':[NAME]})


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','verify','report','run'])
    action=parser.parse_args().action
    if action=='verify': print(json.dumps(verify(),indent=2))
    else: globals()[action]()
