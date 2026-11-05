"""Bounded dent-validation followups and reproducible thesis evidence delivery."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipe_sim.config import Case
from pipe_sim.runner import generate, run, sha256
from pipe_sim.results import extract
from prepare_cases import FINAL
from compare import check_run
from plot_investigation import reference
from comparison_utils import loop_branches, error_metrics
from terminal_progress import solver_directories

PLAN = FINAL/'results/config/investigation.json'
OUT = FINAL/'results/output/deadline_delivery'
TAG = 'wall3_20261018'
PARENTS = ('a37_contact_fixed', 'a37_8pct_contact_fixed')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def prepare():
    plan = json.loads(PLAN.read_text())
    for parent_id in PARENTS:
        name = parent_id+'_'+TAG
        if any(j['id'] == name for j in plan['jobs']):
            continue
        parent = next(j for j in plan['jobs'] if j['id'] == parent_id)
        check_run(parent, parent['entry'])
        case = Case.load(FINAL/parent['case_file'])
        assert case.mesh.wall_layers == 2
        case.mesh.wall_layers = 3
        case.name = name
        case.analysis.timeout_seconds = 3600
        case.notes.append('Deadline validation: wall layers 2 to 3 only; same material, contact, geometry, loading, solver and surface mesh. One-hour operational limit. Compare loading/unloading, residual dent and PEEQ with retained parent; not a burst prediction.')
        case.validate()
        path = FINAL/'sym/cases/investigation'/(name+'.json')
        assert not path.exists()
        case.save(path)
        child = copy.deepcopy(parent)
        child.update(id=name, parent_attempt=parent_id, case_file=str(path.relative_to(FINAL)),
                     case_sha256=sha256(path), run='sym/runs/investigation/'+name,
                     description='Wall refinement only: '+parent_id+' (2 to 3 layers)',
                     changes={'mesh': {'wall_layers': 3}})
        child['entry'].update(id=name, case_file=child['case_file'], case_sha256=child['case_sha256'], assumptions=case.notes)
        plan['jobs'].append(child)
    save(PLAN, plan)
    return [p+'_'+TAG for p in PARENTS]


def execute_queue():
    jobs = json.loads(PLAN.read_text())['jobs']
    for parent in PARENTS:
        name = parent+'_'+TAG
        job = next(j for j in jobs if j['id'] == name)
        directory = FINAL/job['run']
        if (directory/'status.json').exists():
            print('Preserving existing attempt', name, flush=True)
            continue
        while True:
            if json.loads((FINAL/'results/automation/paused.json').read_text())['state'] != 'resumed':
                print('Manual pause: queue exits', flush=True)
                return
            native = solver_directories(set(), run_root=FINAL/'sym')
            memory = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))/1024
            if native is not None and sum(len(v['pids']) for v in native.values()) < 2 and memory >= 13048 and shutil.disk_usage(FINAL).free >= 25*1024**3:
                break
            print('Waiting for a solver slot and resource admission', flush=True)
            time.sleep(10)
        source = FINAL/job['case_file']
        assert sha256(source) == job['case_sha256']
        exe = FINAL/job['solver_executable']
        assert sha256(exe) == job['solver_sha256']
        os.environ['CCX_COMMAND'] = str(exe)
        generate(Case.load(source), directory)
        print('Starting bounded validation', name, flush=True)
        result = run(directory, memory_limit_mb=11000)
        print(json.dumps(result), flush=True)
        if (directory/'job.dat').exists() and (directory/'job.dat').stat().st_size:
            extract(directory)
        report()
    print('Bounded validation queue complete', flush=True)


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = json.loads(PLAN.read_text())
    metrics, provenance = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    for ax, parent in zip(axes, PARENTS):
        job = next(j for j in plan['jobs'] if j['id'] == parent)
        refs = reference(job['entry'])
        for role, style, color in [('experiment','o-','black'), ('paper_fe','--','gray')]:
            ref = refs[role]
            ax.plot(ref.x, ref.y, style, color=color, ms=3, label='Paper experiment' if role=='experiment' else 'Paper FE')
        for name, color in [(parent, '#0072b2'), (parent+'_'+TAG, '#d55e00')]:
            j = next((j for j in plan['jobs'] if j['id']==name), None)
            if not j or not (FINAL/j['run']/'results/history.csv').exists(): continue
            directory = check_run(j, j['entry'])
            case = Case.load(directory/'case.json')
            h = pd.read_csv(directory/'results/history.csv')
            sim = h[h.stage.isin(['indent','withdraw'])].copy()
            sim['x'] = sim.indenter_travel_mm-case.dent.clearance
            sim['y'] = sim.indenter_force_n/1000
            state = json.loads((directory/'status.json').read_text())['state']
            summary = json.loads((directory/'results/summary.json').read_text())
            ax.plot(sim.x, sim.y, color=color, label=f'CalculiX {case.mesh.wall_layers} wall layers ({state})')
            for role, ref in refs.items():
                for branch, part in loop_branches(ref).items():
                    curve = sim[sim.stage==branch]
                    if branch=='withdraw' and len(curve): curve=pd.concat([sim[sim.stage=='indent'].tail(1),curve])
                    values, _ = error_metrics(part, curve)
                    values.update(case_id=name, reference=role, branch=branch,
                                  peak_force_kn=float(sim.y.max()), peak_error_percent=100*(sim.y.max()/ref.y.max()-1),
                                  residual_mm=float(h[np.isclose(h.time,2)].crown_inward_displacement_mm.iloc[-1]) if np.isclose(h.time,2).any() else None,
                                  max_peeq=float(h.max_peeq.max()), solver_state=state,
                                  material_supported_peeq=j['entry']['material']['source_supported_plastic_strain_max'],
                                  contact_checks_passed=summary.get('contact_checks_passed'), validated=False)
                    metrics.append(values)
            for f in [directory/'case.json',directory/'manifest.json',directory/'results/history.csv', directory/'results/summary.json', FINAL/j['entry']['reference_file']]:provenance[str(f.relative_to(FINAL))]=sha256(f)
        ax.set(xlabel='Tool penetration (mm)', ylabel='Force (kN)', title='Shallow dent' if parent==PARENTS[0] else 'Deeper dent')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(OUT/'validation_loops.pdf', metadata={'CreationDate': None, 'ModDate': None});fig.savefig(OUT/'validation_loops.png',dpi=170);plt.close(fig)
    pd.DataFrame(metrics).to_csv(OUT/'validation_metrics.csv',index=False)
    campaign=json.loads((FINAL/'results/config/full_campaign.json').read_text())
    selected=[j for j in campaign['jobs'] if not j.get('superseded_by') and ((j['kind']=='orientation' and 'gouge' in j['id']) or j['id']=='full_zhao2021_p1_adaptive' or (j['kind']=='orientation' and 'combined' in j['id']))]
    # Sized near the thesis text width so labels remain readable after inclusion.
    fig,axes=plt.subplots(1,2,figsize=(8,4.8),layout='constrained');rows=[];histories=[]
    for j in selected:
        directory=FINAL/j['run'];case=Case.load(directory/'case.json')
        manifest=json.loads((directory/'manifest.json').read_text())
        for name,expected in manifest['files'].items():
            assert sha256(directory/name)==expected, str(directory/name)
        state=json.loads((directory/'status.json').read_text())['state']
        if state in ('running','building','generated') or not (directory/'results/history.csv').exists() or not (directory/'results/summary.json').exists():
            print(f'Pending orientation evidence: {j["id"]} ({state}); no extracted terminal history')
            continue
        h=pd.read_csv(directory/'results/history.csv');p=h[h.stage=='pressurise'].sort_values('pressure_mpa')
        if len(p)<2:continue
        combined=case.dent.enabled;angle=case.gouge.angle_deg
        state=json.loads((directory/'status.json').read_text())['state']
        summary=json.loads((directory/'results/summary.json').read_text())
        variant = j['id'].split('_adaptive', 1)[1] if '_adaptive' in j['id'] else ''
        label = f'{angle:g} deg' + (' pressure step 0.005' if 'pressure005' in variant else
                                      ' C3D8' if 'c3d8' in variant else variant.replace('_',' '))
        axes[int(combined)].plot(p.pressure_mpa,p.max_peeq,label=label,
                                 linestyle='--' if variant else '-')
        for pressure in (5.,10.,15.):
            if p.pressure_mpa.min()<=pressure<=p.pressure_mpa.max():
                rows.append(dict(case_id=j['id'],angle_deg=angle,combined=combined,pressure_mpa=pressure,
                    interpolated_max_peeq=float(np.interp(pressure,p.pressure_mpa,p.max_peeq)),
                    crown_inward_mm=float(np.interp(pressure,p.pressure_mpa,p.crown_inward_displacement_mm)),
                    last_pressure_mpa=float(p.pressure_mpa.max()),solver_state=state,
                    contact_checks_passed=summary.get('contact_checks_passed'),
                    withdrawal_completed=summary.get('withdrawal_completed'),validated=False))
        p=p.copy();p['case_id']=j['id'];p['angle_deg']=angle;p['combined']=combined;histories.append(p)
        for f in [directory/'case.json',directory/'manifest.json',directory/'results/history.csv',directory/'results/summary.json']:provenance[str(f.relative_to(FINAL))]=sha256(f)
    for ax,title in zip(axes,['Gouge only','Dent plus gouge']):
        ax.set(xlabel='Applied pressure (MPa)',ylabel='Global maximum equivalent plastic strain',title=title)
        ax.grid(alpha=.2);ax.legend(fontsize=10)
    fig.suptitle('Provisional model response\nCurve endpoints are numerical stops, not burst', fontsize=11)
    fig.savefig(OUT/'orientation_response.pdf', metadata={'CreationDate': None, 'ModDate': None});fig.savefig(OUT/'orientation_response.png',dpi=170);plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT/'common_pressure.csv',index=False)
    pd.concat(histories).to_csv(OUT/'orientation_histories.csv',index=False)
    for name in ('pipe_direction_sampled_early_plastic_20261018T034621Z.json', 'early_plastic_component_sampled_20261018T034621Z.json'):
        evidence=FINAL/'results/output'/name
        if evidence.exists(): provenance[str(evidence.relative_to(FINAL))]=sha256(evidence)
    save(OUT/'provenance.json',dict(sources=provenance,limitations='Input hashes verified; extracted histories snapshotted, not a fresh audit of every native DAT byte. Partial histories and interpolation only; no new physical burst, fracture validation or orientation-strength ranking. Global maxima need not refer to the same material point.'))
    print('Evidence bundle refreshed',OUT,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','queue','report']);args=parser.parse_args()
    if args.action=='prepare':print(prepare())
    elif args.action=='queue':execute_queue()
    else:report()
