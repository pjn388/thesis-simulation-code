"""Bounded production-solver element trials and physical three-way comparisons.

The only numerical model change is C3D20R -> C3D8 at the parent's wall count.
This is a formulation sensitivity, not evidence of mesh convergence or validation.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import campaign
from pipe_sim.config import Case
from pipe_sim.runner import sha256, write_json
from plot_investigation import reference
from comparison_utils import loop_branches, error_metrics

FINAL = campaign.FINAL
TAG = 'c3d8_20261019'
PARENTS = ('orientation_intact_00_adaptive', 'full_zhao2021_p1_adaptive')
OUT = FINAL / 'results/output' / TAG


def prepare():
    plan = json.loads(campaign.PLAN.read_text())
    for parent_id in PARENTS:
        name = parent_id + '_' + TAG
        if any(j['id'] == name for j in plan['jobs']):
            raise FileExistsError(name)
        parent = next(j for j in plan['jobs'] if j['id'] == parent_id)
        source = FINAL / parent['case_file']
        assert sha256(source) == parent['case_sha256']
        case = Case.load(source)
        before = case.to_dict()
        assert case.mesh.order == 2 and case.mesh.wall_layers == 4
        case.mesh.order = 1
        case.name = name
        case.notes.append('19 October physical-comparison trial: only pipe element order changes from C3D20R to C3D8; four wall layers, material, loads, contact, fixtures and PEEQ 0.8 criterion retained. No experimental tangent modification. C3D8 is used by Allouti, but is not the twenty-node formulation in Zhao. Mesh convergence and physical accuracy remain to be assessed.')
        case.validate()
        for key in ('pipe', 'gouge', 'dent', 'material', 'analysis', 'probe_positions'):
            assert case.to_dict()[key] == before[key], key
        assert {k:v for k,v in case.to_dict()['mesh'].items() if k != 'order'} == {k:v for k,v in before['mesh'].items() if k != 'order'}
        path = campaign.CASES / (name + '.json')
        assert not path.exists()
        case.save(path)
        child = copy.deepcopy(parent)
        child.update(id=name, kind='validation' if parent['entry'] else 'formulation',
                     parent_attempt=parent_id, case_file=str(path.relative_to(FINAL)),
                     case_sha256=sha256(path), run='sym/runs/campaign/' + name,
                     changes={'mesh': {'order': 1}},
                     description='Production-solver C3D8 sensitivity; unchanged parent physics')
        child.pop('superseded_by', None)
        if child['entry']:
            child['entry'].update(id=name, case_file=child['case_file'],
                                  case_sha256=child['case_sha256'], assumptions=case.notes)
        plan['jobs'].append(child)
    write_json(campaign.PLAN, plan)
    print('Prepared', [p + '_' + TAG for p in PARENTS], flush=True)


def report():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _report()
        _report("P4")


def _report(specimen="P1"):
    parents = (PARENTS[0], f"full_zhao2021_{specimen.lower()}_adaptive")
    out = OUT if specimen == "P1" else OUT / specimen.lower()
    out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(campaign.PLAN.read_text())
    jobs = {j['id']: j for j in plan['jobs']}
    entry = jobs[parents[1]]['entry']
    refs = reference(entry)
    tests = pd.read_csv(FINAL / 'data/tables/zhao2021/table04_burst_tests.csv')
    measured = float(tests.loc[tests.specimen == specimen, 'burst_pressure_MPa'].iloc[0])
    paper = pd.read_csv(FINAL / 'data/figures/zhao2021/fig14_failure_pressure.csv')
    matched = paper[abs(paper.x - measured) <= paper.x_digitization_uncertainty]
    assert len(matched) == 1
    published = matched.iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout='constrained')
    for role, ref in refs.items():
        axes[0].plot(ref.x, ref.y, 'o-' if role == 'experiment' else '--',
                     ms=2, label='Physical test' if role == 'experiment' else 'Paper FE')
    axes[1].plot(0, measured, 'ko', label='Physical burst')
    axes[1].errorbar(1, published.y, yerr=published.y_digitization_uncertainty,
                     fmt='^', label='Paper FE')
    rows, sources, missing, metrics = [], {}, [], []
    for parent in parents:
        for is_new in (False, True):
            name = parent + '_' + TAG if is_new else parent
            job = jobs[name]
            directory = FINAL / job['run']
            case = Case.load(FINAL / job['case_file'])
            label = 'C3D8 trial' if is_new else 'C3D20R parent'
            hp = directory / 'results/history.csv'
            status = json.loads((directory / 'status.json').read_text()) if (directory / 'status.json').exists() else {'state':'not_run'}
            if not hp.exists():
                if parent == parents[1]: missing.append(label + ': ' + status['state'])
                continue
            manifest = json.loads((directory / 'manifest.json').read_text())
            for filename, expected in manifest['files'].items():
                assert sha256(directory / filename) == expected
            h = pd.read_csv(hp)
            summary = json.loads((directory / 'results/summary.json').read_text())
            row = dict(case_id=name, solver_state=status['state'],
                       last_pressure_mpa=float(h.pressure_mpa.max()),
                       max_peeq=float(h.max_peeq.max()), failure_status=summary.get('status'),
                       criterion_pressure_mpa=None, experimental_error_percent=None, validated=False,
                       pressure_bracket_mpa=summary.get('pressure_bracket_mpa'),
                       residual_crown_displacement_mm=summary.get('residual_crown_displacement_mm'),
                       max_contact_penetration_mm=summary.get('max_nodal_contact_penetration_mm'))
            if parent == parents[1]:
                loop = h[h.stage.isin(['indent','withdraw'])]
                axes[0].plot(loop.indenter_travel_mm-case.dent.clearance,
                             loop.indenter_force_n/1000, label=label+' ('+status['state']+')')
                loop = loop.copy()
                loop['x'] = loop.indenter_travel_mm-case.dent.clearance
                loop['y'] = loop.indenter_force_n/1000
                for role, ref in refs.items():
                    for branch, part in loop_branches(ref).items():
                        curve = loop[loop.stage == branch]
                        if branch == 'withdraw' and len(curve):
                            curve = pd.concat([loop[loop.stage == 'indent'].tail(1), curve])
                        values, _ = error_metrics(part, curve)
                        values.update(case_id=name, reference=role, branch=branch, material_supported_peeq=job['entry']['material']['source_supported_plastic_strain_max'])
                        metrics.append(values)
                row['peak_force_kn'] = float(loop.indenter_force_n.max()/1000) if len(loop) else None
                row['peak_error_percent'] = 100*(row['peak_force_kn']/refs['experiment'].y.max()-1) if len(loop) else None
                usable = (summary.get('status') == 'criterion_crossed' and
                          summary.get('contact_checks_passed') is True and
                          summary.get('withdrawal_completed') is True)
                if usable:
                    pressure = float(summary['burst_pressure_mpa'])
                    lower = float(summary['pressure_bracket_mpa'][0])
                    axes[1].errorbar(3 if is_new else 2, pressure,
                                     yerr=[[pressure-lower],[0]], fmt='s', label=label+' criterion crossing')
                    row.update(criterion_pressure_mpa=pressure, experimental_error_percent=100*(pressure/measured-1))
                else:
                    missing.append(label+': no usable crossing')
                row.update(experiment_mpa=measured, paper_fe_mpa=float(published.y),
                           contact_checks_passed=summary.get('contact_checks_passed'),
                           withdrawal_completed=summary.get('withdrawal_completed'))
            else:
                pressure = h[h.stage == 'pressurise']
                axes[2].plot(pressure.pressure_mpa, pressure.max_peeq, label=label)
            rows.append(row)
            for path in (hp, directory/'status.json', directory/'results/summary.json', directory/'manifest.json'):
                sources[str(path.relative_to(FINAL))] = sha256(path)
    axes[0].set(title=f'Zhao {specimen} denting', xlabel='Tool penetration (mm)', ylabel='Force (kN)')
    axes[1].set(title=f'Zhao {specimen} pressure comparison', ylabel='Pressure (MPa)',
                xticks=[0,1,2,3], xticklabels=['Test','Paper FE','Parent','Trial'], xlim=(-.5,3.5))
    if missing: axes[1].text(.03,.03,'\n'.join(missing),transform=axes[1].transAxes,fontsize=8)
    axes[2].set(title='Intact control: response, not burst', xlabel='Pressure (MPa)', ylabel='Maximum PEEQ')
    for ax in axes: ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out/'physical_comparison.png',dpi=160)
    fig.savefig(out/'physical_comparison.pdf');plt.close(fig)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(metrics).to_csv(out/'branch_metrics.csv',index=False)
    for p in (Path(__file__), FINAL/entry['reference_file'], FINAL/'data/tables/zhao2021/table04_burst_tests.csv', FINAL/'data/figures/zhao2021/fig14_failure_pressure.csv'):
        sources[str(p.relative_to(FINAL))] = sha256(p)
    write_json(out/'provenance.json',dict(sources=sources, limitations='Experimental errors reported without a new acceptance cutoff. Criterion crossing is provisional and requires valid contact/full withdrawal; numerical stops are not burst. Same four wall layers do not establish equivalent discretization accuracy across element types. No isolated experimental tangent used.'))
    print('Comparison refreshed:',out,flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','report']);p.add_argument('case',nargs='?')
    args=p.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'report': report()
    else:
        assert args.case in [name+'_'+TAG for name in PARENTS]
        campaign.execute(args.case)
        report()
