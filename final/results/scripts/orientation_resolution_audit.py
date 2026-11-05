"""Audit terminal frozen-mesh pressure refinements against retained histories."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import campaign
from pipe_sim.runner import sha256, write_json


def audit(angle):
    parent = f'orientation_combined_{angle:02d}_adaptive'
    child = parent + '_pressure005_20261020'
    out = campaign.FINAL / f'results/output/orientation{angle}_pressure005_20261020'
    out.mkdir(exist_ok=True)
    sources, rows, histories = {}, [], {}
    for name in (parent, child):
        directory = campaign.RUNS / name
        state = json.loads((directory/'status.json').read_text())
        assert state['state'] not in ('running', 'building', 'generated')
        manifest = json.loads((directory/'manifest.json').read_text())
        for filename, expected in manifest['files'].items():
            assert sha256(directory/filename) == expected, filename
        summary = json.loads((directory/'results/summary.json').read_text())
        history = pd.read_csv(directory/'results/history.csv')
        pressure = history[history.stage == 'pressurise'].sort_values('pressure_mpa')
        assert pressure.pressure_mpa.is_unique
        histories[name] = pressure
        bracket = summary.get('pressure_bracket_mpa')
        crossing = None
        if bracket:
            crossing = []
            for value in bracket:
                frame = pressure.loc[np.isclose(pressure.pressure_mpa, value, atol=1e-7, rtol=0)]
                assert len(frame) == 1
                crossing.append(float(frame.iloc[0].max_peeq))
            assert crossing[0] < .8 <= crossing[1]
        rows.append(dict(case_id=name, solver_state=state['state'], stop_reason=state.get('stop_reason'),
            elapsed_seconds=state.get('elapsed_seconds'), bracket_mpa=bracket, bracket_peeq=crossing,
            endpoint_mpa=float(pressure.pressure_mpa.max()), max_peeq=summary['maximum_observed_peeq'],
            contact_pass=summary['contact_checks_passed'], withdrawal_completed=summary['withdrawal_completed'],
            penetration_mm=summary['max_nodal_contact_penetration_mm'],
            residual_crown_mm=summary['residual_crown_displacement_mm']))
        for file in ('status.json','manifest.json','results/summary.json','results/history.csv','job.sta'):
            sources[str((directory/file).relative_to(campaign.FINAL))] = sha256(directory/file)
    old, new = (campaign.RUNS/name for name in (parent,child))
    assert sha256(old/'mesh.inp') == sha256(new/'mesh.inp')
    before = json.loads((old/'case.json').read_text())
    after = json.loads((new/'case.json').read_text())
    for key in before:
        if key not in ('name','notes','analysis'):
            assert before[key] == after[key], key
    assert after['analysis']['pressure_max_increment'] == .005
    assert {k:v for k,v in before['analysis'].items() if k != 'pressure_max_increment'} == {k:v for k,v in after['analysis'].items() if k != 'pressure_max_increment'}
    prefix,pressure = (old/'job.inp').read_text().replace(parent,child,1).split('** Stage: pressurise')
    original = f"{before['analysis']['initial_increment']:.12g}, 1., {before['analysis']['min_increment']:.12g}, {before['analysis']['max_increment']:.12g}\n"
    assert pressure.count(original) == 1
    assert (new/'job.inp').read_text() == prefix + '** Stage: pressurise' + pressure.replace(original,'0.005, 1., 1e-06, 0.005\n',1)
    common = []
    upper = min(h.pressure_mpa.max() for h in histories.values())
    for p in (5.,10.,15.,18.,20.,22.,24.,24.5,25.):
        if p > upper or any(p < h.pressure_mpa.min() for h in histories.values()): continue
        row = {'pressure_mpa':p}
        for label, name in (('parent',parent),('refined',child)):
            h = histories[name]
            for column in ('max_peeq','crown_inward_displacement_mm'):
                row[label+'_'+column] = float(np.interp(p,h.pressure_mpa,h[column]))
        common.append(row)
    pd.DataFrame(common).to_csv(out/'common_pressure.csv',index=False)
    siblings = []
    if angle == 60:
        for name in ('mesh_combined60_wall6_20261010','mesh_combined60_fine4_20261010'):
            d = campaign.RUNS/name
            summary = json.loads((d/'results/summary.json').read_text())
            siblings.append(dict(case_id=name, bracket_mpa=summary.get('pressure_bracket_mpa'),
                maximum_observed_peeq=summary['maximum_observed_peeq']))
            sources[str((d/'results/summary.json').relative_to(campaign.FINAL))] = sha256(d/'results/summary.json')
    a,b = [r['bracket_mpa'] for r in rows]
    write_json(out/'audit.json',dict(cases=rows, upper_shift_mpa=b[1]-a[1] if a and b else None,
        material_supported_peeq=0.791057, mesh_siblings=siblings, sources=sources,
        script_sha256=sha256(Path(__file__)), common_pressure_sha256=sha256(out/'common_pressure.csv'),
        limitations='Common-pressure values use linear interpolation of accepted frames. No physical burst reference for these angles. PEEQ0.8 exceeds recovered material support. Pressure resolution is not mesh convergence or physical validation; prior brackets survive later resource/convergence stops. No full angle ranking.'))
    print(out)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('angle',type=int)
    audit(parser.parse_args().angle)
