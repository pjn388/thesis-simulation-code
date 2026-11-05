"""Quantify terminal child mesh sensitivity against immutable retained parents."""
import csv
import json
from pathlib import Path
import numpy as np
from pipe_sim.runner import sha256
from pipe_sim.model import Model
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]

def load(case):
    run = FINAL/'sym/runs/campaign'/case
    if (run/'results/history.csv').exists() and (run/'results/summary.json').exists():
        rows = list(csv.DictReader((run/'results/history.csv').open()))
        h = [dict(stage=r['stage'],time=float(r['time']),pressure_mpa=float(r['pressure_mpa']),
                  tool_travel_mm=float(r['indenter_travel_mm']),tool_force_n=float(r['indenter_force_n']),
                  crown_inward_mm=float(r['crown_inward_displacement_mm']),max_peeq=float(r['max_peeq']),
                  max_peeq_element=int(r['max_peeq_element']),max_peeq_ip=int(r['max_peeq_ip'])) for r in rows]
        summary = json.loads((run/'results/summary.json').read_text())
        quality = dict(residual_crown_mm=summary['residual_crown_displacement_mm'],
                       withdrawal_completed=summary['withdrawal_completed'],contact_checks_passed=summary['contact_checks_passed'])
        evidence = run/'results/history.csv'
    else:
        evidence = FINAL/'results/output/live_cases'/case/'snapshot.json'
        snap = json.loads(evidence.read_text())
        assert all(snap['frozen_inputs_verified'].values())
        h = snap['history']
        quality = {k:snap[k] for k in ['residual_crown_mm','withdrawal_completed','tensile_force_detected']}
        quality['contact_frames'] = len(snap['contacts'])
        quality['max_penetration_mm'] = max(c['penetration_mm'] for c in snap['contacts'])
    accepted = converged_times(run/'job.sta')
    assert all(matches_sta(accepted,r['time']) for r in h)
    return h,quality,dict(path=str(evidence.relative_to(FINAL)),sha256=sha256(evidence),sta_sha256=sha256(run/'job.sta'))


def bracket(h):
    p = [r for r in h if r['stage']=='pressurise']
    for i,r in enumerate(p):
        if r['max_peeq'] >= 0.8:
            return [None if i==0 else p[i-1]['pressure_mpa'],r['pressure_mpa']]
    return None


def hotspots(case,h):
    model = Model.load(FINAL/'sym/runs/campaign'/case/'model.json')
    pressure = [r for r in h if r['stage']=='pressurise']
    crossing = next((r for r in pressure if r['max_peeq']>=0.8),None)
    result = {}
    for label,r in [('crossing',crossing),('terminal',pressure[-1])]:
        if r is None:
            result[label] = None
            continue
        element = r['max_peeq_element']
        conn = model.elements[element][1]
        result[label] = dict(element=element,integration_point=r['max_peeq_ip'],
            initial_corner_centroid_mm=np.mean([model.nodes[n] for n in conn[:8]],axis=0).tolist(),
            in_defect=element in model.elsets.get('DEFECT',[]))
    return result


def main(tag):
    output = FINAL/'results/output'/f'terminal_mesh_comparison_{tag}.json'
    assert not output.exists()
    records = []
    for family,parent in [('p1','full_zhao2021_p1_adaptive'),('combined60','orientation_combined_60_adaptive')]:
        ph,pq,pe = load(parent)
        for variant in ['wall6','fine4']:
            child = f'mesh_{family}_{variant}_20261010'
            ch,cq,ce = load(child)
            assert cq['withdrawal_completed'] and pq['withdrawal_completed']
            branches = {}
            for stage in ['indent','withdraw']:
                p = [r for r in ph if r['stage']==stage]
                c = [r for r in ch if r['stage']==stage]
                c.sort(key=lambda r:r['tool_travel_mm'])
                x = np.array([r['tool_travel_mm'] for r in c]); y = np.array([r['tool_force_n'] for r in c])
                # Consolidate repeated tool positions by their last sampled value.
                unique = {float(a):float(b) for a,b in zip(x,y)}
                x = np.array(sorted(unique)); y = np.array([unique[a] for a in x])
                points = [r for r in p if x[0]<=r['tool_travel_mm']<=x[-1]]
                delta = [float(np.interp(r['tool_travel_mm'],x,y)-r['tool_force_n']) for r in points]
                branches[stage] = dict(parent_samples=len(p),overlap_samples=len(points),
                    force_rmse_kn=float(np.sqrt(np.mean(np.square(delta)))/1000),
                    max_abs_force_difference_kn=float(np.max(np.abs(delta))/1000))
            pp = [r for r in ph if r['stage']=='pressurise']; cp = [r for r in ch if r['stage']=='pressurise']
            common = [r for r in pp if cp[0]['pressure_mpa']<=r['pressure_mpa']<=cp[-1]['pressure_mpa']]
            pressure = []
            for r in common:
                pressure.append(dict(pressure_mpa=r['pressure_mpa'],
                    peeq_difference=float(np.interp(r['pressure_mpa'],[a['pressure_mpa'] for a in cp],[a['max_peeq'] for a in cp])-r['max_peeq']),
                    crown_difference_mm=float(np.interp(r['pressure_mpa'],[a['pressure_mpa'] for a in cp],[a['crown_inward_mm'] for a in cp])-r['crown_inward_mm'])))
            pb,cb = bracket(ph),bracket(ch)
            overlap = None if pb is None or cb is None or pb[0] is None or cb[0] is None else max(pb[0],cb[0])<=min(pb[1],cb[1])
            peakp = max(r['tool_force_n'] for r in ph if r['stage']=='indent')
            peakc = max(r['tool_force_n'] for r in ch if r['stage']=='indent')
            records.append(dict(parent=parent,child=child,parent_evidence=pe,child_evidence=ce,
                branches=branches,parent_hotspots=hotspots(parent,ph),child_hotspots=hotspots(child,ch),peak_force_difference_percent=100*(peakc/peakp-1),
                residual_crown_difference_mm=cq['residual_crown_mm']-pq['residual_crown_mm'],
                parent_quality=pq,child_quality=cq,parent_criterion_bracket_mpa=pb,child_criterion_bracket_mpa=cb,
                brackets_overlap_or_touch=overlap,common_pressure_comparison=pressure,
                child_terminal_pressure_mpa=cp[-1]['pressure_mpa'],child_terminal_peeq=cp[-1]['max_peeq'],
                endpoint_exceeds_supported_peeq=cp[-1]['max_peeq']>0.7910574171740384))
    report = dict(records=records,script_sha256=sha256(__file__),
        limitations='Piecewise linear descriptive comparison at overlapping parent samples only; no pressure or tool-position extrapolation. Printed accepted-time match is precision-limited. Provisional 0.8 criterion, unsupported material tail, assumed gouge/fixture/load conventions remain. Two refined meshes do not establish asymptotic convergence or physical burst. No recovered physical acceptance tolerance; report differences and bracket resolution without claiming pass.')
    output.write_text(json.dumps(report,indent=2)+'\n')
    for r in records:
        print(json.dumps({k:r[k] for k in ['child','peak_force_difference_percent','residual_crown_difference_mm','parent_criterion_bracket_mpa','child_criterion_bracket_mpa','brackets_overlap_or_touch','child_terminal_pressure_mpa','child_terminal_peeq']}))

if __name__ == '__main__':
    import sys
    main(sys.argv[1])
