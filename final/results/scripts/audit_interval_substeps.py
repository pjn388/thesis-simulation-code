"""Controlled assumed-C-path diagnostic; native results remain immutable."""
import json
import sys
import numpy as np
from compare_native_plastic_time import F, STANDARD, QUARTER, load, field, compare
from audit_plastic_return_map import update, tensors, sha256
from audit_internal_pipe_fields import spatial
from pipe_sim.model import Model


def audit(gate, lo, hi):
    evidence, run, records, raw = load(gate)
    model = Model.load(run/'model.json')
    inside = False
    for line in (run/'mesh.inp').read_text().splitlines():
        if line.startswith('**'): continue
        if line.startswith('*'): inside = line.upper().strip() == '*NODE'
        elif inside and line.strip():
            row = line.split(','); model.nodes[int(row[0])] = list(map(float,row[1:4]))
    ids, nodes, grad, weights, checks = spatial(model)
    selection = np.array(ids)-1
    def state(rec):
        s = field(raw,rec,'state',rec['states'])[selection]
        c = np.eye(3)+2*tensors(field(raw,rec,'strain',6)[selection])
        return c, np.eye(3)-2*tensors(s[...,1:7]), s[...,0]
    start = next(r for r in records if abs(24*r['time']-lo)<1e-9)
    end = next(r for r in records if abs(24*r['time']-hi)<1e-9)
    c0, m0, ep0 = state(start); c1, _, _ = state(end)
    material = json.loads((run/'case.json').read_text())['material']
    mu = material['young']/(2*(1+material['poisson']))
    bulk = material['young']/(3*(1-2*material['poisson']))
    lines = (run/'job.inp').read_text().splitlines(); curve = []
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'): break
        curve.append(list(map(float,line.split(','))))
    curve = np.array(curve)
    maxima = dict(stress_mpa=0.,metric=0.,peeq=0.)
    previous_m, previous_ep = m0.copy(), ep0.copy(); count = 0
    for rec in records:
        if not lo+1e-9 < 24*rec['time'] <= hi+1e-9: continue
        c, actual_m, actual_ep = state(rec)
        stress, m, ep, _ = update(c,previous_m,previous_ep,curve,mu,bulk)
        u = field(raw,rec,'u',4)
        deformation = np.eye(3)+np.einsum('eni,epnj->epij',u[nodes-1],grad)
        inverse = np.linalg.inv(deformation)
        actual_s = np.linalg.det(deformation)[...,None,None]*(inverse@tensors(field(raw,rec,'stress',6)[selection])@np.swapaxes(inverse,-1,-2))
        errors = dict(stress_mpa=float(abs(stress-actual_s).max()),metric=float(abs(m-actual_m).max()),peeq=float(abs(ep-actual_ep).max()))
        for k,v in errors.items(): maxima[k] = max(maxima[k],v)
        previous_m, previous_ep = actual_m.copy(), actual_ep.copy(); count += 1
    assert all(maxima[k]<v for k,v in evidence['fixed_limits'].items()), maxima
    def reference(m):
        return float((1.5*mu*(np.cbrt(np.linalg.det(m))-np.cbrt(np.linalg.det(m0)))*weights).sum())
    results = []
    for subdivisions in (1,2,4):
        m, ep = m0.copy(), ep0.copy()
        for i in range(1,subdivisions+1):
            c = c0+(c1-c0)*(i/subdivisions)
            assert np.linalg.eigvalsh(c).min()>0
            stress, m, ep, _ = update(c,m,ep,curve,mu,bulk)
        results.append(dict(subdivisions=subdivisions,reference_increment_n_mm=reference(m),integrated_peeq_increment_mm3=float(((ep-ep0)*weights).sum()),maximum_peeq_increment=float((ep-ep0).max())))
    # Same starting state and accepted grid: isolate actual versus chord C only.
    accepted = [r for r in records if lo+1e-9 < 24*r['time'] <= hi+1e-9]
    paths = []
    for path in ('actual', 'chord'):
        m, ep = m0.copy(), ep0.copy()
        for rec in accepted:
            c = state(rec)[0] if path == 'actual' else c0+(c1-c0)*((24*rec['time']-lo)/(hi-lo))
            assert np.linalg.eigvalsh(c).min()>0
            stress, m, ep, _ = update(c,m,ep,curve,mu,bulk)
        paths.append(dict(path=path,updates=len(accepted),reference_increment_n_mm=reference(m),
            integrated_peeq_increment_mm3=float(((ep-ep0)*weights).sum()),
            final_metric_error_from_native=float(abs(m-previous_m).max()),
            final_peeq_error_from_native=float(abs(ep-previous_ep).max())))
    assert paths[0]['final_metric_error_from_native'] < evidence['fixed_limits']['metric']
    assert paths[0]['final_peeq_error_from_native'] < evidence['fixed_limits']['peeq']
    assert all(sha256(run/n)==h for n,h in evidence['input_output_hashes'].items())
    return dict(run=evidence['run'],gate_sha256=sha256(gate),native_frames=count,native_replay_maximum_errors=maxima,fixed_limits=evidence['fixed_limits'],native_reference_increment_n_mm=reference(previous_m),resolutions=results,accepted_grid_paths=paths,spatial_checks=checks)


if __name__ == '__main__':
    out = F/'results/output'/sys.argv[1]; assert not out.exists()
    source = F/'results/output/shared_plastic_intervals_20261011T160008Z.json'
    retained = json.loads(source.read_text()); assert retained['identity_checks_passed']
    comparison = compare(); assert comparison['provenance_checks_passed']
    interval = retained['top_intervals'][0]; lo,hi = interval['lower_mpa'],interval['upper_mpa']
    lanes = [audit(g,lo,hi) for g in (STANDARD,QUARTER)]
    result = dict(identity_checks_passed=True,physical_acceptance=False,lower_mpa=lo,upper_mpa=hi,lanes=lanes,interval_report_sha256=sha256(source),script_sha256=sha256(__file__),dependencies={n:sha256(F/'results/scripts'/n) for n in ('compare_native_plastic_time.py','audit_plastic_return_map.py','audit_internal_pipe_fields.py')},source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'),limitations='C interpolated linearly between each lane retained endpoints as an explicit diagnostic assumption. Native replay uses actual accepted endpoints and unchanged fixed limits. Substeps do not recover actual midpoint trajectory, native temporal convergence, physical storage/power, cap/contact/MPC/Abaqus equivalence or experimental validation. No production change.')
    out.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
