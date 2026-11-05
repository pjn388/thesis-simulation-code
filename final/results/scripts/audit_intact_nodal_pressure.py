"""Read-only finite-expansion nodal-load reconstruction, with incomplete RF."""
import json
from pathlib import Path
import numpy as np
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks
from audit_nodal_pressure_loads import nodal_loads
from audit_end_surface_traction import face_nodes, FINAL
from audit_intact_end_fields import complete_fields
from audit_pressure_failures import digest, converged_times, matches_sta
from audit_pressure_resultant import reaction_coverage


def main():
    parent_path = FINAL/'results/output/pressure_restraint_candidates/intact_pressure_resultant.json'
    parent = json.loads(parent_path.read_text())
    run = FINAL/parent['run']
    hashes = {n:digest(run/n) for n in ['job.inp','job.dat','job.frd','job.sta','model.json','case.json']}
    assert all(digest(run/n) == h for n,h in parent['native_sha256'].items())
    manifest = json.loads((run/'manifest.json').read_text())
    assert all(digest(run/n) == h for n,h in manifest['files'].items())
    assert json.loads((run/'status.json').read_text())['state'] == 'failed'
    time = parent['accepted_time']
    assert matches_sta(converged_times(run/'job.sta'),time)
    model = Model.load(run/'model.json')
    displacement = None
    with (run/'job.frd').open() as stream:
        for t,kind,labels,rows in complete_fields(stream):
            if kind == 'DISP' and abs(t-time)<1e-6:
                displacement = {n:np.asarray(v) for n,v in rows.items()}
    assert displacement is not None and set(model.nodes).issubset(displacement)
    blocks = {k:rows for k,g,t,rows in dat_blocks(run/'job.dat') if g=='HISTORY' and abs(t-time)<1e-6}
    rf = {int(r[0]):np.asarray(r[1:]) for r in blocks['RF']}
    assert set(rf) == set(model.nsets['HISTORY'])
    coverage = reaction_coverage(model)
    assert not coverage['recorded_constrained_dofs']
    deck = (run/'job.inp').read_text().upper()
    assert deck.count('*DLOAD') == 1 and deck.count('*STEP') == 1
    assert not any(k in deck for k in ['*AMPLITUDE','*CLOAD','*MPC','*EQUATION','*CONTACT'])
    prescribed = {}
    for line in deck.split('*DLOAD\n')[1].split('*')[0].strip().splitlines():
        e,f,p = map(str.strip,line.split(','))
        key = (int(e),int(f[1:]))
        assert key not in prescribed
        prescribed[key] = float(p)*time
    case = json.loads((run/'case.json').read_text())
    ro = case['pipe']['od']/2
    ri = ro-case['pipe']['thickness']
    ratio = ri**2/(ro**2-ri**2)
    expected = {(e,f):parent['pressure_mpa']*(1 if s=='INNER' else -ratio)
                for s in ['INNER','END_MINUS','END_PLUS'] for e,f in model.surfaces[s]}
    assert prescribed.keys() == expected.keys()
    assert all(np.isclose(p,expected[k],rtol=1e-10) for k,p in prescribed.items())
    # FRD DISP has five digits after the decimal in scientific notation.
    # Two coherent rounding perturbations are sensitivity probes, not bounds.
    rounding = {n:np.where(v==0,0,0.5*10.**(np.floor(np.log10(np.maximum(abs(v),1e-300)))-5))
                for n,v in displacement.items()}
    records = []
    for shift in [0,-1,1]:
        current = {n:np.asarray(p)+displacement[n]+shift*rounding[n] for n,p in model.nodes.items()}
        loads = {n:np.zeros(3) for n in model.nodes}
        for (e,f),p in prescribed.items():
            nodes = face_nodes(model,e,f)
            for n,v in zip(nodes,nodal_loads([current[n] for n in nodes],-p)):
                loads[n] += v
        errors = np.array([rf[n]-loads[n] for n in rf])
        records.append(dict(coherent_rounding_shift=shift,
            net_applied_n=sum(loads.values(),np.zeros(3)).tolist(),
            recorded_free_components=3*len(rf),
            recorded_free_max_error_n=float(np.max(abs(errors))),
            recorded_free_rms_error_n=float(np.sqrt(np.mean(errors**2))),
            recorded_nodes=[dict(node=n,rf_n=rf[n].tolist(),applied_n=loads[n].tolist()) for n in sorted(rf)]))
    assert np.allclose(records[0]['net_applied_n'],parent['prescribed_net_force_n'],rtol=1e-9,atol=1e-5)
    assert all(digest(run/n)==h for n,h in hashes.items())
    report = dict(run=parent['run'],accepted_time=time,pressure_mpa=parent['pressure_mpa'],
        records=records,reaction_output_coverage=coverage,native_sha256=hashes,
        input_sha256=manifest['files'],solver_sha256=parent['solver_sha256'],
        parent_report_sha256=digest(parent_path),script_sha256=digest(Path(__file__)),
        sources=['sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f:1360-1486,1597-1615',
                 'sym/vendor/CalculiX/ccx_2.23/doc/ccx/node212.html:62-64,99-138'],
        limitations='Accepted finite-expansion FRD reconstruction, not exported assembly. Only five HISTORY nodes / '
        '15 free components have RF; all 85 constrained components are absent. Coherent plus/minus half-last-digit '
        'FRD probes are not rigorous error bounds. Unloaded free-node agreement cannot validate loaded-node '
        'assembly or recover support reactions. No complete equilibrium, failure cause, mesh convergence, '
        'physical cap behavior, Abaqus equivalence or burst established. Production inputs unchanged.')
    output = FINAL/'results/output/pressure_restraint_candidates/intact_nodal_pressure_loads.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['accepted_time','pressure_mpa','records','limitations']},indent=2))


if __name__ == '__main__':
    main()
