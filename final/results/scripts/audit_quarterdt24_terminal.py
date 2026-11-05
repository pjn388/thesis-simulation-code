"""Verify terminal endpoint-window capture against immutable completed parent."""
import json
from pathlib import Path
from pipe_sim.runner import sha256
from pipe_sim.results import dat_blocks
from pipe_sim.model import Model
from audit_pressure_failures import converged_times, matches_sta
F = Path(__file__).resolve().parents[2]
TAG = 'tangential_capture_quarterdt_p24_20261011T074431Z'

def audit():
    terminal_path = F/'results/output'/f'{TAG}_terminal.json'
    partial_path = F/'results/output'/f'{TAG}_partial.json'
    terminal = json.loads(terminal_path.read_text())
    partial = json.loads(partial_path.read_text())
    parent_path = F/'results/output/plastic_tangential_work_quarterdt_p24_20261011T072144Z_terminal.json'
    parent = json.loads(parent_path.read_text())
    assert terminal['returncode'] == terminal['audit_returncode'] == 0
    assert not terminal['timed_out'] and not terminal['memory_stop']
    assert parent['completed'] and partial['passed']
    assert sha256(F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso') == terminal['solver_sha256'] == parent['solver_sha256']
    for report in (terminal, parent):
        assert all(sha256(F/report['run']/n) == h for n,h in
                   {**report['input_hashes'], **report['output_hashes']}.items())
    assert terminal['input_hashes'] == parent['input_hashes']
    run = F/terminal['run']; model = Model.load(run/'model.json')
    accepted = converged_times(run/'job.sta'); frames = {}
    for kind, group, t, rows in dat_blocks(run/'job.dat'):
        if not matches_sta(accepted,t): continue
        if kind == 'U' and group == 'PIPE_NODES':
            u = {int(r[0]):r[1:] for r in rows}
            frames.setdefault(t,{})['crown_mm'] = -u[model.nsets['CROWN'][0]][1]
            frames[t]['max_constrained_u_mm'] = max(abs(u[n][d]) for n,d in parent['pins'])
        if kind == 'PEEQ' and group == 'PIPE':
            frames.setdefault(t,{})['peeq'] = max(r[-1] for r in rows)
    baseline = {h['time']:h for h in parent['history']}
    comparisons = []
    for h in partial['history']:
        t=h['time']; actual=frames[t]; expected=baseline[t]
        assert h['passed'] and t >= partial['window_minimum_time']
        assert actual['max_constrained_u_mm'] == 0
        assert actual['crown_mm'] == h['crown_inward_mm'] and actual['peeq'] == h['maximum_peeq']
        comparisons.append(dict(time=t, pressure_mpa=h['pressure_mpa'],
            crown_difference_mm=actual['crown_mm']-expected['crown_mm'],
            peeq_difference=actual['peeq']-expected['peeq'], **actual))
    assert len(comparisons)==3 and comparisons[-1]['time']==1
    assert max(abs(c['crown_difference_mm']) for c in comparisons)==0
    assert max(abs(c['peeq_difference']) for c in comparisons)==0
    assert all(sha256(run/n)==h for n,h in partial['hashes'].items())
    return dict(terminal_sha256=sha256(terminal_path), parent_terminal_sha256=sha256(parent_path),
        partial_sha256=sha256(partial_path), run=terminal['run'], parent=parent['run'],
        elapsed_seconds=terminal['elapsed_seconds'], comparisons=comparisons,
        maximum_rounding_excess_n=max(h['max_rounding_excess_n'] for h in partial['history']),
        maximum_local_integration_error_n=max(h['local_load_error_n'] for h in partial['history']),
        verdict='Passed three-frame endpoint-window force audit; printed crown/PEEQ match parent exactly; constrained U zero.',
        successor_blocker='Work gaps remain nonmonotonic after assumed-path temporal quadrature (12.475726/31.585245 Nmm). Earlier force frames, external U/time rounding, actual nonlinear path and internal integration are unbounded. No justified pressure extension or identical failed halfdt retry. Coupled cap tangent/work, contact/MPC and Abaqus equivalence remain missing.',
        source_findings={'internal_energy':'resultsmech.f:1064-1076 uses endpoint-average stress times strain increment; an independent internal temporal integration uncertainty remains.',
            'total_output':'printout.f:553-559 prints total with e13.6; at ~2.281e8 Nmm its half-unit is50 Nmm, worse than summed-element26.415 Nmm bound. TOTALS=YES alone does not unlock precision.',
            'element_output':'printoutelem.f:521 prints each energy with e13.6; immutable ELSE interval audit retained.'},
        limitations='Serial intact no-contact/MPC endpoint window only. No general force transfer, work convergence, production revision, Abaqus equivalence, experimental validation or burst claim.')

if __name__ == '__main__':
    report=audit(); report['script_sha256']=sha256(__file__)
    out=F/'results/output/quarterdt24_terminal_review_20261011T082450Z.json'
    assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
