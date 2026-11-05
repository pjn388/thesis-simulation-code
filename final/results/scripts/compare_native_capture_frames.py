"""Compare verified single-element capture to accepted native DAT frames."""
import json
from pathlib import Path
import numpy as np
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_nodal_pressure_loads import nodal_loads
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]
INPUT = FINAL/'results/output/native_capture_verification_20261011T000006Z.json'
OUTPUT = FINAL/'results/output/native_capture_frames_20261011T000006Z.json'


def main():
    assert not OUTPUT.exists()
    verification = json.loads(INPUT.read_text())
    records = []
    for case in verification['records']:
        run = FINAL/case['run']
        assert all(sha256(run/n) == digest for n,digest in case['hashes'].items())
        captures = json.loads((run/'capture.json').read_text())['records']
        accepted = converged_times(run/'job.sta')
        frames = {}
        for kind, group, time, rows in dat_blocks(run/'job.dat'):
            assert matches_sta(accepted,time)
            frames.setdefault(time,{})[kind] = rows
        plus = run.name.startswith('plus')
        face = [5,6,7,8,13,14,15,16] if plus else [1,4,3,2,12,11,10,9]
        fixed = [1,2,3,4,9,10,11,12] if plus else [5,6,7,8,13,14,15,16]
        free = [] if case['fixed'] else [(n,d) for n in range(1,21) for d in range(3)
                if n not in fixed and (n in face or d==2)]
        history = []
        for time, blocks in sorted(frames.items()):
            candidates = [r for r in captures if abs(r['time']-time)<1e-7]
            assert candidates
            last = candidates[-1]
            native = np.array(last['ff']).reshape(20,3)
            rf = {int(row[0]): np.array(row[1:]) for row in blocks['RF']}
            u = {int(row[0]): np.array(row[1:]) for row in blocks['U']}
            coords = np.array(last['coordinates'])
            current = coords + np.array([u[n] for n in range(1,21)])
            final_load = np.zeros((20,3))
            final_load[np.array(face)-1] = nodal_loads(current[np.array(face)-1],-2*time)
            max_rf_error = max((abs(rf[n][d]-native[n-1,d]) for n,d in free),default=None)
            if free:
                assert max_rf_error < 1e-4
                assert max_rf_error / max(abs(native[n-1,d]) for n,d in free) < 2e-5
            history.append(dict(time=time,pressure_mpa=2*time,
                                free_rf_vs_native_assembly_error_n=max_rf_error,
                                native_assembly_vs_output_geometry_error_n=float(np.max(abs(native-final_load))),
                                assembly_vs_output_u_error_mm=float(np.max(abs(np.array(last['displacements_at_assembly'])-np.array([u[n] for n in range(1,21)])))),
                                assembly_calls_at_time=len(candidates)))
        records.append(dict(run=case['run'],fixed=case['fixed'],history=history))
    report = dict(records=records,input_sha256=sha256(INPUT),script_sha256=sha256(__file__),
                  limitations='Accepted-time grouping of one-element e_c3d calls only; e_c3d_rhs was not exercised. Last assembly is compared with rounded final corrected DAT output. No pipe-wide assembly-cycle identity, MPC transfer, support reaction subtraction, cap coupling/tangent/work, Abaqus equivalence or physical validation.')
    OUTPUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(records),max_free_rf_error_n=max(h['free_rf_vs_native_assembly_error_n'] or 0 for r in records for h in r['history']),max_corrected_geometry_error_n=max(h['native_assembly_vs_output_geometry_error_n'] for r in records for h in r['history']))))


if __name__ == '__main__':
    main()
