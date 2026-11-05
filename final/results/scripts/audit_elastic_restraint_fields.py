"""Read-only full-field elastic audit; FRD nodal stress is extrapolated, not IP stress."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pipe_sim.config import Case
from pipe_sim.model import Model
from pipe_sim.runner import sha256, write_json
from audit_intact_end_fields import complete_fields
from audit_pressure_restraints import audit

FINAL = Path(__file__).resolve().parents[2]


def lame_components(xyz, pressure, outer, inner):
    radius = np.linalg.norm(xyz[:, :2], axis=1)
    a = pressure * inner**2 / (outer**2 - inner**2)
    b = a * outer**2
    return a - b / radius**2, a + b / radius**2, np.full(len(radius), a)


def audit_fields(root):
    pins = audit(root / 'common_adaptive')['candidate_node_dof_coordinates']
    pin_xyz = np.array([p['coordinates_mm'] for p in pins])
    records = []
    for name in ['bottom', 'tangential']:
        run = root / name
        status = json.loads((run / 'status.json').read_text())
        assert status['state'] == 'completed', 'Requires completed native outputs'
        manifest = json.loads((run / 'manifest.json').read_text())
        assert all(sha256(run / n) == h for n, h in manifest['files'].items())
        native = {n: sha256(run / n) for n in ['job.frd', 'job.sta', 'job.dat']}
        model = Model.load(run / 'model.json')
        case = Case.load(run / 'case.json')
        assert not case.material.plastic and case.analysis.max_pressure == 1
        fields = {}
        with (run / 'job.frd').open() as stream:
            for time, kind, labels, rows in complete_fields(stream):
                if abs(time - 1) < 1e-7:
                    fields[kind] = (labels, rows)
        assert set(fields) == {'STRESS', 'DISP'}, 'Missing final full fields'
        assert fields['STRESS'][0] == ['SXX', 'SYY', 'SZZ', 'SXY', 'SYZ', 'SZX']
        ids = sorted({n for e in model.elsets['PIPE'] for n in model.elements[e][1]})
        assert set(ids).issubset(fields['STRESS'][1])
        xyz = np.array([model.nodes[n] for n in ids])
        stress = np.array([fields['STRESS'][1][n] for n in ids])
        radius = np.linalg.norm(xyz[:, :2], axis=1)
        cx, cy = xyz[:, 0] / radius, xyz[:, 1] / radius
        xx, yy, zz, xy, yz, zx = stress.T
        radial = cx**2 * xx + cy**2 * yy + 2 * cx * cy * xy
        hoop = cy**2 * xx + cx**2 * yy - 2 * cx * cy * xy
        expected_r, expected_h, expected_z = lame_components(xyz, 1, case.pipe.od/2, case.pipe.od/2-case.pipe.thickness)
        # Radial theory vanishes at the outside surface; use pressure as denominator.
        distance = np.min(np.linalg.norm(xyz[:, None, :] - pin_xyz[None, :, :], axis=2), axis=1)
        df = pd.DataFrame(dict(node=ids, x=xyz[:,0], y=xyz[:,1], z=xyz[:,2],
                               pin_distance_mm=distance,
                               radial_error_over_pressure=abs(radial-expected_r),
                               hoop_relative_error=abs(hoop-expected_h)/expected_h,
                               axial_relative_error=abs(zz-expected_z)/expected_z))
        metrics = ['radial_error_over_pressure', 'hoop_relative_error', 'axial_relative_error']
        zones = {'all': np.ones(len(df), dtype=bool),
                 'pin_within_15mm': distance <= 15,
                 'end_within_15mm': abs(xyz[:,2]) >= case.pipe.length/2-15,
                 'remote_central_half': abs(xyz[:,2]) < case.pipe.length/4}
        summaries = {}
        for zone, mask in zones.items():
            sample = df.loc[mask]
            summaries[zone] = dict(nodes=len(sample), errors={k: dict(mean=float(sample[k].mean()), p95=float(sample[k].quantile(.95)), maximum=float(sample[k].max())) for k in metrics})
        records.append(dict(variant=name, run=str(run.relative_to(FINAL)), nodes=len(ids), time=1, pressure_mpa=1,
                            zones=summaries, worst_hoop_node=df.loc[df.hoop_relative_error.idxmax()].to_dict(),
                            input_hashes=manifest['files'], native_sha256=native, solver_sha256=status['solver_sha256']))
        assert native == {n: sha256(run/n) for n in native}, 'Native evidence changed during read'
    return dict(records=records, reporter_sha256=sha256(Path(__file__)),
                limitations='FRD stresses are extrapolated/averaged nodal values, not full-field IP stresses. Reference-coordinate Lamé projection; unweighted node statistics. Pin neighborhoods are Euclidean 15mm balls about candidate pins for both matched runs. No new acceptance gate, nonlinear finite-expansion end equilibrium, mesh convergence, stage-specific release, Abaqus verification or experimental validation established. Candidate remains unadopted.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in {'.', '..'}
    result = audit_fields(FINAL / 'sym/runs' / args.run_id)
    out = FINAL / 'results/output/pressure_restraint_candidates' / (args.run_id + '_fields.json')
    write_json(out, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
