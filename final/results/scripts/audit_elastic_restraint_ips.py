"""Audit completed all-PIPE native integration-point stresses against Lamé."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pipe_sim.config import Case
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256, write_json
from audit_elastic_restraint_fields import lame_components
from audit_pressure_restraints import audit
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]


def audit_ips(root):
    pins = audit(root / 'common_adaptive')['candidate_node_dof_coordinates']
    pin_xyz = np.array([pin['coordinates_mm'] for pin in pins])
    records = []
    for name in ['bottom', 'tangential']:
        run = root / name
        status = json.loads((run / 'status.json').read_text())
        assert status['state'] == 'completed', 'Requires completed immutable native output'
        manifest = json.loads((run / 'manifest.json').read_text())
        assert manifest['all_pipe_stress'], 'Requires all-PIPE output variant'
        assert all(sha256(run / n) == h for n, h in manifest['files'].items())
        native = {n: sha256(run / n) for n in ['job.dat', 'job.sta', 'job.frd', 'solver.log']}
        assert matches_sta(converged_times(run / 'job.sta'), 1), 'Final frame not accepted in STA'
        model = Model.load(run / 'model.json')
        case = Case.load(run / 'case.json')
        assert not case.material.plastic and not case.dent.enabled and not case.gouge.enabled
        assert case.analysis.max_pressure == 1 and case.analysis.closed_ends
        blocks = {q: rows for q, group, time, rows in dat_blocks(run / 'job.dat')
                  if group == 'PIPE' and abs(time - 1) < 1e-7 and q in {'S', 'COORD'}}
        assert set(blocks) == {'S', 'COORD'}, 'Missing final stress or coordinates'
        tables = {q: {(int(r[0]), int(r[1])): r[2:] for r in rows}
                  for q, rows in blocks.items()}
        assert all(len(tables[q]) == len(blocks[q]) for q in tables), 'Duplicate IP rows'
        assert tables['S'].keys() == tables['COORD'].keys(), 'Unpaired IP data'
        expected = {(e, ip) for e in model.elsets['PIPE'] for ip in range(1, 9)}
        assert set(tables['S']) == expected, 'Missing C3D20R integration points'
        keys = sorted(expected)
        xyz = np.array([tables['COORD'][key] for key in keys])
        stress = np.array([tables['S'][key] for key in keys])
        assert xyz.shape[1] == 3 and stress.shape[1] == 6
        radius = np.linalg.norm(xyz[:, :2], axis=1)
        cx, cy = xyz[:, 0] / radius, xyz[:, 1] / radius
        # Native DAT order is SXX,SYY,SZZ,SXY,SXZ,SYZ, unlike FRD.
        xx, yy, zz, xy, xz, yz = stress.T
        radial = cx**2 * xx + cy**2 * yy + 2 * cx * cy * xy
        hoop = cy**2 * xx + cx**2 * yy - 2 * cx * cy * xy
        theory_r, theory_h, theory_z = lame_components(xyz, 1, case.pipe.od/2,
                                                       case.pipe.od/2-case.pipe.thickness)
        distance = np.min(np.linalg.norm(xyz[:, None, :] - pin_xyz[None, :, :], axis=2), axis=1)
        frame = pd.DataFrame(dict(element=[k[0] for k in keys], ip=[k[1] for k in keys],
                                  x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], pin_distance_mm=distance,
                                  radial_error_over_pressure=abs(radial-theory_r),
                                  hoop_relative_error=abs(hoop-theory_h)/theory_h,
                                  axial_relative_error=abs(zz-theory_z)/theory_z))
        zones = {'all': np.ones(len(frame), dtype=bool), 'pin_within_15mm': distance <= 15,
                 'end_within_15mm': abs(xyz[:, 2]) >= case.pipe.length/2-15,
                 'remote_central_half': abs(xyz[:, 2]) < case.pipe.length/4}
        metrics = ['radial_error_over_pressure', 'hoop_relative_error', 'axial_relative_error']
        summaries = {zone: dict(points=int(mask.sum()), errors={metric: dict(
            mean=float(frame.loc[mask, metric].mean()),
            p95=float(frame.loc[mask, metric].quantile(.95)),
            maximum=float(frame.loc[mask, metric].max())) for metric in metrics})
            for zone, mask in zones.items()}
        records.append(dict(variant=name, run=str(run.relative_to(FINAL)), points=len(keys),
                            pressure_mpa=1, time=1, zones=summaries,
                            worst_hoop_ip=frame.loc[frame.hoop_relative_error.idxmax()].to_dict(),
                            input_hashes=manifest['files'], native_sha256=native,
                            solver_sha256=status['solver_sha256']))
        assert native == {n: sha256(run / n) for n in native}, 'Native outputs changed'
    return dict(records=records, reporter_sha256=sha256(Path(__file__)), limitations=
                'Unweighted native IP statistics using printed COORD for Lamé projection at 1 MPa. '
                'Radial errors normalized by pressure because outer radial traction vanishes. '
                'C3D20R eight-point coverage required. Low-strain benchmark only; no finite-expansion '
                'end equilibrium, mesh convergence, fixture release, Abaqus or experimental validation. '
                'Candidate remains unadopted; no new acceptance gate.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_id')
    args = parser.parse_args()
    assert Path(args.run_id).name == args.run_id and args.run_id not in {'.', '..'}
    result = audit_ips(FINAL / 'sym/runs' / args.run_id)
    write_json(FINAL / 'results/output/pressure_restraint_candidates' / (args.run_id + '_ips.json'), result)
    print(json.dumps(result, indent=2))
