"""Bounded both-end curved-face check of native free-DOF pressure loads.

This does not capture constrained external loads or the pressure tangent.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from pipe_sim.model import HEX_LOCAL
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_nodal_pressure_loads import nodal_loads
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]


def run(name, elastic_modulus=210000., strict_field=False, max_increment=.25):
    if not np.isfinite(elastic_modulus) or elastic_modulus <= 0:
        raise ValueError("Elastic modulus must be positive and finite")
    if not np.isfinite(max_increment) or not 0 < max_increment <= 1:
        raise ValueError("Maximum increment must be finite and in (0, 1]")
    if Path(name).name != name:
        raise ValueError('Use a new directory name')
    root = FINAL / 'sym/runs/investigation' / name
    root.mkdir(exist_ok=False)
    exe = FINAL / 'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    build = json.loads((exe.parent / 'build.json').read_text())
    assert build['verified'] and sha256(exe) == build['solver_sha256']
    records = []
    for curvature in (0., .04, .08):
        points = (HEX_LOCAL + 1) * np.array([2., 3., 1.])
        top = [i+1 for i, p in enumerate(points) if p[2] == 2]
        bottom = [i+1 for i, p in enumerate(points) if p[2] == 0]
        # Quadratic reference surfaces, constant vertical wall separation.
        points[:, 2] += curvature * ((points[:, 0]-2)**2 + .5*(points[:, 1]-3)**2)
        for side, face, fixed in (
                ('plus', [5,6,7,8,13,14,15,16], bottom),
                ('minus', [1,4,3,2,12,11,10,9], top)):
            directory = root / f'{side}_curvature_{curvature}'
            directory.mkdir()
            lines = ['*HEADING', 'Curved pressure coupon: mm N MPa', '*NODE']
            lines += [f'{i},{p[0]:.16g},{p[1]:.16g},{p[2]:.16g}' for i,p in enumerate(points,1)]
            lines += ['*ELEMENT,TYPE=C3D20R,ELSET=CUBE',
                      '1,'+','.join(map(str,range(1,16)))+',',
                      ','.join(map(str,range(16,21))), '*NSET,NSET=ALL',
                      ','.join(map(str,range(1,17))), ','.join(map(str,range(17,21))),
                      '*MATERIAL,NAME=ELASTIC', '*ELASTIC', f'{elastic_modulus:.16g},0.3',
                      '*SOLID SECTION,ELSET=CUBE,MATERIAL=ELASTIC', '*BOUNDARY']
            lines += [f'{n},1,3,0' for n in fixed]
            # Retain middle-edge lateral pins; loaded face is free in all directions.
            lines += [f'{n},1,2,0' for n in range(1,21) if n not in fixed and n not in face]
            lines += ['*STEP,NLGEOM,INC=100']
            if strict_field:
                # Diagnostic convergence sensitivity, documented in node251.html.
                # Tighten normal/alternative residual and correction criteria.
                lines += ['*CONTROLS,PARAMETERS=FIELD',
                          '1e-8,1e-6,,,1e-8,1e-5,1e-6,1e-12']
            lines += ['*STATIC,SOLVER=PARDISO',
                      f'{max_increment:.16g},1,1e-8,{max_increment:.16g}', '*DLOAD',
                      f'1,P{2 if side == "plus" else 1},2',
                      '*NODE PRINT,NSET=ALL', 'U,RF', '*END STEP']
            deck = directory / 'job.inp'
            deck.write_text('\n'.join(lines)+'\n')
            env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       MKL_THREADING_LAYER='GNU', CCX_NPROC_RESULTS='1',
                       CCX_NPROC_STIFFNESS='1', CCX_NPROC_EQUATION_SOLVER='1')
            with (directory/'solver.log').open('w') as log:
                result = subprocess.run([str(exe), '-i', 'job'], cwd=directory,
                                        env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
            log = (directory/'solver.log').read_text()
            assert result.returncode == 0 and 'Job finished' in log and '*ERROR' not in log
            frames = {}
            sta = converged_times(directory/'job.sta')
            for kind, group, time, rows in dat_blocks(directory/'job.dat'):
                assert matches_sta(sta, time)
                frames.setdefault(time, {})[kind] = rows
            assert abs(max(frames)-1) < 1e-8
            history = []
            for time, blocks in sorted(frames.items()):
                u = {int(r[0]): np.array(r[1:]) for r in blocks['U']}
                rf = {int(r[0]): np.array(r[1:]) for r in blocks['RF']}
                assert set(u) == set(rf) == set(range(1,21))
                applied = {n: np.zeros(3) for n in u}
                for n, v in zip(face, nodal_loads([points[n-1]+u[n] for n in face], -2*time)):
                    applied[n] += v
                free = [(n,d) for n in u for d in range(3)
                        if n not in fixed and (n in face or d == 2)]
                error = max(abs(rf[n][d]-applied[n][d]) for n,d in free)
                scale = max(abs(applied[n][d]) for n,d in free)
                # Native DAT rounding and equilibrium tolerance, not a physical tolerance.
                assert error < 1e-4 and error/max(scale,1e-12) < 2e-5
                assert max(np.linalg.norm(u[n]) for n in fixed) < 1e-12
                history.append(dict(time=time, pressure_mpa=2*time,
                                    free_dof_max_error_n=error, relative_error=error/scale,
                                    max_loaded_face_lateral_load_n=max(abs(applied[n][d]) for n in face for d in (0,1)),
                                    applied_resultant_n=sum(applied.values(),np.zeros(3)).tolist(),
                                    max_displacement_mm=max(float(np.linalg.norm(v)) for v in u.values())))
            records.append(dict(run=str(directory.relative_to(FINAL)), side=side,
                                curvature_per_mm=curvature, history=history,
                                hashes={f: sha256(directory/f) for f in ('job.inp','job.dat','job.sta','solver.log')}))
    report = dict(elastic_modulus_mpa=elastic_modulus, strict_field=strict_field, max_increment=max_increment, records=records, solver_sha256=sha256(exe), script_sha256=sha256(__file__),
                  sources=['sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f:1454-1487,1611-1615'],
                  limitations='One elastic C3D20R with curved reference faces and measured native displacement; middle-edge lateral pins and opposite face fixed. Verifies free components RF against 2x2 current-face loads in both end directions. No constrained-load capture, imposed-motion verification, cap area ratio, tangent, work, pipe convergence, Abaqus equivalence or physical validation established.')
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    output = FINAL/'results/output/pressure_restraint_candidates'/f'{name}.json'
    assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(cases=len(records), max_free_error_n=max(h['free_dof_max_error_n'] for r in records for h in r['history']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--elastic-modulus', type=float, default=210000., help='Manufactured coupon stiffness only; never a pipe material fit')
    parser.add_argument('--strict-field', action='store_true', help='Coupon-only convergence sensitivity; acceptance tolerances unchanged')
    parser.add_argument('--max-increment', type=float, default=.25, help='Coupon-only increment sensitivity')
    args = parser.parse_args()
    run(args.name, args.elastic_modulus, args.strict_field, args.max_increment)
