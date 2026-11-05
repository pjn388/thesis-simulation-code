"""Independent small-strain elastic pressure/work check; no campaign edits."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_nodal_pressure_loads import nodal_loads
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]


def run(name):
    if Path(name).name != name:
        raise ValueError('Use a new single directory name')
    root = FINAL / 'sym/runs/investigation' / name
    root.mkdir(exist_ok=False)
    parent = FINAL / 'sym/runs/investigation/pressure_reaction_coupon_v3_20261008T203005Z/loaded_free/job.inp'
    original = parent.read_text()
    exe = FINAL / 'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    build = json.loads((exe.parent / 'build.json').read_text())
    assert build['verified'] and sha256(exe) == build['solver_sha256']
    # Fixed lateral DOFs and bottom: homogeneous uniaxial strain, fixed face area.
    modulus = 210000 * (1 - .3) / ((1 + .3) * (1 - 2 * .3))
    analytical_work = .5 * (2 * 24) * (2 * 2 / modulus)
    face = [5, 6, 7, 8, 13, 14, 15, 16]
    records = []
    for increment in [.25, .125]:
        directory = root / f'dt_{increment}'
        directory.mkdir()
        deck = original.replace('1,1,1e-8,1', f'{increment},1,1e-8,{increment}')
        deck = deck.replace('*END STEP', '*EL PRINT,ELSET=CUBE\nELSE\n*END STEP')
        (directory / 'job.inp').write_text(deck)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   MKL_THREADING_LAYER='GNU', CCX_NPROC_RESULTS='1',
                   CCX_NPROC_STIFFNESS='1', CCX_NPROC_EQUATION_SOLVER='1')
        with (directory / 'solver.log').open('w') as log:
            result = subprocess.run([str(exe), '-i', 'job'], cwd=directory,
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        log = (directory / 'solver.log').read_text()
        assert result.returncode == 0 and 'Job finished' in log and '*ERROR' not in log
        points = {}
        for line in original.split('*NODE\n')[1].split('*')[0].splitlines():
            values = line.split(',')
            points[int(values[0])] = np.array(list(map(float, values[1:])))
        frames = {}
        for kind, group, time, rows in dat_blocks(directory / 'job.dat'):
            assert matches_sta(converged_times(directory / 'job.sta'), time)
            frames.setdefault(time, {})[kind] = rows
        # ELSE is deliberately the only unrecognised DAT quantity in this coupon.
        assert 'internal energy (element, energy)' in (directory / 'job.dat').read_text()
        previous_u = np.zeros((8, 3))
        previous_load = np.zeros((8, 3))
        work = 0.
        history = []
        for time, blocks in sorted(frames.items()):
            u = {int(row[0]): np.array(row[1:]) for row in blocks['U']}
            rf = {int(row[0]): np.array(row[1:]) for row in blocks['RF']}
            displacement = np.array([u[n] for n in face])
            current = [points[n] + u[n] for n in face]
            loads = nodal_loads(current, -2 * time)
            work += float(np.sum(.5 * (loads + previous_load) * (displacement - previous_u)))
            energy_rows = blocks[None]
            assert len(energy_rows) == 1 and int(energy_rows[0][0]) == 1
            internal = energy_rows[0][1]
            balance = float(sum(rf[n][2] for n in points))
            assert abs(balance) < 1e-4
            assert abs(sum(loads[:, 2]) + 48 * time) < 1e-8
            assert abs(work / internal - 1) < 2e-4
            history.append(dict(time=time, pressure_mpa=2*time, external_work_n_mm=work,
                                internal_energy_n_mm=internal, rf_balance_n=balance))
            previous_u, previous_load = displacement, loads
        assert abs(max(frames) - 1) < 1e-8
        assert abs(work / analytical_work - 1) < 2e-4
        records.append(dict(run=str(directory.relative_to(FINAL)), increment=increment,
                            history=history, analytical_work_n_mm=analytical_work,
                            final_relative_work_energy_error=work/internal-1,
                            final_relative_analytical_error=work/analytical_work-1,
                            hashes={f: sha256(directory/f) for f in ['job.inp', 'job.dat', 'job.sta', 'solver.log']}))
    report = dict(records=records, parent=str(parent.relative_to(FINAL)),
                  parent_sha256=sha256(parent), solver_sha256=sha256(exe),
                  script_sha256=sha256(__file__),
                  source='sym/vendor/CalculiX/ccx_2.23/doc/ccx/node280.html:179-186',
                  limitations='One C3D20R element, elastic uniaxial strain, tiny deformation, planar constant-area pressure. Checks energy output, pressure sign, face quadrature and increment sensitivity. No pipe mesh convergence, end-cap work, plastic work, alternate equilibrium branch, Abaqus equivalence or physical burst validation.')
    (root / 'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    output = FINAL / 'results/output/pressure_restraint_candidates' / f'{name}.json'
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    run(parser.parse_args().name)
