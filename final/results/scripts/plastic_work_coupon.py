"""Bounded independent finite-strain plastic load/unload work diagnostic."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from pipe_sim.config import Case
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]


def run(name):
    if Path(name).name != name:
        raise ValueError('Use a new directory name')
    root = FINAL / 'sym/runs/investigation' / name
    root.mkdir(exist_ok=False)
    parent = FINAL / 'sym/runs/investigation/material_coupon/job.inp'
    case_path = FINAL / 'sym/cases/investigation/a37_curve_only.json'
    case = Case.load(case_path)
    original = parent.read_text()
    exe = FINAL / 'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    build = json.loads((exe.parent / 'build.json').read_text())
    assert build['verified'] and sha256(exe) == build['solver_sha256']
    # Estimate stress-free residual travel using the independently checked law.
    # Terminal force is measured below: this estimate is not fitted after solving.
    plastic = np.array(case.material.plastic)
    stress = plastic[0, 0]
    for _ in range(50):
        stress = float(np.interp(.02 - stress / case.material.young,
                                 plastic[:, 1], plastic[:, 0]))
    residual = float(np.expm1(.02 - stress / case.material.young))
    records = []
    for increment in [.025, .0125]:
        directory = root / f'dt_{increment}'
        directory.mkdir()
        deck = original.split('*STEP')[0]
        for travel in [float(np.expm1(.02)), residual]:
            deck += '\n'.join(['*STEP, NLGEOM, INC=1000', '*STATIC',
                f'{increment},1.,1.e-8,{increment}', '*BOUNDARY', f'RIGHT,1,1,{travel:.12g}',
                '*NODE PRINT, NSET=ALL', 'U, RF', '*EL PRINT, ELSET=CUBE',
                'S, PEEQ, ELSE', '*END STEP', ''])
        (directory / 'job.inp').write_text(deck)
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   MKL_THREADING_LAYER='GNU', CCX_NPROC_RESULTS='1',
                   CCX_NPROC_STIFFNESS='1', CCX_NPROC_EQUATION_SOLVER='1')
        with (directory / 'solver.log').open('w') as log:
            result = subprocess.run([str(exe), '-i', 'job'], cwd=directory, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=60)
        log = (directory / 'solver.log').read_text()
        assert result.returncode == 0 and 'Job finished' in log and '*ERROR' not in log
        frames = {}
        accepted = converged_times(directory / 'job.sta')
        for kind, group, time, rows in dat_blocks(directory / 'job.dat'):
            assert matches_sta(accepted, time)
            frames.setdefault(time, {})[kind] = rows
        assert 'internal energy (element, energy)' in (directory / 'job.dat').read_text()
        previous_u = previous_force = work = 0.
        history = []
        for time, fields in sorted(frames.items()):
            u = {int(r[0]): r[1:] for r in fields['U']}
            rf = {int(r[0]): r[1:] for r in fields['RF']}
            displacement = u[2][0]
            force = sum(rf[n][0] for n in (2, 3, 6, 7))
            work += .5 * (force + previous_force) * (displacement - previous_u)
            energy_rows = fields[None]
            assert len(energy_rows) == 1 and int(energy_rows[0][0]) == 1
            internal = energy_rows[0][1]
            peeq = float(np.mean([r[2] for r in fields['PEEQ']]))
            balance = sum(rf[n][0] for n in rf)
            assert abs(balance) < 1e-3
            assert peeq < plastic[-1, 1]
            history.append(dict(time=time, displacement_mm=displacement, force_n=force,
                peeq=peeq, external_work_n_mm=work, internal_energy_n_mm=internal,
                relative_work_energy_error=work/internal-1 if internal else None,
                reaction_balance_n=balance))
            previous_u, previous_force = displacement, force
        assert abs(max(frames)-2) < 1e-8
        peak = max(abs(r['force_n']) for r in history)
        loading = next(r for r in history if abs(r['time']-1) < 1e-8)
        final = history[-1]
        checks = dict(work_energy_within_two_percent=abs(work/internal-1) < .02,
            near_zero_terminal_force=abs(force)/peak < .01,
            retained_positive_plastic_work=internal > 0 and internal < loading['internal_energy_n_mm'],
            elastic_unloading_peeq_change=abs(peeq-loading['peeq']))
        records.append(dict(run=str(directory.relative_to(FINAL)), increment=increment,
            history=history, checks=checks, hashes={f: sha256(directory/f)
                for f in ['job.inp', 'job.dat', 'job.sta', 'solver.log']}))
    coarse, fine = [r['history'][-1] for r in records]
    report = dict(records=records, parent=str(parent.relative_to(FINAL)),
        parent_sha256=sha256(parent), material_case_sha256=sha256(case_path),
        solver_sha256=sha256(exe), script_sha256=sha256(__file__),
        unload_target_mm=residual, prescribed_peak_true_strain=.02,
        refinement=dict(final_work_relative_difference=fine['external_work_n_mm']/coarse['external_work_n_mm']-1,
            final_energy_relative_difference=fine['internal_energy_n_mm']/coarse['internal_energy_n_mm']-1,
            work_energy_error_reduced=abs(fine['relative_work_energy_error']) < abs(coarse['relative_work_energy_error'])),
        sources=['sym/vendor/CalculiX/ccx_2.23/doc/ccx/node280.html:179-186',
                 'sym/vendor/CalculiX/ccx_2.23/src/resultsmech.f:1044-1084'],
        limitations='One C3D8 uniaxial coupon at supported plastic strain, two increments. ELSE is accumulated internal work including plastic dissipation, not solely recoverable elastic energy. Two-percent work and one-percent terminal-force checks are diagnostic tolerances, not experimental acceptance. No finite-cap work, pressure-load revision, pipe mesh convergence, Abaqus equivalence or physical burst validation.')
    (root / 'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    output = FINAL / 'results/output' / f'{name}.json'
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, indent=2))
    print(json.dumps([r['checks'] for r in records]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    run(parser.parse_args().name)
