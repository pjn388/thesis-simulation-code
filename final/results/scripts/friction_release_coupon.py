"""Bounded full-overlap contact shear/release control; not a sparse-face test."""
import argparse
import json
import subprocess
from pathlib import Path

from contact_coupon import deck, FINAL
from pipe_sim.results import dat_blocks
from pipe_sim.runner import sha256, solver_environment
from audit_pressure_failures import converged_times, matches_sta


def run(name):
    if Path(name).name != name:
        raise ValueError('name must be a single directory name')
    root = FINAL / 'sym/runs/investigation' / name
    root.mkdir(parents=True, exist_ok=False)
    exe = FINAL / 'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    build = json.loads((exe.parent / 'build.json').read_text())
    assert build['verified'] and sha256(exe) == build['solver_sha256']
    records = []
    for mu in (0., .3):
        directory = root / f'mu_{mu:g}'
        directory.mkdir()
        header = deck('TABULAR').split('*STEP')[0]
        if mu > 0:
            header = header.replace('*CONTACT PAIR', f'*FRICTION\n{mu},200000\n*CONTACT PAIR')
        # Keep normal overlap fixed during shear. Small shift preserves almost
        # full overlap; it does not deliberately exercise sparse old-point maps.
        paths = [(0., -.11), (.0005, -.11), (.005, -.11), (.005, 0.), (0., 0.)]
        lines = [header]
        for x, z in paths:
            lines += ['*STEP,NLGEOM,INC=200', '*STATIC,SOLVER=PARDISO',
                      '.1,1,1e-8,.1', '*BOUNDARY', f'100,1,1,{x}',
                      f'100,3,3,{z}', '*NODE PRINT,NSET=REF,FREQUENCY=1',
                      'U,RF', '*END STEP']
        inp = directory / 'job.inp'
        inp.write_text('\n'.join(lines) + '\n')
        initial = sha256(inp)
        with (directory / 'solver.log').open('w') as log:
            result = subprocess.run([str(exe), '-i', 'job'], cwd=directory,
                                    env=solver_environment(1), stdout=log,
                                    stderr=subprocess.STDOUT, timeout=90)
        log = (directory / 'solver.log').read_text()
        assert result.returncode == 0 and 'Job finished' in log and '*ERROR' not in log
        assert sha256(inp) == initial
        sta = converged_times(directory / 'job.sta')
        rows = {}
        for kind, group, time, data in dat_blocks(directory / 'job.dat'):
            if kind in ('U', 'RF') and data:
                rows.setdefault(time, {'time': time})[kind] = data[0][1:4]
        accepted = [r for t, r in sorted(rows.items())
                    if 'U' in r and 'RF' in r and matches_sta(sta, t)]
        endpoints = [next(r for r in accepted if abs(r['time']-t) < 1e-7)
                     for t in range(1, 6)]
        normal = -endpoints[0]['RF'][2]
        checks = {
            'normal_2000n': abs(normal - 2000.) < .02,
            'coulomb_bound': all(abs(r['RF'][0]) <= mu * max(0., -r['RF'][2]) + .05
                                 for r in accepted if 1 <= r['time'] <= 3),
            'released_zero': max(abs(v) for r in accepted if r['time'] >= 4
                                 for v in r['RF']) < .05,
            'shear_response': (abs(endpoints[2]['RF'][0]) < .05 if mu == 0.
                               else abs(endpoints[2]['RF'][0]) > 500.),
        }
        records.append(dict(mu=mu, run=str(directory.relative_to(FINAL)),
                            accepted_frames=len(accepted), endpoints=endpoints,
                            checks=checks, input_sha256=initial,
                            native_sha256={n: sha256(directory/n) for n in
                                           ('job.sta', 'job.dat', 'solver.log')}))
    report = dict(records=records, solver_sha256=sha256(exe),
                  script_sha256=sha256(__file__),
                  limitations='Rigid master and fixed elastic slave, nearly full overlap. '
                  'Checks integrated forces, Coulomb bound and release only; does not '
                  'verify sparse-face remapping, pipe unloading, mesh convergence or burst.')
    (root / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    out = FINAL / 'results/output/investigation' / f'{name}.json'
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert all(all(r['checks'].values()) for r in records), 'Coupon checks failed; preserve evidence'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    run(parser.parse_args().name)
