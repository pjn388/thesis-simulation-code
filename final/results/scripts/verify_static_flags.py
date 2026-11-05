"""Bounded runtime test of static pressure branch eligibility, not tangent validity."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
from verify_shared_node_capture import FINAL, EXE, coupon
from pipe_sim.runner import sha256


def run(tag):
    capture = Path(__file__).with_name('capture_static_flags_gdb.py')
    source = FINAL/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    assert sha256(EXE) == json.loads((EXE.parent/'build.json').read_text())['solver_sha256']
    root = FINAL/'sym/runs/investigation'/('static_flags_'+tag)
    root.mkdir(exist_ok=False)
    cases = []
    for side in ('minus', 'plus'):
        assert json.loads((FINAL/'results/automation/paused.json').read_text())['state'] == 'resumed'
        folder = root/side
        folder.mkdir()
        deck, _, _, _, _ = coupon(side, False, .04)
        (folder/'job.inp').write_text(deck)
        (folder/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            MKL_THREADING_LAYER='GNU', CCX_NPROC_RESULTS='1', CCX_NPROC_STIFFNESS='1',
            CCX_NPROC_EQUATION_SOLVER='1', THESIS_FLAG_SOURCE=str(source),
            THESIS_FLAG_OUTPUT=str(folder/'flags.json'))
        with (folder/'solver.log').open('w') as stream:
            result = subprocess.run(['gdb', '--batch', '-x', 'capture.gdb', '--args',
                str(EXE), '-i', 'job'], cwd=folder, env=env, stdout=stream,
                stderr=subprocess.STDOUT, timeout=120)
        assert result.returncode == 0 and 'exited normally' in (folder/'solver.log').read_text()
        data = json.loads((folder/'flags.json').read_text())
        assert data['records']
        assert all(r['nelem'] in (1, 2) and r['nmethod'] == 1 and r['nload'] == 2
                   for r in data['records'])  # Cross-check ABI mapping against the frozen deck.
        combinations = Counter((r['mass'], r['rhsi'], r['stiffness'], r['buckling'])
                               for r in data['records'])
        eligible = sum(r['rhsi'] != 1 and r['mass'] == 1 and 1 in r['iperturb']
                       for r in data['records'])
        cases.append(dict(side=side, run=str(folder.relative_to(FINAL)),
            calls=len(data['records']), rhs_calls=data['rhs_calls'],
            combinations=[dict(mass=k[0], rhsi=k[1], stiffness=k[2], buckling=k[3], count=v)
                          for k, v in sorted(combinations.items())],
            cited_pressure_stiffness_branch_eligible_calls=eligible,
            maximum_increment=max(r['iinc'] for r in data['records']),
            hashes={p.name:sha256(p) for p in (folder/'job.inp', folder/'flags.json', folder/'job.sta')}))
    report = dict(cases=cases, native_execution=True, physical_acceptance=False,
        limitation='Entry arguments establish branch eligibility only on two elastic free curved-face coupons. No matrix contribution or residual derivative captured; not production pipe/contact/MPC or Abaqus equivalence. ABI cross-checks do not replace debug symbols.',
        next='Capture fixed-geometry native element matrix differences with/without pressure and compare residual derivatives; audit alternative assembly before proposing a production revision. Hardening remains independent.',
        provenance={str(p.relative_to(FINAL)):sha256(p) for p in (Path(__file__), capture, source, EXE,
            Path(__file__).with_name('verify_shared_node_capture.py'))})
    output = FINAL/'results/output'/('static_flags_'+tag+'.json')
    assert not output.exists()
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(cases))


if __name__ == '__main__':
    import sys
    run(sys.argv[1])
