"""Guarded reference checks, then one unchanged intact-pipe convergence experiment.

This uses an explicitly experimental binary outside production runner acceptance.
Never changes the accepted solver manifest or calls a numerical endpoint burst.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

from pipe_sim.runner import sha256, solver_environment
from pipe_sim.results import dat_blocks, extract
from terminal_progress import solver_directories
FINAL = Path(__file__).resolve().parents[2]

BUILD = FINAL/'sym/vendor/ccx_static_linesearch_v3_20261019'
ROOT = FINAL/'sym/runs/investigation/static_linesearch_strict_20261019'
OUTPUT = FINAL/'results/output/static_linesearch_strict_20261019.json'


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def launch(directory, exe, enabled, timeout):
    assert json.loads((FINAL/'results/automation/paused.json').read_text())['state']=='resumed'
    solvers = solver_directories(set(), run_root=FINAL/'sym')
    assert solvers is not None and sum(len(v['pids']) for v in solvers.values()) < 2
    mem = 1024*int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
    assert mem > 13*1024**3 and shutil.disk_usage(FINAL).free > 25*1024**3
    env = solver_environment(2)
    env.pop('THESIS_STATIC_LINESEARCH', None)
    if enabled:
        env['THESIS_STATIC_LINESEARCH'] = '1'
    state = dict(state='running', started_utc=datetime.now(timezone.utc).isoformat(),
                 solver_sha256=sha256(exe), command=[str(exe), '-i', 'job'],
                 experimental_static_linesearch=enabled, timeout_seconds=timeout,
                 memory_limit_mb=11000, solver_threads=2, validated=False)
    save(directory/'status.json', state)
    start = time.monotonic(); peak = 0
    with (directory/'solver.log').open('w') as log:
        process = subprocess.Popen(state['command'], cwd=directory, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    rss = int(next(s.split()[1] for s in Path(f'/proc/{process.pid}/status').read_text().splitlines() if s.startswith('VmRSS:')))/1024
                    peak = max(peak, rss)
                except (FileNotFoundError, ProcessLookupError, StopIteration):
                    pass
                if peak > 11000:
                    raise RuntimeError('11,000 MiB memory guard')
                if time.monotonic()-start > timeout:
                    raise TimeoutError('bounded experiment time guard')
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            state.update(state='completed' if process.returncode == 0 else 'failed', returncode=process.returncode)
        except Exception as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            state.update(state='timed_out' if isinstance(exc, TimeoutError) else 'failed', error=str(exc))
        finally:
            state.update(elapsed_seconds=time.monotonic()-start, peak_observed_rss_mb=peak,
                         finished_utc=datetime.now(timezone.utc).isoformat())
            save(directory/'status.json', state)
    return state


def fields(path):
    return {(q, g, t): np.array(a) for q, g, t, a in dat_blocks(path) if q and a}


def comparison(reference, actual):
    shared = reference.keys() & actual.keys()
    assert shared
    errors = []
    for key in shared:
        a, b = actual[key], reference[key]
        assert a.shape == b.shape and np.isfinite(a).all()
        errors.append(float(np.max(abs(a-b)/np.maximum(np.max(abs(b), axis=0), 1.))))
    groups = {(q, g) for q, g, _ in reference}
    for q, g in groups:
        terminal = max(t for qq, gg, t in reference if (qq, gg)==(q, g))
        assert (q, g, terminal) in shared, 'Each observable must reach its reference endpoint'
    return dict(shared_blocks=len(shared), maximum_normalized_difference=max(errors), passed=max(errors)<1e-5)


def main():
    ROOT.mkdir(exist_ok=False)
    metadata = json.loads((BUILD/'build.json').read_text())
    exe = Path(metadata['executable'])
    parent_exe = FINAL/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==metadata['solver_sha256'] and sha256(parent_exe)==metadata['parent_solver_sha256']
    report = dict(state='running', experimental=True, physical_validation=False, coupon_checks=[],
                  sources={str(p.relative_to(FINAL)):sha256(p) for p in (Path(__file__), BUILD/'build.json', exe, parent_exe)})
    save(ROOT/'status.json', dict(state='running', kind='bounded_convergence_experiment', supervisor_pid=os.getpid()))
    save(OUTPUT, report)
    try:
        cases = [('elastic', 'sym/runs/elastic_001'), ('material', 'sym/runs/investigation/material_coupon'),
                 ('contact', 'sym/runs/investigation/contact_coupon/tabular_fixed')]
        for name, source in cases:
            reference = None
            for variant, binary, enabled in [('parent', parent_exe, False), ('disabled', exe, False), ('enabled', exe, True)]:
                folder = ROOT/(name+'_'+variant); folder.mkdir()
                for p in (FINAL/source).glob('*.inp'):
                    shutil.copy2(p, folder/p.name)
                deck = folder/'job.inp'
                # Match equilibrium precision in all three branches, rather than
                # compare loosely converged states against a stricter output gate.
                controls = '*CONTROLS, PARAMETERS=FIELD\n1e-08,1e-05,,,1e-8,1e-5,1e-6,1e-12\n'
                deck.write_text(deck.read_text().replace('*STATIC', controls+'*STATIC,SOLVER=PARDISO'))
                save(folder/'input_manifest.json', {p.name:sha256(p) for p in folder.glob('*.inp')})
                state = launch(folder, binary, enabled, 120)
                text = (folder/'solver.log').read_text()
                assert state['state']=='completed' and 'Job finished' in text and '*ERROR' not in text, str(folder)
                actual = fields(folder/'job.dat')
                if reference is None:
                    reference = actual
                    continue
                check = comparison(reference, actual)
                check.update(case=name, variant=variant, run=str(folder.relative_to(FINAL)),
                             line_search_evaluations=text.count('line search factor='))
                report['coupon_checks'].append(check); save(OUTPUT, report)
                assert check['passed'], check
        report['coupon_checks_passed'] = True
        # Clone an existing failed full-pressure case, with byte-identical physics inputs.
        parent = FINAL/'sym/runs/campaign/orientation_intact_00_adaptive'
        original = json.loads((parent/'manifest.json').read_text())
        for name, digest in original['files'].items():
            assert sha256(parent/name)==digest, name
        case = json.loads((parent/'case.json').read_text())
        assert not case['dent']['enabled'] and not case['gouge']['enabled']
        directory = ROOT/'intact'; directory.mkdir()
        for name in original['files']:
            shutil.copy2(parent/name, directory/name)
        shutil.copy2(parent/'manifest.json', directory/'manifest.json')
        save(directory/'experiment.json', dict(parent=str(parent.relative_to(FINAL)),
             parent_manifest_sha256=sha256(parent/'manifest.json'), solver_sha256=sha256(exe),
             changed='Opt-in static line-search algorithm only; all manifest inputs byte-identical',
             runtime_guard_seconds=1800, acceptance='Numerical convergence experiment only; not production or physical burst acceptance'))
        report.update(pipe_run=str(directory.relative_to(FINAL)), parent_run=str(parent.relative_to(FINAL)))
        save(OUTPUT, report)
        report['pipe_status'] = launch(directory, exe, True, 1800)
        if (directory/'job.dat').exists() and (directory/'job.dat').stat().st_size:
            extract(directory)
        report['state'] = 'completed'
        report['line_search_evaluations'] = (directory/'solver.log').read_text().count('line search factor=')
        report['next'] = 'Compare accepted pressure, common-load response and iteration/cutback cost with unchanged parent. Gate production continuation on evidence; solver stop is not burst.'
    except Exception as exc:
        report.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(OUTPUT, report)
        save(ROOT/'status.json', dict(state=report['state'], kind='bounded_convergence_experiment',
                                     report=str(OUTPUT.relative_to(FINAL)), finished_utc=report['finished_utc']))


if __name__ == '__main__':
    main()
