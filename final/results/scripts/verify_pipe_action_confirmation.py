"""One bounded second-direction native confirmation of assembled action closure.

Four fixed-history native probes, followed by the independent all-touched-element
assembly audit. Original native approximation must remain reported separately.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np

import verify_pipe_sampled_direction as native
import audit_pipe_assembled_action as audit
from terminal_progress import solver_directories


def direction(base):
    mapping = np.array(base['mapping']).reshape(-1, 4)[:, 1:]
    connect = np.array(base['connectivity'])
    # Different reproducible direction and wider spatial coverage than the parent.
    selected = [1, 721, 1441, 2017, 2500, 2880]
    nodes = np.unique(np.concatenate([connect[base['ipkon'][e-1]:base['ipkon'][e-1]+20]-1 for e in selected]))
    indices = np.unique(mapping[nodes]); indices = indices[indices > 0]-1
    result = np.zeros(base['neq'])
    result[indices] = np.random.default_rng(29019).normal(size=len(indices))
    result /= np.linalg.norm(result)
    return result, selected, indices


def main(tag):
    root = native.F/'sym/runs/investigation'/('pipe_action_confirmation_'+tag)
    root.mkdir(exist_ok=False)
    state = dict(state='running', kind='bounded_native_diagnostic', started_utc=datetime.now(timezone.utc).isoformat(),
                 supervisor_pid=os.getpid(), native_probe_limit_seconds=600, native_probe_count=4,
                 seed=29019, step_sizes=[1e-6, 5e-7], physical_validation=False)
    status = root/'status.json'
    def save():
        tmp = status.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=2)+'\n'); tmp.replace(status)
    save()
    try:
        assert json.loads((native.F/'results/automation/paused.json').read_text())['state']=='resumed'
        solvers = solver_directories(set(), run_root=native.F/'sym')
        assert solvers is not None and sum(len(v['pids']) for v in solvers.values()) < 2
        available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
        assert available > 4*1024**2 and shutil.disk_usage(native.F).free > 25*1024**3
        prerequisite = native.F/'results/output/pipe_assembled_action_20261019T0012.json'
        assert json.loads(prerequisite.read_text())['checks_passed']
        native.sampled_direction = direction
        gate = native.F/'results/output'/('pipe_direction_'+tag+'.json')
        try:
            native.main(tag, 1e-6, 'results/output/pipe_solver_early_plastic_20261017T054741Z.json', 600)
        except AssertionError:
            # The uncorrected native comparison is expected to fail; only continue
            # if all four coherent probes produced its final immutable evidence.
            if not gate.exists():
                raise
        report = json.loads(gate.read_text())
        assert len(report['cases']) == 4
        report.update(confirmation=dict(seed=29019, selected_elements=[1,721,1441,2017,2500,2880],
                                       prerequisite_sha256=native.sha256(prerequisite),
                                       wrapper_sha256=native.sha256(__file__)))
        gate.write_text(json.dumps(report, indent=2)+'\n')
        audit.main(tag, str(gate.relative_to(native.F)))
        state.update(state='completed', corrected_action_checks_passed=True,
                     original_native_checks_passed=report['checks_passed'],
                     report='results/output/pipe_assembled_action_'+tag+'.json')
    except Exception as exc:
        state.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == '__main__':
    main(sys.argv[1])
