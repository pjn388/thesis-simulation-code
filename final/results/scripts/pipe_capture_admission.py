"""Read-only preflight for the proposed bounded pipe capture; never launches it.

A passing snapshot is necessary, not permission to run: the future service must
recheck under launch serialization and verify the remaining small-control gates.
"""
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from terminal_progress import solver_directories

FINAL = Path(__file__).resolve().parents[2]


def assess(pause_state, available_mib, free_bytes, native_pids):
    reasons = []
    if pause_state != 'resumed':
        reasons.append('manual pause or unknown pause state')
    if not math.isfinite(available_mib) or available_mib < 13048:
        reasons.append('MemAvailable below 13048 MiB or invalid')
    if free_bytes < 40 * 1024**3:
        reasons.append('disk below 40 GiB reserve')
    # Count processes, not run directories: duplicate solvers must count too.
    if len(native_pids) >= 2:
        reasons.append('two or more native solver processes already present')
    return dict(resource_snapshot_passed=not reasons, blockers=reasons,
                pause_state=pause_state, available_mib=available_mib,
                free_gib=free_bytes / 1024**3, native_pids=sorted(native_pids))


def snapshot():
    pause = json.loads((FINAL/'results/automation/paused.json').read_text())['state']
    memory = next(int(s.split()[1])/1024 for s in Path('/proc/meminfo').read_text().splitlines()
                  if s.startswith('MemAvailable:'))
    found = solver_directories(set(), FINAL/'sym/runs')
    if found is None:
        raise RuntimeError('native process observation unavailable')
    pids = [p for row in found.values() for p in row['pids']]
    result = assess(pause, memory, shutil.disk_usage(FINAL).free, pids)
    result.update(utc=datetime.now(timezone.utc).isoformat(),
                  scope='Read-only resource snapshot; not an atomic launch admission',
                  pending=['predictor/constraint verification', 'explicit node-set/MPC connectivity',
                           'serialized service launch with fresh admission and frozen identity checks'])
    return result


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2, allow_nan=False))
