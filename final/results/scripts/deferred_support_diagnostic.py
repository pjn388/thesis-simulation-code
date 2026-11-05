"""Bounded output-only support diagnostic after lane A drains; no production edits."""
import argparse
import fcntl
import json
import shutil
import subprocess
import time
from pathlib import Path

from pipe_sim.runner import run, sha256, write_json
from pipe_sim.results import extract
from prepare_reaction_benchmark import verify
from campaign import available_memory_mb, request_capacity_review

FINAL = Path(__file__).resolve().parents[2]
SOURCE = FINAL/'sym/runs/intact_support_output_20261008T193004Z'
TARGET = FINAL/'sym/runs/investigation/intact_support_bounded_20261010T170002Z'
RECORD = FINAL/'results/output/support_followup_20261010T170002Z.json'
LANE = 'thesis-campaign-large.service'


def paused():
    path = FINAL/'results/automation/paused.json'
    return path.exists() and json.loads(path.read_text()).get('state') in ('paused_by_user', 'pausing')


def native_count():
    count = 0
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if (process/'comm').read_text().strip().startswith('ccx'):
                count += 1
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return count


def admission(lane_active, count, memory, disk, user_paused):
    return not lane_active and count < 2 and memory >= 13048 and disk >= 25 and not user_paused


def prepare():
    evidence = verify(SOURCE)
    TARGET.mkdir(parents=True, exist_ok=False)
    for name in ('case.json', 'model.json', 'mesh.inp', 'job.inp'):
        shutil.copy2(SOURCE/name, TARGET/name)
    case = json.loads((TARGET/'case.json').read_text())
    case['analysis'].update(threads=2, timeout_seconds=1800)
    write_json(TARGET/'case.json', case)
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    manifest.update(parent=str(SOURCE.relative_to(FINAL)),
                    parent_manifest_sha256=sha256(SOURCE/'manifest.json'),
                    controlled_change='Prepared support U/RF output; operational threads=2 and timeout=1800 only',
                    files={name: sha256(TARGET/name) for name in manifest['files']})
    write_json(TARGET/'manifest.json', manifest)
    write_json(TARGET/'status.json', dict(state='generated', launch_deferred=True, reason='Waiting for lane A, native capacity, memory/disk and user pause checks'))
    write_json(RECORD, dict(state='queued', lane=LANE, run=str(TARGET.relative_to(FINAL)),
                           verification=evidence, timeout_seconds=1800, memory_limit_mb=11000,
                           comparison='Accepted common-pressure histories against intact parent; constrained U/RF and geometrically integrated pressure loads. RF includes loading plus reaction; subtraction needs verified assembled-load equivalence. No raw RF reaction/burst claim.',
                           limitation='Same unvalidated physics; output diagnostic only. Finite-cap native coupling remains unresolved.'))


def execute():
    with (TARGET/'followup.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if json.loads((TARGET/'status.json').read_text())['state'] != 'generated':
            raise RuntimeError('Preserve terminal/live attempt; no duplicate launch')
        while True:
            active = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', LANE]).returncode == 0
            count = native_count()
            memory = available_memory_mb()
            disk = shutil.disk_usage(FINAL).free/1024**3
            held = paused()
            record = json.loads(RECORD.read_text())
            record.update(state='waiting_capacity', lane_active=active, native_count=count,
                          available_mib=memory, disk_free_gib=disk, user_paused=held)
            write_json(RECORD, record)
            if admission(active, count, memory, disk, held):
                break
            time.sleep(15)
        # Recheck pause immediately before entering the guarded runner.
        if paused():
            raise RuntimeError('User pause arrived at launch boundary; diagnostic remains generated')
        record['state'] = 'running'
        write_json(RECORD, record)
        status = run(TARGET, memory_limit_mb=11000)
        record.update(state='finished', solver_status=status)
        if (TARGET/'job.sta').exists():
            record['summary'] = extract(TARGET)
        write_json(RECORD, record)
        request_capacity_review()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'execute'])
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else execute()
