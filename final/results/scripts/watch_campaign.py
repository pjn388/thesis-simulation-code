"""Refresh lightweight native progress; rebuild plots when attempts change state."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from campaign import FINAL, OUT, PLAN
from pipe_sim.config import Case
from pipe_sim.results import stage_at


def converged_increment(path):
    if not path.exists():
        return None
    for line in reversed(path.read_text(errors='replace').splitlines()):
        fields=line.split()
        if len(fields)!=7 or not fields[0].isdigit() or not fields[2].isdigit():
            continue  # U is an unsuccessful attempt, never a converged pressure.
        try:
            return {'step':int(fields[0]), 'increment':int(fields[1]), 'time':float(fields[4])}
        except ValueError:
            continue
    return None


def snapshot():
    rows=[]
    for job in json.loads(PLAN.read_text())['jobs']:
        if job.get('superseded_by'):
            continue
        directory=FINAL/job['run']
        status_path=directory/'status.json'
        state=json.loads(status_path.read_text()) if status_path.exists() else {'state':'not_run'}
        row={'case_id':job['id'], 'state':state['state'], 'stage':None, 'applied_pressure_mpa':None}
        summary=directory/'results/summary.json'
        row['extracted_result_mtime']=summary.stat().st_mtime_ns if summary.exists() else None
        increment=converged_increment(directory/'job.sta')
        if increment:
            case=Case.load(FINAL/job['case_file'])
            stage,pressure=stage_at(case,increment['time'])
            row.update(increment,stage=stage,applied_pressure_mpa=pressure)
        rows.append(row)
    now=datetime.now(timezone.utc).isoformat()
    lines=['# Live campaign progress','',f'Updated {now}. Monitor PID {os.getpid()}.',
           'Stages below are the last converged native increments. Applied pressure is **not burst pressure**.',
           'Full comparison plots and [summary](SUMMARY.md) refresh when an attempt changes state.','',
           '| Case | State | Last converged stage | Applied pressure (MPa) |',
           '|---|---|---|---:|']
    for row in rows:
        pressure=row['applied_pressure_mpa']
        value=f'{pressure:.5g}' if pressure is not None else '—'
        lines.append(f"| {row['case_id']} | {row['state']} | {row['stage'] or '—'} | {value} |")
    OUT.mkdir(exist_ok=True)
    for name,contents in [('PROGRESS.md','\n'.join(lines)+'\n'),('progress.json',json.dumps({'updated_utc':now,'pid':os.getpid(),'jobs':rows},indent=2)+'\n')]:
        temporary=OUT/(name+'.tmp');temporary.write_text(contents);temporary.replace(OUT/name)
    return rows


def watch(once=False):
    previous=None
    while True:
        try:
            rows=snapshot()
            states=tuple((row['case_id'],row['state'],row['extracted_result_mtime']) for row in rows)
            if states!=previous:
                # Read-only plotting runs in a fresh process, using current scripts.
                with (OUT/'monitor-report.log').open('a') as log:
                    result=subprocess.run([sys.executable,str(Path(__file__).with_name('campaign.py')),'report'],stdout=log,stderr=subprocess.STDOUT)
                if result.returncode==0:
                    previous=states
            if once or all(row['state'] in ('completed','failed','interrupted') for row in rows):
                break
        except (OSError,ValueError) as exc:
            # A state file may briefly be between writes; retry without changing jobs.
            print(f'Monitor retry: {exc}',flush=True)
        time.sleep(60)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    watch(parser.parse_args().once)
