"""Supervise an existing orphaned solver without restarting its analysis."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import time

from campaign import FINAL, PLAN
from pipe_sim.config import Case
from pipe_sim.decks import steps
from pipe_sim.results import extract
from pipe_sim.runner import sha256, write_json, runtime_budget
from watch_campaign import converged_increment


def process_identity(pid):
    """Include kernel start ticks to distinguish PID reuse."""
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z':
            return None
        return stat[19]
    except FileNotFoundError:
        return None


def finished_normally(log, increment, expected_time):
    return ('Job finished' in log and '*ERROR' not in log and increment is not None
            and abs(increment['time']-expected_time)<1e-5)


def active_elapsed(state, now):
    """Exclude explicitly recorded user pauses from the original runtime budget."""
    start=datetime.fromisoformat(state['started_utc']).timestamp()
    return max(0.,now-start-float(state.get('total_paused_seconds',0.)))


def recover(name,pid):
    job=next(j for j in json.loads(PLAN.read_text())['jobs'] if j['id']==name)
    directory=(FINAL/job['run']).resolve()
    with (directory/'recovery.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((directory/'status.json').read_text())
        case=Case.load(directory/'case.json')
        if state['state']=='running':
            state.update(runtime_budget(directory, case.analysis.timeout_seconds))
            identity=process_identity(pid)
            if identity is not None:
                if Path(f'/proc/{pid}/cwd').resolve()!=directory:
                    raise ValueError('PID working directory does not match this run')
                if sha256(Path(f'/proc/{pid}/exe'))!=state['solver_sha256']:
                    raise ValueError('PID executable does not match the recorded solver')
            state.update(supervisor_recovered_utc=datetime.now(timezone.utc).isoformat(),
                         supervisor_pid=os.getpid(),solver_pid=pid,solver_start_ticks=identity)
            write_json(directory/'status.json',state)
            start=datetime.fromisoformat(state['started_utc']).timestamp()
            while identity is not None and process_identity(pid)==identity:
                fields=Path(f'/proc/{pid}/status').read_text().splitlines()
                rss=next((float(line.split()[1])/1024 for line in fields if line.startswith('VmRSS:')),0.)
                state['peak_observed_rss_mb']=max(state.get('peak_observed_rss_mb',0.),rss)
                elapsed=active_elapsed(state,time.time())
                memory_exceeded = rss>state.get('memory_limit_mb',float('inf'))
                if memory_exceeded or elapsed>state['timeout_seconds']:
                    state['stop_reason']=('Memory guard' if memory_exceeded else 'Runtime budget')+' exceeded after supervisor recovery; not physical failure'
                    os.kill(pid,signal.SIGTERM)
                    for _ in range(10):
                        if process_identity(pid)!=identity:break
                        time.sleep(1)
                    if process_identity(pid)==identity:os.kill(pid,signal.SIGKILL)
                    break
                time.sleep(5)
            log=(directory/'solver.log').read_text(errors='replace')
            complete=finished_normally(log,converged_increment(directory/'job.sta'),len(steps(case)))
            state.update(state='interrupted' if state.get('stop_reason') else ('completed' if complete else 'failed'),
                         elapsed_seconds=active_elapsed(state,time.time()),returncode=None,
                         completion_evidence='Recovered non-child process: exit code unavailable; native completion marker, absence of errors and final converged time checked')
            write_json(directory/'status.json',state)
        if not (directory/'results/summary.json').exists() and (directory/'job.dat').exists() and (directory/'job.dat').stat().st_size:
            try:extract(directory)
            except ValueError as exc:write_json(directory/'extraction_error.json',{'error':str(exc)})
        print(json.dumps(state),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_id');parser.add_argument('pid',type=int)
    args=parser.parse_args();recover(args.case_id,args.pid)
