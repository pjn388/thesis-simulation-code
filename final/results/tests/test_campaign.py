import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from campaign import usable_predictions
from watch_campaign import converged_increment
from recover_campaign_run import finished_normally
from review_live_withdrawal import matches_sta, bounded_lines
import numpy as np
import hashlib


def test_live_snapshot_printed_time_boundary():
    # DAT 2.451563 vs STA 2 + .451562 differs by one printed time unit.
    assert matches_sta(np.array([2.451562]), 2.451563)
    assert not matches_sta(np.array([2.451562]), 2.451564)


def test_live_snapshot_drops_partial_line_and_bounds_growth(tmp_path):
    native = tmp_path/'job.dat'
    prefix = b'complete\npartial'
    native.write_bytes(prefix)
    evidence = {}
    reader = bounded_lines(native, evidence)
    assert next(reader) == 'complete\n'
    with native.open('ab') as stream:
        stream.write(b' completed later\nnew line\n')
    assert list(reader) == []
    assert evidence['job.dat']['prefix_bytes'] == len(prefix)
    assert evidence['job.dat']['prefix_sha256'] == hashlib.sha256(prefix).hexdigest()
    assert evidence['job.dat']['size_after_read'] > len(prefix)


def test_failed_increment_is_not_a_pressure_result(tmp_path):
    sta=tmp_path/'job.sta'
    sta.write_text('STEP INC ATT ITRS TOT_TIME STEP_TIME INC_TIME\n'
                   '3 10 1 8 2.4 0.4 0.1\n'
                   '3 11 1U 60 2.5 0.5 0.1\n')
    assert converged_increment(sta)=={'step':3,'increment':10,'time':2.4}


def test_criterion_requires_valid_contact_and_completed_withdrawal():
    good=dict(failure_status='criterion_crossed',contact_checks_passed=True,
              dent=True,withdrawal_completed=True,criterion_pressure_mpa=15.)
    cases=[good,dict(good,contact_checks_passed=False),dict(good,withdrawal_completed=False),
           dict(good,failure_status='criterion_not_reached'),dict(good,criterion_pressure_mpa=None),
           dict(good,dent=False,withdrawal_completed=False)]
    assert usable_predictions(pd.DataFrame(cases)).index.tolist()==[0,5]


def test_recovered_run_requires_native_completion_and_final_increment():
    assert finished_normally('Job finished',{'time':3.},3.)
    assert not finished_normally('Job finished',{'time':2.4},3.)
    assert not finished_normally('*ERROR\nJob finished',{'time':3.},3.)
    assert not finished_normally('Job finished',None,3.)
    assert not finished_normally('still solving',{'time':3.},3.)


def test_parallel_queues_wait_for_memory_and_keep_separate_progress(tmp_path,monkeypatch):
    import json
    import campaign
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'jobs':[{'id':'one','run':'runs/one'}]}))
    monkeypatch.setattr(campaign,'PLAN',plan);monkeypatch.setattr(campaign,'FINAL',tmp_path)
    monkeypatch.setattr(campaign,'OUT',tmp_path/'out')
    memory=iter([1000,20000]);actions=[]
    monkeypatch.setattr(campaign,'available_memory_mb',lambda:next(memory))
    monkeypatch.setattr(campaign.time,'sleep',lambda seconds:actions.append('wait'))
    monkeypatch.setattr(campaign,'execute',lambda name:actions.append('solve'))
    monkeypatch.setattr(campaign.subprocess,'run',lambda *a,**kw:None)
    monkeypatch.setattr(campaign,'request_capacity_review',lambda:actions.append('review'))
    campaign.queue(['one'],queue_id='b')
    assert actions==['wait','solve','review']
    assert json.loads((tmp_path/'out/queue_b.json').read_text())['state']=='finished'
    assert not (tmp_path/'out/queue.json').exists()


def test_overlapping_launches_solve_case_only_once(tmp_path,monkeypatch):
    import json,time
    from concurrent.futures import ThreadPoolExecutor
    import campaign
    directory=tmp_path/'run';directory.mkdir()
    (directory/'status.json').write_text('{"state":"generated"}')
    case=tmp_path/'case.json';case.write_text('{}')
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'jobs':[{'id':'one','run':'run','case_file':'case.json','case_sha256':campaign.sha256(case)}]}))
    monkeypatch.setattr(campaign,'PLAN',plan);monkeypatch.setattr(campaign,'FINAL',tmp_path)
    monkeypatch.setattr(campaign,'OUT',tmp_path/'out')
    from types import SimpleNamespace
    monkeypatch.setattr(campaign.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    calls=[]
    def fake_run(path,**kwargs):
        calls.append(path);time.sleep(.05)
        result={'state':'completed'};(directory/'status.json').write_text(json.dumps(result));return result
    monkeypatch.setattr(campaign,'run',fake_run)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(campaign.execute,['one','one']))
    assert calls==[directory]


def test_capacity_hook_does_not_launch_local_services(monkeypatch):
    import campaign
    monkeypatch.setattr(campaign.subprocess, 'run', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Unexpected service launch')))
    assert campaign.request_capacity_review() is None
