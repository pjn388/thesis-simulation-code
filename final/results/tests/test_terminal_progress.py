import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from terminal_progress import last_increment, snapshot, task_row, overall_progress, overall_lines
from progress_inventory import unit_state, management_snapshot


def fixture_job(root, name='example', dent=True, state='running', total=2.5):
    directory = root / name
    directory.mkdir()
    case = {'dent': {'enabled': dent}, 'analysis': {'max_pressure': 35}}
    (directory / 'case.json').write_text(json.dumps(case))
    (directory / 'status.json').write_text(json.dumps({'state': state}))
    (directory / 'job.sta').write_text(f'3 2 1 4 {total} 0.5 0.1\n')
    return {'id': name, 'kind': 'iteration', 'run': name, 'case_file': name + '/case.json'}


def test_failed_and_partial_native_lines_do_not_advance_progress(tmp_path):
    sta = tmp_path / 'job.sta'
    sta.write_text('3 2 1 4 2.5 0.5 0.1\n3 3 1U 60 2.7 0.7 0.2\n3 3 2 4 2.6 0.6 0.1')
    assert last_increment(sta) == 2.5


def test_pressure_only_and_dented_runs_use_their_actual_stage_count(tmp_path):
    dent = task_row(fixture_job(tmp_path, state='failed'), tmp_path)
    assert dent['fraction'] == pytest.approx(2.5/3)
    assert dent['pressure'] == 17.5
    assert dent['state'] == 'FAILED'
    plain = task_row(fixture_job(tmp_path, name='plain', dent=False, total=.5), tmp_path)
    assert plain['fraction'] == .5
    assert plain['pressure'] == 17.5


def test_transient_status_write_and_stale_running_are_distinguished(tmp_path):
    job = fixture_job(tmp_path)
    assert task_row(job, tmp_path, live=set())['state'] == 'NO SOLVER?'
    status = tmp_path / 'example/status.json'
    status.write_text('{')
    assert task_row(job, tmp_path)['state'] == 'READ ERROR'
    status.write_text('{"state":"running"}')
    assert task_row(job, tmp_path, live={tmp_path / 'example'})['state'] == 'RUNNING'


def test_refresh_discovers_new_tasks_and_excludes_superseded_attempts(tmp_path):
    first = fixture_job(tmp_path)
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'jobs': [first]}))
    kwargs = dict(plan_path=plan, root=tmp_path, include_earlier=False)
    assert len(snapshot(**kwargs)[1]) == 1
    second = fixture_job(tmp_path, name='replacement')
    first['superseded_by'] = 'replacement'
    plan.write_text(json.dumps({'jobs': [first, second]}))
    assert [r['id'] for r in snapshot(**kwargs)[1]] == ['replacement']
    assert len(snapshot(**kwargs, all_attempts=True)[1]) == 2


def test_discovers_unregistered_coupon_and_deduplicates_registries(tmp_path):
    first=fixture_job(tmp_path)
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'jobs':[first]}))
    config=tmp_path/'results/config';config.mkdir(parents=True)
    (config/'runs.json').write_text(json.dumps([dict(first,case_id='alias')]))
    coupon=tmp_path/'sym/runs/new_coupon';coupon.mkdir(parents=True)
    (coupon/'job.inp').write_text('*STEP\n*STATIC\n.1,2\n*END STEP\n')
    (coupon/'job.sta').write_text('1 1 1 2 1.0 1.0 1.0\n')
    (coupon/'solver.log').write_text('Job finished\n')
    _,rows=snapshot(plan_path=plan,root=tmp_path,include_earlier=False)
    assert len(rows)==2
    native=next(r for r in rows if r['id']=='new_coupon')
    assert native['fraction']==.5
    assert native['state']=='SOLVED'  # Native completion, not diagnostic pass.
    assert not native['campaign']
    assert len(snapshot(plan_path=plan,root=tmp_path,include_earlier=False,include_other=False)[1])==1


def test_generated_diagnostic_is_not_misreported_as_queued(tmp_path):
    job=fixture_job(tmp_path,state='generated');job.update(campaign=False,kind='additional')
    assert task_row(job,tmp_path)['state']=='PREPARED'
    job=fixture_job(tmp_path,name='paused');job.update(campaign=False,kind='additional')
    live={tmp_path/'paused':{'pids':[123], 'paused':True}}
    assert task_row(job,tmp_path,live=live)['state']=='PAUSED'


def test_executed_superseded_run_remains_visible_without_double_count(tmp_path):
    base=tmp_path/'sym/runs';base.mkdir(parents=True)
    old=fixture_job(tmp_path,name='sym/runs/old',state='failed')
    new=fixture_job(tmp_path,name='sym/runs/new',state='running')
    old['superseded_by']=new['id']
    plan=tmp_path/'plan.json';plan.write_text(json.dumps({'jobs':[old,new]}))
    _,rows=snapshot(plan_path=plan,root=tmp_path,include_earlier=False)
    assert len(rows)==2
    assert sum(r['campaign'] for r in rows)==1
    assert next(r for r in rows if r['id']==old['id'])['state']=='SUPERSEDED'


def test_management_failure_and_orphaned_timer_are_not_healthy():
    assert unit_state({'LoadState':'loaded','ActiveState':'failed'})=='FAILED'
    assert unit_state({'LoadState':'not-found','ActiveState':'active'})=='ORPHANED'
    assert unit_state({'LoadState':'loaded','ActiveState':'active','FreezerState':'frozen'})=='PAUSED'


def test_publication_snapshot_does_not_query_private_services(tmp_path, monkeypatch):
    import progress_inventory
    monkeypatch.setattr(progress_inventory.subprocess, 'run', lambda *a, **kw: (_ for _ in ()).throw(AssertionError('Unexpected service query')))
    assert management_snapshot(tmp_path) == {'units': [], 'errors': [], 'review': None, 'review_counts': {}}


def test_overall_estimate_counts_required_cases_and_live_work_only(tmp_path):
    rows=[]
    for name,state,total in [('done','completed',3), ('live','running',1.5),
                             ('failed','failed',2.99), ('stopped','interrupted',2.9),
                             ('queued','generated',0)]:
        rows.append(task_row(fixture_job(tmp_path,name=name,state=state,total=total),tmp_path))
    historical=dict(rows[0],id='historical',superseded=True)
    diagnostic=dict(rows[0],id='diagnostic',campaign=False)
    earlier=dict(rows[0],id='earlier',campaign=False,earlier=True)
    result=overall_progress(rows+[historical,diagnostic,earlier,rows[0]])
    assert result==dict(total=5,completed=1,underway=1,attention=2,fraction=.3)
    rows[1]['state']='PAUSED'
    assert overall_progress(rows)['fraction']==.3
    rows[1]['state']='NO SOLVER?'
    assert overall_progress(rows)['fraction']==.2


def test_overall_completion_requires_native_evidence_and_handles_empty_plan(tmp_path):
    row=task_row(fixture_job(tmp_path,state='completed',total=3),tmp_path)
    assert overall_progress([row])['fraction']==1
    row.update(fraction=None,error='Missing native progress')
    assert overall_progress([row])['completed']==0
    assert overall_progress([row])['attention']==1
    assert overall_progress([])['fraction'] is None
    assert 'n/a' in overall_lines([],80)[0]
    row.update(fraction=1,error=None,state='RUNNING')
    assert '99.9%' in overall_lines([row],80)[0]


def test_overall_bar_remains_pinned_when_scrolling(monkeypatch):
    import curses
    from types import SimpleNamespace
    import terminal_progress
    header=overall_lines([],100)
    monkeypatch.setattr(terminal_progress,'refresh',lambda args,width:('now',header+[str(i) for i in range(50)]))
    monkeypatch.setattr(curses,'curs_set',lambda value:None)
    class Screen:
        def __init__(self):self.frames=[];self.keys=iter([curses.KEY_END,ord('q')])
        def keypad(self,value):pass
        def timeout(self,value):pass
        def getmaxyx(self):return 24,100
        def erase(self):self.frame={}
        def addnstr(self,y,x,text,width):self.frame[y]=text[:width]
        def refresh(self):self.frames.append(self.frame.copy())
        def getch(self):return next(self.keys)
    screen=Screen()
    terminal_progress.terminal(screen,SimpleNamespace(interval=10))
    assert all(frame[0]==header[0] for frame in screen.frames)
    assert screen.frames[0][7]=='0'
    assert screen.frames[1][7]=='34'
