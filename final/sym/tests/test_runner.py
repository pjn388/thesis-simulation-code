import json
from pathlib import Path
import pytest
from pipe_sim.config import Case
from pipe_sim.runner import run, sha256, write_json


def prepared(tmp_path):
    Case().save(tmp_path/'case.json')
    (tmp_path/'job.inp').write_text('*HEADING\nfixture\n')
    write_json(tmp_path/'manifest.json',{'solver':'calculix','files':{n:sha256(tmp_path/n) for n in ['case.json','job.inp']}})
    write_json(tmp_path/'status.json',{'state':'generated'})


def test_zero_exit_does_not_hide_solver_error(tmp_path,monkeypatch):
    prepared(tmp_path)
    fake=tmp_path/'fake_ccx';fake.write_text('#!/bin/sh\necho "*ERROR in analysis"\necho "Job finished"\n')
    fake.chmod(0o755);monkeypatch.setenv('CCX_COMMAND',str(fake))
    assert run(tmp_path)['state']=='failed'
    with pytest.raises(ValueError,match='state is failed'):run(tmp_path)


def test_changed_inputs_cannot_reuse_manifest(tmp_path):
    prepared(tmp_path);(tmp_path/'job.inp').write_text('changed')
    with pytest.raises(ValueError,match='input changed'):run(tmp_path)


def test_memory_guard_is_an_interruption_not_burst(tmp_path,monkeypatch):
    prepared(tmp_path)
    fake=tmp_path/'memory_ccx';fake.write_text('#!/bin/sh\nsleep 60\n')
    fake.chmod(0o755);monkeypatch.setenv('CCX_COMMAND',str(fake))
    status=run(tmp_path,memory_limit_mb=.001)
    assert status['state']=='interrupted'
    assert 'Memory guard' in status['stop_reason']


def test_pardiso_requires_verified_binary(tmp_path,monkeypatch):
    from pipe_sim import runner
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.delenv('CCX_COMMAND',raising=False)
    folder=tmp_path/'vendor/ccx_pardiso';folder.mkdir(parents=True)
    binary=folder/'ccx_2.23_pardiso';binary.write_text('binary')
    metadata=dict(verified=False,solver_sha256=sha256(binary));write_json(folder/'build.json',metadata)
    case=Case();case.analysis.linear_solver='pardiso'
    with pytest.raises(ValueError,match='verified'):runner.case_solver_path(case)
    metadata['verified']=True;write_json(folder/'build.json',metadata)
    assert runner.case_solver_path(case)==str(binary)


def test_tabular_contact_rejects_unverified_binary(tmp_path,monkeypatch):
    from pipe_sim import runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    stock=tmp_path/'stock';stock.write_text('stock binary')
    fixed=tmp_path/'vendor/ccx_tabular_fix/ccx_2.23_tabularfix';fixed.parent.mkdir(parents=True);fixed.write_text('verified binary')
    write_json(fixed.parent/'build.json',dict(base_solver_sha256=sha256(stock),solver_sha256=sha256(fixed)))
    monkeypatch.setattr(runner,'solver_path',lambda solver:str(stock));monkeypatch.delenv('CCX_COMMAND',raising=False)
    case=Case();case.dent.normal_law='compression_only';case.dent.contact_formulation='surface_to_surface'
    assert runner.case_solver_path(case)==str(fixed)
    monkeypatch.setenv('CCX_COMMAND',str(stock))
    with pytest.raises(ValueError,match='not been verified'):runner.case_solver_path(case)
    case.dent.normal_law='linear';assert runner.case_solver_path(case)==str(stock)


def test_campaign_budget_is_scoped_and_preserves_frozen_case(tmp_path,monkeypatch):
    from pipe_sim import runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    policy=tmp_path/'campaign_runtime.json'
    write_json(policy,{'timeout_seconds':28800})
    assert runner.runtime_budget(tmp_path/'runs/diagnostic',1800)['timeout_seconds']==1800
    campaign=tmp_path/'runs/campaign/example';campaign.mkdir(parents=True)
    prepared(campaign)
    case=Case.load(campaign/'case.json');case.analysis.timeout_seconds=.01;case.save(campaign/'case.json')
    manifest=json.loads((campaign/'manifest.json').read_text())
    manifest['files']['case.json']=sha256(campaign/'case.json');write_json(campaign/'manifest.json',manifest)
    original=sha256(campaign/'case.json')
    fake=tmp_path/'fake_ccx';fake.write_text('#!/bin/sh\nsleep 0.1\necho "Job finished"\n');fake.chmod(0o755)
    monkeypatch.setenv('CCX_COMMAND',str(fake))
    status=run(campaign)
    assert status['state']=='completed'  # Survived the original 0.01-second cap.
    assert status['timeout_seconds']==28800
    assert status['case_timeout_seconds']==.01
    assert status['runtime_policy_sha256']==sha256(policy)
    assert sha256(campaign/'case.json')==original


@pytest.mark.parametrize('value',[0,-1,float('inf'),float('nan'),True,'28800'])
def test_invalid_runtime_budget_is_rejected(tmp_path,monkeypatch,value):
    from pipe_sim import runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    (tmp_path/'campaign_runtime.json').write_text(json.dumps({'timeout_seconds':value}))
    with pytest.raises(ValueError,match='finite and positive'):
        runner.runtime_budget(tmp_path/'runs/campaign/example',14400)


def test_extended_policy_still_enforces_a_finite_timeout(tmp_path,monkeypatch):
    from pipe_sim import runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    write_json(tmp_path/'campaign_runtime.json',{'timeout_seconds':.05})
    campaign=tmp_path/'runs/campaign/example';campaign.mkdir(parents=True);prepared(campaign)
    fake=tmp_path/'fake_ccx';fake.write_text('#!/bin/sh\nsleep 60\n');fake.chmod(0o755)
    monkeypatch.setenv('CCX_COMMAND',str(fake))
    status=run(campaign)
    assert status['state']=='interrupted'
    assert status['stop_reason']=='Runtime budget exceeded; not physical failure'


@pytest.mark.parametrize('cpus,expected',[(1,1),(5,4),(8,7)])
def test_campaign_auto_threads_follow_affinity_and_reserve_one_cpu(tmp_path,monkeypatch,cpus,expected):
    from pipe_sim import runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(cpus)))
    write_json(tmp_path/'campaign_runtime.json',dict(threads='auto',reserve_cpus=1))
    assert runner.solver_thread_settings(tmp_path/'runs/campaign/example',3)['solver_threads']==expected
    assert runner.solver_thread_settings(tmp_path/'runs/diagnostic',2)['solver_threads']==min(2,cpus)


@pytest.mark.parametrize('settings',[dict(threads=0),dict(threads=2.5),dict(threads=True),dict(threads='all'),dict(reserve_cpus=-1)])
def test_invalid_thread_policy_rejected(tmp_path,monkeypatch,settings):
    from pipe_sim import runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    write_json(tmp_path/'campaign_runtime.json',settings)
    with pytest.raises(ValueError):runner.solver_thread_settings(tmp_path/'runs/campaign/example',3)


def test_native_launch_gets_explicit_thread_budget_and_preserves_case(tmp_path,monkeypatch):
    from pipe_sim import runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(5)))
    monkeypatch.setenv('CCX_NPROC_EQUATION_SOLVER','1')
    write_json(tmp_path/'campaign_runtime.json',dict(timeout_seconds=28800,threads='auto',reserve_cpus=1))
    directory=tmp_path/'runs/campaign/example';directory.mkdir(parents=True);prepared(directory)
    before=sha256(directory/'case.json')
    fake=tmp_path/'fake_ccx'
    fake.write_text('#!/bin/sh\necho "$OMP_NUM_THREADS $MKL_NUM_THREADS $CCX_NPROC_EQUATION_SOLVER $CCX_NPROC_STIFFNESS $CCX_NPROC_RESULTS"\necho "Job finished"\n')
    fake.chmod(0o755);monkeypatch.setenv('CCX_COMMAND',str(fake))
    state=run(directory)
    assert state['state']=='completed'
    assert state['solver_threads']==4
    assert (directory/'solver.log').read_text().startswith('4 4 4 4 4\n')
    assert sha256(directory/'case.json')==before


@pytest.mark.parametrize('cpus,expected',[(1,1),(5,2),(8,3)])
def test_parallel_campaign_shares_cpu_budget(tmp_path,monkeypatch,cpus,expected):
    from pipe_sim import runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    monkeypatch.setattr(runner.os,'sched_getaffinity',lambda pid:set(range(cpus)))
    write_json(tmp_path/'campaign_runtime.json',dict(threads='auto',reserve_cpus=1,parallel_jobs=2))
    settings=runner.solver_thread_settings(tmp_path/'runs/campaign/example',3)
    assert settings['solver_threads']==expected
    assert settings['parallel_jobs']==2


@pytest.mark.parametrize('jobs',[0,-1,True,1.5])
def test_invalid_parallel_budget_rejected(tmp_path,monkeypatch,jobs):
    from pipe_sim import runner
    monkeypatch.setattr(runner,'ROOT',tmp_path)
    write_json(tmp_path/'campaign_runtime.json',dict(threads='auto',parallel_jobs=jobs))
    with pytest.raises(ValueError,match='parallel_jobs'):
        runner.solver_thread_settings(tmp_path/'runs/campaign/example',3)
