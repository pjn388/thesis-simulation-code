import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from .geometry import build_model
from .decks import write_deck, steps

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def runtime_budget(directory, case_timeout_seconds):
    """Operational campaign budget, separate from immutable physical inputs.

    Read once when a runner or recovery supervisor attaches. Other runs retain
    their case budget. Record the exact policy hash with the lifecycle status.
    """
    budget = dict(timeout_seconds=float(case_timeout_seconds),
                  case_timeout_seconds=float(case_timeout_seconds))
    policy = ROOT/'campaign_runtime.json'
    if Path(directory).resolve().is_relative_to((ROOT/'runs/campaign').resolve()) and policy.exists():
        raw = policy.read_bytes()
        config = json.loads(raw)
        value = config['timeout_seconds']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Campaign timeout_seconds must be finite and positive')
        budget.update(timeout_seconds=float(value), runtime_policy=str(policy),
                      runtime_policy_sha256=hashlib.sha256(raw).hexdigest())
    return budget


def solver_thread_settings(directory, case_threads):
    """Choose threads at native launch; never change an existing solver."""
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    requested = case_threads
    policy = ROOT/'campaign_runtime.json'
    reserved = 0
    parallel_jobs = 1
    if Path(directory).resolve().is_relative_to((ROOT/'runs/campaign').resolve()) and policy.exists():
        config = json.loads(policy.read_text())
        requested = config.get('threads', case_threads)
        reserved = config.get('reserve_cpus', 0)
        parallel_jobs = config.get('parallel_jobs', 1)
        if isinstance(reserved, bool) or not isinstance(reserved, int) or reserved < 0:
            raise ValueError('reserve_cpus must be a nonnegative integer')
        if isinstance(parallel_jobs, bool) or not isinstance(parallel_jobs, int) or parallel_jobs < 1:
            raise ValueError('parallel_jobs must be a positive integer')
    if requested == 'auto':
        effective = max(1, (available-reserved)//parallel_jobs)
    else:
        if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
            raise ValueError('threads must be a positive integer or auto')
        effective = min(requested, max(1, available-reserved))
    return dict(solver_threads=effective, case_threads=case_threads,
                requested_threads=requested, available_cpus=available, reserve_cpus=reserved,
                parallel_jobs=parallel_jobs)


def solver_environment(threads):
    """Explicit limits prevent inherited CCX caps overriding the chosen budget."""
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'CCX_NPROC_RESULTS', 'CCX_NPROC_STIFFNESS',
                'CCX_NPROC_EQUATION_SOLVER', 'MKL_NUM_THREADS'):
        env[key] = str(threads)
    env.update(OPENBLAS_NUM_THREADS='1', MKL_THREADING_LAYER='GNU')
    return env


def solver_path(solver='calculix'):
    if solver != 'calculix': raise ValueError('Only CalculiX is supported')
    executable=os.environ.get('CCX_COMMAND') or shutil.which('ccx')
    local=ROOT/'vendor/CalculiX/ccx_2.23/src/ccx_2.23'
    if not executable and local.exists():executable=str(local)
    if not executable:raise FileNotFoundError(f'{solver} executable unavailable. See README.md')
    return str(Path(executable).resolve()) if Path(executable).exists() else executable


def case_solver_path(case,solver='calculix'):
    """Guard the verified 2.23 face-to-face TABULAR workaround from stock use."""
    if solver != 'calculix': raise ValueError('Only CalculiX is supported')
    if case.analysis.linear_solver=='pardiso':
        directory=ROOT/'vendor/ccx_pardiso';metadata=directory/'build.json'
        if not metadata.exists():raise ValueError('Build and verify the isolated PARDISO executable first')
        build=json.loads(metadata.read_text());executable=Path(os.environ.get('CCX_COMMAND') or directory/'ccx_2.23_pardiso')
        if not build.get('verified') or sha256(executable)!=build['solver_sha256']:
            raise ValueError('PARDISO executable must match the verified build')
        return str(executable)
    executable=solver_path(solver)
    if case.dent.normal_law!='compression_only' or case.dent.contact_formulation!='surface_to_surface':
        return executable
    fixed=ROOT/'vendor/ccx_tabular_fix/ccx_2.23_tabularfix'
    metadata=ROOT/'vendor/ccx_tabular_fix/build.json'
    if not fixed.exists() or not metadata.exists():
        raise ValueError('Compression-only face-to-face contact needs the verified isolated solver; run scripts/build_contact_fix.py and the contact coupon. See docs/contact-fix.md.')
    build=json.loads(metadata.read_text());current=sha256(executable)
    if not os.environ.get('CCX_COMMAND') and current==build['base_solver_sha256']:
        executable=str(fixed)
    if sha256(executable)!=build['solver_sha256']:
        raise ValueError('This executable has not been verified for TABULAR face-to-face contact. See docs/contact-fix.md; stock 2.23 is known to fail.')
    return executable


def generate(case, directory, solver='calculix'):
    if solver != 'calculix': raise ValueError('Only CalculiX is supported')
    directory=Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f'Refusing to overwrite existing run: {directory}')
    directory.mkdir(parents=True,exist_ok=True)
    case.save(directory/'case.json')
    write_json(directory/'status.json',{'state':'building'})
    try:
        model=build_model(case)
        model.save(directory/'model.json')
        model.export_vtu(directory/'mesh.vtu')
        from .preview import render_model
        render_model(case,model,directory/'geometry.png')
        write_deck(case,model,directory,solver)
        manifest={'case':case.name,'solver':solver,'created_utc':datetime.now(timezone.utc).isoformat(),
                  'python':platform.python_version(),'platform':platform.platform(),
                  'units':{'length':'mm','force':'N','stress':'MPa'},'mesh':model.metadata,
                  'stages':[name for name,_ in steps(case)],'validation_status':'unvalidated',
                  'files':{n:sha256(directory/n) for n in ['case.json','model.json','mesh.inp','job.inp']}}
        if (ROOT/'uv.lock').exists():manifest['uv_lock_sha256']=sha256(ROOT/'uv.lock')
        manifest['source_hashes']={str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'src/pipe_sim').glob('*.py')}
        write_json(directory/'manifest.json',manifest)
        write_json(directory/'status.json',{'state':'generated'})
        return directory
    except Exception as e:
        write_json(directory/'status.json',{'state':'build_failed','error':str(e)})
        raise


def run(directory,memory_limit_mb=None):
    from .config import Case
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['solver'] != 'calculix': raise ValueError('Only CalculiX is supported')
    for name, digest in manifest['files'].items():
        if sha256(directory/name)!=digest:raise ValueError(f'Run input changed after generation: {name}')
    state=json.loads((directory/'status.json').read_text())['state']
    if state!='generated':raise ValueError(f'Run state is {state}; generate a new directory to rerun')
    case=Case.load(directory/'case.json');solver=manifest['solver']
    exe=case_solver_path(case,solver)
    threading = solver_thread_settings(directory, case.analysis.threads)
    cmd=[exe,'-i','job']
    env=solver_environment(threading['solver_threads'])
    started=time.time()
    status={'state':'running','command':cmd,'started_utc':datetime.now(timezone.utc).isoformat()}
    status.update(runtime_budget(directory, case.analysis.timeout_seconds))
    status.update(threading)
    status['thread_environment'] = {key: env[key] for key in (
        'OMP_NUM_THREADS', 'CCX_NPROC_RESULTS', 'CCX_NPROC_STIFFNESS',
        'CCX_NPROC_EQUATION_SOLVER', 'MKL_NUM_THREADS', 'MKL_THREADING_LAYER', 'OPENBLAS_NUM_THREADS')}
    timeout_seconds = status['timeout_seconds']
    if memory_limit_mb is not None:status['memory_limit_mb']=memory_limit_mb
    if Path(exe).is_file():status['solver_sha256']=sha256(exe)
    write_json(directory/'status.json',status)
    with (directory/'solver.log').open('w') as log:
        process=subprocess.Popen(cmd,cwd=directory,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            while True:
                remaining=timeout_seconds-(time.time()-started)
                if remaining<=0:
                    status['stop_reason']='Runtime budget exceeded; not physical failure'
                    raise subprocess.TimeoutExpired(cmd,timeout_seconds)
                if memory_limit_mb is not None:
                    try:
                        fields=Path(f'/proc/{process.pid}/status').read_text().splitlines()
                        rss=next(float(line.split()[1])/1024 for line in fields if line.startswith('VmRSS:'))
                        status['peak_observed_rss_mb']=max(status.get('peak_observed_rss_mb',0.),rss)
                        if rss>memory_limit_mb:
                            status['stop_reason']='Memory guard exceeded; not physical failure'
                            raise subprocess.TimeoutExpired(cmd,timeout_seconds)
                    except (FileNotFoundError,StopIteration):pass
                try:
                    returncode=process.wait(timeout=min(5.,remaining));break
                except subprocess.TimeoutExpired:continue
        except (subprocess.TimeoutExpired,KeyboardInterrupt):
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            status.update(state='interrupted',elapsed_seconds=time.time()-started)
            status.setdefault('stop_reason','Time limit or interrupt; not physical failure')
            write_json(directory/'status.json',status)
            return status
    log=(directory/'solver.log').read_text(errors='replace')
    complete=returncode==0 and 'Job finished' in log and '*ERROR' not in log
    status.update(state='completed' if complete else 'failed',returncode=returncode,elapsed_seconds=time.time()-started)
    write_json(directory/'status.json',status)
    return status


def collect_external(directory):
    """Record returned native results from a remote workstation without rerunning it."""
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['solver'] != 'calculix': raise ValueError('Only CalculiX is supported')
    for name,digest in manifest['files'].items():
        if sha256(directory/name)!=digest:raise ValueError(f'Returned input differs from generated input: {name}')
    text=(directory/'solver.log').read_text(errors='replace')
    complete='Job finished' in text and '*ERROR' not in text
    result={'state':'completed' if complete else 'failed','execution':'external',
            'collected_utc':datetime.now(timezone.utc).isoformat(),
            'solver_version':'External installation; record its version separately with the returned files.'}
    write_json(directory/'status.json',result)
    return result
