"""One guarded capture at the intact parent's first previously unaccepted load.

No perturbation sweep and no solver/physics revision. Stop before factorization.
"""
import json, os, shutil, subprocess
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F, sha256
from terminal_progress import solver_directories

def main():
    root=F/'sym/runs/investigation/late_local_inputs_20261019'
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    live=solver_directories(set(),run_root=F/'sym');assert live is not None and sum(len(v['pids']) for v in live.values())<2
    assert int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))>5*1024**2
    assert shutil.disk_usage(F).free>25*1024**3
    parent=F/'sym/runs/campaign/orientation_intact_00_adaptive'
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    manifest=json.loads((parent/'manifest.json').read_text())
    for name,digest in manifest['files'].items():assert sha256(parent/name)==digest
    root.mkdir(exist_ok=False)
    state=dict(state='running',kind='bounded_native_diagnostic',supervisor_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),parent=str(parent.relative_to(F)),minimum_time=.75049,timeout_seconds=900,physical_validation=False)
    def save(): (root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    try:
        for name in manifest['files']:shutil.copy2(parent/name,root/name)
        capture=Path(__file__).with_name('capture_late_local_gdb.py')
        (root/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_MIN_TIME='.75049',THESIS_OUTPUT=str(root/'assembly.json'))
        with (root/'solver.log').open('w') as log:
            subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900,check=True)
        d=json.loads((root/'assembly.json').read_text());g=d['global_matrix'];r=d['residual']
        assert len(d['ips'])==16 and all('current' in ip for ip in d['ips'])
        assert d['time']>=.75049 and d['nmpc']==0
        np.testing.assert_array_equal(r['b'],np.array(r['fext'])-r['internal'])
        np.testing.assert_array_equal(g['rhs'],r['b'])
        assert all(np.isfinite(g[k]).all() for k in ('ad','au','rhs'))
        report=dict(checks_passed=True,run=str(root.relative_to(F)),time=d['time'],cycle=d['cycle'],neq=d['neq'],residual_norm=float(np.linalg.norm(r['b'])),maximum_prior_peeq=float(np.array(d['previous_state']).reshape(d['ne'],8,d['nstate'])[:,:,0].max()),sources={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,exe,root/'job.inp',root/'mesh.inp',root/'assembly.json')},next='Compare native local inputs and returned PEEQ/stress against retained scalar replay. Binary has no debug locals: derive branch and scalar exit only after input identity passes; no tolerance change authorized.',limitations='ABI/residual capture only. No tangent correction, stability, material/fixture validation, burst or Abaqus acceptance.')
        (F/'results/output/late_local_inputs_20261019.json').write_text(json.dumps(report,indent=2)+'\n')
        state['state']='completed'
    except Exception as exc:
        state.update(state='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:
        state['finished_utc']=datetime.now(timezone.utc).isoformat();save()
if __name__=='__main__': main()
