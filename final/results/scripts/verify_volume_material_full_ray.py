"""Four bounded native full-ray probes of revised solver; no pressure acceptance."""
import gc,json,os,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scipy.sparse import coo_matrix,diags
from audit_plastic_return_map import F,sha256
from terminal_progress import solver_directories

def main():
    gate=F/'results/output/volume_material_full_ray_20261019.json'
    accepted=json.loads(gate.read_text());assert accepted['checks_passed']
    for name,h in accepted['sources'].items():assert sha256(F/name)==h,name
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    live=solver_directories(set(),run_root=F/'sym');assert live is not None and sum(len(v['pids']) for v in live.values())<2
    assert int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))>5*1024**2
    assert shutil.disk_usage(F).free>25*1024**3
    base_run=F/'sym/runs/investigation/volume_material_capture_v5_20261019'
    base=json.loads((base_run/'assembly.json').read_text());g=base['global_matrix'];n=g['neq']
    direction=np.array(json.loads((base_run/'full_direction.json').read_text()))
    rows=np.array(g['irow'])-1;cols=np.repeat(np.arange(n),np.diff(g['jq']))
    matrix=coo_matrix((g['au'],(rows,cols)),shape=(n,n))+coo_matrix((g['au_upper'],(cols,rows)),shape=(n,n))+diags(g['ad'])
    predicted=matrix@direction
    exe=F/'sym/vendor/ccx_volume_material_v5_20261019/ccx_2.23_material'
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    root=F/'sym/runs/investigation/volume_material_full_ray_fd_20261019';root.mkdir(exist_ok=False)
    source=Path(__file__).with_name('capture_pipe_solver_gdb.py');code=source.read_text()
    # Observe fixed old history across actual results evaluation. Current state is
    # intentionally recomputed at the imposed displacement, not restored here.
    anchor="snapshot['ips']=ips"
    code=code.replace(anchor,"""assert read(p[63],snapshot['state_count'],'d')==snapshot['previous_state']
        assert read(p[74],snapshot['stress_count'],'d')==snapshot['previous_stress']
        snapshot['fixed_old_history_unchanged']=True
        snapshot['ips']=ips""")
    code=code.replace('(7,10,29,33,34,65)','(7,10,29,33,34,63,65,74)')
    assert code!=source.read_text()
    capture=root/'capture_native.py';capture.write_text(code)
    report=dict(state='running',checks_passed=False,kind='bounded_native_diagnostic',supervisor_pid=os.getpid(),cases=[],comparisons=[],relative_limit=2e-7,sources={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),source,capture,exe,exe.parent/'build.json',gate)},limitations='Four fixed-history full-ray probes. Old-history immutability is not rejected-trial rollback. Separate follower derivative and explicit residual-evaluation rollback remain unaccepted. No pressure continuation, convergence benefit or physical burst acceptance.')
    out=F/'results/output/volume_material_full_ray_fd_20261019.json'
    def save():
        out.write_text(json.dumps(report,indent=2)+'\n')
        (root/'status.json').write_text(json.dumps({k:report[k] for k in ('state','kind','supervisor_pid')})+'\n')
    save();vectors={};external={}
    try:
        for h in (1e-6,-1e-6,5e-7,-5e-7):
            assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
            probe=root/f'h{h}';probe.mkdir()
            for name in ('job.inp','mesh.inp'):shutil.copy2(base_run/name,probe/name)
            shutil.copy2(base_run/'full_direction.json',probe/'direction.json')
            (probe/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_VOLUME_MATERIAL='1',THESIS_MIN_TIME='.75049',THESIS_OUTPUT=str(probe/'assembly.json'),THESIS_DIRECTION_FILE=str(probe/'direction.json'),THESIS_AMPLITUDE=str(h))
            with (probe/'solver.log').open('w') as log:subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=probe,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            d=json.loads((probe/'assembly.json').read_text())
            for k in ('previous_state','previous_stress','displacement_before','mapping','time','ttime','coordinates','ipkon','connectivity','cycle'):assert d[k]==base[k],k
            assert d['fixed_old_history_unchanged']
            assert d['displacement_imposed']==d['displacement_after']==d['residual']['displacement']
            np.testing.assert_array_equal(d['residual']['b'],np.array(d['residual']['fext'])-d['internal'])
            vectors[h]=np.array(d['internal']);external[h]=np.array(d['residual']['fext'])
            report['cases'].append(dict(run=str(probe.relative_to(F)),h=h,assembly_sha256=sha256(probe/'assembly.json'),fixed_old_history_unchanged=True));save();del d;gc.collect()
        previous=None
        for h in (1e-6,5e-7):
            fd=(vectors[h]-vectors[-h])/(2*h);ext=(external[h]-external[-h])/(2*h)
            error=float(np.linalg.norm(fd-predicted)/np.linalg.norm(fd))
            change=None if previous is None else float(np.linalg.norm(fd-previous)/np.linalg.norm(fd))
            report['comparisons'].append(dict(h=h,native_internal_relative_error=error,step_change=change,follower_norm=float(np.linalg.norm(ext)),passed=error<2e-7 and (change is None or change<2e-7)));previous=fd
        report.update(state='completed',checks_passed=all(c['passed'] for c in report['comparisons']))
        save();assert report['checks_passed'],'Preserved full-ray native FD failure; unchanged gates.'
    except Exception as exc:report.update(state='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:report['finished_utc']=datetime.now(timezone.utc).isoformat();save()
if __name__=='__main__':main()
