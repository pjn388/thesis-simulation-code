"""One late fixed-history central-difference verification; no solver revision."""
import gc,json,os,shutil,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256
from audit_pipe_assembled_action import assemble_actions
from verify_pipe_direction import action
from terminal_progress import solver_directories


def main():
    gate=F/'results/output/late_intact_components_native_table_20261019.json'
    accepted=json.loads(gate.read_text());assert accepted['checks_passed']
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    live=solver_directories(set(),run_root=F/'sym');assert live is not None and sum(len(v['pids']) for v in live.values())<2
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))
    assert available>5*1024**2 and shutil.disk_usage(F).free>25*1024**3
    run=F/'sym/runs/investigation/late_local_inputs_20261019'
    parent=F/'sym/runs/campaign/orientation_intact_00_adaptive'
    manifest=json.loads((parent/'manifest.json').read_text())
    for n,h in manifest['files'].items():assert sha256(parent/n)==h and sha256(run/n)==h
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    base=json.loads((run/'assembly.json').read_text());mapping=np.array(base['mapping']).reshape(-1,4)[:,1:]
    # Exactly the localized largest-correction node used by the passed retained audit.
    node=accepted['selected_node']-1;indices=mapping[node];indices=indices[indices>0]-1
    from scipy.sparse import coo_matrix,diags
    from scipy.sparse.linalg import spsolve
    g=base['global_matrix'];rows=np.array(g['irow'])-1;cols=np.repeat(np.arange(g['neq']),np.diff(g['jq']))
    off=coo_matrix((g['au'],(rows,cols)),shape=(g['neq'],g['neq'])).tocsc()
    correction=spsolve(off+off.T+diags(g['ad']),g['rhs']);direction=np.zeros(g['neq']);direction[indices]=correction[indices];direction/=np.linalg.norm(direction)
    table=np.array(base['ips'][0]['plconloc'][:400]).reshape(-1,2)[:,::-1];deck=(run/'job.inp').read_text()
    exact,surrogate,external,coverage=assemble_actions(base,direction,table,208000,.3,deck)
    duration=float(deck.split('*STATIC, SOLVER=PARDISO\n')[1].splitlines()[0].split(',')[1]);external*=2.3/duration
    predicted=exact-external;native=action(g,direction)
    assert coverage['maximum_stress_replay_error_mpa']<1e-4 and np.linalg.norm(surrogate-native)/np.linalg.norm(native)<2e-7
    root=F/'sym/runs/investigation/late_native_action_20261019';root.mkdir(exist_ok=False)
    state=dict(state='running',kind='bounded_native_diagnostic',supervisor_pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),probe_count=4,probe_timeout_seconds=600,physical_validation=False)
    def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    save();cases=[];vectors={}
    try:
        capture=Path(__file__).with_name('capture_pipe_solver_gdb.py')
        for h in (1e-6,-1e-6,5e-7,-5e-7):
            assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
            probe=root/f'h{h}';probe.mkdir()
            for n in manifest['files']:shutil.copy2(parent/n,probe/n)
            (probe/'direction.json').write_text(json.dumps(direction.tolist()))
            (probe/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_CYCLE=str(base['cycle']),THESIS_OUTPUT=str(probe/'assembly.json'),THESIS_DIRECTION_FILE=str(probe/'direction.json'),THESIS_AMPLITUDE=str(h))
            with (probe/'solver.log').open('w') as log:subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=probe,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True)
            d=json.loads((probe/'assembly.json').read_text())
            for k in ('previous_state','previous_stress','displacement_before','mapping','time','ttime','coordinates','ipkon','connectivity'):assert d[k]==base[k],k
            assert d['displacement_imposed']==d['displacement_after']==d['residual']['displacement']
            np.testing.assert_array_equal(d['residual']['b'],np.array(d['residual']['fext'])-d['internal'])
            vectors[h]=np.array(d['internal'])-d['residual']['fext']
            cases.append(dict(run=str(probe.relative_to(F)),h=h,assembly_sha256=sha256(probe/'assembly.json')))
            state['completed_probes']=len(cases);save();del d;gc.collect()
        comparisons=[];previous=None
        for h in (1e-6,5e-7):
            fd=(vectors[h]-vectors[-h])/(2*h);denom=np.linalg.norm(fd)
            err=float(np.linalg.norm(fd-predicted)/denom);change=None if previous is None else float(np.linalg.norm(fd-previous)/denom)
            comparisons.append(dict(h=h,corrected_net_relative_error=err,original_native_relative_error=float(np.linalg.norm(fd-native)/denom),step_change=change,passed=err<2e-7 and (change is None or change<2e-7)));previous=fd
        passed=all(r['passed'] for r in comparisons)
        report=dict(checks_passed=passed,relative_limit=2e-7,cases=cases,comparisons=comparisons,coverage=coverage,time=base['time'],cycle=base['cycle'],sources={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,gate,run/'assembly.json',exe)},limitations='One localized late first-iteration direction with fixed history. No global convergence causation, terminal-state stability, production correction or physical burst acceptance.')
        with (F/'results/output/late_native_action_20261019.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
        assert passed,'Preserved failed finite-difference gate; no relaxation.'
        state['state']='completed'
    except Exception as exc:state.update(state='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:state['finished_utc']=datetime.now(timezone.utc).isoformat();save()

if __name__=='__main__':main()
