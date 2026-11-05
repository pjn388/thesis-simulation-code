"""First sparse actual-pipe matrix/residual ABI gate; no correction/FD claim."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256

def main(tag,minimum_time=None,early_plastic=False):
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    gate=F/'results/output/plastic_pipe_replay_20261011T101242Z.json';parent=json.loads(gate.read_text());assert parent['checks_passed']
    source_run=F/parent['run']
    for n,h in parent['input_output_hashes'].items():assert sha256(source_run/n)==h
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso';assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    root=F/'sym/runs/investigation'/('pipe_solver_'+tag);root.mkdir(exist_ok=False)
    (root/'job.inp').write_bytes((source_run/'job.inp').read_bytes())
    (root/'mesh.inp').write_bytes((source_run/'mesh.inp').read_bytes())
    assert sha256(root/'mesh.inp')==sha256(source_run/'mesh.inp')
    capture=Path(__file__).with_name('capture_pipe_solver_gdb.py')
    (root/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_CYCLE='2',THESIS_OUTPUT=str(root/'assembly.json'))
    if minimum_time is not None:env['THESIS_MIN_TIME']=str(minimum_time)
    with (root/'solver.log').open('w') as out:r=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=600)
    assert r.returncode==0,str(root)
    data=json.loads((root/'assembly.json').read_text());g=data['global_matrix'];res=data['residual'];n=data['neq']
    assert data['nmpc']==0 and data['displacement_after']==data['displacement_imposed']==res['displacement']
    assert res['internal']==data['internal'];np.testing.assert_array_equal(res['b'],np.array(res['fext'])-res['internal']);assert g['rhs']==res['b']
    assert g['symmetryflag']==0 and g['inputformat']==0 and g['jq'][0]==1 and g['jq'][-1]==g['nzs']+1
    assert len(g['ad'])==n and len(g['au'])==len(g['irow'])==g['nzs']
    for col in range(n):
        rows=g['irow'][g['jq'][col]-1:g['jq'][col+1]-1];assert rows==sorted(set(rows)) and all(col+1<v<=n for v in rows)
    for key in ('ad','au','rhs'):assert np.isfinite(g[key]).all()
    prior=np.array(data['previous_state']).reshape(data['ne'],8,data['nstate'])[:,:,0]
    prior_gate=1e-8 if early_plastic else 1e-3
    assert not early_plastic or minimum_time is not None
    if minimum_time is not None:
        assert data['time']>=float(minimum_time) and prior.max()>prior_gate
    for ip in data['ips']:
        expected=np.array(data['previous_state']).reshape(data['ne'],8,data['nstate'])[ip['element']-1,ip['ip']-1]
        np.testing.assert_array_equal(ip['previous'],expected)
    if early_plastic:
        assert any(ip['previous'][0]>prior_gate for ip in data['ips']), 'Selected IPs must include positive plastic history'
    report=dict(checks_passed=True,run=str(root.relative_to(F)),parent_run=parent['run'],cycle=data['cycle'],time=data['time'],ttime=data['ttime'],nk=data['nk'],ne=data['ne'],neq=n,nzs=g['nzs'],nmpc=data['nmpc'],max_external_force=float(max(map(abs,res['fext']))),residual_norm=float(np.linalg.norm(res['b'])),hashes={p.name:sha256(p) for p in (root/'job.inp',root/'mesh.inp',root/'capture.gdb',root/'assembly.json',root/'solver.log')},provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,exe,gate,F/'sym/vendor/CalculiX/ccx_2.23/src/results.c',F/'sym/vendor/CalculiX/ccx_2.23/src/calcresidual.c',F/'sym/vendor/CalculiX/ccx_2.23/src/pardiso.c')},limitations='Unperturbed early iteration only. Sparse structure and residual ABI identities, not element assembly/FD/correction acceptance. Pressure derivative, nonuniform IP mapping and plastic iteration selection pending. No production change or convergence/burst/experimental validation.')
    report.update(selection_minimum_time=minimum_time,early_plastic_selection=early_plastic,selection_prior_peeq_gate=prior_gate,maximum_prior_peeq=float(prior.max()),plastic_prior_ip_count=int((prior>0).sum()),captured_ip_count=len(data['ips']),captured_maximum_prior_peeq=max((ip['previous'][0] for ip in data['ips']),default=0))
    path=F/'results/output'/('pipe_solver_'+tag+'.json');assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main(sys.argv[1],float(sys.argv[2]) if len(sys.argv)>2 else None,'--early-plastic' in sys.argv[3:])
