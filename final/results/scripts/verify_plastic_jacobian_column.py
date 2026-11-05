"""Replay one immutable coupon to a later assembly; measure fixed-history derivative."""
import json,os,sys,subprocess
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256

def main(tag):
    parent=Path(os.environ.get('THESIS_COUPON_PARENT',str(F/'sym/runs/investigation/material_tangent_20261017T040006Z')))
    gate=Path(os.environ.get('THESIS_COUPON_GATE',str(F/'results/output/material_tangent_20261017T040006Z.json')))
    reference=json.loads(gate.read_text())
    for n,h in reference['hashes'].items():assert sha256(parent/n)==h
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    root=F/'sym/runs/investigation'/('plastic_jacobian_'+tag);root.mkdir(exist_ok=False)
    capture=Path(__file__).with_name('capture_plastic_solver_gdb.py');source=F/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    cases=[];comparisons=[]
    for cycle in (30,60):
        states={};direction=None
        for h in (0.,1e-6,-1e-6,5e-7,-5e-7):
            assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
            run=root/f'c{cycle}_h{h}';run.mkdir();(run/'job.inp').write_bytes((parent/'job.inp').read_bytes())
            (run/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
            if direction is None:direction=np.random.default_rng(int(os.environ.get('THESIS_DIRECTION_SEED','1741'))).normal(size=24);direction/=np.linalg.norm(direction)
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_CYCLE=str(cycle),THESIS_DIRECTION=json.dumps(direction.tolist()),THESIS_AMPLITUDE=str(h),THESIS_OUTPUT=str(run/'assembly.json'),THESIS_ELEMENT_SOURCE=str(source))
            with (run/'solver.log').open('w') as out:r=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=run,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120)
            assert r.returncode==0,str(run)
            data=json.loads((run/'assembly.json').read_text());states[h]=data
            np.testing.assert_allclose(data['displacement_after'],data['displacement_imposed'],rtol=0,atol=1e-15)
            assert data['displacement_after']==data['assembly_displacement']
            assert data['stress']==data['assembly_stress']
            assert sorted(ip['ip'] for ip in data['ips'])==list(range(1,9))
            ns=data['state_count']//8
            for ip in data['ips']:assert ip['previous']==data['previous_state'][(ip['ip']-1)*ns:ip['ip']*ns]
            if h==0:
                direction=np.eye(data['neq'])[int(os.environ['THESIS_BASIS_COLUMN'])]
            else:
                base=states[0.]
                for key in ('previous_state','previous_stress','displacement_before','time','ttime','mapping'):assert data[key]==base[key],key
                assert [p['previous'] for p in data['ips']]==[p['previous'] for p in base['ips']]
            cases.append(dict(run=str(run.relative_to(F)),cycle=cycle,h=h,neq=data['neq'],time=data['time'],ttime=data['ttime'],previous_peeq=[p['previous'][0] for p in data['ips']],hashes={n:sha256(run/n) for n in ('job.inp','capture.gdb','assembly.json','solver.log')}))
        base=states[0.];mapping=np.array(base['mapping']).reshape(8,4)[:,1:].ravel();s=np.array(base['element_matrix']).reshape(60,60,order='F')[:24,:24];s=np.triu(s)+np.triu(s,1).T
        k=np.zeros((base['neq'],base['neq']))
        for i,a in enumerate(mapping):
            for j,b in enumerate(mapping):
                if a>0 and b>0:k[a-1,b-1]+=s[i,j]
        g=base['global_matrix'];assert g['symmetryflag']==0 and g['inputformat']==0
        assert g['jq'][0]==1 and g['jq'][-1]==g['nzs']+1
        solver_k=np.diag(g['ad'])
        for column in range(g['neq']):
            for offset in range(g['jq'][column]-1,g['jq'][column+1]-1):
                row=g['irow'][offset]-1
                assert column<row<g['neq']
                solver_k[row,column]=g['au'][offset];solver_k[column,row]=g['au'][offset]
        matrix_error=float(np.max(np.abs(k-solver_k)))
        np.testing.assert_allclose(k,solver_k,rtol=2e-14,atol=1e-10)
        for data in states.values():
            r=data['residual'];assert r['nmethod']==1
            assert r['displacement']==data['assembly_displacement']
            assert r['internal']==data['internal']
            assert np.max(np.abs(r['fext']))==0
            np.testing.assert_array_equal(r['b'],-np.array(r['internal']))
            np.testing.assert_array_equal(data['global_matrix']['rhs'],r['b'])
        comparisons.append(dict(cycle=cycle,kind='actual_solver_matrix_and_residual',matrix_max_absolute_error=matrix_error,matrix_relative_error=float(np.linalg.norm(k-solver_k)/np.linalg.norm(solver_k)),external_force_max=0,residual_identity_exact=True))
        prev=None
        for h in (1e-6,5e-7):
            derivative=(np.array(states[h]['internal'])-states[-h]['internal'])/(2*h);expected=k@direction
            comparisons.append(dict(cycle=cycle,h=h,relative_error=float(np.linalg.norm(derivative-expected)/np.linalg.norm(expected)),derivative=derivative.tolist(),matrix_direction=expected.tolist(),step_change=None if prev is None else float(np.linalg.norm(derivative-prev)/np.linalg.norm(derivative))))
            prev=derivative
    report=dict(basis_column=int(os.environ['THESIS_BASIS_COLUMN']),direction_seed=int(os.environ.get('THESIS_DIRECTION_SEED','1741')),cases=cases,comparisons=comparisons,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,source,source.with_name('results.c'),exe,gate)},limitations='Native single C3D8 coupon; same prior history all8IPs, free displacement direction before results, projected internal force and same-cycle element tangent. Actual PARDISO-entry matrix and calcresidual capture; zero external loads, residual identity checked. No pressure/contact/MPC/cap/pipe/Abaqus/energy/experimental acceptance. Deliberately stopped before selected factorization; not completed solve.')
    output=F/'results/output'/('plastic_jacobian_'+tag+'.json');assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main(sys.argv[1])
