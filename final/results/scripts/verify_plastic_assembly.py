"""Replay one immutable coupon to a later assembly; measure fixed-history derivative."""
import json,os,sys,subprocess
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256

def main(tag):
    parent=F/'sym/runs/investigation/material_tangent_20261017T040006Z'
    gate=F/'results/output/material_tangent_20261017T040006Z.json'
    reference=json.loads(gate.read_text())
    for n,h in reference['hashes'].items():assert sha256(parent/n)==h
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    root=F/'sym/runs/investigation'/('plastic_assembly_'+tag);root.mkdir(exist_ok=False)
    capture=Path(__file__).with_name('capture_plastic_assembly_gdb.py');source=F/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    cases=[];comparisons=[]
    for cycle in (30,60):
        states={};direction=None
        for h in (0.,1e-6,-1e-6,5e-7,-5e-7):
            assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
            run=root/f'c{cycle}_h{h}';run.mkdir();(run/'job.inp').write_bytes((parent/'job.inp').read_bytes())
            (run/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
            if direction is None:direction=np.random.default_rng(1741).normal(size=24);direction/=np.linalg.norm(direction)
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_CYCLE=str(cycle),THESIS_DIRECTION=json.dumps(direction.tolist()),THESIS_AMPLITUDE=str(h),THESIS_OUTPUT=str(run/'assembly.json'),THESIS_ELEMENT_SOURCE=str(source))
            with (run/'solver.log').open('w') as out:r=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=run,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120)
            assert r.returncode==0,str(run)
            data=json.loads((run/'assembly.json').read_text());states[h]=data
            np.testing.assert_allclose(data['displacement_after'],data['displacement_imposed'],rtol=0,atol=1e-15)
            assert data['displacement_after']==data['assembly_displacement']
            assert data['stress']==data['assembly_stress']
            assert sorted(ip['ip'] for ip in data['ips'])==list(range(1,9))
            if h==0:
                direction=np.random.default_rng(1741).normal(size=data['neq']);direction/=np.linalg.norm(direction)
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
        prev=None
        for h in (1e-6,5e-7):
            derivative=(np.array(states[h]['internal'])-states[-h]['internal'])/(2*h);expected=k@direction
            comparisons.append(dict(cycle=cycle,h=h,relative_error=float(np.linalg.norm(derivative-expected)/np.linalg.norm(expected)),derivative=derivative.tolist(),matrix_direction=expected.tolist(),step_change=None if prev is None else float(np.linalg.norm(derivative-prev)/np.linalg.norm(derivative))))
            prev=derivative
    report=dict(cases=cases,comparisons=comparisons,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,source,source.with_name('results.c'),exe,gate)},limitations='Native single C3D8 coupon; same prior history all8IPs, free displacement direction before results, projected internal force and same-cycle element tangent. Zero external loads; no global PARDISO matrix capture, pressure/contact/MPC/cap/pipe/Abaqus/energy/experimental acceptance. Deliberately stopped after selected assembly; not completed solve.')
    output=F/'results/output'/('plastic_assembly_'+tag+'.json');assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main(sys.argv[1])
