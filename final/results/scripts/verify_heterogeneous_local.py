"""Compare native local stress perturbations to independent analytic derivatives."""
import json,os,sys,subprocess
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256,tensors
from audit_material_linearization import derivative,vector

def main(tag):
    parent=F/'sym/runs/investigation/heterogeneous_coupon_v2_20261017T043331Z'
    gate=F/'results/output/heterogeneous_coupon_v2_20261017T043331Z.json'
    for n,h in json.loads(gate.read_text())['hashes'].items():assert sha256(parent/n)==h
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso';capture=Path(__file__).with_name('capture_heterogeneous_local_gdb.py')
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    root=F/'sym/runs/investigation'/('heterogeneous_local_'+tag);root.mkdir(exist_ok=False)
    lines=(parent/'job.inp').read_text().splitlines();curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve);cases=[];comparisons=[]
    for cycle in (30,60):
        retained=json.loads((F/f'sym/runs/investigation/plastic_jacobian_v2_20261017T043331Z_b0/c{cycle}_h0.0/assembly.json').read_text())
        for ip in (1,6):
            previous=None;states={}
            probes=[(-1,0.,0)]+[(col,h,s) for col in range(6) for h in (1e-7,5e-8) for s in (-1,1)]
            for col,h,sign in probes:
                assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
                run=root/f'c{cycle}_ip{ip}_d{col}_h{h}_s{sign}';run.mkdir();(run/'job.inp').write_bytes((parent/'job.inp').read_bytes())
                (run/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
                direction=np.zeros(6)
                if col>=0:direction[col]=1 if col<3 else .5
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_CYCLE=str(cycle),THESIS_IP=str(ip),THESIS_DIRECTION=json.dumps(direction.tolist()),THESIS_AMPLITUDE=str(h*sign),THESIS_OUTPUT=str(run/'local.json'))
                with (run/'solver.log').open('w') as log:r=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=60)
                assert r.returncode==0
                records=json.loads((run/'local.json').read_text())['records'];last=records[-1]
                if previous is None:
                    previous=records;reference=last
                    old=next(p for p in retained['ips'] if p['ip']==ip)
                    for a,b in (('previous','previous'),('strain','strain'),('stress','stress'),('stiffness','stiffness')):assert last[a]==old[b]
                else:
                    assert records[:-1]==previous[:-1]
                    for key in ('previous','strain_before','elastic','cycle','time','ttime','icmd'):assert last[key]==reference[key]
                    np.testing.assert_array_equal(last['strain'],np.array(reference['strain'])+h*sign*direction)
                    states[col,h,sign]=np.array(last['stress'])
                cases.append(dict(run=str(run.relative_to(F)),cycle=cycle,ip=ip,col=col,h=h,sign=sign,prior_calls_identical=len(records)-1,hashes={n:sha256(run/n) for n in ('job.inp','capture.gdb','local.json','solver.log')}))
            e=np.array(reference['strain']);prev=np.array(reference['previous']);c=np.eye(3)+2*tensors(e[None])[0];m=np.eye(3)-2*tensors(prev[None,1:7])[0]
            young,nu=reference['elastic'];mu=young/(2*(1+nu));bulk=young/(3*(1-2*nu));exact=[];native=np.zeros((6,6))
            for (i,j),v in zip([(i,j) for j in range(6) for i in range(j+1)],reference['stiffness']):native[i,j]=native[j,i]=v
            for col in range(6):
                d=np.eye(6)[col]/(1 if col<3 else 2)
                exact.append(vector(derivative(c,m,prev[0],curve,mu,bulk,2*tensors(d[None])[0])))
            exact=np.array(exact).T;lastfd=None
            for h in (1e-7,5e-8):
                fd=np.array([(states[col,h,1]-states[col,h,-1])/(2*h) for col in range(6)]).T
                comparisons.append(dict(cycle=cycle,ip=ip,h=h,previous_peeq=prev[0],analytic_native_fd_relative=float(np.linalg.norm(exact-fd)/np.linalg.norm(fd)),native_tangent_fd_relative=float(np.linalg.norm(native-fd)/np.linalg.norm(fd)),step_change=None if lastfd is None else float(np.linalg.norm(fd-lastfd)/np.linalg.norm(fd)),native_fd_matrix=fd.tolist(),analytic_matrix=exact.tolist(),native_matrix=native.tolist()))
                lastfd=fd
    result=dict(cases=cases,comparisons=comparisons,limits=dict(relative=2e-7),checks_passed=all(r['analytic_native_fd_relative']<2e-7 for r in comparisons),provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,gate,exe,Path(__file__).with_name('audit_material_linearization.py'),Path(__file__).with_name('audit_plastic_return_map.py'))},limitations='Native local derivatives at two IPs from exact same preceding calls and prior state. Probes intentionally stop after local return. No equilibrium, pipe causality or production acceptance.')
    out=F/'results/output'/('heterogeneous_local_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if 'matrix' not in k} for r in comparisons],indent=2),flush=True);assert result['checks_passed']
if __name__=='__main__':main(sys.argv[1])
