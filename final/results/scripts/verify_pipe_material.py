"""Native local derivative screen at retained C3D20R pipe endpoint histories.

Standalone injected material calls, not native pipe iteration/assembly acceptance.
"""
import json,os,re,subprocess,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,tensors,sha256,update
from audit_material_linearization import derivative,vector

def main(tag):
    gate=F/'results/output/plastic_pipe_replay_20261011T101242Z.json';evidence=json.loads(gate.read_text());assert evidence['checks_passed']
    run=F/evidence['run']
    for n,h in evidence['input_output_hashes'].items():assert sha256(run/n)==h
    source=F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f';assert sha256(source)==evidence['source_sha256']
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso';assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    root=F/'sym/runs/investigation'/('pipe_material_'+tag);root.mkdir(exist_ok=False)
    raw=np.memmap(run/'fields.bin',dtype='<f8',mode='r');records=json.loads((run/'fields.json').read_text())['records']
    def field(rec,name,width):
        m=rec['fields'][name];return raw[m['offset']//8:m['offset']//8+m['count']].reshape((rec['elements'],rec['ips'],width))
    final=records[-1];end=field(final,'state',final['states']);element,ip=np.unravel_index(end[...,0].argmax(),end[...,0].shape)
    assert final['ips']==8 and end[element,ip,0]>.01
    zero=np.zeros(final['states']);previous=field(records[-2],'state',final['states'])[element,ip].copy()
    strain=field(final,'strain',6)[element,ip].copy();delta=strain-field(records[-2],'strain',6)[element,ip]
    selections=[dict(name='elastic',frame=0,previous=zero.tolist(),strain=field(records[0],'strain',6)[element,ip].tolist()),dict(name='plastic',frame=len(records)-1,previous=previous.tolist(),strain=strain.tolist()),dict(name='synthetic_unloading',frame=len(records)-1,previous=end[element,ip].tolist(),strain=(strain-.5*delta).tolist())]
    material=json.loads((run/'case.json').read_text())['material'];young,nu=material['young'],material['poisson'];mu=young/(2*(1+nu));bulk=young/(3*(1-2*nu))
    lines=(run/'job.inp').read_text().splitlines();curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve)
    coupon=F/'sym/runs/investigation/heterogeneous_coupon_v2_20261017T043331Z/job.inp'
    deck=coupon.read_text();deck=re.sub(r'(?m)^202500.0,0.3$',f'{young},{nu}',deck)
    deck=re.sub(r'\*PLASTIC\n[^*]*','*PLASTIC\n'+''.join(f'{a:.17g},{b:.17g}\n' for a,b in curve),deck)
    capture=Path(__file__).with_name('capture_pipe_material_gdb.py');cases=[];comparisons=[]
    for selection in selections:
        states={};reference=None
        for col,h,sign in [(-1,0.,0)]+[(col,h,s) for col in range(6) for h in (1e-7,5e-8) for s in (-1,1)]:
            assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
            probe=root/f"{selection['name']}_d{col}_h{h}_s{sign}";probe.mkdir();(probe/'job.inp').write_text(deck)
            inp=dict(selection);direction=np.zeros(6)
            if col>=0:direction[col]=1 if col<3 else .5
            inp['strain']=(np.array(selection['strain'])+h*sign*direction).tolist();(probe/'state.json').write_text(json.dumps(inp,indent=2)+'\n')
            (probe/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_STATE=str(probe/'state.json'),THESIS_OUTPUT=str(probe/'local.json'))
            with (probe/'solver.log').open('w') as log:r=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=probe,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=30)
            assert r.returncode==0
            local=json.loads((probe/'local.json').read_text());assert local['previous']==selection['previous'] and local['strain']==inp['strain'] and local['elastic']==[young,nu]
            if col<0:reference=local
            else:states[col,h,sign]=np.array(local['stress'])
            cases.append(dict(run=str(probe.relative_to(F)),selection=selection['name'],col=col,h=h,sign=sign,hashes={n:sha256(probe/n) for n in ('job.inp','state.json','capture.gdb','local.json','solver.log')}))
        prev=np.array(selection['previous']);c=np.eye(3)+2*tensors(np.array(selection['strain'])[None])[0];m=np.eye(3)-2*tensors(prev[None,1:7])[0]
        predicted,_,qp,active=update(c,m,np.array(prev[0]),curve,mu,bulk);stress_error=float(np.max(np.abs(vector(predicted)-reference['stress'])))
        exact=np.array([vector(derivative(c,m,prev[0],curve,mu,bulk,2*tensors((np.eye(6)[col]/(1 if col<3 else 2))[None])[0])) for col in range(6)]).T
        native=np.zeros((6,6))
        for (i,j),v in zip([(i,j) for j in range(6) for i in range(j+1)],reference['stiffness']):native[i,j]=native[j,i]=v
        assert bool(active)==(selection['name']=='plastic')
        last=None
        for h in (1e-7,5e-8):
            fd=np.array([(states[col,h,1]-states[col,h,-1])/(2*h) for col in range(6)]).T
            comparisons.append(dict(selection=selection['name'],element=int(element+1),ip=int(ip+1),frame=selection['frame'],previous_peeq=float(prev[0]),current_peeq=float(qp),active=bool(active),h=h,stress_error_mpa=stress_error,analytic_fd_relative=float(np.linalg.norm(exact-fd)/np.linalg.norm(fd)),native_fd_relative=float(np.linalg.norm(native-fd)/np.linalg.norm(fd)),step_change=None if last is None else float(np.linalg.norm(fd-last)/np.linalg.norm(fd)),native_matrix=native.tolist(),analytic_matrix=exact.tolist(),fd_matrix=fd.tolist()));last=fd
    passed=all(r['analytic_fd_relative']<2e-7 and r['stress_error_mpa']<1e-4 for r in comparisons)
    result=dict(checks_passed=passed,cases=cases,comparisons=comparisons,selections=selections,limits=dict(relative=2e-7,stress_mpa=1e-4),parent_run=str(run.relative_to(F)),parent_hashes=evidence['input_output_hashes'],provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,gate,source,exe,coupon,Path(__file__).with_name('audit_material_linearization.py'),Path(__file__).with_name('audit_plastic_return_map.py'))},limitations='Native standalone material calls with exact injected pipe endpoint previous histories and X52 curve; no replay of actual native pipe iterations. Synthetic unloading is explicitly constructed from final endpoint and reversed half last strain increment. No actual pipe stiffness/matrix/residual, convergence causality, substantial failure-state plasticity, cap/contact/MPC/energy/Abaqus or experimental acceptance.')
    out=F/'results/output'/('pipe_material_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if 'matrix' not in k} for r in comparisons],indent=2),flush=True);assert passed
if __name__=='__main__':main(sys.argv[1])
