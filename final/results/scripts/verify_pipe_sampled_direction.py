"""Sampled-element pipe fixed-history FD; unchanged global derivative gates.

Direction covers the captured elements1,2017; original sparse probe is retained.
"""
import argparse,json,os,subprocess,sys,gc
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256,tensors
from audit_internal_pipe_fields import shape,source_basis
from verify_heterogeneous_coupon import points

def mapping(data):
    xyz=np.array(data['coordinates']).reshape(-1,3);u=np.array(data['displacement_after']).reshape(-1,4)[:,1:]
    connect=np.array(data['connectivity']);nodes=np.array([connect[k:k+20] for k in data['ipkon']])-1
    native={(p['element'],p['ip']):p for p in data['ips']};assert len(native)==len(data['ips'])
    assert len(native) in (16,data['ne']*8)
    prior=np.array(data['previous_state']).reshape(data['ne'],8,data['nstate'])
    for p in data['ips']:assert p['previous']==prior[p['element']-1,p['ip']-1].tolist()
    selected=np.array([0,2016]);nodes=nodes[selected];maximum=0.;span=0.;allstrain=[]
    for ip,q in enumerate(points()):
        d=np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T
        _,source=source_basis(q);np.testing.assert_allclose(d,source,rtol=0,atol=1e-14)
        jac=np.einsum('eni,nj->eij',xyz[nodes],d);assert (np.linalg.det(jac)>0).all()
        grad=np.einsum('ni,eij->enj',d,np.linalg.inv(jac));deform=np.eye(3)+np.einsum('eni,enj->eij',u[nodes],grad)
        metric=np.einsum('eji,ejk->eik',deform,deform)
        strain=np.array([native[e+1,ip+1]['strain'] for e in selected]);allstrain.append(strain)
        target=np.eye(3)+2*tensors(strain);maximum=max(maximum,float(abs(metric-target).max()))
        np.testing.assert_allclose(metric,target,rtol=0,atol=2e-12)
    span=float(np.ptp(np.array(allstrain)[:,:,0]));assert span>1e-8
    return dict(maximum_metric_error=maximum,nonuniform_strain_span=span,elements=len(selected),ips=8,limit=2e-12)

def action(g,direction):
    result=np.array(g['ad'])*direction
    rows=np.array(g['irow'])-1;cols=np.repeat(np.arange(g['neq']),np.diff(g['jq']));values=np.array(g['au'])
    np.add.at(result,rows,values*direction[cols]);np.add.at(result,cols,values*direction[rows]);return result

def sampled_direction(base):
    direction=np.zeros(base['neq'])
    nodal_mapping=np.array(base['mapping']).reshape(-1,4)[:,1:]
    connect=np.array(base['connectivity']);selected=sorted(set(p['element'] for p in base['ips']))
    nodes=np.unique(np.concatenate([connect[base['ipkon'][e-1]:base['ipkon'][e-1]+20]-1 for e in selected]))
    indices=np.unique(nodal_mapping[nodes]);indices=indices[indices>0]-1
    assert len(indices)>0
    direction[indices]=np.random.default_rng(1741).normal(size=len(indices));direction/=np.linalg.norm(direction)
    return direction,selected,indices

def main(tag,step=1e-7,parent_gate='results/output/pipe_solver_v3_20261017T045902Z.json',probe_timeout=180):
    assert 0<step<1e-2 and 0<probe_timeout<=600
    gate=F/parent_gate;parent=json.loads(gate.read_text());assert parent['checks_passed']
    run=F/parent['run']
    for n,h in parent['hashes'].items():assert sha256(run/n)==h
    base=json.loads((run/'assembly.json').read_text());mapcheck=mapping(base)
    prior=np.array(base['previous_state']).reshape(base['ne'],8,base['nstate'])[:,:,0]
    if parent.get('early_plastic_selection'):
        assert prior.max()>1e-8 and any(p['previous'][0]>1e-8 for p in base['ips'])
        assert base['time']>=parent['selection_minimum_time']
    direction,selected,indices=sampled_direction(base)
    predicted=action(base['global_matrix'],direction)
    capture=Path(__file__).with_name('capture_pipe_solver_gdb.py');exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==parent['provenance'][str(exe.relative_to(F))]
    root=F/'sym/runs/investigation'/('pipe_direction_'+tag);root.mkdir(exist_ok=False)
    states={};cases=[]
    for h in (step,-step,step/2,-step/2):
        assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
        probe=root/f'h{h}';probe.mkdir()
        for name in ('job.inp','mesh.inp'):(probe/name).write_bytes((run/name).read_bytes())
        (probe/'direction.json').write_text(json.dumps(direction.tolist()))
        (probe/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_CYCLE=str(base['cycle']),THESIS_OUTPUT=str(probe/'assembly.json'),THESIS_DIRECTION_FILE=str(probe/'direction.json'),THESIS_AMPLITUDE=str(h))
        with (probe/'solver.log').open('w') as out:r=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=probe,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=probe_timeout)
        assert r.returncode==0,str(probe)
        data=json.loads((probe/'assembly.json').read_text())
        for key in ('previous_state','previous_stress','displacement_before','mapping','time','ttime','coordinates','ipkon','connectivity'):assert data[key]==base[key],key
        assert data['displacement_imposed']==data['displacement_after']==data['residual']['displacement']
        old={(p['element'],p['ip']):p['previous'] for p in base['ips']}
        assert len(data['ips'])==16
        for p in data['ips']:assert p['previous']==old[p['element'],p['ip']]
        mapping(data)
        assert data['residual']['internal']==data['internal'];np.testing.assert_array_equal(data['residual']['b'],np.array(data['residual']['fext'])-data['internal']);assert data['global_matrix']['rhs']==data['residual']['b']
        states[h]=(np.array(data['internal']),np.array(data['residual']['fext']))
        cases.append(dict(run=str(probe.relative_to(F)),h=h,hashes={n:sha256(probe/n) for n in ('job.inp','mesh.inp','direction.json','capture.gdb','assembly.json','solver.log')}));del data;gc.collect()
    rows=[];previous=None
    for h in (step,step/2):
        internal=(states[h][0]-states[-h][0])/(2*h);external=(states[h][1]-states[-h][1])/(2*h);net=internal-external
        err=float(np.linalg.norm(net-predicted)/np.linalg.norm(net));step=None if previous is None else float(np.linalg.norm(net-previous)/np.linalg.norm(net))
        rows.append(dict(h=h,native_net_relative_error=err,internal_only_relative_error=float(np.linalg.norm(internal-predicted)/np.linalg.norm(internal)),external_action_norm=float(np.linalg.norm(external)),internal_action_norm=float(np.linalg.norm(internal)),step_change=step,passed=err<2e-7 and (step is None or step<2e-7)));previous=net
    report=dict(direction_elements=selected,direction_equations=len(indices),checks_passed=all(r['passed'] for r in rows),mapping_checks=mapcheck,cases=cases,comparisons=rows,parent_gate=str(gate.relative_to(F)),parent_sha256=sha256(gate),maximum_prior_peeq=float(prior.max()),early_plastic_selection=parent.get('early_plastic_selection',False),probe_timeout_seconds=probe_timeout,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,exe,Path(__file__).with_name('audit_internal_pipe_fields.py'),Path(__file__).with_name('verify_heterogeneous_coupon.py'))},limitations='One selected actual pipe iteration with fixed captured prior history, one sparse direction, no MPC/contact. Native pressure force derivative measured separately. No plastic constitutive correction or convergence-causation, production load, Abaqus or experimental validation claim. Unchanged2e-7 derivative gate; report preserved even on failure.')
    output=F/'results/output'/('pipe_direction_'+tag+'.json');assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows));assert report['checks_passed']
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag');parser.add_argument('step',type=float,nargs='?',default=1e-7)
    parser.add_argument('--parent-gate',default='results/output/pipe_solver_v3_20261017T045902Z.json')
    parser.add_argument('--probe-timeout',type=int,default=180)
    options=parser.parse_args()
    main(options.tag,options.step,options.parent_gate,options.probe_timeout)
