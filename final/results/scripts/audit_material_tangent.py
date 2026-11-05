"""Native local material tangent versus independent fixed-history central differences.

Measures discrepancies without fitting tolerances or changing native inputs.
"""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F, tensors, sha256, update


def main(tag):
    parent=F/'sym/runs/investigation/plastic_state_coupon_20261011T095613Z'
    gate=F/'results/output/accumulated_plastic_history_20261017T033006Z.json'
    reference=json.loads(gate.read_text())
    row=next(r for r in reference['trajectories'] if r['lane']=='standard')
    assert row['checks_passed']
    for name,h in row['input_output_hashes'].items():assert sha256(parent/name)==h
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    run=F/'sym/runs/investigation'/('material_tangent_'+tag)
    run.mkdir(exist_ok=False)
    (run/'job.inp').write_bytes((parent/'job.inp').read_bytes())
    capture=Path(__file__).with_name('capture_material_tangent_gdb.py')
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    (run/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_TANGENT_OUTPUT=str(run/'tangent.json'))
    with (run/'solver.log').open('w') as stream:
        result=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=run,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    assert result.returncode==0 and 'exited normally' in (run/'solver.log').read_text()
    lines=(run/'job.inp').read_text().splitlines();curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve)
    records=json.loads((run/'tangent.json').read_text())['records']
    comparisons=[]
    # Tensor components: 11,22,33,12,13,23. Packed upper triangle by columns.
    pairs=[(i,j) for j in range(6) for i in range(j+1)]
    weights=np.array([1,1,1,2,2,2])
    directions=np.random.default_rng(1741).normal(size=(3,6))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    for index,r in enumerate(records):
        e=np.array(r['emec']);previous=np.array(r['previous_state'])
        metric=np.eye(3)-2*tensors(previous[None,1:7])[0]
        q=np.array(previous[0]);young,nu=r['elastic'];mu=young/(2*(1+nu));bulk=young/(3*(1-2*nu))
        def predict(v):return update(np.eye(3)+2*tensors(v[None])[0],metric,q,curve,mu,bulk)
        stress,newmetric,newq,active=predict(e)
        stress_vector=np.array([stress[0,0],stress[1,1],stress[2,2],stress[0,1],stress[0,2],stress[1,2]])
        stress_error=float(abs(stress_vector-r['stress']).max())
        k=np.zeros((6,6))
        for (i,j),v in zip(pairs,r['stiffness']):k[i,j]=k[j,i]=v
        errors=[]
        for direction in directions:
            expected=k@(weights*direction)
            for h in (1e-6,5e-7):
                plus=predict(e+h*direction);minus=predict(e-h*direction)
                derivative=(plus[0]-minus[0])/(2*h)
                dv=np.array([derivative[0,0],derivative[1,1],derivative[2,2],derivative[0,1],derivative[0,2],derivative[1,2]])
                errors.append(dict(h=h,relative_error=float(np.linalg.norm(dv-expected)/np.linalg.norm(expected)),same_branch=bool(plus[3]==minus[3]==active)))
        comparisons.append(dict(index=index,time=r['time'],ttime=r['ttime'],plastic=bool(active),previous_peeq=float(q),current_peeq=float(newq),stress_error_mpa=stress_error,derivatives=errors))
    report=dict(run=str(run.relative_to(F)),parent=str(parent.relative_to(F)),parent_deck_identical=sha256(run/'job.inp')==sha256(parent/'job.inp'),calls=len(records),plastic_calls=sum(r['plastic'] for r in comparisons),max_stress_error_mpa=max(r['stress_error_mpa'] for r in comparisons),max_relative_derivative_error=max(d['relative_error'] for r in comparisons for d in r['derivatives']),comparisons=comparisons,directions=directions.tolist(),hashes={name:sha256(run/name) for name in ('job.inp','job.sta','job.dat','solver.log','tangent.json','capture.gdb')},provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,gate,exe,Path(__file__).with_name('audit_plastic_return_map.py'),F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f')},limitations='Local derivative diagnostic, fixed exact previous native IP state for each trial; independent central differences use retained isotropic return map. Not global residual or native perturbed assembly. No contact/MPC/cap/Abaqus/energy partition/experimental acceptance. Relative differences reported without post hoc tolerance fitting.')
    output=F/'results/output'/('material_tangent_'+tag+'.json');assert not output.exists()
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','hashes','provenance','directions')},indent=2))
if __name__=='__main__':main(sys.argv[1])
