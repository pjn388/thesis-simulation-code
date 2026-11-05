"""Native local stress derivatives with identical replayed previous plastic history."""
import json,os,sys,subprocess
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256


def main(tag):
    parent_report=F/'results/output'/('material_tangent_'+tag+'.json')
    report=json.loads(parent_report.read_text());parent=F/report['run']
    for name,h in report['hashes'].items():assert sha256(parent/name)==h
    for name,h in report['provenance'].items():assert sha256(F/name)==h
    reference=json.loads((parent/'tangent.json').read_text())['records'][70]
    assert reference['current_state'][0]>reference['previous_state'][0]>0
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    capture=Path(__file__).with_name('capture_material_trial_gdb.py')
    root=F/'sym/runs/investigation'/('material_trials_'+tag);root.mkdir(exist_ok=False)
    states={};cases=[]
    for i,direction in enumerate(report['directions']):
        for h in (1e-6,1e-7):
            for sign in (-1,1):
                assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
                run=root/f'd{i}_h{h}_s{sign}';run.mkdir()
                (run/'job.inp').write_bytes((parent/'job.inp').read_bytes())
                (run/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_TANGENT_OUTPUT=str(run/'tangent.json'),THESIS_TRIAL_INDEX='70',THESIS_TRIAL_DIRECTION=json.dumps(direction),THESIS_TRIAL_AMPLITUDE=str(sign*h))
                with (run/'solver.log').open('w') as stream:
                    result=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=run,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
                assert result.returncode==0
                records=json.loads((run/'tangent.json').read_text())['records'];assert len(records)==71
                r=records[-1]
                assert r['previous_state']==reference['previous_state']
                assert r['time']==reference['time'] and r['ttime']==reference['ttime']
                np.testing.assert_array_equal(r['emec'],np.array(reference['emec'])+sign*h*np.array(direction))
                # Every preceding local stress/state/stiffness call replays identically.
                assert records[:-1]==json.loads((parent/'tangent.json').read_text())['records'][:70]
                states[i,h,sign]=np.array(r['stress'])
                cases.append(dict(run=str(run.relative_to(F)),direction=i,h=h,sign=sign,previous_state_identical=True,prior_calls_identical=70,hashes={n:sha256(run/n) for n in ('job.inp','tangent.json','solver.log','capture.gdb')}))
    k=np.zeros((6,6))
    for (i,j),v in zip([(i,j) for j in range(6) for i in range(j+1)],reference['stiffness']):k[i,j]=k[j,i]=v
    comparisons=[]
    for i,direction in enumerate(report['directions']):
        earlier=None
        for h in (1e-6,1e-7):
            derivative=(states[i,h,1]-states[i,h,-1])/(2*h)
            expected=k@(np.array([1,1,1,2,2,2])*direction)
            comparisons.append(dict(direction=i,h=h,native_tangent_relative_error=float(np.linalg.norm(derivative-expected)/np.linalg.norm(expected)),derivative=derivative.tolist(),step_change_relative=None if earlier is None else float(np.linalg.norm(derivative-earlier)/np.linalg.norm(derivative))))
            earlier=derivative
    output=F/'results/output'/('material_trials_'+tag+'.json');assert not output.exists()
    result=dict(cases=cases,comparisons=comparisons,index=70,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,capture.with_name('capture_material_tangent_gdb.py'),parent_report,exe)},limitation='Native material-call derivative only, exact same70prior calls and previous plastic state; current local strain perturbed then intentionally terminated. No global residual/matrix or native equilibrium solution after perturbation, no claim of solver defect or pipe cause. Production and Abaqus unchanged.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main(sys.argv[1])
