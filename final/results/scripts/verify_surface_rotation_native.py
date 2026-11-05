"""Isolated native pressure-only whole-element manufactured rotation controls."""
import json, os, subprocess, sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F, sha256
from audit_quadratic_pressure_derivative import assemble
from audit_end_surface_traction import face_nodes
from pipe_sim.model import Model

def main(tag):
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    parent=F/'results/output/surface_metric_native_20261017T052958Z.json'
    prior=json.loads(parent.read_text()); assert prior['checks_passed']
    for run, hashes in prior['run_hashes'].items():
        for name,h in hashes.items(): assert sha256(F/run/name)==h
    baseline=F/'results/output/pipe_solver_v3_20261017T045902Z.json'
    base=json.loads(baseline.read_text()); run=F/base['run']
    for name,h in base['hashes'].items(): assert sha256(run/name)==h
    old=json.loads((F/'results/output/plastic_pipe_replay_20261011T101242Z.json').read_text())
    modelpath=F/old['run']/'model.json'; model=Model.load(modelpath)
    source=F/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    capture=Path(__file__).with_name('capture_surface_rotation_gdb.py')
    element=prior['element']; root=F/'sym/runs/investigation'/('surface_rotation_'+tag); root.mkdir(exist_ok=False)
    rotations=[np.eye(3)]
    for axis,angle in ((np.array([1.,2.,3.]),.7),(np.array([2.,-1.,.5]),1.1)):
        axis/=np.linalg.norm(axis); a,b,c=axis; cross=np.array([[0,-c,b],[c,0,-a],[-b,a,0]])
        rotations.append(np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*cross)
    records=[]; hashes={}; matrices={}
    for ri,rotation in enumerate(rotations):
        assert np.linalg.norm(rotation.T@rotation-np.eye(3))<2e-12 and np.linalg.det(rotation)>0
        for metric in (False,True):
            probes=[]
            for zero in (False,True):
                assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
                probe=root/f'r{ri}_metric{int(metric)}_zero{int(zero)}'; probe.mkdir()
                for name in ('job.inp','mesh.inp'): (probe/name).write_bytes((run/name).read_bytes())
                (probe/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_ELEMENT_SOURCE=str(source),THESIS_ELEMENT=str(element),THESIS_ZERO_PRESSURE=str(int(zero)),THESIS_ROTATION=json.dumps(rotation.tolist()),THESIS_METRIC=str(int(metric)),THESIS_OUTPUT=str(probe/'capture.json'))
                with (probe/'solver.log').open('w') as out:
                    result=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=probe,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=150)
                assert result.returncode==0, str(probe)
                probes.append(json.loads((probe/'capture.json').read_text()))
                hashes[str(probe.relative_to(F))]={p.name:sha256(p) for p in probe.iterdir() if p.is_file()}
            a,b=probes
            for key in ('manufactured','coordinates','displacement','stress','previous_state','nodes','time','loads','load_elements','sideload','flags_before'):
                assert a[key]==b[key], key
            state=a['manufactured']
            np.testing.assert_allclose(state['coordinates'],np.array(a['coordinates'])@rotation.T,rtol=0,atol=1e-12)
            assert not np.any(state['displacement']) and not np.any(state['stress']) and not np.any(state['previous_state'])
            assert state['flags']==dict(mass=[1],rhsi=[0],stiffness=[1],buckling=[0])
            native=(np.array(a['matrix'])-b['matrix']).reshape(60,60,order='F')
            face=face_nodes(model,element,1); ids=np.array([a['nodes'].index(n) for n in face]); dofs=(3*ids[:,None]+np.arange(3)).ravel()
            selected=[i for i in range(len(a['loads'])//2) if a['load_elements'][2*i]==element]; assert len(selected)==1
            pressure=a['loads'][2*selected[0]]
            _,raw=assemble(np.array(state['coordinates'])[ids],2); raw*=pressure/2.3
            full=np.zeros((60,60)); full[np.ix_(dofs,dofs)]=raw; norm=np.linalg.norm(full); symmetric=(full+full.T)/2
            block=np.kron(np.eye(20),rotation); matrices[ri,metric]=native
            rotation_error=float(np.linalg.norm(block.T@native@block-matrices[0,metric])/np.linalg.norm(matrices[0,metric]))
            massdiff=float(np.linalg.norm(np.array(a['mass_matrix'])-b['mass_matrix'])); assert massdiff==0
            records.append(dict(rotation_index=ri,metric=metric,rotation=rotation.tolist(),native_vs_symmetric_relative=float(np.linalg.norm(native-symmetric)/norm),native_vs_raw_relative=float(np.linalg.norm(native-full)/norm),native_rotation_relative=rotation_error,native_symmetry_relative=float(np.linalg.norm(native-native.T)/norm),mass_pair_difference=massdiff,pressure=pressure))
    passed=all(r['native_vs_symmetric_relative']<2e-7 and r['native_rotation_relative']<2e-7 for r in records if r['metric'])
    report=dict(checks_passed=passed,records=records,relative_gate=2e-7,run_hashes=hashes,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,parent,baseline,exe,source,source.with_name('shape8q.f'),modelpath)},method='Twelve serial disposable native element replays: three proper whole-element rotations, original/metric mapping, loaded/zero-pressure pairs. Reference=current, zero displacement/stress/history with stiffness1 initializing local displacement; loaded/zero subtraction isolates pressure contribution; mass1/rhsi0 triggers excluded branch. Kill before global assembly. Rotations preserve topology and nonfolding. Original failures and raw asymmetry retained.',limitations='Manufactured pressure-only local derivative, not constitutive/plastic history transfer or production static assembly. No global load activation, cap/contact/MPC, convergence causation, physical validation, burst or Abaqus equivalence.')
    output=F/'results/output'/('surface_rotation_'+tag+'.json'); assert not output.exists(); output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(records),flush=True);assert passed
if __name__=='__main__':main(sys.argv[1])
