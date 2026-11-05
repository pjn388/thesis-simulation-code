"""Native isolated pressure stiffness, compared with independent area derivative."""
import json,os,re,subprocess,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256
from audit_quadratic_pressure_derivative import assemble
from audit_end_surface_traction import face_nodes
from pipe_sim.model import Model

def main(tag):
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    gate=F/'results/output/pipe_pressure_matrix_20261017T051901Z.json';accepted=json.loads(gate.read_text());assert accepted['checks_passed']
    baselinegate=F/'results/output/pipe_solver_v3_20261017T045902Z.json';baseline=json.loads(baselinegate.read_text());assert baseline['checks_passed']
    run=F/baseline['run']
    for n,h in baseline['hashes'].items():assert sha256(run/n)==h
    base=json.loads((run/'assembly.json').read_text())
    source=F/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso';assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    original=json.loads((F/'results/output/plastic_pipe_replay_20261011T101242Z.json').read_text());modelpath=F/original['run']/'model.json';model=Model.load(modelpath)
    loads=[x.split(',') for x in (run/'job.inp').read_text().split('*DLOAD\n')[1].split('*')[0].strip().splitlines()]
    element=int(loads[0][0]);capture=Path(__file__).with_name('capture_surface_metric_gdb.py')
    root=F/'sym/runs/investigation'/('surface_metric_native_'+tag);root.mkdir(exist_ok=False)
    probes=[];hashes={}
    for zero in (False,True):
        probe=root/('zero' if zero else 'loaded');probe.mkdir()
        for name in ('job.inp','mesh.inp'):(probe/name).write_bytes((run/name).read_bytes())
        (probe/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_ELEMENT_SOURCE=str(source),THESIS_ELEMENT=str(element),THESIS_ZERO_PRESSURE=str(int(zero)),THESIS_OUTPUT=str(probe/'capture.json'))
        with (probe/'solver.log').open('w') as out:r=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=probe,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=150)
        assert r.returncode==0,str(probe)
        probes.append(json.loads((probe/'capture.json').read_text()));hashes[str(probe.relative_to(F))]={p.name:sha256(p) for p in probe.iterdir() if p.is_file()}
    a,b=probes
    assert len(a['surface_metric_replacements'])==len(b['surface_metric_replacements'])==4
    for key in ('nodes','time','coordinates','displacement','stress','previous_state','loads','load_elements','sideload','flags_before'):assert a[key]==b[key],key
    nodes=np.array(a['nodes'])-1
    np.testing.assert_array_equal(a['coordinates'],np.array(base['coordinates']).reshape(-1,3)[nodes])
    np.testing.assert_array_equal(a['displacement'],np.array(base['displacement_after']).reshape(-1,4)[nodes])
    assert a['time']==[base['time']]
    native=(np.array(a['matrix'])-b['matrix']).reshape(60,60,order='F')
    independent=np.zeros((60,60));current=np.array(a['coordinates'])+np.array(a['displacement'])[:,1:]
    selectedloads=[]
    for e,f,p in loads:
        if int(e)!=element:continue
        face=list(face_nodes(model,element,int(f.strip()[1:])));ids=np.array([a['nodes'].index(n) for n in face]);dofs=(3*ids[:,None]+np.arange(3)).ravel()
        _,k=assemble(current[ids],2)
        # Native s adds minus the derivative of inward external force.
        k*=float(p)*base['time']/2.3
        independent[np.ix_(dofs,dofs)]+=k;selectedloads.append([int(e),f,float(p)])
    norm=np.linalg.norm(independent);symmetric=(independent+independent.T)/2
    error=float(np.linalg.norm(native-symmetric)/norm)
    report=dict(checks_passed=error<2e-7,native_branch_relative_limit=2e-7,element=element,time=base['time'],loads=selectedloads,native_norm=float(np.linalg.norm(native)),independent_norm=float(norm),native_symmetry=float(np.linalg.norm(native-native.T)/norm),native_vs_raw_relative=float(np.linalg.norm(native-independent)/norm),native_vs_symmetrized_relative=error,mass_matrix_pair_difference=float(np.linalg.norm(np.array(a['mass_matrix'])-b['mass_matrix'])),run_hashes=hashes,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,source,source.with_name('shape8q.f'),exe,gate,baselinegate,run/'assembly.json',modelpath)},method='Surface metric replaces only returned shape8q gradients in four calls, with paired state identity checked. Two disposable native replays at same early pipe iteration, same selected loaded C3D20R. Set mass1 rhsi0 in isolated element only; retain stiffness and all material/geometric inputs. Loaded minus zero-pressure s cancels common contributions; sm retained separately. Stop immediately after element return; never proceed altered flags to global assembly. Compare minus external derivative to s, with raw and symmetric forms retained.',limitations='One loaded element and early elastic geometry only. Not an extracted full-pipe matrix, native unsymmetrized term, plastic correction, convergence-causation test, production patch, finite-cap work or physical/Abaqus/burst acceptance.')
    output=F/'results/output'/('surface_metric_native_'+tag+'.json');assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('run_hashes','provenance')}));assert report['checks_passed']
if __name__=='__main__':main(sys.argv[1])
