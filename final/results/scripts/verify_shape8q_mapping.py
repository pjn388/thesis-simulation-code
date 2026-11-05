"""Actual binary surface gradients and admissible rigid-rotation controls."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256
from audit_pressure_branch_mapping import global_gradients,branch
from audit_quadratic_pressure_derivative import shape,assemble

def main(tag):
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    parent=F/'results/output/pressure_branch_v2_20261017T052035Z.json'
    prior=json.loads(parent.read_text());assert not prior['checks_passed']
    run=F/next(iter(prior['run_hashes']))
    for name,h in prior['run_hashes'][str(run.relative_to(F))].items():assert sha256(run/name)==h
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    uv=np.array([[-1,-1],[1,-1],[1,1],[-1,1],[0,-1],[1,0],[0,1],[-1,0]],float)
    flat=np.column_stack((1.2*uv[:,0]+.1*uv[:,1],.9*uv[:,1],np.zeros(8)))
    rotations=[np.eye(3)]
    for axis,angle in ((np.array([1.,2.,3.]),.7),(np.array([2.,-1.,.5]),1.1)):
        axis/=np.linalg.norm(axis);a,b,c=axis;cross=np.array([[0,-c,b],[c,0,-a],[-b,a,0]])
        rotations.append(np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*cross)
    root=F/'sym/runs/investigation'/('shape8q_mapping_'+tag);root.mkdir(exist_ok=False)
    capture=Path(__file__).with_name('capture_shape8q_gdb.py');records=[];hashes={}
    for label,coords,rotation in [('retained',None,None)]+[('planar_'+str(i),flat@r.T,r) for i,r in enumerate(rotations)]:
        probe=root/label;probe.mkdir()
        for name in ('job.inp','mesh.inp'):(probe/name).write_bytes((run/name).read_bytes())
        (probe/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_ELEMENT_SOURCE=str(F/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'),THESIS_SHAPE_COORDS=json.dumps(None if coords is None else coords.ravel().tolist()),THESIS_OUTPUT=str(probe/'capture.json'))
        if label=='retained':
            retained=F/'sym/runs/investigation/shape8q_mapping_v2_20261017T052506Z/retained/capture.json'
            (probe/'capture.json').write_bytes(retained.read_bytes())
            (probe/'reused.json').write_text(json.dumps(dict(path=str(retained.relative_to(F)),sha256=sha256(retained)))+'\n')
        else:
            with (probe/'solver.log').open('w') as out:
                result=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=probe,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=150)
            assert result.returncode==0,str(probe)
        points=json.loads((probe/'capture.json').read_text());errors=[];absolute=[];chains=[];native=[];areas=[]
        for point in points:
            x=np.array(point['coordinates']).reshape(8,3);r,s=point['r'],point['s'];n,dr,ds=shape(r,s)
            shp=np.array(point['shp']).reshape(7,8,order='F');xs=np.array(point['xs']).reshape(3,7,order='F')
            _,gradient,area,tangent,inverse=global_gradients(x,r,s)
            pairs=((shp[:3].T,gradient),(shp[3],n),(xs[:,:2],tangent),(np.array(point['area']),area))
            absolute.append([float(np.max(abs(a-b))) for a,b in pairs])
            errors.append(max(float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(b))) for a,b in pairs))
            chain=shp[:3].T@tangent;expected=np.column_stack((dr,ds))
            chains.append(float(np.linalg.norm(chain-expected)/np.linalg.norm(expected)))
            native.append(shp[:3].T);areas.append(area)
        assert max(errors)<2e-12
        record=dict(label=label,native_source_max_scaled=max(errors),native_source_absolute_by_quantity=np.max(absolute,axis=0).tolist(),quantity_order=['global_gradient','shape_value','reference_tangent','area_vector'],relative_reproduction_limit=2e-12,native_chain_relative=max(chains),quadrature=points)
        if coords is not None:
            normal=rotation@np.array([0.,0.,1.]);signed=np.array(areas)@normal
            assert min(signed)>1.;record['minimum_signed_jacobian']=float(min(signed))
            # Affine planar mapping is nonfolded everywhere, not only at Gauss points.
            record['affine_reference_current_identical']=True
            record['native_gradient_rotation_error']=float(max(np.linalg.norm(g@rotation-base) for g,base in zip(native,planar_native))) if label!='planar_0' else 0.
            mapped,_=branch(coords,coords,2.3);_,raw=assemble(coords,2)
            record['source_branch_vs_symmetric_relative']=float(np.linalg.norm(mapped-(raw+raw.T)/2)/np.linalg.norm(raw))
            block=np.kron(np.eye(8),rotation)
            record['source_branch_rotation_relative']=float(np.linalg.norm(block.T@mapped@block-planar_matrix)/np.linalg.norm(planar_matrix)) if label!='planar_0' else 0.
            record['independent_rotation_relative']=float(np.linalg.norm(block.T@raw@block-planar_raw)/np.linalg.norm(planar_raw)) if label!='planar_0' else 0.
            if label=='planar_0':planar_native=native;planar_matrix=mapped;planar_raw=raw
            assert record['independent_rotation_relative']<2e-12
        records.append(record);hashes[str(probe.relative_to(F))]={p.name:sha256(p) for p in probe.iterdir() if p.is_file()}
    report=dict(checks_passed=True,native_tangent_accepted=False,records=records,run_hashes=hashes,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),capture,parent,exe,F/'sym/vendor/CalculiX/ccx_2.23/src/shape8q.f',Path(__file__).with_name('audit_pressure_branch_mapping.py'))},method='Unmodified existing solver binary, source-sized shape8q shp7x8 xs3x7. Original retained first four global-gradient calls, then separate manufactured affine planar coordinate replays at same call sites. Every process killed after fourth shape8q return, before element assembly. Planar reference=current and positive constant area establish nonfolded controls. Raw native and independent chain/rotation errors retained without rescaling.',limitations='Diagnostic source/binary mapping reproduction only. Failed native tangent gate remains failed. Rotated branch matrices use source algebra checked previously, not a new native e_c3d matrix extraction. No production change, convergence causation, plastic/cap/contact/MPC/Abaqus or experimental validation.')
    out=F/'results/output'/('shape8q_mapping_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='quadrature'} for r in records]))
if __name__=='__main__':main(sys.argv[1])
