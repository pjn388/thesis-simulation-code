"""Bounded C3D20R plastic/unloading native matrix and corrected directional FD gate."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256,tensors
from audit_material_linearization import derivative,vector
from audit_internal_pipe_fields import shape,source_basis
from verify_heterogeneous_coupon import points
from pipe_sim.model import HEX_LOCAL

def correction(data,curve):
    xyz=(HEX_LOCAL+1)/2;u=np.array(data['displacement_after']).reshape(20,4)[:,1:]
    ips={p['ip']:p for p in data['ips']};assert sorted(ips)==list(range(1,9))
    assembled=np.zeros((60,60));checks=[]
    for index,q in enumerate(points(),1):
        natural=np.array([shape(q.astype(complex)+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T
        values,primary=source_basis(q)
        np.testing.assert_allclose(natural,primary,rtol=0,atol=1e-14)
        np.testing.assert_allclose(values,shape(q),rtol=0,atol=1e-14)
        jac=xyz.T@natural;grad=natural@np.linalg.inv(jac);deformation=np.eye(3)+u.T@grad
        ip=ips[index];c=np.eye(3)+2*tensors(np.array(ip['strain'])[None])[0]
        error=float(np.max(abs(deformation.T@deformation-c)))
        np.testing.assert_allclose(deformation.T@deformation,c,rtol=0,atol=2e-14)
        previous=np.array(ip['previous']);metric=np.eye(3)-2*tensors(previous[None,1:7])[0]
        delta=np.zeros((6,6));mu=202500/2.6;bulk=202500/1.2
        for col in range(6):
            v=np.eye(6)[col]/(1 if col<3 else 2);dc=2*tensors(v[None])[0]
            delta[:,col]=vector(derivative(c,metric,previous[0],curve,mu,bulk,dc)-derivative(c,metric,previous[0],curve,mu,bulk,dc,True))
        b=np.zeros((6,60))
        for node in range(20):
            for axis in range(3):
                de=(np.outer(deformation[axis],grad[node])+np.outer(grad[node],deformation[axis]))/2
                b[:,3*node+axis]=vector(de)*[1,1,1,2,2,2]
        assert np.linalg.det(jac)>0 and np.linalg.det(deformation)>0
        assembled+=b.T@delta@b*np.linalg.det(jac)
        checks.append(dict(ip=index,strain_mapping_error=error,previous_peeq=float(previous[0]),current_volume_ratio=float(np.linalg.det(deformation))))
    mapping=np.array(data['mapping']).reshape(20,4)[:,1:].ravel();out=np.zeros((data['neq'],data['neq']))
    for i,a in enumerate(mapping):
        for j,b in enumerate(mapping):
            if a>0 and b>0:out[a-1,b-1]+=assembled[i,j]
    return out,checks

def main(tag):
    root=F/'sym/runs/investigation'/('c3d20_plastic_'+tag);root.mkdir(exist_ok=False)
    parent=F/'sym/runs/investigation/material_tangent_20261017T040006Z'
    gate=F/'results/output/material_tangent_20261017T040006Z.json'
    for n,h in json.loads(gate.read_text())['hashes'].items():assert sha256(parent/n)==h
    text=(parent/'job.inp').read_text();curve=np.array([list(map(float,s.split(','))) for s in text.split('*PLASTIC\n')[1].split('*')[0].strip().splitlines()])
    xyz=(HEX_LOCAL+1)/2
    deck='*HEADING\nC3D20R plastic/unloading assembly diagnostic\n*NODE\n'+'\n'.join(f'{i+1},'+','.join(map(str,x)) for i,x in enumerate(xyz))
    deck+='\n*ELEMENT,TYPE=C3D20R,ELSET=CUBE\n1,'+','.join(map(str,range(1,16)))+',\n'+','.join(map(str,range(16,21)))+'\n*NSET,NSET=ALL\n'+','.join(map(str,range(1,17)))+'\n'+','.join(map(str,range(17,21)))+'\n'
    for name,axis,value in [('LEFT',0,0),('RIGHT',0,1),('Y0',1,0),('Z0',2,0)]:
        deck+=f'*NSET,NSET={name}\n'+','.join(str(i+1) for i,x in enumerate(xyz) if x[axis]==value)+'\n'
    deck+=text[text.index('*MATERIAL'):]
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso';assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    capture=Path(__file__).with_name('capture_element_solver_gdb.py');source=F/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f'
    cases=[];rows=[]
    for cycle in (30,60):
        states={}
        for h in (0.,1e-7,-1e-7,5e-8,-5e-8):
            assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
            run=root/f'c{cycle}_h{h}';run.mkdir();(run/'job.inp').write_text(deck)
            (run/'capture.gdb').write_text('set pagination off\nset confirm off\nsource '+str(capture)+'\nquit\n')
            direction=np.zeros(60) if h==0 else direction
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1',THESIS_CYCLE=str(cycle),THESIS_DIRECTION=json.dumps(direction.tolist()),THESIS_AMPLITUDE=str(h),THESIS_OUTPUT=str(run/'assembly.json'),THESIS_ELEMENT_SOURCE=str(source))
            with (run/'solver.log').open('w') as out:r=subprocess.run(['gdb','--batch','-x','capture.gdb','--args',str(exe),'-i','job'],cwd=run,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=90)
            assert r.returncode==0,str(run)
            data=json.loads((run/'assembly.json').read_text());states[h]=data
            assert data['nk']==20 and data['mi'][0]==8 and len(data['ips'])==8
            np.testing.assert_allclose(data['displacement_after'],data['displacement_imposed'],rtol=0,atol=1e-15)
            assert data['displacement_after']==data['assembly_displacement']
            assert data['stress']==data['assembly_stress']
            if h==0:
                base=data;direction=np.random.default_rng(1741).normal(size=data['neq']);direction/=np.linalg.norm(direction)
            else:
                for key in ('previous_state','previous_stress','displacement_before','mapping','time','ttime'):assert data[key]==base[key],key
                assert [p['previous'] for p in data['ips']]==[p['previous'] for p in base['ips']]
            residual=data['residual'];assert residual['internal']==data['internal'] and residual['displacement']==data['assembly_displacement']
            np.testing.assert_array_equal(residual['b'],-np.array(data['internal']));np.testing.assert_array_equal(data['global_matrix']['rhs'],residual['b']);assert max(map(abs,residual['fext']))==0
            cases.append(dict(run=str(run.relative_to(F)),cycle=cycle,h=h,hashes={n:sha256(run/n) for n in ('job.inp','capture.gdb','assembly.json','solver.log')}))
        mapping=np.array(base['mapping']).reshape(20,4)[:,1:].ravel();s=np.array(base['element_matrix']).reshape(60,60,order='F');s=np.triu(s)+np.triu(s,1).T
        k=np.zeros((base['neq'],base['neq']))
        for i,a in enumerate(mapping):
            for j,b in enumerate(mapping):
                if a>0 and b>0:k[a-1,b-1]+=s[i,j]
        g=base['global_matrix'];assert g['symmetryflag']==0 and g['inputformat']==0 and g['jq'][0]==1 and g['jq'][-1]==g['nzs']+1
        native=np.diag(g['ad'])
        for col in range(g['neq']):
            for off in range(g['jq'][col]-1,g['jq'][col+1]-1):
                row=g['irow'][off]-1;assert col<row<g['neq'];native[row,col]=native[col,row]=g['au'][off]
        np.testing.assert_allclose(k,native,rtol=2e-14,atol=1e-10)
        delta,checks=correction(base,curve);prev=None
        for h in (1e-7,5e-8):
            fd=(np.array(states[h]['internal'])-states[-h]['internal'])/(2*h)
            err=float(np.linalg.norm(fd-(k+delta)@direction)/np.linalg.norm(fd));step=None if prev is None else float(np.linalg.norm(fd-prev)/np.linalg.norm(fd))
            rows.append(dict(cycle=cycle,time=base['time'],ttime=base['ttime'],h=h,corrected_relative_error=err,native_relative_error=float(np.linalg.norm(fd-k@direction)/np.linalg.norm(fd)),step_change=step,matrix_max_error=float(np.max(abs(k-native))),mapping_checks=checks,correction_asymmetry_norm=float(np.linalg.norm(delta-delta.T)),passed=err<2e-7 and (step is None or step<2e-7)))
            prev=fd
    provenance=(Path(__file__),capture,source,source.with_name('results.c'),source.with_name('shape20h.f'),source.with_name('gauss.f'),exe,gate,Path(__file__).with_name('audit_material_linearization.py'),Path(__file__).with_name('audit_internal_pipe_fields.py'),Path(__file__).with_name('verify_heterogeneous_coupon.py'))
    report=dict(checks_passed=all(r['passed'] for r in rows),cases=cases,comparisons=rows,provenance={str(p.relative_to(F)):sha256(p) for p in provenance},limitations='Single C3D20R, one free direction, plastic loading/unloading, same native prior history and actual matrix/residual. Independent diagnostic constitutive correction only; no solver patch, pipe convergence causation, pressure/contact/MPC/cap work/Abaqus or experimental validation. Full pipe matrices must remain sparse.')
    output=F/'results/output'/('c3d20_plastic_'+tag+'.json');assert not output.exists();output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows));assert report['checks_passed']
if __name__=='__main__':main(sys.argv[1])
