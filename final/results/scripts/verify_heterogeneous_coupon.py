"""Nonuniform C3D8 trajectory, native IP mapping and fixed-history Jacobian."""
import json,os,re,sys,subprocess
from pathlib import Path
import numpy as np
from pipe_sim.results import dat_blocks
from audit_plastic_return_map import F,tensors,sha256
from audit_material_linearization import derivative,vector
import verify_plastic_jacobian as full

XYZ=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)

def points():
    source=F/'sym/vendor/CalculiX/ccx_2.23/src/gauss.f'
    text=source.read_text().split('gauss3d2=reshape((/',1)[1].split('/)',1)[0]
    values=re.findall(r'[-+]?\d+\.\d+d[+-]?\d+',text)
    points=np.array([float(v.replace('d','e')) for v in values]).reshape(8,3)
    assert len(np.unique(points,axis=0))==8
    return points

def geometry(point,u):
    signs=2*XYZ-1
    natural=np.array([[signs[a,i]*np.prod([1+signs[a,j]*point[j] for j in range(3) if j!=i])/8 for i in range(3)] for a in range(8)])
    jac=XYZ.T@natural;grad=natural@np.linalg.inv(jac);deformation=np.eye(3)+u.T@grad
    assert np.linalg.det(jac)>0 and np.linalg.det(deformation)>0
    return jac,grad,deformation

mapping_checks=[]
def correction(data,run):
    lines=(run/'job.inp').read_text().splitlines();curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve);u=np.array(data['displacement_after']).reshape(8,4)[:,1:]
    ips={p['ip']:p for p in data['ips']};assert sorted(ips)==list(range(1,9))
    assert np.ptp([p['previous'][0] for p in ips.values()])>1e-6
    assembled=np.zeros((24,24));rows=[]
    for index,point in enumerate(points(),1):
        ip=ips[index];jac,grad,deformation=geometry(point,u)
        c=np.eye(3)+2*tensors(np.array(ip['strain'])[None])[0]
        error=float(np.max(np.abs(deformation.T@deformation-c)))
        np.testing.assert_allclose(deformation.T@deformation,c,rtol=0,atol=2e-14)
        previous=np.array(ip['previous']);metric=np.eye(3)-2*tensors(previous[None,1:7])[0]
        mu=202500/(2*1.3);bulk=202500/(3*.4);delta=np.zeros((6,6))
        for col in range(6):
            v=np.eye(6)[col]/(1 if col<3 else 2);dc=2*tensors(v[None])[0]
            delta[:,col]=vector(derivative(c,metric,previous[0],curve,mu,bulk,dc)-derivative(c,metric,previous[0],curve,mu,bulk,dc,True))
        b=np.zeros((6,24))
        for node in range(8):
            for axis in range(3):
                de=(np.outer(deformation[axis],grad[node])+np.outer(grad[node],deformation[axis]))/2
                b[:,3*node+axis]=vector(de)*np.array([1,1,1,2,2,2])
        assembled+=b.T@delta@b*np.linalg.det(jac)
        rows.append(dict(native_ip=index,natural=point.tolist(),strain_mapping_error=error,reference_jacobian=float(np.linalg.det(jac)),deformation_jacobian=float(np.linalg.det(deformation)),previous_peeq=previous[0]))
    mapping=np.array(data['mapping']).reshape(8,4)[:,1:].ravel();projected=np.zeros((data['neq'],data['neq']))
    for i,a in enumerate(mapping):
        for j,b in enumerate(mapping):
            if a>0 and b>0:projected[a-1,b-1]+=assembled[i,j]
    mapping_checks.append(dict(cycle=data['cycle'],points=rows))
    return projected

def main(tag):
    assert json.loads((F/'results/automation/paused.json').read_text())['state']=='resumed'
    parent=F/'sym/runs/investigation/material_tangent_20261017T040006Z'
    oldgate=F/'results/output/material_tangent_20261017T040006Z.json'
    for n,h in json.loads(oldgate.read_text())['hashes'].items():assert sha256(parent/n)==h
    run=F/'sym/runs/investigation'/('heterogeneous_coupon_'+tag);run.mkdir(exist_ok=False)
    text=(parent/'job.inp').read_text()
    # Change prescribed axial travel only: y=0 is 90%, y=1 is 110%.
    for value in ('0.0202013400268','0.018478569623'):
        replacement='\n'.join(f'{node},1,1,{float(value)*factor:.14g}' for node,factor in ((2,.9),(3,1.1),(6,.9),(7,1.1)))
        assert text.count('RIGHT,1,1,'+value)==1
        text=text.replace('RIGHT,1,1,'+value,replacement)
    (run/'job.inp').write_text(text)
    exe=F/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert sha256(exe)==json.loads((exe.parent/'build.json').read_text())['solver_sha256']
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MKL_THREADING_LAYER='GNU',CCX_NPROC_RESULTS='1',CCX_NPROC_STIFFNESS='1',CCX_NPROC_EQUATION_SOLVER='1')
    with (run/'solver.log').open('w') as log:
        r=subprocess.run([str(exe),'-i','job'],cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert r.returncode==0 and 'Job finished' in (run/'solver.log').read_text() and '*ERROR' not in (run/'solver.log').read_text()
    frames={}
    for kind,group,t,data in dat_blocks(run/'job.dat'):frames.setdefault(t,{})[kind]=data
    assert max(frames)==2. and len(frames)==80
    rows=[]
    for t,fields in frames.items():
        u=np.array([r[1:] for r in sorted(fields['U'],key=lambda r:r[0])]);ep=np.array([r[2] for r in fields['PEEQ']])
        js=[float(np.linalg.det(geometry(p,u)[2])) for p in points()]
        assert min(js)>0 and min(ep)>=0 and max(ep)<.197798202029
        rows.append(dict(time=t,minimum_deformation_jacobian=min(js),maximum_peeq=float(max(ep)),peeq_spread=float(np.ptp(ep))))
    assert max(r['peeq_spread'] for r in rows)>1e-4
    gate=F/'results/output'/('heterogeneous_coupon_'+tag+'.json')
    report=dict(run=str(run.relative_to(F)),parent=str(parent.relative_to(F)),trajectory=rows,completed=True,hashes={n:sha256(run/n) for n in ('job.inp','solver.log','job.dat','job.sta')},provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),exe,oldgate,F/'sym/vendor/CalculiX/ccx_2.23/src/gauss.f',F/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f')},limitations='Manufactured heterogeneous load/unload; original material, constraints and increments. Not experimental validation; no production load change.')
    assert not gate.exists();gate.write_text(json.dumps(report,indent=2)+'\n')
    os.environ.update(THESIS_COUPON_PARENT=str(run),THESIS_COUPON_GATE=str(gate))
    full.correction=correction
    full.main(tag)
    out=F/'results/output'/('heterogeneous_mapping_'+tag+'.json')
    out.write_text(json.dumps(dict(mapping_checks=mapping_checks,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),gate,F/'sym/vendor/CalculiX/ccx_2.23/src/gauss.f',F/'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f')},limitations='Native IP mapping and positive quadrature Jacobians. No pipe-state, pressure/contact/MPC/energy/cap/Abaqus acceptance.'),indent=2)+'\n')
if __name__=='__main__':main(sys.argv[1])
