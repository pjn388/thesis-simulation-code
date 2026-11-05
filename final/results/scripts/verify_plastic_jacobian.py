"""Full fixed-history coupon Jacobian and local native Newton iteration audit."""
import itertools,json,os,sys,subprocess
from pathlib import Path
import numpy as np
from audit_material_linearization import derivative,vector
from audit_plastic_return_map import F,tensors,sha256

def correction(data,run):
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    signs=2*xyz-1;ips=data['ips']
    for key in ('strain','previous'):
        np.testing.assert_allclose([p[key] for p in ips],np.broadcast_to(ips[0][key],(8,len(ips[0][key]))),rtol=0,atol=2e-14)
    lines=(run/'job.inp').read_text().splitlines();curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve);u=np.array(data['displacement_after']).reshape(8,4)[:,1:]
    c=np.eye(3)+2*tensors(np.array(ips[0]['strain'])[None])[0]
    previous=np.array(ips[0]['previous']);metric=np.eye(3)-2*tensors(previous[None,1:7])[0]
    mu=202500/(2*1.3);bulk=202500/(3*.4);delta=np.zeros((6,6))
    for col in range(6):
        v=np.eye(6)[col]/(1 if col<3 else 2);dc=2*tensors(v[None])[0]
        delta[:,col]=vector(derivative(c,metric,previous[0],curve,mu,bulk,dc)-derivative(c,metric,previous[0],curve,mu,bulk,dc,True))
    correction=np.zeros((24,24))
    for point in itertools.product((-1/np.sqrt(3),1/np.sqrt(3)),repeat=3):
        natural=np.array([[signs[a,i]*np.prod([1+signs[a,j]*point[j] for j in range(3) if j!=i])/8 for i in range(3)] for a in range(8)])
        jac=xyz.T@natural;grad=natural@np.linalg.inv(jac);deformation=np.eye(3)+u.T@grad
        np.testing.assert_allclose(deformation.T@deformation,c,rtol=0,atol=2e-14)
        b=np.zeros((6,24))
        for node in range(8):
            for axis in range(3):
                de=(np.outer(deformation[axis],grad[node])+np.outer(grad[node],deformation[axis]))/2
                b[:,3*node+axis]=vector(de)*np.array([1,1,1,2,2,2])
        correction+=b.T@delta@b*np.linalg.det(jac)
    mapping=np.array(data['mapping']).reshape(8,4)[:,1:].ravel();projected=np.zeros((data['neq'],data['neq']))
    for i,a in enumerate(mapping):
        for j,b in enumerate(mapping):
            if a>0 and b>0:projected[a-1,b-1]+=correction[i,j]
    return projected

def main(tag):
    script=Path(__file__).with_name('verify_plastic_jacobian_column.py')
    reports=[]
    for column in range(8):
        env=dict(os.environ,THESIS_BASIS_COLUMN=str(column))
        subprocess.run([sys.executable,str(script),tag+'_b'+str(column)],env=env,check=True)
        reports.append(F/'results/output'/('plastic_jacobian_'+tag+'_b'+str(column)+'.json'))
    rows=[];provenance={};hash_count=0
    evidence=[json.loads(p.read_text()) for p in reports]
    for p,e in zip(reports,evidence):
        provenance[str(p.relative_to(F))]=sha256(p)
        for path,h in e['provenance'].items():assert sha256(F/path)==h;hash_count+=1
        for case in e['cases']:
            for n,h in case['hashes'].items():assert sha256(F/case['run']/n)==h;hash_count+=1
    for cycle in (30,60):
        bases=[]
        for e in evidence:
            c=next(c for c in e['cases'] if c['cycle']==cycle and c['h']==0)
            bases.append(json.loads((F/c['run']/'assembly.json').read_text()))
        for b in bases[1:]:
            for key in ('previous_state','previous_stress','displacement_before','assembly_displacement','global_matrix','ips'):assert b[key]==bases[0][key],key
        b=bases[0];g=b['global_matrix'];k=np.diag(g['ad'])
        for col in range(8):
            for offset in range(g['jq'][col]-1,g['jq'][col+1]-1):
                r=g['irow'][offset]-1;k[r,col]=k[col,r]=g['au'][offset]
        c=next(c for c in evidence[0]['cases'] if c['cycle']==cycle and c['h']==0)
        delta=correction(b,F/c['run']);consistent=k+delta
        iteration=np.eye(8)-np.linalg.solve(k,consistent)
        eigen=np.linalg.eigvals(iteration)
        prev=None
        for h in (1e-6,5e-7):
            fd=np.column_stack([next(x['derivative'] for x in e['comparisons'] if x['cycle']==cycle and x.get('h')==h) for e in evidence])
            errors=[]
            for col in range(8):
                absolute=float(np.linalg.norm(fd[:,col]-consistent[:,col]));relative=absolute/max(np.linalg.norm(fd[:,col]),1e-30)
                assert relative<2e-7 or absolute<1e-7
                errors.append(dict(column=col,relative_error=relative,absolute_error_N_per_mm=absolute))
            rows.append(dict(cycle=cycle,h=h,native_relative_error=float(np.linalg.norm(fd-k)/np.linalg.norm(fd)),corrected_relative_error=float(np.linalg.norm(fd-consistent)/np.linalg.norm(fd)),columns=errors,step_change=None if prev is None else float(np.linalg.norm(fd-prev)/np.linalg.norm(fd)),iteration_spectral_radius=float(max(abs(eigen))),iteration_spectral_norm=float(np.linalg.norm(iteration,2)),native_condition_number=float(np.linalg.cond(k)),correction_relative_norm=float(np.linalg.norm(delta)/np.linalg.norm(k)),correction_asymmetry_norm=float(np.linalg.norm(delta-delta.T)),finite_difference_jacobian=fd.tolist(),native_matrix=k.tolist(),analytic_correction=delta.tolist()))
            prev=fd
    result=dict(comparisons=rows,provenance=provenance,script_sha256=sha256(__file__),hashes_verified=hash_count,limits=dict(column_relative_error=2e-7,near_zero_absolute_N_per_mm=1e-7),limitations='Fixed per-IP history single C3D8 coupon, eight free basis directions. Residual is negative internal force, with zero external load. Local linear Newton estimate only; no nonlinear convergence guarantee or pipe causation. No production patch, pressure/contact/MPC/cap/energy/Abaqus/experimental acceptance.')
    out=F/'results/output'/('plastic_jacobian_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict((k,v) for k,v in r.items() if k not in ('columns','finite_difference_jacobian','native_matrix','analytic_correction')) for r in rows],indent=2))
if __name__=='__main__':main(sys.argv[1])
