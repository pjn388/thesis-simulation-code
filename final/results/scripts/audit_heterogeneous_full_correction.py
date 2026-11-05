"""Independently assemble full analytic minus actual native local tangent."""
import json,sys
from pathlib import Path
import numpy as np
from verify_heterogeneous_coupon import F,geometry,points,tensors,sha256,vector

def main(tag):
    local=F/'results/output/heterogeneous_material_v2_20261017T043331Z.json'
    global_report=F/'results/output/heterogeneous_jacobian_audit_v2_20261017T043331Z.json'
    ld=json.loads(local.read_text());gd=json.loads(global_report.read_text());count=0
    for report in (ld,gd):
        for n,h in report['provenance'].items():assert sha256(F/n)==h;count+=1
    rows=[];provenance={}
    for cycle in (30,60):
        base=F/f'sym/runs/investigation/plastic_jacobian_v2_20261017T043331Z_b0/c{cycle}_h0.0/assembly.json'
        d=json.loads(base.read_text());provenance[str(base.relative_to(F))]=sha256(base)
        u=np.array(d['displacement_after']).reshape(8,4)[:,1:];assembled=np.zeros((24,24));contributions=[]
        for index,point in enumerate(points(),1):
            ip=next(p for p in d['ips'] if p['ip']==index);r=next(r for r in ld['comparisons'] if r['cycle']==cycle and r['ip']==index)
            jac,grad,deformation=geometry(point,u)
            np.testing.assert_allclose(deformation.T@deformation,np.eye(3)+2*tensors(np.array(ip['strain'])[None])[0],rtol=0,atol=2e-14)
            native=np.zeros((6,6))
            for (i,j),v in zip([(i,j) for j in range(6) for i in range(j+1)],ip['stiffness']):native[i,j]=native[j,i]=v
            np.testing.assert_array_equal(native,r['native_matrix']);delta=np.array(r['full_matrix'])-native;b=np.zeros((6,24))
            for node in range(8):
                for axis in range(3):
                    de=(np.outer(deformation[axis],grad[node])+np.outer(grad[node],deformation[axis]))/2
                    b[:,3*node+axis]=vector(de)*[1,1,1,2,2,2]
            contribution=b.T@delta@b*np.linalg.det(jac);assembled+=contribution
            contributions.append(dict(ip=index,norm_N_per_mm=float(np.linalg.norm(contribution))))
        mapping=np.array(d['mapping']).reshape(8,4)[:,1:].ravel();projected=np.zeros((d['neq'],d['neq']))
        for i,a in enumerate(mapping):
            for j,b in enumerate(mapping):
                if a>0 and b>0:projected[a-1,b-1]+=assembled[i,j]
        for old in (r for r in gd['comparisons'] if r['cycle']==cycle):
            native=np.array(old['native_matrix']);fd=np.array(old['finite_difference']);predicted=native+projected;columns=[]
            for col in range(d['neq']):
                absolute=float(np.linalg.norm(fd[:,col]-predicted[:,col]));relative=absolute/np.linalg.norm(fd[:,col]);columns.append(dict(column=col,absolute_error_N_per_mm=absolute,relative_error=float(relative),passed=bool(relative<2e-7 or absolute<1e-7)))
            rows.append(dict(cycle=cycle,h=old['h'],passed=all(c['passed'] for c in columns),columns=columns,native_relative_error=old['native_relative_error'],corrected_relative_error=float(np.linalg.norm(fd-predicted)/np.linalg.norm(fd)),fd_asymmetry_norm=float(np.linalg.norm(fd-fd.T)),predicted_asymmetry_norm=float(np.linalg.norm(predicted-predicted.T)),full_correction=projected.tolist(),unexplained_matrix=(fd-predicted).tolist(),ip_contributions=contributions))
    provenance.update({str(p.relative_to(F)):sha256(p) for p in (Path(__file__),local,global_report,Path(__file__).with_name('verify_heterogeneous_coupon.py'))})
    result=dict(checks_passed=all(r['passed'] for r in rows),comparisons=rows,hashes_verified=count,provenance=provenance,limits=dict(relative=2e-7,near_zero_absolute_N_per_mm=1e-7),limitations='Independent B-transpose*(full analytic minus actual native local)*B integration at all8IPs. No fitting or symmetrization. Native local probes independently required; no installed correction or production/pipe acceptance.')
    out=F/'results/output'/('heterogeneous_full_correction_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('columns','full_correction','unexplained_matrix','ip_contributions')} for r in rows],indent=2));assert result['checks_passed']
if __name__=='__main__':main(sys.argv[1])
