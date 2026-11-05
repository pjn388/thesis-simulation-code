"""Attribute heterogeneous assembled remainder to each captured local tangent."""
import json,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,tensors,sha256,update
from audit_material_linearization import derivative,vector

def main(tag):
    rows=[];provenance={}
    for cycle in (30,60):
        run=F/f'sym/runs/investigation/plastic_jacobian_{tag}_b0/c{cycle}_h0.0';path=run/'assembly.json';d=json.loads(path.read_text());provenance[str(path.relative_to(F))]=sha256(path)
        lines=(run/'job.inp').read_text().splitlines();curve=[]
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'):break
            curve.append(list(map(float,line.split(','))))
        curve=np.array(curve)
        for ip in d['ips']:
            assert ip['icmd']==0
            c=np.eye(3)+2*tensors(np.array(ip['strain'])[None])[0];prev=np.array(ip['previous']);m=np.eye(3)-2*tensors(prev[None,1:7])[0]
            mu=202500/(2*1.3);bulk=202500/(3*.4);k=np.zeros((6,6))
            for (i,j),v in zip([(i,j) for j in range(6) for i in range(j+1)],ip['stiffness']):k[i,j]=k[j,i]=v
            exact=[];frozen=[];fd=[]
            for col in range(6):
                v=np.eye(6)[col]/(1 if col<3 else 2);dc=2*tensors(v[None])[0];h=1e-7
                exact.append(vector(derivative(c,m,prev[0],curve,mu,bulk,dc)));frozen.append(vector(derivative(c,m,prev[0],curve,mu,bulk,dc,True)))
                fd.append(vector((update(c+h*dc,m,np.array(prev[0]),curve,mu,bulk)[0]-update(c-h*dc,m,np.array(prev[0]),curve,mu,bulk)[0])/(2*h)))
            exact=np.array(exact).T;frozen=np.array(frozen).T;fd=np.array(fd).T
            stress=vector(update(c,m,np.array(prev[0]),curve,mu,bulk)[0]);se=float(np.max(np.abs(stress-ip['stress'])))
            ef=float(np.linalg.norm(exact-fd)/np.linalg.norm(fd));assert ef<2e-7 and se<1e-4
            rows.append(dict(cycle=cycle,ip=ip['ip'],stress_max_error_mpa=se,analytic_fd_relative_error=ef,frozen_yield_j_vs_native=float(np.linalg.norm(frozen-k)/np.linalg.norm(k)),full_vs_native=float(np.linalg.norm(exact-k)/np.linalg.norm(k)),native_matrix=k.tolist(),full_matrix=exact.tolist(),frozen_matrix=frozen.tolist()))
    for p in (Path(__file__),Path(__file__).with_name('audit_material_linearization.py'),Path(__file__).with_name('audit_plastic_return_map.py'),F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'):provenance[str(p.relative_to(F))]=sha256(p)
    out=F/'results/output'/f'heterogeneous_material_{tag}.json';assert not out.exists();out.write_text(json.dumps(dict(comparisons=rows,provenance=provenance,limitations='Offline analytic constitutive derivative and independent central differences at captured heterogeneous IP states; no native local strain perturbations. Frozen-yield-J sufficiency is tested, not assumed. No patch/pipe/experimental acceptance.'),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if 'matrix' not in k} for r in rows],indent=2))
if __name__=='__main__':main(sys.argv[1])
