"""Independent C3D8 nodal assembly of full-minus-frozen determinant derivative."""
import itertools,json,sys
from pathlib import Path
import numpy as np
from audit_material_linearization import derivative,vector
from audit_plastic_return_map import F,tensors,sha256

def audit(tag):
    reports=[F/'results/output'/n for n in ('plastic_solver_20261017T042704Z.json','plastic_solver_direction2_20261017T042704Z.json')]
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]],float)
    signs=2*xyz-1;rows=[];provenance={}
    for report in reports:
        evidence=json.loads(report.read_text());provenance[str(report.relative_to(F))]=sha256(report)
        seed=evidence.get('direction_seed',1741)
        for cycle in (30,60):
            case=next(c for c in evidence['cases'] if c['cycle']==cycle and c['h']==0)
            run=F/case['run']
            for n,h in case['hashes'].items():assert sha256(run/n)==h
            data=json.loads((run/'assembly.json').read_text());ips=data['ips']
            # This cube's unperturbed history is homogeneous: no assumed IP ordering.
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
            direction=np.random.default_rng(seed).normal(size=data['neq']);direction/=np.linalg.norm(direction)
            for comparison in evidence['comparisons']:
                if comparison['cycle']!=cycle or 'h' not in comparison:continue
                fd=np.array(comparison['derivative']);native=np.array(comparison['matrix_direction']);predicted=projected@direction
                error=float(np.linalg.norm(fd-native-predicted)/np.linalg.norm(fd));measured=np.linalg.norm(fd-native)
                rows.append(dict(seed=seed,cycle=cycle,h=comparison['h'],native_relative_error=comparison['relative_error'],corrected_relative_error=error,unexplained_fraction_of_discrepancy=None if cycle==60 else float(np.linalg.norm(fd-native-predicted)/measured),correction_action=predicted.tolist()))
                assert error<2e-7
    result=dict(comparisons=rows,provenance=provenance,script_sha256=sha256(__file__),derivation_sha256=sha256(Path(__file__).with_name('audit_material_linearization.py')),limits=dict(corrected_relative_error=2e-7),limitations='Independent full-minus-frozen determinant-hardening derivative assembled with reference C3D8 quadrature and finite-strain B, homogeneous previous native state. Two free directions, plastic and unloading control. Explanatory diagnostic only; nonsymmetric correction not installed. No pipe convergence causation, pressure/contact/MPC/cap/work/Abaqus or experimental validation.')
    out=F/'results/output'/('plastic_nodal_correction_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':audit(sys.argv[1])
