"""Refine retained local tangent differences, separating hardening knot crossings."""
import json,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,tensors,sha256,update


def main(tag):
    evidence=F/'results/output'/('material_tangent_'+tag+'.json')
    report=json.loads(evidence.read_text());run=F/report['run']
    for name,h in report['hashes'].items():assert sha256(run/name)==h
    for name,h in report['provenance'].items():assert sha256(F/name)==h
    lines=(run/'job.inp').read_text().splitlines();curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve);records=json.loads((run/'tangent.json').read_text())['records']
    # Worst directional case plus final loading, elastic unloading and interior plastic.
    worst=max(report['comparisons'],key=lambda r:max(d['relative_error'] for d in r['derivatives']))['index']
    indices=sorted(set([worst,70,118,171]))
    comparisons=[]
    for index in indices:
        r=records[index];e=np.array(r['emec']);previous=np.array(r['previous_state'])
        metric=np.eye(3)-2*tensors(previous[None,1:7])[0];q=np.array(previous[0])
        young,nu=r['elastic'];mu=young/(2*(1+nu));bulk=young/(3*(1-2*nu))
        def predict(v):return update(np.eye(3)+2*tensors(v[None])[0],metric,q,curve,mu,bulk)
        def vector(s):return np.array([s[0,0],s[1,1],s[2,2],s[0,1],s[0,2],s[1,2]])
        base=predict(e);k=np.zeros((6,6))
        for (i,j),v in zip([(i,j) for j in range(6) for i in range(j+1)],r['stiffness']):k[i,j]=k[j,i]=v
        for h in (1e-6,5e-7,1e-7,5e-8,1e-8):
            derivative=np.zeros((6,6));segments=[];branches=[]
            for column in range(6):
                perturb=np.eye(6)[column]*h
                plus,minus=predict(e+perturb),predict(e-perturb)
                derivative[:,column]=(vector(plus[0])-vector(minus[0]))/(2*h*(1 if column<3 else 2))
                segments.append([int(np.searchsorted(curve[:,1],float(s[2]),side='right')) for s in (minus,base,plus)])
                branches.append([bool(s[3]) for s in (minus,base,plus)])
            symmetric=(derivative+derivative.T)/2
            comparisons.append(dict(index=index,time=r['time'],ttime=r['ttime'],plastic=bool(base[3]),h=h,current_peeq=float(base[2]),previous_peeq=float(q),hardening_segments=segments,branch_flags=branches,relative_full_matrix_error=float(np.linalg.norm(derivative-k)/np.linalg.norm(derivative)),relative_symmetric_part_error=float(np.linalg.norm(symmetric-k)/np.linalg.norm(derivative)),relative_antisymmetric_part=float(np.linalg.norm(derivative-symmetric)/np.linalg.norm(derivative)),native_matrix=k.tolist(),independent_matrix=derivative.tolist()))
    output=F/'results/output'/('material_tangent_steps_'+tag+'.json');assert not output.exists()
    result=dict(comparisons=comparisons,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),evidence,Path(__file__).with_name('audit_plastic_return_map.py'))},limitations='Retained exact local previous history; full six-component derivative refinement. Native tangent is packed symmetric. No implication of assembled solver defect, cap/contact/MPC/Abaqus or thermodynamic acceptance. No tolerance fitted.')
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k not in ('native_matrix','independent_matrix','hardening_segments','branch_flags')} for r in comparisons],indent=2))
if __name__=='__main__':main(sys.argv[1])
