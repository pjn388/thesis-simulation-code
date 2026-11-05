"""Retain full heterogeneous Jacobian comparison, including failed gates."""
import json,sys
import numpy as np
from verify_heterogeneous_coupon import F,correction,mapping_checks,sha256

def main(tag):
    paths=[F/'results/output'/f'plastic_jacobian_{tag}_b{col}.json' for col in range(8)]
    evidence=[json.loads(p.read_text()) for p in paths];hash_count=0
    for e in evidence:
        for name,h in e['provenance'].items():assert sha256(F/name)==h;hash_count+=1
        for c in e['cases']:
            for name,h in c['hashes'].items():assert sha256(F/c['run']/name)==h;hash_count+=1
    rows=[]
    for cycle in (30,60):
        bases=[];runs=[]
        for e in evidence:
            c=next(c for c in e['cases'] if c['cycle']==cycle and c['h']==0);runs.append(F/c['run']);bases.append(json.loads((runs[-1]/'assembly.json').read_text()))
        for b in bases[1:]:
            for key in ('previous_state','previous_stress','displacement_before','assembly_displacement','global_matrix','ips'):assert b[key]==bases[0][key],key
        b=bases[0];g=b['global_matrix'];k=np.diag(g['ad']);n=g['neq']
        assert n==8
        for col in range(n):
            for off in range(g['jq'][col]-1,g['jq'][col+1]-1):
                r=g['irow'][off]-1;k[r,col]=k[col,r]=g['au'][off]
        delta=correction(b,runs[0]);predicted=k+delta;prev=None
        for h in (1e-6,5e-7):
            fd=np.column_stack([next(x['derivative'] for x in e['comparisons'] if x['cycle']==cycle and x.get('h')==h) for e in evidence]);columns=[]
            for col in range(n):
                absolute=float(np.linalg.norm(fd[:,col]-predicted[:,col]));relative=absolute/np.linalg.norm(fd[:,col]);columns.append(dict(column=col,absolute_error_N_per_mm=absolute,relative_error=float(relative),passed=bool(relative<2e-7 or absolute<1e-7)))
            iteration=np.eye(n)-np.linalg.solve(k,fd)
            rows.append(dict(cycle=cycle,h=h,columns=columns,passed=all(c['passed'] for c in columns),native_relative_error=float(np.linalg.norm(fd-k)/np.linalg.norm(fd)),corrected_relative_error=float(np.linalg.norm(fd-predicted)/np.linalg.norm(fd)),step_change=None if prev is None else float(np.linalg.norm(fd-prev)/np.linalg.norm(fd)),native_iteration_spectral_radius=float(max(abs(np.linalg.eigvals(iteration)))),correction_asymmetry_norm=float(np.linalg.norm(delta-delta.T)),fd_asymmetry_norm=float(np.linalg.norm(fd-fd.T)),unexplained_matrix=(fd-predicted).tolist(),finite_difference=fd.tolist(),native_matrix=k.tolist(),analytic_correction=delta.tolist()))
            prev=fd
    result=dict(checks_passed=all(r['passed'] for r in rows),comparisons=rows,mapping_checks=mapping_checks,hashes_verified=hash_count,limits=dict(relative=2e-7,near_zero_absolute_N_per_mm=1e-7),provenance={str(p.relative_to(F)):sha256(p) for p in paths+[__import__('pathlib').Path(__file__),__import__('pathlib').Path(__file__).with_name('verify_heterogeneous_coupon.py')]},limitations='Failure audit, unchanged gates. All-column native fixed-history residual Jacobian; actual native matrix/residual identities checked separately in column reports. No fit or production patch; local contraction not nonlinear or pipe causation.')
    out=F/'results/output'/f'heterogeneous_jacobian_audit_{tag}.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict((k,v) for k,v in r.items() if k not in ('columns','unexplained_matrix','finite_difference','native_matrix','analytic_correction')) for r in rows],indent=2))
if __name__=='__main__':main(sys.argv[1])
