"""Explain injected X52 local tangent; explicitly not assembled pipe Newton behavior."""
import json,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,tensors,sha256
from audit_material_linearization import derivative,vector

def main(tag):
    parent=F/'results/output'/('pipe_material_'+tag+'.json');evidence=json.loads(parent.read_text());assert evidence['checks_passed']
    for case in evidence['cases']:
        for n,h in case['hashes'].items():assert sha256(F/case['run']/n)==h
    for n,h in evidence['provenance'].items():assert sha256(F/n)==h
    run=F/evidence['parent_run']
    for n,h in evidence['parent_hashes'].items():assert sha256(run/n)==h
    material=json.loads((run/'case.json').read_text())['material'];young,nu=material['young'],material['poisson'];mu=young/(2*(1+nu));bulk=young/(3*(1-2*nu))
    lines=(run/'job.inp').read_text().splitlines();curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve);rows=[]
    for state in evidence['selections']:
        row=next(r for r in evidence['comparisons'] if r['selection']==state['name']);exact=np.array(row['analytic_matrix']);native=np.array(row['native_matrix'])
        prev=np.array(state['previous']);c=np.eye(3)+2*tensors(np.array(state['strain'])[None])[0];m=np.eye(3)-2*tensors(prev[None,1:7])[0]
        frozen=np.array([vector(derivative(c,m,prev[0],curve,mu,bulk,2*tensors((np.eye(6)[col]/(1 if col<3 else 2))[None])[0],True)) for col in range(6)]).T
        sym=(frozen+frozen.T)/2;det=exact-frozen;anti=(frozen-frozen.T)/2
        relative=float(np.linalg.norm(native-sym)/np.linalg.norm(native));closure=float(np.linalg.norm(exact-native-det-anti)/np.linalg.norm(exact))
        iteration=np.eye(6)-np.linalg.solve(native,exact)
        rows.append(dict(selection=state['name'],symmetric_frozen_vs_native_relative=relative,decomposition_closure_relative=closure,full_vs_native_relative=float(np.linalg.norm(exact-native)/np.linalg.norm(exact)),determinant_term_rank=int(np.linalg.matrix_rank(det,tol=1e-6)),isolated_material_newton_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(iteration)))),isolated_material_newton_norm2=float(np.linalg.norm(iteration,2)),native_condition2=float(np.linalg.cond(native))))
    passed=all(r['symmetric_frozen_vs_native_relative']<2e-12 and r['decomposition_closure_relative']<2e-12 for r in rows)
    out=F/'results/output'/('pipe_material_tangent_'+tag+'.json');assert not out.exists()
    result=dict(checks_passed=passed,comparisons=rows,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),parent,Path(__file__).with_name('audit_material_linearization.py'),F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f')},source_reference='incplas.f:791-1019 symmetric packed21 tangent and averaged noncoaxial xn/xd terms',limitations='Isolated6component fixed-history material Newton operator only; no geometric/external/assembled pipe matrix or residual, global Newton contraction, failure-state plasticity, production revision or physical validation. Engineering shear conventions retained; analytic matrices remain unsymmetrized.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2));assert passed
if __name__=='__main__':main(sys.argv[1])
