"""Explain native local tangent via frozen yield-J and symmetric projection.

This is an audit identity; no projection is applied to an independent Jacobian.
incplas.f:791-1019 stores only21 symmetric entries; the xn/xd cross terms
at lines 891 onward explicitly average the two orders. The full local map
need not have that major symmetry with determinant-dependent yield stress.
"""
import json,sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F,sha256

def main(tag):
    parent=F/'results/output/heterogeneous_material_v2_20261017T043331Z.json';d=json.loads(parent.read_text())
    for n,h in d['provenance'].items():assert sha256(F/n)==h
    rows=[]
    for r in d['comparisons']:
        full=np.array(r['full_matrix']);frozen=np.array(r['frozen_matrix']);native=np.array(r['native_matrix'])
        symmetric=(frozen+frozen.T)/2;yield_term=full-frozen;skew_term=(frozen-frozen.T)/2
        error=float(np.linalg.norm(symmetric-native)/np.linalg.norm(native));closure=float(np.linalg.norm(full-native-yield_term-skew_term)/np.linalg.norm(full))
        rows.append(dict(cycle=r['cycle'],ip=r['ip'],native_vs_symmetric_frozen_relative=error,decomposition_closure_relative=closure,yield_determinant_relative=float(np.linalg.norm(yield_term)/np.linalg.norm(full)),frozen_antisymmetric_relative=float(np.linalg.norm(skew_term)/np.linalg.norm(full)),passed=bool(error<2e-7 and closure<2e-7)))
    result=dict(checks_passed=all(r['passed'] for r in rows),comparisons=rows,identity='full-native = (full-frozen_yield_J) + (frozen_yield_J-frozen_yield_J.T)/2, to native floating precision',limits=dict(relative=2e-7),provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),parent,F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f')},limitations='Offline diagnostic decomposition on retained states. Symmetric projection describes native storage, not an approved change to analytic or global matrices. No proof of pipe stagnation or production suitability.')
    out=F/'results/output'/('heterogeneous_symmetry_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2));assert result['checks_passed']
if __name__=='__main__':main(sys.argv[1])
