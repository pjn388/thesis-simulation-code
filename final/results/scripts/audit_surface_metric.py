"""Diagnostic surface metric replay; never modifies production mapping."""
import json, sys
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F, sha256
import audit_pressure_branch_mapping as mapping
from audit_quadratic_pressure_derivative import shape, assemble

def metric_gradients(x, r, s):
    n, dr, ds = shape(r, s)
    tangent = np.column_stack((dr @ x, ds @ x))
    inverse = np.linalg.solve(tangent.T @ tangent, tangent.T)
    return n, np.column_stack((dr, ds)) @ inverse, np.cross(tangent[:, 0], tangent[:, 1]), tangent, inverse

def main(tag):
    parent = F/'results/output/shape8q_mapping_v3_20261017T052506Z.json'
    prior = json.loads(parent.read_text())
    for run, hashes in prior['run_hashes'].items():
        for name, h in hashes.items():
            assert sha256(F/run/name) == h
    records = []
    original = mapping.global_gradients
    try:
        mapping.global_gradients = metric_gradients
        for record in prior['records']:
            errors = []; gradients = []
            for point in record['quadrature']:
                x = np.array(point['coordinates']).reshape(8, 3)
                r, s = point['r'], point['s']
                _, dr, ds = shape(r, s)
                _, g, _, tangent, inverse = metric_gradients(x, r, s)
                errors.append(float(np.linalg.norm(g @ tangent - np.column_stack((dr, ds)))/np.linalg.norm(np.column_stack((dr, ds)))))
                gradients.append(g)
                assert np.linalg.norm(inverse @ tangent-np.eye(2)) < 2e-12
            matrix, _ = mapping.branch(x, x, 2.3)
            _, raw = assemble(x, 2)
            result = dict(label=record['label'], chain_relative=max(errors), symmetric_relative=float(np.linalg.norm(matrix-(raw+raw.T)/2)/np.linalg.norm(raw)), raw_relative=float(np.linalg.norm(matrix-raw)/np.linalg.norm(raw)))
            if record['label']=='planar_0':
                flat=x; base=matrix; base_gradients=gradients
            elif record['label'].startswith('planar_'):
                # Recover the affine rigid rotation on the tangent plane; normal fixes orientation.
                def frame(y):
                    _, dr, ds=shape(0.,0.);t=np.column_stack((dr@y,ds@y));return np.column_stack((t,np.cross(t[:,0],t[:,1])))
                rotation=frame(x) @ np.linalg.inv(frame(flat))
                assert np.linalg.norm(rotation.T@rotation-np.eye(3))<2e-12
                block=np.kron(np.eye(8),rotation)
                result['rotation_relative']=float(np.linalg.norm(block.T@matrix@block-base)/np.linalg.norm(base))
                result['gradient_rotation_relative']=float(max(np.linalg.norm(g@rotation-b)/np.linalg.norm(b) for g,b in zip(gradients,base_gradients)))
            records.append(result)
    finally:
        mapping.global_gradients=original
    passed=all(r['chain_relative']<2e-12 and r['symmetric_relative']<2e-7 and r.get('rotation_relative',0)<2e-12 and r.get('gradient_rotation_relative',0)<2e-12 for r in records)
    report=dict(checks_passed=passed,records=records,relative_tangent_limit=2e-7,reconstruction_rotation_limit=2e-12,provenance={str(p.relative_to(F)):sha256(p) for p in (Path(__file__),parent,Path(mapping.__file__),Path(__file__).with_name('audit_quadratic_pressure_derivative.py'))},limitations='Algebra only on retained curved reference=current and three affine controls; native tangent gate remains failed. No production patch, branch activation, global assembly, convergence causation, plastic/cap/contact/MPC/Abaqus or experimental validation.')
    out=F/'results/output'/('surface_metric_'+tag+'.json');assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));assert passed
if __name__=='__main__':main(sys.argv[1])
