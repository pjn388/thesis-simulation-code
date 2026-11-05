"""Check retained native probes against independently integrated face loads."""
import json
import sys
from pathlib import Path
import numpy as np
from audit_nodal_pressure_loads import nodal_loads
from verify_shared_node_capture import FINAL, coupon
from verify_finite_residual import matrix
from pipe_sim.runner import sha256


def main(tag):
    path = FINAL/'results/output'/f'coherent_residual_{tag}.json'
    report = json.loads(path.read_text())
    for p,h in report['provenance'].items():
        assert sha256(FINAL/p) == h,p
    states, errors = {}, []
    for case in report['cases']:
        folder = FINAL/case['run']
        for p,h in case['hashes'].items():
            assert sha256(folder/p) == h,p
        data = json.loads((folder/'matrix.json').read_text())
        r = data['residual']
        _,nodes,elements,face,pins = coupon(case['side'],False,.04)
        mapping = np.array(r['mapping']).reshape(len(nodes),4)[:,1:]
        u = np.array(r['displacement']).reshape(len(nodes),4)[:,1:]
        external = np.zeros(57)
        for record in data['records']:
            conn = np.array(record['connectivity'])-1
            xyz = np.array(record['coordinates'])+u[conn]
            force = nodal_loads(xyz[face],-record['xload'][0])
            for local,value in zip(face,force):
                for eq,f in zip(mapping[conn[local]],value):
                    if eq > 0:
                        external[eq-1] += f
        assert np.max(abs(external-r['fext'])) < 1e-10
        errors.append(float(np.max(abs(external-r['fext']))))
        states[case['side'],case['pressure'],case['amplitude']] = (data,external)
    comparisons = []
    direction = np.array(report['direction'])
    for side in ('minus','plus'):
        for pressure in (0,2):
            base,_ = states[side,pressure,0.]
            k = matrix(base['global_matrix'])
            for h in (1e-4,5e-5):
                (plus,fp),(minus,fm) = (states[side,pressure,s*h] for s in (1,-1))
                dload = (fp-fm)/(2*h)
                dr = (np.array(plus['residual']['residual'])-minus['residual']['residual'])/(2*h)
                predicted = dload-k@direction
                error = float(np.linalg.norm(dr-predicted)/np.linalg.norm(predicted))
                assert np.isfinite(error)
                comparisons.append(dict(side=side,pressure=pressure,h=h,complete_residual_with_independent_pressure_derivative_relative_error=error))
    output = FINAL/'results/output'/f'coherent_residual_independent_{tag}.json'
    assert not output.exists()
    output.write_text(json.dumps(dict(passes_2e8=all(c['complete_residual_with_independent_pressure_derivative_relative_error'] < 2e-8 for c in comparisons),comparisons=comparisons,maximum_external_force_error_n=max(errors),provenance={str(p.relative_to(FINAL)):sha256(p) for p in (Path(__file__),path,Path(__file__).with_name('audit_nodal_pressure_loads.py'))},acceptance='Measured finite-state errors without relaxing2e-8 acceptance. All20 captures/provenance hashes verified; independent quadratic face integration reproduces assembled external force; complete residual directional derivative matches independent external derivative minus native finite-state matrix action; see passes_2e8. One direction/two h/both faces per seed, finite elastic state only. No native patch, later-state or physical acceptance.'),indent=2)+'\n')
    print(output)
    print(json.dumps(comparisons))

if __name__ == '__main__':
    for seed in (1741,1742,1743):
        main(sys.argv[1]+'_s'+str(seed))
