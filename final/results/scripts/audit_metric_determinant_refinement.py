"""Measure determinant drift without normalizing retained native plastic states."""
import json
import numpy as np
from audit_plastic_return_map import F, sha256

GATE = F/'results/output/elastic_unloading_coupon_20261011T230008Z.json'
CANDIDATES = F/'results/output/unloading_candidates_20261011T233004Z.json'
OUT = F/'results/output/metric_determinant_refinement_20261012T000008Z.json'


def run():
    gate = json.loads(GATE.read_text())
    candidates = json.loads(CANDIDATES.read_text())
    assert gate['frozen_state_work_checks_passed'] and candidates['identity_checks_passed']
    assert candidates['gate_sha256'] == sha256(GATE)
    records = []
    for retained, comparison in zip(gate['records'], candidates['records'], strict=True):
        assert retained['run'] == comparison['run']
        directory = F/retained['run']
        assert all(sha256(directory/n) == h for n,h in retained['hashes'].items())
        assert retained['replay_checks_passed']
        loading = retained['history'][:retained['divisions']//2]
        unloading = retained['history'][retained['divisions']//2:]
        start = loading[-1]
        determinants = [float(np.linalg.det(h['metric'])) for h in retained['history']]
        assert max(abs(d-h['metric_determinant']) for d,h in zip(determinants,retained['history'])) < 1e-12
        assert max(abs(h['metric_determinant']-start['metric_determinant']) for h in unloading) < 1e-10
        records.append(dict(run=retained['run'],divisions=retained['divisions'],
            loading_intervals=len(loading),hashes=retained['hashes'],
            loaded_metric_determinant=start['metric_determinant'],
            loaded_determinant_departure=start['metric_determinant']-1,
            maximum_loading_determinant_departure=max(abs(h['metric_determinant']-1) for h in loading),
            loaded_peeq=start['maximum_peeq'],
            maximum_elastic_j_work_error_n_mm=comparison['maximum_elastic_j_work_error_n_mm'],
            maximum_total_j_work_error_n_mm=comparison['maximum_total_j_work_error_n_mm']))
    differences = [records[i+1]['loaded_determinant_departure']-records[i]['loaded_determinant_departure'] for i in range(2)]
    result = dict(records=records,identity_checks_passed=True,physical_acceptance=False,
        nested_determinant_differences=differences,
        successive_difference_magnitude_ratio=abs(differences[1]/differences[0]),
        determinant_departures_decrease=all(abs(b['loaded_determinant_departure'])<abs(a['loaded_determinant_departure']) for a,b in zip(records,records[1:])),
        gate_sha256=sha256(GATE),candidates_sha256=sha256(CANDIDATES),script_sha256=sha256(__file__),
        acceptance='Immutable native hashes; retained native replay gates; recomputed determinant agreement1e-12; frozen unloading determinant1e-10. Refinement trends measured without fitting or enforcing convergence.',
        references='incplas.f:156-189,775-790; retained unloading replay and candidate comparison reports.',
        limitations='Three loading resolutions on one artificial homogeneous trajectory cannot establish an asymptotic determinant limit or causation of energy discrepancy. Native metrics not normalized. No physical storage/dissipation, production cap/contact/MPC/Abaqus or burst acceptance.',
        next='Inspect native metric update and continuum isochoric-flow convention to explain measured determinant refinement before proposing a new bounded trajectory or production revision.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': run()
