"""Immutable diagnostic of spherical reconstruction; never replace native states."""
import json
from pathlib import Path
import numpy as np
from shear_unloading_coupon import F, MU, BULK, audit, sha256

TAG = 'spherical_metric_audit_20261012T013001Z'


def reconstruct(f, m):
    b = f @ m @ f.T
    d = b - np.trace(b)/3*np.eye(3)
    values = np.linalg.eigvalsh(d)
    low = -values.min()
    high = max(1., low + 1.)
    while np.prod(values + high) < 1.: high *= 2
    for _ in range(80):
        q = (low + high)/2
        if np.prod(values + q) < 1.: low = q
        else: high = q
    q = (low + high)/2
    corrected = d + q*np.eye(3)
    inverse = np.linalg.inv(f)
    metric = inverse @ corrected @ inverse.T
    i2 = -.5*np.trace(d@d)
    cubic = q**3 + i2*q + np.linalg.det(d) - 1.
    assert abs(cubic) < 1e-12
    assert abs(np.linalg.det(metric)-1.) < 1e-12
    assert np.linalg.eigvalsh(metric).min() > 0
    assert abs((corrected-np.trace(corrected)/3*np.eye(3))-d).max() < 1e-12
    return metric, d, q, abs(cubic)


def run():
    recovery = F/'results/output/shear_unloading_recovery_20261012T003001Z.json'
    refinement = F/'results/output/shear_metric_refinement_20261012T010000Z.json'
    retained = json.loads(recovery.read_text())
    refined = json.loads(refinement.read_text())
    assert retained['immutable_replay_checks_passed']
    assert not retained['original_monotonic_work_gate_passed']
    assert refined['identity_checks_passed']
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    assert sha256(source) == retained['manifest']['source_sha256']
    records = list(retained['records'])
    for item in refined['records'][3:]:
        directory = F/item['run']
        lines = (directory/'job.inp').read_text().splitlines()
        curve = []
        for line in lines[lines.index('*PLASTIC')+1:]:
            if line.startswith('*'): break
            curve.append(list(map(float,line.split(','))))
        records.append(audit(directory,item['divisions'],np.array(curve)))
    summaries = []
    for record in records:
        assert all(sha256(F/record['run']/n)==h for n,h in record['hashes'].items())
        reconstructed = []; normalization_error = 0.; cubic_error = 0.; state_change = 0.
        for h in record['history']:
            f = np.eye(3); f[0,1] = .03*h['path_fraction']
            m = np.array(h['metric'])
            candidate,d,q,error = reconstruct(f,m)
            reconstructed.append(candidate)
            state_change = max(state_change,float(abs(candidate-m).max()))
            cubic_error = max(cubic_error,error)
            # Uniform determinant normalization scales the deviatoric stress too.
            scale = np.linalg.det(m)**(-1/3)
            normalization_error = max(normalization_error,float(abs(MU*(scale-1)*d).max()))
        loaded = record['divisions']//2-1
        unloading_change = max(float(abs(m-reconstructed[loaded]).max())
                               for m in reconstructed[loaded+1:])
        assert unloading_change > 1e-12
        summaries.append(dict(run=record['run'],divisions=record['divisions'],hashes=record['hashes'],
            maximum_metric_change=state_change,maximum_cubic_residual=cubic_error,
            maximum_uniform_normalization_stress_change_mpa=normalization_error,
            postprocessed_metric_change_during_native_elastic_unloading=unloading_change,
            loaded_native_determinant=record['history'][loaded]['metric_determinant']))
    paper = F/'sym/runs/investigation/source_evidence_20261012T013001Z/geers_eccomas2000.pdf'
    report = dict(checks_passed=True,physical_acceptance=False,records=summaries,
        provenance=dict(script_sha256=sha256(__file__),paper_path=str(paper.relative_to(F)),
            paper_sha256=sha256(paper),source_sha256=sha256(source),
            recovery_sha256=sha256(recovery),refinement_sha256=sha256(refinement),
            dependencies={n:sha256(F/'results/scripts'/n) for n in
                ('shear_unloading_coupon.py','audit_plastic_return_map.py','review_plastic_state_capture.py')}),
        reference=dict(title='Geers et al., Large deformation gradient-enhanced hyperelasto-plasticity for ductile failure, ECCOMAS 2000',
            url='https://congress2.cimne.com/eccomas/proceedings/eccomas2000/pdf/90.pdf',
            equations='12/13 (p5/6), 26-32 (p7/8), 36-40 (p9)'),
        inference='Native C:deltaM=0 maps to spatial trace(F deltaM F^T)=0 (eq12), distinct from inverse-Be contraction (eq13). Spherical reconstruction preserves instantaneous deviatoric stress; uniform metric normalization changes it. Per-state spherical reconstruction changes a native frozen metric on unloading, so it cannot be accepted as a history-consistent correction.',
        limitations='Read-only hypothetical algebra, no corrected trajectory or solver. Paper effective-plastic-strain convention differs; its hardening/multiplier/tangent must be mapped together, not imported as a calibrated replacement. No physical storage/dissipation, experimental, pressure-cap/contact/MPC/Abaqus or burst acceptance. Original monotonic-work gate remains failed.',
        next='Derive a coupled reference material-point update and tangent with explicit hardening/PEEQ convention, then test objectivity, yield consistency, frozen elastic unloading and finite-difference tangent before any native fork.')
    out = F/'results/output'/f'{TAG}.json'
    assert not out.exists()
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),records=[{k:v for k,v in r.items() if k!='hashes'} for r in summaries])))


if __name__ == '__main__': run()
