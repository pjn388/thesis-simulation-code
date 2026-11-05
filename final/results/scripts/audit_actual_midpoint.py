"""Compare immutable actual C paths with assumed chords; no physical acceptance."""
import json
import sys
import numpy as np
from compare_native_plastic_time import F, STANDARD, QUARTER, load, field, compare
from audit_plastic_return_map import tensors, sha256


def audit(gate, lo, hi):
    evidence, run, records, raw = load(gate)
    def c(rec):
        return np.eye(3)+2*tensors(field(raw, rec, 'strain', 6))
    start = next(r for r in records if abs(24*r['time']-lo)<1e-9)
    end = next(r for r in records if abs(24*r['time']-hi)<1e-9)
    c0, c1 = c(start), c(end)
    mids = []
    for rec in records:
        p = 24*rec['time']
        if not lo+1e-9 < p < hi-1e-9:
            continue
        fraction = (p-lo)/(hi-lo)
        chord = c0+(c1-c0)*fraction
        actual = c(rec)
        mids.append(dict(pressure_mpa=p, fraction=fraction,
                         maximum_c_component_chord_error=float(abs(actual-chord).max()),
                         relative_frobenius_chord_error=float((np.linalg.norm(actual-chord,axis=(-2,-1))/np.linalg.norm(actual,axis=(-2,-1))).max())))
    # Exact additive determinant identity and second-order expansion of det(I+A).
    selected = [r for r in records if lo-1e-9 <=24*r['time']<=hi+1e-9]
    determinant = []
    for previous, current in zip(selected, selected[1:]):
        def metric(rec):
            s = field(raw, rec, 'state', rec['states'])
            return np.eye(3)-2*tensors(s[...,1:7])
        m0, m1 = metric(previous), metric(current)
        a = np.linalg.solve(m0,m1-m0)
        trace = np.trace(a,axis1=-2,axis2=-1)
        second = .5*(trace**2-np.trace(a@a,axis1=-2,axis2=-1))
        cubic = np.linalg.det(a)
        exact = np.linalg.det(m1)/np.linalg.det(m0)-1
        error = float(abs(exact-(trace+second+cubic)).max())
        assert error<1e-12, error
        determinant.append(dict(lower_mpa=24*previous['time'],upper_mpa=24*current['time'],
            maximum_exact_relative_determinant_change=float(abs(exact).max()),
            maximum_first_order_trace=float(abs(trace).max()),
            maximum_second_order_term=float(abs(second).max()),
            maximum_cubic_term=float(abs(cubic).max()),identity_error=error))
    assert all(sha256(run/n)==h for n,h in evidence['input_output_hashes'].items())
    return dict(run=evidence['run'],gate_sha256=sha256(gate),interior_frames=mids,additive_determinant=determinant)


if __name__=='__main__':
    out = F/'results/output'/sys.argv[1]
    assert not out.exists()
    retained = F/'results/output/interval_substeps_20261011T163001Z.json'
    source = json.loads(retained.read_text())
    assert source['identity_checks_passed']
    assert compare()['provenance_checks_passed']
    lo, hi = source['lower_mpa'],source['upper_mpa']
    result = dict(provenance_checks_passed=True,physical_acceptance=False,
        lower_mpa=lo,upper_mpa=hi,lanes=[audit(g,lo,hi) for g in (STANDARD,QUARTER)],
        retained_substeps_sha256=sha256(retained),script_sha256=sha256(__file__),
        dependencies={n:sha256(F/'results/scripts'/n) for n in ('compare_native_plastic_time.py','audit_plastic_return_map.py')},
        source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'),
        limitations='All captured integration points; extrema need not coincide. Actual accepted C is compared with each lane own endpoint chord. Algebraic determinant identity is not physical energy or temporal convergence. No normalization, tolerance fitting, production load revision or burst claim.')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
