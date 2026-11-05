"""Isolate conservative elastic trial quadrature; never integrate plastic states."""
import json
import sys
from pathlib import Path

import numpy as np
from compare_native_plastic_time import STANDARD, QUARTER, load, field
from audit_plastic_return_map import F, tensors, sha256
from audit_internal_pipe_fields import spatial
from pipe_sim.model import Model


def elastic(c, metric, mu, bulk):
    """Potential and 2*dW/dC for incplas.f's fixed-metric elastic trial."""
    j = np.sqrt(np.linalg.det(c))
    assert np.all(j > 0)
    ci = np.linalg.inv(c)
    trace = np.einsum('...ij,...ij->...', c, metric)/j**(2/3)
    potential = mu/2*trace + bulk/4*(j*j-1)-bulk/2*np.log(j)
    stress = mu*(metric/j[..., None, None]**(2/3)-trace[..., None, None]*ci/3)
    stress += bulk/2*(j*j-1)[..., None, None]*ci
    return potential, stress


def audit(gate):
    evidence, run, records, raw = load(gate)
    model = Model.load(run/'model.json')
    inside = False
    for line in (run/'mesh.inp').read_text().splitlines():
        if line.startswith('**'): continue
        if line.startswith('*'): inside = line.upper().strip() == '*NODE'
        elif inside and line.strip():
            row = line.split(',')
            model.nodes[int(row[0])] = list(map(float, row[1:4]))
    ids, nodes, gradients, weights, checks = spatial(model)
    material = json.loads((run/'case.json').read_text())['material']
    mu = material['young']/(2*(1+material['poisson']))
    bulk = material['young']/(3*(1-2*material['poisson']))
    metric = np.broadcast_to(np.eye(3), (len(ids), 8, 3, 3)).copy()
    old_f = metric.copy()
    counts = (1, 2, 4, 8)
    work = {n: 0. for n in counts}
    exact = 0.
    # Independent analytic derivative check at a manufactured SPD C and metric.
    ctest = np.diag([1.03, .98, 1.01]); mtest = np.diag([.99, 1.02, 1.])
    direction = np.array([[.3, .1, .2], [.1, -.2, .05], [.2, .05, .4]])
    eps = 1e-5
    pplus, _ = elastic(ctest+eps*direction, mtest, mu, bulk)
    pminus, _ = elastic(ctest-eps*direction, mtest, mu, bulk)
    _, s = elastic(ctest, mtest, mu, bulk)
    derivative_error = float(abs((pplus-pminus)/(2*eps)-.5*np.sum(s*direction)))
    assert derivative_error < 1e-4
    history = []
    for rec in records:
        new_f = np.eye(3)+np.einsum('eni,epnj->epij', field(raw, rec, 'u', 4)[nodes-1], gradients)
        old_c = np.swapaxes(old_f, -1, -2)@old_f
        new_c = np.swapaxes(new_f, -1, -2)@new_f
        p0, s0 = elastic(old_c, metric, mu, bulk)
        p1, _ = elastic(new_c, metric, mu, bulk)
        exact += float(((p1-p0)*weights).sum())
        for n in counts:
            previous_c, previous_s = old_c, s0
            for k in range(1, n+1):
                f = old_f+(new_f-old_f)*k/n
                c = np.swapaxes(f, -1, -2)@f
                _, stress = elastic(c, metric, mu, bulk)
                work[n] += float((.25*np.einsum('epij,epij->ep', stress+previous_s, c-previous_c)*weights).sum())
                previous_c, previous_s = c, stress
        history.append(dict(time=rec['time'], exact_frozen_trial_work_n_mm=exact,
                            quadrature_minus_exact_n_mm={n: work[n]-exact for n in counts}))
        # Each interval resets to its retained native starting plastic metric.
        state = field(raw, rec, 'state', rec['states'])[np.array(ids)-1]
        metric = np.eye(3)-2*tensors(state[..., 1:7])
        old_f = new_f
    errors = [abs(work[n]-exact) for n in counts]
    decreasing = all(b < a for a, b in zip(errors, errors[1:]))
    assert decreasing, 'Preserve nonmonotone quadrature outcome for investigation.'
    assert all(sha256(run/n) == h for n, h in evidence['input_output_hashes'].items())
    return dict(run=evidence['run'], accepted_frames=len(records), elements=len(ids),
                derivative_error_n_mm_per_mm3=derivative_error, derivative_limit=1e-4,
                exact_frozen_trial_work_n_mm=exact, resolutions=[dict(substeps=n,
                trial_work_n_mm=work[n], quadrature_minus_exact_n_mm=work[n]-exact) for n in counts],
                quadrature_error_decreases=bool(decreasing), history=history,
                gate_sha256=sha256(gate), input_output_hashes=evidence['input_output_hashes'], spatial_checks=checks)


if __name__ == '__main__':
    out = Path(sys.argv[1]); assert not out.exists()
    result = dict(trajectories=[audit(g) for g in (STANDARD, QUARTER)],
                  script_sha256=sha256(__file__), source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'),
                  sources='incplas.f elastic trial and bulk term; resultsmech.f endpoint-average material work. W=mu/2 tr(C M) J^(-2/3)+K/4(J^2-1)-K/2 ln J; S=2 dW/dC at fixed M.',
                  limitations='Counterfactual elastic trial with native plastic metric frozen separately per output interval; resets exclude metric-evolution work. Sum of interval potential differences is not a single endpoint stored energy, plastic work, dissipation, actual native trajectory, correction or bound on external/native gaps. Affine F interpolation assumed. No physical validation, cap/contact/MPC or Abaqus equivalence.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({**result, 'trajectories': [{k:v for k,v in t.items() if k not in ('history','input_output_hashes')} for t in result['trajectories']]}, indent=2))
