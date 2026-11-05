"""Reconstruct endpoint PK2 stress from native plastic states, not a path replay."""
import json
import sys
import numpy as np
from audit_internal_pipe_fields import F, tensors, spatial
from pipe_sim.model import Model
from pipe_sim.runner import sha256


def audit(lane, gate=None, out=None):
    if lane == 'coupon':
        gate = F/'results/output/plastic_state_coupon_20261011T095613Z_state.json'
        run = F/'sym/runs/investigation/plastic_state_coupon_20261011T095613Z'
        young, poisson = 202500., .3
        xyz = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    else:
        assert lane == 'pipe'
        if gate is None:
            gate = F/'results/output/plastic_state_capture_20261011T095613Z_audit.json'
        evidence = json.loads(gate.read_text())
        run = F/evidence['run']
        model = Model.load(run/'model.json')
        inside = False
        for line in (run/'mesh.inp').read_text().splitlines():
            if line.startswith('**'):
                continue
            if line.startswith('*'):
                inside = line.upper().strip() == '*NODE'
            elif inside and line.strip():
                values = line.split(',')
                model.nodes[int(values[0])] = list(map(float, values[1:4]))
        ids, nodes, grad, weights, checks = spatial(model)
        material = json.loads((run/'case.json').read_text())['material']
        young, poisson = material['young'], material['poisson']
    assert json.loads(gate.read_text())['checks_passed']
    # Restrict the identity to the retained isotropic, residual-stress-free decks.
    deck = (run/'job.inp').read_text().upper()
    assert '*INITIAL CONDITIONS' not in deck and '*TEMPERATURE' not in deck
    assert 'HARDENING=' not in deck
    hashes = {name:sha256(run/name) for name in ('job.inp','job.dat','job.sta','fields.bin','fields.json')}
    raw = np.memmap(run/'fields.bin', dtype='<f8', mode='r')
    records = json.loads((run/'fields.json').read_text())['records']
    mu = young/(2*(1+poisson))
    bulk = young/(3*(1-2*poisson))
    maximum = 0.
    history = []
    previous_peeq = None
    minimum_increment = 0.
    minimum_metric_eigenvalue = float('inf')
    maximum_backstress = 0.
    for rec in records:
        def field(name, width):
            meta = rec['fields'][name]
            return raw[meta['offset']//8:meta['offset']//8+meta['count']].reshape((-1,width)) if name == 'u' else raw[meta['offset']//8:meta['offset']//8+meta['count']].reshape((rec['elements'],rec['ips'],width))
        u = field('u',4)[:,1:]
        if lane == 'coupon':
            coef = np.linalg.lstsq(np.c_[np.ones(8),xyz],u,rcond=None)[0]
            deformation = np.broadcast_to(np.eye(3)+coef[1:].T,(1,8,3,3))
        else:
            deformation = np.eye(3)+np.einsum('eni,epnj->epij',u[nodes-1],grad)
        state = field('state',rec['states'])
        strain, stress = tensors(field('strain',6)), tensors(field('stress',6))
        c = np.eye(3)+2*strain
        inverse_c = np.linalg.inv(c)
        jacobian = np.linalg.det(deformation)
        metric = np.eye(3)-2*tensors(state[...,1:7])
        trace = np.einsum('...ij,...ij->...',c,metric)/3
        # incplas.f:68-74,156-189,774-790,1025-1032: the final
        # deviatoric plastic metric encodes the return correction.
        predicted = mu/jacobian[...,None,None]**(2/3)*(metric-trace[...,None,None]*inverse_c)+bulk/2*(jacobian**2-1)[...,None,None]*inverse_c
        inverse_f = np.linalg.inv(deformation)
        actual = jacobian[...,None,None]*(inverse_f@stress@np.swapaxes(inverse_f,-1,-2))
        error = float(abs(predicted-actual).max())
        maximum = max(maximum,error)
        minimum_metric_eigenvalue = min(minimum_metric_eigenvalue,float(np.linalg.eigvalsh(metric).min()))
        maximum_backstress = max(maximum_backstress,float(abs(state[...,7:13]).max()))
        peeq = state[...,0]
        if previous_peeq is not None:
            minimum_increment = min(minimum_increment,float((peeq-previous_peeq).min()))
        previous_peeq = peeq.copy()
        history.append(dict(time=rec['time'],maximum_pk2_error_mpa=error,maximum_peeq=float(peeq.max())))
    assert maximum < 1e-7 and minimum_metric_eigenvalue > 0 and maximum_backstress == 0 and minimum_increment > -1e-12
    assert all(sha256(run/name)==value for name,value in hashes.items())
    source = F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'
    result = dict(checks_passed=True,lane=lane,run=str(run.relative_to(F)),accepted_frames=len(records),maximum_pk2_error_mpa=maximum,minimum_metric_eigenvalue=minimum_metric_eigenvalue,maximum_backstress=maximum_backstress,minimum_peeq_increment=minimum_increment,history=history,hashes=hashes,gate_sha256=sha256(gate),source_sha256=sha256(source),script_sha256=sha256(__file__),limitations='Endpoint constitutive identity only. Does not reconstruct intermediate return-map trajectory, establish plastic work convergence, cap/contact/MPC/Abaqus transfer or physical burst.')
    if out is None:
        out = F/'results/output'/f'plastic_state_stress_{lane}_20261011T100320Z.json'
    assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ('history','hashes')},indent=2))


if __name__ == '__main__':
    audit(sys.argv[1])
