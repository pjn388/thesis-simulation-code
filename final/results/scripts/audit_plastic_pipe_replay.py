"""Gated previous-state and accumulated pipe replay; no native/model changes."""
import json
import numpy as np
from audit_plastic_return_map import update, F, tensors, sha256
from audit_internal_pipe_fields import spatial
from pipe_sim.model import Model


def audit(gate=None, out=None):
    if gate is None:
        gate = F/'results/output/plastic_state_stress_pipe_20261011T100320Z.json'
    evidence = json.loads(gate.read_text())
    assert evidence['checks_passed']
    coupon = F/'results/output/plastic_coupon_substeps_half_20261011T100722Z.json'
    assert json.loads(coupon.read_text())['checks_passed']
    run = F/evidence['run']
    hashes = {n:sha256(run/n) for n in ('job.inp','mesh.inp','model.json','case.json','job.dat','job.sta','fields.bin','fields.json')}
    assert all(hashes[n] == h for n,h in evidence['hashes'].items())
    deck = (run/'job.inp').read_text()
    assert all(s not in deck.upper() for s in ('*INITIAL CONDITIONS','*TEMPERATURE','*CREEP','HARDENING='))
    lines = deck.splitlines(); curve = []
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'): break
        curve.append(list(map(float,line.split(','))))
    curve = np.array(curve)
    assert np.all(np.diff(curve[:,1]) > 0) and np.all(np.diff(curve[:,0]) >= 0)
    model = Model.load(run/'model.json'); inside = False
    for line in (run/'mesh.inp').read_text().splitlines():
        if line.startswith('**'): continue
        if line.startswith('*'): inside = line.upper().strip() == '*NODE'
        elif inside and line.strip():
            values = line.split(','); model.nodes[int(values[0])] = list(map(float,values[1:4]))
    ids,nodes,grad,weights,checks = spatial(model)
    material = json.loads((run/'case.json').read_text())['material']
    mu = material['young']/(2*(1+material['poisson']))
    bulk = material['young']/(3*(1-2*material['poisson']))
    records = json.loads((run/'fields.json').read_text())['records']
    raw = np.memmap(run/'fields.bin',dtype='<f8',mode='r')
    shape = (len(ids),8)
    native_metric = free_metric = np.broadcast_to(np.eye(3),shape+(3,3)).copy()
    native_peeq = free_peeq = np.zeros(shape)
    previous_stress = np.zeros(shape+(3,3)); previous_strain = previous_stress.copy()
    energy = np.zeros(shape)
    maxima = {mode:dict(stress_mpa=0.,metric=0.,peeq=0.) for mode in ('previous_native','accumulated')}
    maximum_energy_error = 0.; history = []
    for rec in records:
        def field(name,width):
            m = rec['fields'][name]; values = raw[m['offset']//8:m['offset']//8+m['count']]
            return values.reshape((-1,width)) if name == 'u' else values.reshape((rec['elements'],rec['ips'],width))[np.array(ids)-1,:8]
        u = field('u',4)[:,1:]
        deformation = np.eye(3)+np.einsum('eni,epnj->epij',u[nodes-1],grad)
        strain = tensors(field('strain',6)); c = np.eye(3)+2*strain
        inverse = np.linalg.inv(deformation)
        actual_stress = np.linalg.det(deformation)[...,None,None]*(inverse@tensors(field('stress',6))@np.swapaxes(inverse,-1,-2))
        state = field('state',rec['states'])
        actual_metric = np.eye(3)-2*tensors(state[...,1:7]); actual_peeq = state[...,0]
        frame = dict(time=rec['time'],maximum_peeq=float(actual_peeq.max()))
        for mode,metric,peeq in (('previous_native',native_metric,native_peeq),('accumulated',free_metric,free_peeq)):
            stress,predicted_metric,predicted_peeq,active = update(c,metric,peeq,curve,mu,bulk)
            errors = dict(stress_mpa=float(abs(stress-actual_stress).max()),metric=float(abs(predicted_metric-actual_metric).max()),peeq=float(abs(predicted_peeq-actual_peeq).max()))
            for key,value in errors.items(): maxima[mode][key] = max(maxima[mode][key],value)
            frame[mode] = dict(errors=errors,plastic_ips=int(active.sum()))
            if mode == 'accumulated':
                free_metric,free_peeq = predicted_metric,predicted_peeq
                energy += .5*np.einsum('epij,epij->ep',stress+previous_stress,strain-previous_strain)
                previous_stress = stress
        error = float(abs(energy-field('energy',2)[...,0]).max())
        maximum_energy_error = max(maximum_energy_error,error)
        frame.update(maximum_energy_density_error=error,replayed_energy_n_mm=float((energy*weights).sum()),native_energy_n_mm=float((field('energy',2)[...,0]*weights).sum()))
        history.append(frame)
        previous_strain = strain
        native_metric,native_peeq = actual_metric.copy(),actual_peeq.copy()
    limits = dict(stress_mpa=1e-4,metric=1e-8,peeq=1e-8)
    passed = all(maxima[mode][key] < limit for mode in maxima for key,limit in limits.items()) and maximum_energy_error < 1e-8
    assert all(sha256(run/n) == h for n,h in hashes.items())
    result = dict(checks_passed=passed,accepted_frames=len(records),maximum_errors=maxima,fixed_limits=limits,maximum_energy_density_error=maximum_energy_error,energy_density_limit=1e-8,history=history,spatial_checks=checks,input_output_hashes=hashes,gate_sha256=sha256(gate),coupon_gate_sha256=sha256(coupon),return_map_script_sha256=sha256(F/'results/scripts/audit_plastic_return_map.py'),source_sha256=sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f'),script_sha256=sha256(__file__),run=str(run.relative_to(F)),limitations='Endpoint constitutive and native endpoint-average energy identities only. No intermediate-path/work convergence, physical dissipation, cap/contact/MPC/Abaqus equivalence or experimental burst acceptance.')
    if out is None:
        out = F/'results/output/plastic_pipe_replay_20261011T101242Z.json'
    assert not out.exists(); out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('history','input_output_hashes')},indent=2))
    assert passed, 'Preserve failed identity; investigate without fitting or relaxing limits.'


if __name__ == '__main__': audit()
