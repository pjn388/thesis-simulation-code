"""Selected-IP replay of immutable full-ray captures; no native experiment."""
import json
from pathlib import Path
import numpy as np
from audit_plastic_return_map import F, tensors, update, sha256
from audit_material_linearization import derivative, vector
from audit_internal_pipe_fields import shape
from verify_heterogeneous_coupon import points


def main():
    root = F/'sym/runs/investigation'
    basepath = root/'volume_material_capture_v5_20261019/assembly.json'
    tablepath = root/'late_local_inputs_20261019/assembly.json'
    gate = F/'results/output/volume_material_full_ray_fd_20261019.json'
    evidence = json.loads(gate.read_text())
    assert evidence['state'] == 'failed'
    base = json.loads(basepath.read_text())
    table = json.loads(tablepath.read_text())
    curve = np.array(table['ips'][0]['plconloc'][:400]).reshape(-1, 2)[:, ::-1]
    del table
    direction = np.array(json.loads((basepath.parent/'full_direction.json').read_text()))
    xyz = np.array(base['coordinates']).reshape(-1, 3)
    mapping = np.array(base['mapping']).reshape(-1, 4)[:, 1:]
    du = np.zeros_like(xyz)
    du[mapping > 0] = direction[mapping[mapping > 0]-1]
    selected = [(238, 2), (1441, 4), (2365, 3)]
    mu = 208000/2.6; bulk = 208000/1.2
    rows = []
    sources = [Path(__file__), basepath, tablepath, gate]
    for h in (1e-6, -1e-6, 5e-7, -5e-7):
        path = root/f'volume_material_full_ray_fd_20261019/h{h}/assembly.json'
        case = next(c for c in evidence['cases'] if c['h'] == h)
        assert sha256(path) == case['assembly_sha256']
        frame = json.loads(path.read_text()); sources.append(path)
        assert frame['previous_state'] == base['previous_state']
        u = np.array(frame['displacement_after']).reshape(-1, 4)[:, 1:]
        states = np.array(frame['current_state']).reshape(base['ne'], 8, base['nstate'])
        prior = np.array(base['previous_state']).reshape(base['ne'], 8, base['nstate'])
        strain = tensors(np.array(frame['strain']).reshape(base['ne'], 8, 6))
        stress = tensors(np.array(frame['stress']).reshape(base['ne'], 8, 6))
        for element, ip in selected:
            e = element-1; i = ip-1; k = base['ipkon'][e]
            nodes = np.array(base['connectivity'][k:k+20])-1
            q = points()[i].astype(complex)
            natural = np.array([shape(q+1e-30j*np.eye(3)[j]).imag/1e-30 for j in range(3)]).T
            grad = natural@np.linalg.inv(xyz[nodes].T@natural)
            deform = np.eye(3)+u[nodes].T@grad
            c = deform.T@deform
            native_c = np.eye(3)+2*strain[e, i]
            old = prior[e, i]; metric = np.eye(3)-2*tensors(old[None, 1:7])[0]
            s, m, p, active = update(c, metric, np.array(old[0]), curve, mu, bulk)
            sn, mn, pn, an = update(native_c, metric, np.array(old[0]), curve, mu, bulk)
            df = du[nodes].T@grad
            tangent = derivative(c, metric, old[0], curve, mu, bulk, df.T@deform+deform.T@df)
            rows.append(dict(h=h, element=element, ip=ip, plastic=bool(active),
                stored_output_strain_plastic=bool(an),
                geometry_c_error=float(abs(c-native_c).max()),
                geometry_stress_error_mpa=float(abs(s-stress[e,i]).max()),
                stored_output_strain_stress_error_mpa=float(abs(sn-stress[e,i]).max()),
                geometry_peeq_error=float(abs(p-states[e,i,0])),
                stored_output_strain_peeq_error=float(abs(pn-states[e,i,0])),
                native_peeq=float(states[e,i,0]), replay_peeq=float(p),
                native_stress=vector(stress[e,i]).tolist(), replay_stress=vector(s).tolist(),
                stored_output_strain_replay_stress=vector(sn).tolist(), analytic_action=vector(tangent).tolist()))
        del frame
    comparisons = []
    for element, ip in selected:
        local = {r['h']: r for r in rows if (r['element'],r['ip']) == (element,ip)}
        for h in (1e-6, 5e-7):
            fd = lambda key: (np.array(local[h][key])-local[-h][key])/(2*h)
            native = fd('native_stress'); replay = fd('replay_stress')
            ns = fd('stored_output_strain_replay_stress')
            analytic = np.array(local[h]['analytic_action'])
            comparisons.append(dict(element=element, ip=ip, h=h,
                native_vs_geometry_replay_fd=float(np.linalg.norm(native-replay)/np.linalg.norm(analytic)),
                native_vs_stored_output_strain_replay_fd=float(np.linalg.norm(native-ns)/np.linalg.norm(analytic)),
                geometry_replay_vs_analytic_fd=float(np.linalg.norm(replay-analytic)/np.linalg.norm(analytic))))
    report = dict(kind='selected_ip_precision_and_branch_replay', acceptance=False,
        rows=rows, comparisons=comparisons,
        sources={str(p.relative_to(F)):sha256(p) for p in sources},
        limitations='Three selected IPs only. Stored output strain is not verified current mechanical strain: resultsmech.f writes eei only for iout>0. Its replay is an invalid-input control, not branch evidence. Reconstructed strain is not native local branch input capture. No native branch exit, global FD acceptance, rollback, pressure benefit or physical burst established.',
        next='Compare selected-IP geometry and return-map replay with higher precision on retained frames before any native-input capture; do not rerun full-ray probes or change gates.')
    out = F/'results/output/volume_material_full_ray_local_replay_20261019.json'
    with out.open('x') as f: json.dump(report,f,indent=2); f.write('\n')
    print(json.dumps(dict(comparisons=comparisons, maxima={k:max(r[k] for r in rows) for k in ('geometry_c_error','geometry_stress_error_mpa','stored_output_strain_stress_error_mpa','geometry_peeq_error','stored_output_strain_peeq_error')}),indent=2))


if __name__ == '__main__': main()
