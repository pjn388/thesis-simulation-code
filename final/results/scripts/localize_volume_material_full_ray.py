"""Localize failed two-amplitude native FD using immutable captures only."""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix, diags
from audit_plastic_return_map import F, sha256


def main():
    gate = F/'results/output/volume_material_full_ray_fd_20261019.json'
    failed = json.loads(gate.read_text())
    assert failed['state'] == 'failed' and not failed['checks_passed']
    root = F/'sym/runs/investigation/volume_material_full_ray_fd_20261019'
    basepath = F/'sym/runs/investigation/volume_material_capture_v5_20261019/assembly.json'
    base = json.loads(basepath.read_text())
    paths = [root/f'h{h}'/'assembly.json' for h in (1e-6, -1e-6, 5e-7, -5e-7)]
    frames = [json.loads(p.read_text()) for p in paths]
    for p, d, case in zip(paths, frames, failed['cases']):
        assert sha256(p) == case['assembly_sha256']
        for k in ('previous_state', 'previous_stress', 'displacement_before', 'mapping', 'time', 'cycle'):
            assert d[k] == base[k], k
    g = base['global_matrix']; n = g['neq']
    rows = np.array(g['irow'])-1; cols = np.repeat(np.arange(n), np.diff(g['jq']))
    matrix = coo_matrix((g['au'], (rows, cols)), shape=(n,n)) + coo_matrix((g['au_upper'], (cols, rows)), shape=(n,n)) + diags(g['ad'])
    direction = np.array(json.loads((basepath.parent/'full_direction.json').read_text()))
    predicted = matrix@direction
    fd = [(np.array(frames[i]['internal'])-frames[i+1]['internal'])/(2*h) for i,h in ((0,1e-6),(2,5e-7))]
    error = [v-predicted for v in fd]
    mapping = np.array(base['mapping']); inverse = {int(eq): (i//4+1, i%4) for i,eq in enumerate(mapping) if eq>0}
    top = np.argsort(np.abs(error[1]))[-20:][::-1]
    equations = [dict(equation=int(i+1), node=inverse[int(i+1)][0], dof=inverse[int(i+1)][1], error_large=float(error[0][i]), error_small=float(error[1][i])) for i in top]
    # Native stress ordering is six components per IP, eight IPs per element.
    stress_fd = [(np.array(frames[i]['stress'])-frames[i+1]['stress'])/(2*h) for i,h in ((0,1e-6),(2,5e-7))]
    delta = (stress_fd[1]-stress_fd[0]).reshape(base['ne'],8,6)
    energy = np.sum(delta*delta,axis=(1,2)); order=np.argsort(energy)[::-1]
    elements=[]
    for e in order[:20]:
        ip,component=np.unravel_index(np.argmax(np.abs(delta[e])),(8,6))
        elements.append(dict(element=int(e+1),fraction_of_stress_amplitude_difference=float(energy[e]/energy.sum()),ip=int(ip+1),component=int(component+1),stress_values_mpa=[float(np.array(d['stress']).reshape(base['ne'],8,6)[e,ip,component]) for d in frames]))
    report=dict(kind='retained_failure_localization',checks_passed=True,acceptance=False,
        native_fd_errors=[float(np.linalg.norm(e)/np.linalg.norm(v)) for e,v in zip(error,fd)],
        amplitude_difference=float(np.linalg.norm(fd[1]-fd[0])/np.linalg.norm(fd[1])),
        error_vector_cosine=float(error[0]@error[1]/np.linalg.norm(error[0])/np.linalg.norm(error[1])),
        error_norm_ratio_small_over_large=float(np.linalg.norm(error[1])/np.linalg.norm(error[0])),
        stress_amplitude_difference_norm=float(np.linalg.norm(delta)),
        top20_element_fraction=float(energy[order[:20]].sum()/energy.sum()),
        top_equations=equations,top_elements=elements,
        sources={str(p.relative_to(F)):sha256(p) for p in [Path(__file__),gate,basepath,*paths]},
        limitations='Localization only; does not accept the tangent, establish branch causation, rollback or physical burst. No new native probes.',
        next='Inspect retained native stress/PEEQ and actual return-map branch at localized IPs. Opposite-sign or amplitude-dependent local responses must be explained before any native successor; no tolerance relaxation or amplitude sweep.')
    out=F/'results/output/volume_material_full_ray_localization_20261019.json'
    with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('sources','top_equations','top_elements')},indent=2))
    print(json.dumps(elements[:5],indent=2))

if __name__ == '__main__':main()
