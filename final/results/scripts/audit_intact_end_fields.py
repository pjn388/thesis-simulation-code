"""Audit saved intact nodal fields; extrapolated FRD stress is not IP stress."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pipe_sim.model import Model
from pipe_sim.results import NUMBER, stress_invariants
from audit_pressure_failures import digest

FINAL = Path(__file__).resolve().parents[2]


def complete_fields(lines):
    """Yield only terminated DISP/STRESS blocks, retaining native component order."""
    time = None
    kind = None
    rows = {}
    labels = []
    for line in lines:
        if line.lstrip().startswith('100CL'):
            time = float(line.split()[2])
        elif line.startswith(' -4'):
            kind = line.split()[1]
            rows, labels = {}, []
        elif line.startswith(' -5') and kind:
            labels.append(line.split()[1])
        elif line.startswith(' -1') and kind in ('DISP', 'STRESS'):
            values = re.findall(NUMBER, line[3:])
            rows[int(values[0])] = [float(v) for v in values[1:]]
        elif line.startswith(' -3'):
            if kind in ('DISP', 'STRESS'):
                yield time, kind, labels, rows
            kind = None


def main():
    run = FINAL / 'sym/runs/campaign/orientation_intact_00_adaptive'
    status = json.loads((run / 'status.json').read_text())
    assert status['state'] == 'failed', 'Audit requires retained terminal intact attempt'
    manifest = json.loads((run / 'manifest.json').read_text())
    assert all(digest(run / name) == sha for name, sha in manifest['files'].items())
    model = Model.load(run / 'model.json')
    history = pd.read_csv(run / 'results/history.csv')
    latest = history.iloc[-1]
    fields = {}
    with (run / 'job.frd').open() as stream:
        for time, kind, labels, rows in complete_fields(stream):
            if abs(time - latest.time) < 1e-6:
                fields[kind] = (labels, rows)
    assert set(fields) == {'DISP', 'STRESS'}
    labels, stress = fields['STRESS']
    assert labels == ['SXX', 'SYY', 'SZZ', 'SXY', 'SYZ', 'SZX']
    displacement = fields['DISP'][1]
    pipe_nodes = sorted({n for e in model.elsets['PIPE'] for n in model.elements[e][1]})
    assert set(pipe_nodes).issubset(stress) and set(pipe_nodes).issubset(displacement)
    rows = []
    for node in pipe_nodes:
        x, y, z = model.nodes[node]
        ux, uy, uz = displacement[node]
        xx, yy, zz, xy, yz, zx = stress[node]
        radius = np.hypot(x, y)
        vm, eta, principal = stress_invariants([xx, yy, zz, xy, zx, yz])
        rows.append(dict(node=node, x=x, y=y, z=z,
                         zone='end' if abs(z) >= 1900 else 'centre' if abs(z) <= 100 else 'remote',
                         radial_displacement_mm=(x*ux+y*uy)/radius,
                         axial_displacement_mm=uz,
                         hoop_stress_reference_basis_mpa=(y*y*xx+x*x*yy-2*x*y*xy)/radius**2,
                         axial_stress_mpa=zz, nodal_mises_mpa=vm,
                         nodal_triaxiality=eta, nodal_max_principal_mpa=principal))
    df = pd.DataFrame(rows)
    hotspot = int(latest.max_peeq_element)
    out = FINAL / 'results/output/live_cases/orientation_intact_00_adaptive'
    out.mkdir(parents=True, exist_ok=True)
    df[df.node.isin(model.elements[hotspot][1])].to_csv(out/'endpoint_hotspot_nodes.csv', index=False)
    columns = ['radial_displacement_mm', 'axial_displacement_mm',
               'hoop_stress_reference_basis_mpa', 'axial_stress_mpa', 'nodal_mises_mpa']
    result = dict(run=str(run.relative_to(FINAL)), time=float(latest.time),
                  pressure_mpa=float(latest.pressure_mpa),
                  integration_point_hotspot=dict(element=hotspot, ip=int(latest.max_peeq_ip),
                                                 peeq=float(latest.max_peeq)),
                  solver_sha256=status['solver_sha256'],
                  evidence_sha256={name: digest(run/name) for name in
                                   ['job.frd', 'job.sta', 'model.json', 'job.inp', 'results/history.csv']},
                  zone_statistics=df.groupby('zone')[columns].agg(['min','mean','max']).to_json(),
                  maximum_nodal_mises=df.loc[df.nodal_mises_mpa.idxmax()].to_dict(),
                  limitations='FRD nodal stresses are extrapolated/averaged, not matched integration-point stresses. '
                  'Hoop projection uses reference coordinates. Zone means are unweighted nodal means. '
                  'No end artifact, physical limit point, burst, or mesh convergence established.')
    result['zone_statistics'] = json.loads(result['zone_statistics'])
    (out/'endpoint_fields.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
