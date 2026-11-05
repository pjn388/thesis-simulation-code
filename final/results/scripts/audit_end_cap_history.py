"""Read-only finite-expansion cap diagnostic; no replacement load is prescribed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pipe_sim.model import Model
from pipe_sim.results import dat_blocks
from audit_end_cap_resultant import FINAL, areas, contours
from audit_end_surface_traction import face_nodes
from audit_intact_end_fields import complete_fields
from audit_nodal_pressure_loads import nodal_loads
from audit_pressure_failures import digest, converged_times, matches_sta


def audit(run, elastic=False):
    manifest = json.loads((run/'manifest.json').read_text())
    assert all(digest(run/f) == h for f, h in manifest['files'].items())
    status = json.loads((run/'status.json').read_text())
    assert status['state'] in ('completed', 'failed', 'interrupted')
    files = ['manifest.json', 'case.json', 'model.json', 'job.inp', 'job.sta',
             'job.dat' if elastic else 'job.frd']
    hashes = {f: digest(run/f) for f in files}
    case = json.loads((run/'case.json').read_text())
    model = Model.load(run/'model.json')
    ro = case['pipe']['od']/2
    ri = ro-case['pipe']['thickness']
    ratio = ri**2/(ro**2-ri**2)
    # Verify the actual frozen traction rather than assume current code generated it.
    deck = (run/'job.inp').read_text().upper()
    lines = deck.split('*DLOAD\n')[1].split('*')[0].strip().splitlines()
    loads = {(int(e), int(f.strip()[1:])): float(p)
             for e, f, p in (line.split(',') for line in lines)}
    end_edges = {s: contours(model, s, ri, ro) for s in ('END_MINUS', 'END_PLUS')}
    for s in end_edges:
        for e, f in model.surfaces[s]:
            assert np.isclose(loads[e, f], -case['analysis']['max_pressure']*ratio,
                              rtol=1e-11)
    accepted = converged_times(run/'job.sta')
    if elastic:
        frames = ((t, {int(r[0]): np.asarray(r[1:]) for r in rows})
                  for k, g, t, rows in dat_blocks(run/'job.dat') if k == 'U' and g == 'BENCH')
        pressures = None
    else:
        history = pd.read_csv(run/'results/history.csv')
        pressure_history = history[history.stage == 'pressurise']
        pressures = [(float(r.time), float(r.pressure_mpa)) for r in pressure_history.itertuples()]
        hashes['results/history.csv'] = digest(run/'results/history.csv')
        stream = (run/'job.frd').open()
        frames = ((t, rows) for t, k, labels, rows in complete_fields(stream) if k == 'DISP')
    records = []
    last_u = None
    try:
        for time, u in frames:
            if pressures is None:
                pressure = time*case['analysis']['max_pressure']
            else:
                matching = [p for t, p in pressures if abs(t-time) <= 1e-6]
                if not matching:
                    continue
                assert len(matching) == 1
                pressure = matching[0]
            assert matches_sta(accepted, time)
            last_u = u
            for surface, edges in end_edges.items():
                nodes = {n for rows in edges.values() for edge in rows for n in edge}
                assert nodes.issubset(u)
                current = {n: np.asarray(model.nodes[n])+u[n] for n in nodes}
                projected = areas(edges, current)
                wall = projected['outer']-projected['inner']
                assert wall > 0 and projected['inner'] > 0
                records.append(dict(time=time, pressure_mpa=pressure, surface=surface,
                    current_bore_projected_area_mm2=projected['inner'],
                    current_wall_projected_area_mm2=wall,
                    current_area_ratio=projected['inner']/wall,
                    prescribed_reference_ratio=ratio,
                    ideal_closed_cap_axial_resultant_n=pressure*projected['inner'],
                    prescribed_traction_axial_resultant_n=pressure*ratio*wall,
                    relative_ideal_cap_mismatch=ratio*wall/projected['inner']-1))
        assert records
        if pressures is not None:
            assert len(records) == 2*len(pressures), 'Missing accepted pressure displacement frame'
    finally:
        if not elastic:
            stream.close()
    # Independent solver-order (2x2) surface load reconstruction at the last frame.
    checks = []
    for record in records[-2:]:
        surface = record['surface']
        for order in (2, 3):
            resultant = np.zeros(3)
            for e, f in model.surfaces[surface]:
                nodes = face_nodes(model, e, f)
                xyz = [np.asarray(model.nodes[n])+last_u[n] for n in nodes]
                resultant += nodal_loads(xyz, record['pressure_mpa']*ratio, order).sum(axis=0)
            sign = -1 if surface == 'END_MINUS' else 1
            error = sign*resultant[2]/record['prescribed_traction_axial_resultant_n']-1
            if order == 3:
                assert abs(error) < 1e-10, (surface, order, error)
            checks.append(dict(surface=surface, quadrature_order=order,
                               contour_vs_surface_relative_error=float(error),
                               prescribed_traction_resultant_n=resultant.tolist()))
    assert all(digest(run/f) == h for f, h in hashes.items()), 'Native evidence changed'
    return dict(run=str(run.relative_to(FINAL)), solver_state=status['state'],
                solver_sha256=status['solver_sha256'], accepted_pressure_frames=len(records)//2,
                history=records, endpoint_quadrature_checks=checks, evidence_sha256=hashes)


def main():
    parent = FINAL/'results/output/pressure_restraint_candidates/pipe_work_v3_20261010T080005Z.json'
    elastic = json.loads(parent.read_text())
    records = [audit(FINAL/r['run'], elastic=True) for r in elastic['records']]
    records.append(audit(FINAL/'sym/runs/campaign/orientation_intact_00_adaptive'))
    source_paths = ['sym/src/pipe_sim/decks.py',
                    'sym/vendor/CalculiX/ccx_2.23/src/e_c3d.f',
                    'data/evidence/zhao2021/text.txt']
    report = dict(records=records, script_sha256=digest(Path(__file__)),
        parent_elastic_report_sha256=digest(parent),
        source_sha256={f: digest(FINAL/f) for f in source_paths},
        source_references=['decks.py:92-96 reference-area ratio',
            'e_c3d.f:1360-1486,1597-1615 current-geometry pressure quadrature',
            'zhao2021/text.txt:313-317 axial tensile end stress; finite-area convention unspecified'],
        interpretation='A fixed reference-area-ratio follower traction need not equal pressure times current bore area. '
            'Both are candidate load conventions; the paper does not settle the finite-expansion convention.',
        limitations='Ideal cap comparison is geometric and axial only, not a verified physical cap or fixture. '
            'Rounded native displacements; no cap replacement work or full reaction balance computed. '
            'Elastic work balance verifies the prescribed traction, not the ideal cap. '
            'No proof of cutback causation, plastic-work balance, alternate branch, burst, mesh convergence '
            'or Abaqus equivalence. Frozen production loads and adapters remain unchanged.')
    out = FINAL/'results/output/pressure_restraint_candidates/end_cap_history_20261010T083002Z.json'
    assert not out.exists()
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps([dict(run=r['run'], frames=r['accepted_pressure_frames'],
        endpoint_mismatch=[v['relative_ideal_cap_mismatch'] for v in r['history'][-2:]]) for r in records]))


if __name__ == '__main__':
    main()
