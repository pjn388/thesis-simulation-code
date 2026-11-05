"""Freeze retained mesh parents and audit one-parameter children without extracting live runs."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from pipe_sim.runner import sha256
from audit_pressure_failures import converged_times, matches_sta

FINAL = Path(__file__).resolve().parents[2]


def audit(tag):
    plan = json.loads((FINAL/'results/config/full_campaign.json').read_text())
    jobs = {j['id']: j for j in plan['jobs']}
    records = []
    for child in jobs.values():
        if child.get('study') != 'mesh_sensitivity_'+tag:
            continue
        parent = jobs[child['parent_attempt']]
        parent_case = json.loads((FINAL/parent['case_file']).read_text())
        child_case = json.loads((FINAL/child['case_file']).read_text())
        for job in (parent, child):
            assert sha256(FINAL/job['case_file']) == job['case_sha256']
        changed = child['parameter_changes']
        assert len(changed) == 1
        key, values = next(iter(changed.items()))
        parameter = key.removeprefix('mesh.')
        assert parameter in ('wall_layers', 'fine_size')
        assert parent_case['mesh'][parameter] == values['old']
        assert child_case['mesh'][parameter] == values['new']
        child_case['mesh'][parameter] = values['old']
        for key in ('name', 'notes'):
            child_case[key] = parent_case[key]
        assert child_case == parent_case, child['id']
        run = FINAL/parent['run']
        manifest = json.loads((run/'manifest.json').read_text())
        assert all(sha256(run/f) == h for f, h in manifest['files'].items())
        summary = json.loads((run/'results/summary.json').read_text())
        rows = list(csv.DictReader((run/'results/history.csv').open()))
        accepted = converged_times(run/'job.sta')
        assert all(matches_sta(accepted, float(r['time'])) for r in rows)
        pressure = [r for r in rows if r['stage'] == 'pressurise']
        indent = [r for r in rows if r['stage'] == 'indent']
        material_source = json.loads(parent_case['material']['source'])
        supported_extent = material_source['source_supported_plastic_strain_max']
        live = FINAL/child['run']
        native = None
        if (live/'manifest.json').exists():
            m = json.loads((live/'manifest.json').read_text())
            assert all(sha256(live/f) == h for f, h in m['files'].items())
            times = converged_times(live/'job.sta') if (live/'job.sta').exists() else []
            native = dict(mesh=m['mesh'], status=json.loads((live/'status.json').read_text()),
                          accepted_frames=len(times), last_accepted_workflow_time=float(times[-1]) if len(times) else None,
                          manifest_sha256=sha256(live/'manifest.json'))
        records.append(dict(child=child['id'], parent=parent['id'], changes=changed,
                            child_case_sha256=child['case_sha256'], parent_case_sha256=parent['case_sha256'],
                            parent_run=parent['run'], child_run=child['run'], live_snapshot=native,
                            parent_metrics=dict(peak_indent_force_n=max(float(r['indenter_force_n']) for r in indent),
                                                residual_crown_mm=summary['residual_crown_displacement_mm'],
                                                provisional_criterion_bracket_mpa=summary.get('pressure_bracket_mpa'),
                                                crossing_centroid_mm=summary.get('failure_element_initial_centroid_mm'),
                                                last_pressure_mpa=float(pressure[-1]['pressure_mpa']),
                                                last_peeq=float(pressure[-1]['max_peeq']),
                                                supported_plastic_strain_extent=supported_extent,
                                                endpoint_exceeds_supported_extent=float(pressure[-1]['max_peeq']) > supported_extent,
                                                contact_checks_passed=summary['contact_checks_passed'],
                                                max_penetration_mm=summary['max_nodal_contact_penetration_mm']),
                            parent_evidence_sha256={f: sha256(run/f) for f in ('manifest.json', 'job.sta', 'results/history.csv', 'results/summary.json')}))
    assert len(records) == 4
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), records=records,
                comparison_protocol=[
                    'Require frozen hashes, unchanged physical inputs and accepted STA frames before comparing.',
                    'Compare loading and unloading separately at common tool travel within each branch; report overlap and force RMSE against retained parent and paper curves without shifting curves.',
                    'Report peak force and residual crown differences, tensile tool force, penetration and supported material extent; incomplete withdrawal cannot pass residual comparison.',
                    'Compare PEEQ and crown at common accepted pressures only; no endpoint extrapolation. Compare hotspot coordinates/defect membership rather than element IDs across meshes.',
                    'Report first sampled criterion brackets and their width; overlapping brackets do not resolve a crossing difference. Missing crossings remain missing.',
                    'No physical acceptance tolerance is recovered from sources. Quantify sensitivity, retain both parents, and do not call two mesh resolutions asymptotic convergence or experimental validation.'],
                limitations='Early indentation only; no child force/strain extraction or mesh convergence conclusion. Prescribed load/cap, material, gouge and fixture uncertainties remain.',
                script_sha256=sha256(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tag')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), 'Use a new evidence filename'
    args.output.write_text(json.dumps(audit(args.tag), indent=2)+'\n')
    print(args.output)
