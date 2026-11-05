"""Independently bracket frozen common-load samples against original extracted frames.

Audits retained CSVs, not native DAT extraction or physical validity. No solves.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

FINAL = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def audit(snapshot):
    provenance = json.loads((snapshot/'provenance.json').read_text())['sources']
    identities = []
    for name, expected in provenance.items():
        identities.append(dict(path=name, passed=digest(FINAL/name) == expected))
    plan = {job['id']: job for job in json.loads((FINAL/'results/config/full_campaign.json').read_text())['jobs']}
    records = []
    for sample in rows(snapshot/'common_pressure.csv'):
        job = plan[sample['case_id']]
        directory = FINAL/job['run']
        original = rows(directory/'results/history.csv')
        frames = [r for r in original if r['stage'] == 'pressurise']
        pressures = [float(r['pressure_mpa']) for r in frames]
        target = float(sample['pressure_mpa'])
        increasing = all(b > a for a, b in zip(pressures, pressures[1:]))
        assert increasing and pressures[0] <= target <= pressures[-1]
        low = max((r for r in frames if float(r['pressure_mpa']) <= target), key=lambda r: float(r['pressure_mpa']))
        high = min((r for r in frames if float(r['pressure_mpa']) >= target), key=lambda r: float(r['pressure_mpa']))
        a, b = float(low['pressure_mpa']), float(high['pressure_mpa'])
        fraction = (target-a)/(b-a) if b > a else 0.
        errors = {}
        for source, column in [('max_peeq', 'interpolated_max_peeq'), ('crown_inward_displacement_mm', 'crown_inward_mm')]:
            value = float(low[source]) + fraction*(float(high[source])-float(low[source]))
            errors[column] = abs(value-float(sample[column]))
        case = json.loads((directory/'case.json').read_text())
        support = json.loads(case['material']['source'])['source_supported_plastic_strain_max']
        hotspot = lambda r: [int(r['max_peeq_element']), int(r['max_peeq_ip'])]
        records.append(dict(case_id=sample['case_id'], angle_deg=float(sample['angle_deg']), combined=sample['combined']=='True',
            pressure_mpa=target, lower_pressure_mpa=a, upper_pressure_mpa=b, gap_mpa=b-a,
            lower_peeq=float(low['max_peeq']), upper_peeq=float(high['max_peeq']),
            lower_hotspot=hotspot(low), upper_hotspot=hotspot(high), bracket_hotspot_changed=hotspot(low)!=hotspot(high),
            history_hotspot_count=len({tuple(hotspot(r)) for r in frames}),
            pressure_start_peeq=float(frames[0]['max_peeq']),
            source_supported_peeq=support, sample_exceeds_material_support=float(sample['interpolated_max_peeq'])>support,
            errors=errors, passed=max(errors.values())<1e-12))
    common15 = [r for r in records if r['pressure_mpa']==15.]
    return dict(utc=datetime.now(timezone.utc).isoformat(), snapshot=str(snapshot.relative_to(FINAL)),
        snapshot_hashes={name:digest(snapshot/name) for name in ['common_pressure.csv','orientation_histories.csv','provenance.json']},
        source_identities=identities, records=records,
        summary=dict(samples=len(records), at15=len(common15), maximum_absolute_error=max(max(r['errors'].values()) for r in records),
            at15_gap_range_mpa=[min(r['gap_mpa'] for r in common15), max(r['gap_mpa'] for r in common15)],
            at15_bracket_hotspot_changes=sum(r['bracket_hotspot_changed'] for r in common15),
            at15_material_exceedances=sum(r['sample_exceeds_material_support'] for r in common15)),
        passed=all(r['passed'] for r in records) and all(r['passed'] for r in identities),
        limitations='Linear interpolation of global maxima is not a tracked material-point strain or a newly converged frame. Bracket identity does not rule out an unsampled hotspot switch. No native DAT re-extraction, fracture criterion or physical validation audited.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(args.snapshot.resolve())
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summary'], indent=2))
    raise SystemExit(0 if result['passed'] else 1)
