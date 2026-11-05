"""Audit stage-global PEEQ and hotspot identities in retained combined histories."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from audit_common_pressure import audit, digest, rows, FINAL


def inspect(snapshot):
    common = audit(snapshot)
    jobs = {j['id']: j for j in json.loads((FINAL/'results/config/full_campaign.json').read_text())['jobs']}
    records = []
    for sample in common['records']:
        if not sample['combined'] or sample['pressure_mpa'] != 15.:
            continue
        directory = FINAL/jobs[sample['case_id']]['run']
        history = rows(directory/'results/history.csv')
        stages = {}
        for stage in ('indent', 'withdraw', 'pressurise'):
            frames = [r for r in history if r['stage'] == stage]
            assert frames
            frame = frames[0] if stage == 'pressurise' else frames[-1]
            stages[stage] = {k: float(frame[k]) for k in ('time', 'pressure_mpa', 'max_peeq')}
            stages[stage]['hotspot'] = [int(frame['max_peeq_element']), int(frame['max_peeq_ip'])]
        assert stages['withdraw']['pressure_mpa'] == 0.
        records.append(dict(case_id=sample['case_id'], angle_deg=sample['angle_deg'],
            history_path=str((directory/'results/history.csv').relative_to(FINAL)),
            history_sha256=digest(directory/'results/history.csv'), stages=stages,
            at15_bracket=sample,
            withdrawal_to_first_pressure_hotspot_changed=stages['withdraw']['hotspot'] != stages['pressurise']['hotspot'],
            withdrawal_to_15_bracket_hotspot_changed=any(stages['withdraw']['hotspot'] != sample[k] for k in ('lower_hotspot','upper_hotspot'))))
    assert len(records) == 7
    return dict(utc=datetime.now(timezone.utc).isoformat(), snapshot=str(snapshot.relative_to(FINAL)),
        source_identities=common['source_identities'], records=records,
        summary=dict(cases=len(records), withdrawal_peeq_range=[min(r['stages']['withdraw']['max_peeq'] for r in records),max(r['stages']['withdraw']['max_peeq'] for r in records)],
            withdrawal_to_first_pressure_hotspot_changes=sum(r['withdrawal_to_first_pressure_hotspot_changed'] for r in records),
            withdrawal_to_15_bracket_hotspot_changes=sum(r['withdrawal_to_15_bracket_hotspot_changed'] for r in records)),
        passed=common['passed'], limitations='Stage-global maxima only; no subtraction defines local pressure-induced strain. Equal hotspot IDs at sampled frames do not establish continuous material-point tracking. Retained extracted CSV audit, not native DAT re-extraction, physical acceptance or burst validation.')

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=inspect(args.snapshot.resolve())
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['summary'],indent=2))
    raise SystemExit(0 if result['passed'] else 1)
