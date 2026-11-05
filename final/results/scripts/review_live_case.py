"""Read-only bounded denting review for an existing campaign case."""
import argparse
import csv
import json
from pathlib import Path

from review_live_withdrawal import FINAL, snapshot, compare_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case_id')
    args = parser.parse_args()
    if Path(args.case_id).name != args.case_id:
        parser.error('case_id must be a campaign directory name')
    run = FINAL / 'sym/runs/campaign' / args.case_id
    out = FINAL / 'results/output/live_cases' / args.case_id
    out.mkdir(parents=True, exist_ok=True)
    result = snapshot(run)
    plan = json.loads((FINAL / 'results/config/full_campaign.json').read_text())
    job = next(job for job in plan['jobs'] if job['id'] == args.case_id)
    if (job.get('entry') or {}).get('comparison_kind') == 'pressure_strain':
        result['branch_comparison'] = {
            'status': 'pending_pressure_gauge_overlay',
            'reason': 'Pressure-strain references are not dent force/travel loops. '
                      'Use the gauge comparison after complete probe extraction.'}
    elif not job.get('entry'):
        result['branch_comparison'] = {
            'status': 'no_literature_loop_reference',
            'reason': 'Numerical baseline/orientation case has no physical force/travel reference.'}
    else:
        result = compare_snapshot(result, run=run, out=out)
    result['first_sampled_material_exceedance'] = next(
        (row for row in result['history'] if row['max_peeq'] > result['material_supported_peeq']), None)
    (out / 'snapshot.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'history.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result['history'][0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(result['history'])
    print(json.dumps({key: result[key] for key in [
        'run', 'solver_state', 'latest_frame', 'maximum_peeq', 'peak_tool_force_n',
        'withdrawal_completed', 'residual_crown_mm', 'tensile_force_detected',
        'first_sampled_material_exceedance', 'branch_comparison']}))


if __name__ == '__main__':
    main()
