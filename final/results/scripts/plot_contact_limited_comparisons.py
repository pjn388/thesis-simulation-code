"""Replot frozen X52/A37 denting curves without post-contact free withdrawal.

Display only: preserve the first recorded unloaded frame and omit later frames.
The endpoint is sampled, not an interpolated estimate of exact contact loss.
Run from any directory with final/sym/.venv/bin/python.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comparison_utils import canonical_curve

FINAL = Path(__file__).resolve().parents[2]
OUT = FINAL / 'results/output/contact_limited_20261026'
FORCE_TOLERANCE_N = 1e-6


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources, records = {}, []

    def read_json(path):
        sources[str(path.relative_to(FINAL))] = digest(path)
        return json.loads(path.read_text())

    def read_csv(path):
        sources[str(path.relative_to(FINAL))] = digest(path)
        return pd.read_csv(path)

    def simulation(run, expected_hash):
        directory = FINAL / run
        path = directory / 'results/history.csv'
        assert digest(path) == expected_hash, f'Frozen history changed: {path}'
        history = read_csv(path)
        case = read_json(directory / 'case.json')
        curve = history[history.stage.isin(['indent', 'withdraw'])].copy().reset_index(drop=True)
        force = curve.indenter_force_n.to_numpy()
        candidates = np.flatnonzero((curve.stage == 'withdraw') & (np.abs(force) <= FORCE_TOLERANCE_N))
        assert len(candidates), f'No recorded unloaded withdrawal frame: {run}'
        end = int(candidates[0])
        assert end > 0 and force[end - 1] > FORCE_TOLERANCE_N
        assert np.all(np.abs(force[end:]) <= FORCE_TOLERANCE_N), f'Recontact or nonzero tail: {run}'
        trimmed = curve.iloc[:end + 1].copy()
        assert trimmed.indenter_force_n.max() == curve.indenter_force_n.max()
        pd.testing.assert_frame_equal(trimmed[trimmed.stage == 'indent'], curve[curve.stage == 'indent'])
        trimmed['x'] = trimmed.indenter_travel_mm - case['dent']['clearance']
        trimmed['y'] = trimmed.indenter_force_n / 1000
        trimmed.to_csv(OUT / (directory.name + '_display.csv'), index=False)
        records.append(dict(run=run, original_points=len(curve), displayed_points=len(trimmed),
                            omitted_points=len(curve) - len(trimmed),
                            last_loaded_tool_penetration_mm=float(trimmed.x.iloc[-2]),
                            first_unloaded_tool_penetration_mm=float(trimmed.x.iloc[-1]),
                            first_unloaded_force_n=float(force[end]),
                            peak_force_unchanged=True, loading_branch_unchanged=True))
        return trimmed

    def references(entry, ax):
        frame = read_csv(FINAL / entry['reference_file']).fillna({'panel': ''})
        for key, style, color, label in [
            ('experimental_series', 'o-', 'black', 'Original experiment'),
            ('numerical_series', '--', '0.55', 'Published FE'),
        ]:
            selected = frame[(frame.panel == entry['panel']) & (frame.series == entry[key])]
            curve = canonical_curve(selected, 'mm', 'kN')
            ax.plot(curve.x, curve.y, style, color=color, ms=3, label=label)

    def finish(fig, name):
        fig.supxlabel('Tool penetration (mm)')
        fig.supylabel('Force (kN)')
        fig.savefig(OUT / (name + '.png'), dpi=180)
        fig.savefig(OUT / (name + '.pdf'), metadata={'CreationDate': None, 'ModDate': None})
        plt.close(fig)

    jobs = {j['id']: j for j in read_json(FINAL / 'results/config/full_campaign.json')['jobs']}
    frozen = read_json(FINAL / 'results/output/pressure005_20261020/provenance.json')['sources']
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for specimen, ax in zip(('p1', 'p2', 'p3', 'p4'), axes.flat):
        parent = f'full_zhao2021_{specimen}_adaptive'
        references(jobs[parent]['entry'], ax)
        for name, color, label in [(parent, '#0072b2', 'Our CalculiX: parent'),
                                   (parent + '_pressure005_20261020', '#d55e00', 'Our CalculiX: refined pressure steps')]:
            run = jobs[name]['run']
            curve = simulation(run, frozen[run + '/results/history.csv'])
            ax.plot(curve.x, curve.y, color=color, label=label)
        ax.set_title(specimen.upper())
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('X52 — Zhao (2021): loading and unloading to first recorded zero force')
    finish(fig, 'x52_load_displacement')

    provenance = read_json(FINAL / 'results/output/campaign/dent_validation/provenance.json')
    entries = {j['id']: j['entry'] for j in read_json(FINAL / 'results/config/investigation.json')['jobs']}
    entries.update({e['id']: e for e in read_json(FINAL / 'results/config/literature_cases.json')})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout='constrained')
    for depth, record, ax in zip((3, 8, 10, 13, 16), provenance, axes.flat):
        assert digest(FINAL / record['reference_file']) == record['reference_sha256']
        references(entries[record['case_id']], ax)
        curve = simulation(record['run'], record['history_sha256'])
        ax.plot(curve.x, curve.y, color='#0072b2', label='Our CalculiX')
        ax.set_title(f'{depth}% OD dent depth')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.04, .9,
        'Display ends at the first recorded\nzero-force withdrawal frame.\n\n'
        'Later free tool travel is omitted.\nLoading and loaded unloading are unchanged.\n\n'
        'Endpoint resolution is limited by saved frames.\n\n'
        'Denting-response comparisons;\nburst validation remains unresolved.', va='top', fontsize=11)
    fig.suptitle('A37 — Allouti (2012): loading and unloading to first recorded zero force')
    finish(fig, 'a37_load_displacement')

    for path, original_hash in sources.items():
        assert digest(FINAL / path) == original_hash, f'Source changed during plotting: {path}'
    (OUT / 'provenance.json').write_text(json.dumps(dict(
        policy='Keep all indentation and withdrawal through first recorded zero-force withdrawal frame; omit only subsequent zero-force free travel. No interpolation or source edits.',
        force_tolerance_n=FORCE_TOLERANCE_N, sources=sources, curves=records,
        script_sha256=digest(Path(__file__).resolve())), indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
