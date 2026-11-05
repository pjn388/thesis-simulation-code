"""Read failed/interrupted campaign attempts without editing native runs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parents[2]
OUTPUT = FINAL / 'results/output/pressure_failure_audit'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def converged_times(path):
    times = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 7 and fields[0].isdigit() and 'U' not in fields[2]:
            # Workflow steps each last one unit. STA total time has poorer
            # printed precision than step time after indentation/withdrawal.
            times.append(int(fields[0]) - 1 + float(fields[5]))
    return np.asarray(times)


def matches_sta(times, t):
    # DAT and STA can differ by one printed unit; slack covers binary rounding.
    return bool(np.any(np.abs(times - t) <= 1e-6 + 1e-12))


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    records = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for run in sorted((FINAL / 'sym/runs/campaign').iterdir()):
        status_path = run / 'status.json'
        summary_path = run / 'results/summary.json'
        if not status_path.exists() or not summary_path.exists():
            continue
        status = json.loads(status_path.read_text())
        if status['state'] not in ('failed', 'interrupted'):
            continue
        case = json.loads((run / 'case.json').read_text())
        summary = json.loads(summary_path.read_text())
        history = pd.read_csv(run / 'results/history.csv')
        pressure = history[history.stage == 'pressurise'].copy()
        if pressure.empty:
            continue
        sta = converged_times(run / 'job.sta')
        # Native STA and DAT use different printed precision. This is a
        # precision-limited consistency check, not unique increment matching.
        matched = [matches_sta(sta, t) for t in history.time]
        if not all(matched):
            raise ValueError(f'{run.name}: extracted history has a non-STA frame')
        manifest = json.loads((run / 'manifest.json').read_text())
        checks = {name: digest(run / name) == sha
                  for name, sha in manifest['files'].items()}
        if not all(checks.values()):
            raise ValueError(f'{run.name}: frozen input hash mismatch')
        material = json.loads(case['material']['source'])
        supported = material['source_supported_plastic_strain_max']
        endpoint = pressure.iloc[-1]
        plastic = np.asarray(case['material']['plastic'])
        upper = int(np.searchsorted(plastic[:, 1], endpoint.max_peeq, side='right'))
        bracket = plastic[max(0, upper - 1):min(len(plastic), upper + 1)]
        tangent = None if len(bracket) != 2 else float(
            np.diff(bracket[:, 0])[0] / np.diff(bracket[:, 1])[0])
        # Secants are descriptive observations, not a stiffness eigenvalue,
        # a mathematical limit point or a physical instability/burst test.
        dp = float(pressure.pressure_mpa.diff().iloc[-1])
        de = float(pressure.max_peeq.diff().iloc[-1])
        du = float(pressure.crown_inward_displacement_mm.diff().iloc[-1])
        above = history[history.max_peeq > supported]
        records.append(dict(
            case_id=run.name, run=str(run.relative_to(FINAL)),
            solver_state=status['state'], solver_sha256=status['solver_sha256'],
            stop_reason=status.get('stop_reason'),
            returncode=status.get('returncode'),
            elapsed_seconds=status.get('elapsed_seconds'),
            peak_observed_rss_mb=status.get('peak_observed_rss_mb'),
            native_error_lines=[line.strip() for line in
                (run / 'solver.log').read_text().splitlines()
                if '*ERROR' in line or 'too many cutbacks' in line.lower()],
            criterion_status=summary['status'],
            first_sampled_criterion_pressure_mpa=summary.get('burst_pressure_mpa'),
            criterion_pressure_bracket_mpa=summary.get('pressure_bracket_mpa'),
            criterion_is_validated_burst=False,
            frozen_inputs_verified=checks, history_frames=len(history),
            all_history_times_match_converged_sta_within_1e_6=all(matched),
            last_pressure_mpa=float(endpoint.pressure_mpa),
            max_observed_peeq=summary['maximum_observed_peeq'],
            supported_plastic_strain_max=supported,
            supported_range_exceeded=bool(len(above)),
            first_above_supported_range=None if above.empty else
                above.iloc[0][['time', 'stage', 'pressure_mpa', 'max_peeq']].to_dict(),
            criterion_peeq=case['analysis']['failure_peeq'],
            endpoint_plastic_bracket=bracket.tolist(),
            endpoint_table_hardening_mpa_per_plastic_strain=tangent,
            last_secant_mpa_per_peeq=None if de == 0 else dp / de,
            last_secant_mpa_per_outward_crown_mm=None if du == 0 else dp / -du,
            residual_crown_mm=summary['residual_crown_displacement_mm'],
            contact_checks_passed=summary['contact_checks_passed'],
            pressure_contact_applicable=case['dent']['enabled'],
            max_peeq_element=int(endpoint.max_peeq_element),
            max_peeq_ip=int(endpoint.max_peeq_ip),
            source=material,
            evidence_sha256={name: digest(run / name) for name in
                ('manifest.json', 'job.sta', 'solver.log',
                 'results/history.csv', 'results/summary.json')},
            numerical_stop_is_burst=False))
        label = run.name.replace('full_allouti2012_', 'A37 ').replace(
            '_percent_od_figure_curve', '%').replace('orientation_gouge_00_adaptive', 'Zhao gouge-only 0°')
        line, = axes[0].plot(pressure.max_peeq, pressure.pressure_mpa, '.-', label=label)
        axes[0].axvline(supported, color=line.get_color(), linestyle=':', alpha=.5)
        axes[1].plot(-pressure.crown_inward_displacement_mm,
                     pressure.pressure_mpa, '.-', label=label)
    axes[0].set_xlabel('Maximum observed PEEQ (dotted: material data extent)')
    axes[1].set_xlabel('Crown outward displacement from initial surface (mm)')
    for ax in axes:
        ax.set_ylabel('Applied pressure (MPa)')
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Converged pressure histories: numerical endpoints are not burst')
    fig.savefig(OUTPUT / 'pressure_histories.png', dpi=180)
    plt.close(fig)
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                 interpretation='Observed secants do not establish a limit point; material strain extent is not a fracture criterion.',
                 records=records)
    (OUTPUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(f'Audited {len(records)} failed/interrupted pressure histories; frozen inputs and STA times verified.')


if __name__ == '__main__':
    main()
