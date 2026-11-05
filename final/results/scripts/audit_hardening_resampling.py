"""Read-only reproduction of CCX plcopy's >200-point hardening resampling."""
import json
from pathlib import Path

import numpy as np
from pipe_sim.runner import sha256

FINAL = Path(__file__).resolve().parents[2]


def resample(table):
    table = np.asarray(table, dtype=float)
    assert np.all(np.diff(table[:, 1]) > 0)
    if len(table) <= 200:
        return table.copy()
    strain = np.linspace(table[0, 1], table[-1, 1] - 1e-10, 200)
    return np.column_stack((np.interp(strain, table[:, 1], table[:, 0]), strain))


def main():
    records = []
    for name in ['orientation_intact_00_adaptive', 'orientation_gouge_00_adaptive',
                 'orientation_gouge_90_adaptive', 'full_allouti2012_3_percent_od_figure_curve']:
        run = FINAL/'sym/runs/campaign'/name
        manifest = json.loads((run/'manifest.json').read_text())
        assert all(sha256(run/n) == h for n, h in manifest['files'].items())
        case = json.loads((run/'case.json').read_text())
        original = np.asarray(case['material']['plastic'])
        effective = resample(original)
        # Difference between two piecewise-linear laws attains its extrema
        # at their combined knots. This covers the whole table, not a grid.
        knots = np.unique(np.concatenate((original[:, 1], effective[:, 1])))
        raw = np.interp(knots, original[:, 1], original[:, 0])
        internal = np.interp(knots, effective[:, 1], effective[:, 0])
        i = int(np.argmax(abs(internal-raw)))
        records.append(dict(case_id=name, input_points=len(original),
                            internal_points=len(effective),
                            max_absolute_flow_stress_difference_mpa=float(abs(internal-raw)[i]),
                            strain_at_max_difference=float(knots[i]),
                            max_relative_flow_stress_difference=float(np.max(abs(internal-raw)/raw)),
                            internal_table_stress_plastic_strain=effective.tolist(),
                            frozen_input_sha256=manifest['files']))
    sources = ['sym/vendor/CalculiX/ccx_2.23/src/plcopy.f',
               'sym/vendor/CalculiX/ccx_2.23/src/plinterpol.f',
               'sym/vendor/CalculiX/ccx_2.23/src/materialdata_me.f',
               'sym/vendor/CalculiX/ccx_2.23/src/statics.f',
               'sym/vendor/CalculiX/ccx_2.23/doc/ccx/node350.html']
    result = dict(records=records, source_sha256={p:sha256(FINAL/p) for p in sources},
                  script_sha256=sha256(Path(__file__)),
                  limitations='Source-based plcopy reproduction, not an independently executed Fortran material coupon. No temperature interpolation, tangent convergence, failure cause, burst or experimental validation established. No inputs changed. Effective law may smooth local plateau/tangent changes despite small stress error.')
    output = FINAL/'results/output/pressure_restraint_candidates/hardening_resampling.json'
    output.write_text(json.dumps(result, indent=2)+'\n')
    for r in records:
        print(r['case_id'], r['input_points'], r['internal_points'],
              r['max_absolute_flow_stress_difference_mpa'], r['max_relative_flow_stress_difference'])


if __name__ == '__main__':
    main()
