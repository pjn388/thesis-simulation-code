"""Aggregate immutable plastic attribution at shared accepted pressures only."""
import json
import sys
from pathlib import Path
from pipe_sim.runner import sha256
from compare_native_plastic_time import F, compare


def summarize(history, lo, hi):
    rows = [h for h in history if lo + 1e-10 < h['pressure_mpa'] <= hi + 1e-10]
    plastic = [h for h in rows if h['plastic']]
    return dict(frames=len(rows), plastic_frames=len(plastic),
        reference_n_mm=sum(h['reference_increment_n_mm'] for h in rows),
        integrated_peeq_mm3=sum(h['volume_integrated_peeq_increment_mm3'] for h in rows),
        squared_peeq_mm3=sum(h['volume_integrated_squared_peeq_increment_mm3'] for h in rows),
        max_peeq_increment=max((h['maximum_peeq_increment'] for h in rows), default=0),
        max_active_volume_fraction=max((h['active_volume_fraction'] for h in rows), default=0))


def audit():
    source = F/'results/output/plastic_increment_attribution_20261011T153002Z.json'
    data = json.loads(source.read_text())
    assert data['identity_checks_passed'] and not data['physical_acceptance']
    assert sha256(F/'results/scripts/audit_plastic_increment_attribution.py') == data['script_sha256']
    assert all(sha256(F/'results/scripts'/n) == h for n,h in data['dependencies'].items())
    assert sha256(F/'sym/vendor/CalculiX/ccx_2.23/src/incplas.f') == data['source_sha256']
    comparison = compare()
    assert comparison['provenance_checks_passed']
    histories = [lane['history'] for lane in data['lanes']]
    pressures = [0.] + [h['pressure_mpa'] for h in data['shared_history']]
    intervals = []
    for lo, hi in zip(pressures, pressures[1:]):
        lanes = [summarize(h, lo, hi) for h in histories]
        intervals.append(dict(lower_mpa=lo, upper_mpa=hi, lanes=lanes,
            reference_difference_n_mm=lanes[1]['reference_n_mm']-lanes[0]['reference_n_mm']))
    for i,lane in enumerate(data['lanes']):
        for key, endpoint in [('reference_n_mm','accumulated_reference_n_mm'),
                              ('integrated_peeq_mm3','integrated_final_peeq_mm3'),
                              ('squared_peeq_mm3','accumulated_squared_peeq_increment_mm3')]:
            assert abs(sum(r['lanes'][i][key] for r in intervals)-lane['endpoint'][endpoint]) < .01
        assert sum(r['lanes'][i]['frames'] for r in intervals) == len(histories[i])
    total = sum(r['reference_difference_n_mm'] for r in intervals)
    assert abs(total-data['shared_history'][-1]['accumulated_reference_difference_n_mm']) < .01
    ranked = sorted(intervals, key=lambda r: abs(r['reference_difference_n_mm']), reverse=True)
    return dict(identity_checks_passed=True,physical_acceptance=False,
        attribution_sha256=sha256(source),script_sha256=sha256(__file__),
        provenance=comparison['gate_hashes'],runs=comparison['runs'],
        shared_intervals=len(intervals),endpoint_difference_n_mm=total,
        absolute_interval_difference_n_mm=sum(abs(r['reference_difference_n_mm']) for r in intervals),
        top_intervals=ranked[:5],intervals=intervals,
        limitations='Shared accepted endpoints only; no state interpolation. Aggregated reference and squared PEEQ increments are descriptive, not physical work or causation. No new solve, load/material correction, physical energy, convergence or cap/contact/MPC/Abaqus acceptance.')


if __name__ == '__main__':
    out=Path(sys.argv[1]); assert not out.exists()
    result=audit(); out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('intervals','limitations')},indent=2))
