"""Retained axial formulation comparison; no physical burst inferred."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

FINAL = Path(__file__).resolve().parents[2]
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def report():
    out = FINAL/'results/output/c3d8_20261019/axial'
    out.mkdir(exist_ok=True)
    names = ['orientation_gouge_00_adaptive', 'orientation_gouge_00_adaptive_c3d8_20261020']
    sources, rows, samples, cases = {}, [], [], []
    plan = json.loads((FINAL/'results/config/full_campaign.json').read_text())
    supported = next(j for j in plan['jobs'] if j['id']=='full_zhao2021_p1_adaptive')['entry']['material']['source_supported_plastic_strain_max']
    for name in names:
        run = FINAL/'sym/runs/campaign'/name
        case = json.loads((run/'case.json').read_text()); cases.append(case)
        manifest = json.loads((run/'manifest.json').read_text())
        for f, expected in manifest['files'].items():
            assert digest(run/f)==expected, f
        h = pd.read_csv(run/'results/history.csv')
        assert h.pressure_mpa.is_monotonic_increasing
        summary = json.loads((run/'results/summary.json').read_text())
        status = json.loads((run/'status.json').read_text())
        assert status['state']=='failed' and summary['burst_pressure_mpa'] is None
        assert 'increment size smaller than minimum' in (run/'solver.log').read_text()
        rows.append(dict(case_id=name, last_pressure_mpa=h.pressure_mpa.max(),max_peeq=h.max_peeq.max(),source_supported_peeq=supported,within_source_strain_range=bool(h.max_peeq.max()<=supported),criterion_pressure_mpa=None,physical_burst_accepted=False,solver_sha256=status['solver_sha256']))
        for pressure in (5.,10.,15.,18.,18.9):
            assert h.pressure_mpa.min()<=pressure<=h.pressure_mpa.max()
            samples.append(dict(case_id=name,pressure_mpa=pressure,**{col:float(np.interp(pressure,h.pressure_mpa,h[col])) for col in ('max_peeq','crown_inward_displacement_mm')}))
        for f in ('case.json','manifest.json','status.json','solver.log','job.sta','results/history.csv','results/summary.json'):
            sources[str((run/f).relative_to(FINAL))]=digest(run/f)
    for key in ('pipe','gouge','dent','material','analysis','probe_positions'):
        assert cases[0][key]==cases[1][key],key
    assert {k:v for k,v in cases[0]['mesh'].items() if k!='order'}=={k:v for k,v in cases[1]['mesh'].items() if k!='order'}
    assert [c['mesh']['order'] for c in cases]==[2,1]
    assert len({r['solver_sha256'] for r in rows})==1
    binary=FINAL/'sym/vendor/ccx_pardiso/ccx_2.23_pardiso'
    assert digest(binary)==rows[0]['solver_sha256']
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(samples).to_csv(out/'common_pressure.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(dict(sources=sources,checks_passed=True,interpolation='Linear between accepted frames; no extrapolation.',decision='No criterion crossing or physical benefit established. Retain response evidence; no angle expansion, unchanged rerun or mesh sweep.',limitations='C3D8 and C3D20R are different formulations; same nominal mesh is not mesh convergence. No physical gouge-only test identified for this control; Zhao P1 dented test is not its burst reference.'),indent=2)+'\n')
    print(pd.DataFrame(rows).to_string(index=False)); print(pd.DataFrame(samples).to_string(index=False))
if __name__=='__main__': report()
