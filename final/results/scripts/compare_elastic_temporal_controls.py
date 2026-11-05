"""Descriptive three-resolution elastic comparison, never a burst acceptance test."""
import json,sys
import numpy as np
from audit_internal_pipe_fields import F
from pipe_sim.model import Model
from pipe_sim.runner import sha256
SCRIPT_HASH=sha256(__file__)
def compare(lanes):
    records=[];states={}
    for lane in lanes:
        p=F/'results/output'/f'elastic_conservative_path_{lane}_20261011T094510Z.json';e=json.loads(p.read_text());assert e['checks_passed']
        r=F/e['run'];assert all(sha256(r/n)==h for n,h in e['input_output_hashes'].items())
        model=Model.load(r/'model.json');n=model.nsets['CROWN'][0];raw=np.memmap(r/'fields.bin',dtype='<f8',mode='r');captured=json.loads((r/'fields.json').read_text())['records'];times=[];crown=[]
        for rec in captured:
            m=rec['fields']['u'];u=raw[m['offset']//8:m['offset']//8+m['count']].reshape((-1,4));times.append(rec['time']);crown.append(-float(u[n-1,2]))
        dt=np.diff([0.]+times);states[lane]=(times,crown)
        forcep=F/'results/output'/f'internal_endpoint_force_componentwise_{lane}_20261011T093005Z.json';force=json.loads(forcep.read_text());assert force['diagnostic_limit_passed'] and force['run']==e['run'] and force['input_output_hashes']==e['input_output_hashes']
        records.append(dict(lane=lane,run=e['run'],accepted_frames=len(captured),minimum_accepted_increment=float(dt.min()),maximum_accepted_increment=float(dt.max()),final_crown_inward_mm=crown[-1],final=e['history'][-1],maximum_conservative_path_potential_difference_n_mm=e['maximum_path_minus_endpoint_potential_n_mm'],endpoint_free_force_residual_n=force['maximum_free_component_residual_n'],path_evidence_sha256=sha256(p),force_evidence_sha256=sha256(forcep)))
    baseline=states['elastic'];comparisons=[]
    for lane in lanes:
        if lane=='elastic':continue
        t,c=states[lane];mask=[i for i,x in enumerate(baseline[0]) if t[0]<=x<=t[-1]]
        delta=np.interp(np.array(baseline[0])[mask],t,c)-np.array(baseline[1])[mask]
        comparisons.append(dict(lane=lane,parent='elastic',overlap_parent_samples=len(mask),parent_samples=len(baseline[0]),maximum_interpolated_crown_difference_mm=float(abs(delta).max()),final_crown_difference_mm=c[-1]-baseline[1][-1],note='Descriptive linear interpolation only within shared pressure extent; no curve shifts/extrapolation.'))
    return dict(records=records,common_pressure_comparisons=comparisons,script_sha256=SCRIPT_HASH,verdict='Conservative elastic identities and componentwise endpoint equilibrium pass; work gaps and actual accepted step sequences reported without a fitted work tolerance.',limitations='Artificial elastic controls isolate numerical path assumptions, not plastic constitutive accuracy or experimental validity. Two/three resolutions do not validate finite-cap/contact/MPC/Abaqus equivalence.')
if __name__=='__main__':
    lanes=sys.argv[1:];r=compare(lanes);out=F/'results/output'/('elastic_temporal_comparison_'+'_'.join(lanes)+'_20261011T094510Z.json');assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
