"""Accumulated isotropic coupon replay and assumed affine-F path sensitivity."""
import json
import sys
import numpy as np
from audit_plastic_return_map import update, F, tensors, sha256


def audit(lane="standard"):
    gate=F/'results/output/plastic_return_map_coupon_20261011T100722Z.json'
    assert json.loads(gate.read_text())['checks_passed']
    assert lane in ('standard','half')
    run=F/'sym/runs/investigation'/('plastic_state_coupon_20261011T095613Z' if lane=='standard' else 'plastic_state_coupon_half_20261011T100722Z')
    evidence_path=F/'results/output'/('plastic_state_stress_coupon_20261011T100320Z.json' if lane=='standard' else 'plastic_state_coupon_half_20261011T100722Z_audit.json')
    evidence=json.loads(evidence_path.read_text())
    assert evidence['checks_passed']
    assert all(sha256(run/n)==v for n,v in evidence['hashes'].items())
    lines=(run/'job.inp').read_text().splitlines();curve=[]
    for line in lines[lines.index('*PLASTIC')+1:]:
        if line.startswith('*'):break
        curve.append(list(map(float,line.split(','))))
    curve=np.array(curve)
    raw=np.memmap(run/'fields.bin',dtype='<f8',mode='r')
    records=json.loads((run/'fields.json').read_text())['records']
    xyz=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    frames=[]
    for rec in records:
        def field(name,width):
            m=rec['fields'][name];return raw[m['offset']//8:m['offset']//8+m['count']].reshape((-1,width)) if name=='u' else raw[m['offset']//8:m['offset']//8+m['count']].reshape((1,8,width))
        coef=np.linalg.lstsq(np.c_[np.ones(8),xyz],field('u',4)[:,1:],rcond=None)[0]
        deformation=np.eye(3)+coef[1:].T;inv=np.linalg.inv(deformation)
        frames.append(dict(F=deformation,metric=np.eye(3)-2*tensors(field('state',rec['states'])[...,1:7]),peeq=field('state',rec['states'])[...,0],stress=np.linalg.det(deformation)*(inv@tensors(field('stress',6))@inv.T),energy=float(field('energy',2)[...,0].mean())))
    results=[]
    for count in (1,2,4,8,16):
        metric=np.broadcast_to(np.eye(3),(1,8,3,3)).copy();peeq=np.zeros((1,8))
        previous_f=np.eye(3);previous_e=np.zeros((3,3));previous_s=np.zeros((1,8,3,3));work=0.
        errors=dict(stress_mpa=0.,metric=0.,peeq=0.)
        for frame in frames:
            for step in range(1,count+1):
                deformation=previous_f+(frame['F']-previous_f)*(step/count)
                c=deformation.T@deformation;e=(c-np.eye(3))/2
                stress,metric,peeq,_=update(np.broadcast_to(c,(1,8,3,3)),metric,peeq,curve,202500/(2*1.3),202500/(3*.4))
                work+=float((.5*np.einsum('...ij,ij->...',stress+previous_s,e-previous_e)).mean())
                previous_s,previous_e=stress,e
            for key,value in dict(stress_mpa=float(abs(stress-frame['stress']).max()),metric=float(abs(metric-frame['metric']).max()),peeq=float(abs(peeq-frame['peeq']).max())).items():errors[key]=max(errors[key],value)
            previous_f=frame['F']
        results.append(dict(substeps_per_native_increment=count,work_n_mm=work,maximum_native_endpoint_differences=errors,final_peeq=float(peeq.max())))
    limits=dict(stress_mpa=1e-4,metric=1e-8,peeq=1e-8)
    passed=all(results[0]['maximum_native_endpoint_differences'][k]<v for k,v in limits.items()) and abs(results[0]['work_n_mm']-frames[-1]['energy'])<1e-8
    assert all(sha256(run/n)==v for n,v in evidence['hashes'].items())
    result=dict(lane=lane,field_evidence_sha256=sha256(evidence_path),checks_passed=passed,accepted_frames=len(frames),baseline_fixed_limits=limits,native_final_energy_n_mm=frames[-1]['energy'],native_energy_replay_error_n_mm=abs(results[0]['work_n_mm']-frames[-1]['energy']),resolutions=results,gate_sha256=sha256(gate),return_map_script_sha256=sha256(F/'results/scripts/audit_plastic_return_map.py'),script_sha256=sha256(__file__),limitations='Only one-substep accumulated identity is an acceptance gate. F is assumed linear between retained endpoints; subdivisions change plastic state evolution. Work sensitivity is not native temporal convergence or physical plastic dissipation. No fixture/cap/contact/MPC/Abaqus or experimental validation.')
    out=F/'results/output'/('plastic_coupon_substeps_20261011T100722Z.json' if lane=='standard' else 'plastic_coupon_substeps_half_20261011T100722Z.json');assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert passed


if __name__=='__main__':audit(sys.argv[1] if len(sys.argv)>1 else 'standard')
