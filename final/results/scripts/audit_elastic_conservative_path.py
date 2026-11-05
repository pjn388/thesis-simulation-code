"""Exact conservative elastic work along assumed linear output U; no native rerun."""
import json,sys
import numpy as np
from audit_internal_pipe_fields import F,spatial
from pipe_sim.model import Model
from pipe_sim.runner import sha256
SCRIPT_HASH=sha256(__file__)

def audit(lane):
    assert lane in ('elastic','elastic_quarter','elastic_half')
    evidence=F/'results/output'/f'internal_pipe_work_verified_{lane}_20261011T093005Z.json'
    report=json.loads(evidence.read_text());assert report['field_checks_passed']
    r=F/report['run'];hashes=report['input_output_hashes']
    assert all(sha256(r/n)==h for n,h in hashes.items())
    case=json.loads((r/'case.json').read_text());assert case['material']['plastic']==[]
    model=Model.load(r/'model.json');inside=False
    for line in (r/'mesh.inp').read_text().splitlines():
        if line.startswith('**'):continue
        if line.startswith('*'):inside=line.upper().strip()=='*NODE';continue
        if inside and line.strip():
            tokens=line.split(',');model.nodes[int(tokens[0])]=list(map(float,tokens[1:4]))
    ids,nodes,grad,weights,checks=spatial(model)
    young=case['material']['young'];nu=case['material']['poisson'];mu=young/(2*(1+nu));lam=young*nu/((1+nu)*(1-2*nu))
    raw=np.memmap(r/'fields.bin',dtype='<f8',mode='r');records=json.loads((r/'fields.json').read_text())['records']
    previous=np.broadcast_to(np.eye(3),(len(ids),8,3,3)).copy();work=0.;maximum=0.;history=[]
    for rec,h in zip(records,report['history']):
        assert rec['time']==h['time']
        m=rec['fields']['u'];u=raw[m['offset']//8:m['offset']//8+m['count']].reshape((-1,4))[:,1:]
        current=np.eye(3)+np.einsum('eni,epnj->epij',u[nodes-1],grad);delta=current-previous
        # F linear => E quadratic, S(E) quadratic, dE linear: 2pt Gauss exact cubic work.
        increment=0.
        for a,w in zip((.5-.5/np.sqrt(3),.5+.5/np.sqrt(3)),(.5,.5)):
            deformation=previous+a*delta
            E=(np.swapaxes(deformation,-1,-2)@deformation-np.eye(3))/2
            S=lam*np.trace(E,axis1=-2,axis2=-1)[...,None,None]*np.eye(3)+2*mu*E
            dE=(np.swapaxes(deformation,-1,-2)@delta+np.swapaxes(delta,-1,-2)@deformation)/2
            increment+=w*float((np.einsum('epij,epij->ep',S,dE)*weights).sum())
        work+=increment
        E=(np.swapaxes(current,-1,-2)@current-np.eye(3))/2
        exact=float(((.5*lam*np.trace(E,axis1=-2,axis2=-1)**2+mu*np.einsum('epij,epij->ep',E,E))*weights).sum())
        difference=work-exact;maximum=max(maximum,abs(difference))
        history.append(dict(time=rec['time'],conservative_path_work_n_mm=work,endpoint_potential_n_mm=exact,path_minus_potential_n_mm=difference,path_minus_native_n_mm=work-h['internal_energy_n_mm'],external_minus_conservative_path_n_mm=sum(h['external_work_by_surface_n_mm'].values())-work,linear_stress_path_minus_conservative_path_n_mm=h['linear_output_path_material_work_n_mm']-work))
        previous=current
    assert len(history)==len(report['history']) and maximum<1e-5
    assert all(sha256(r/n)==h for n,h in hashes.items())
    return dict(lane=lane,run=report['run'],accepted_frames=len(history),checks_passed=True,maximum_path_minus_endpoint_potential_n_mm=maximum,history=history,field_evidence_sha256=sha256(evidence),input_output_hashes=hashes,script_sha256=SCRIPT_HASH,spatial_checks=checks,sources=report['sources'],verdict='Actual elastic S(E) integrated along linear U output paths equals conservative endpoint potential. Linear interpolation of endpoint stress is a different constitutive path; external-minus-native gap remains an assumed trajectory diagnostic, not evidence of physical energy imbalance.',limitations='Native equilibrium trajectory between outputs is not observed. No plastic constitutive path, pressure cap/contact/MPC/Abaqus or experimental validation. Does not independently prove external-load path correctness.')
if __name__=='__main__':
    report=audit(sys.argv[1]);out=F/'results/output'/f'elastic_conservative_path_{sys.argv[1]}_20261011T094510Z.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('history','input_output_hashes')}|{'final':report['history'][-1]},indent=2))
