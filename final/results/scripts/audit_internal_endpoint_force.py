"""Independent total-Lagrangian endpoint equilibrium on captured pipe fields."""
import json,sys
import numpy as np
from audit_internal_pipe_fields import F,spatial,tensors
from pipe_sim.model import Model
from pipe_sim.runner import sha256
from audit_end_surface_traction import shape8,face_nodes

def audit(lane):
    evidence=F/'results/output'/f'internal_pipe_work_verified_{lane}_20261011T093005Z.json'
    report=json.loads(evidence.read_text());assert report['field_checks_passed'];r=F/report['run']
    assert all(sha256(r/n)==h for n,h in report['input_output_hashes'].items())
    model=Model.load(r/'model.json');inside=False
    for line in (r/'mesh.inp').read_text().splitlines():
        if line.startswith('**'):continue
        if line.startswith('*'):inside=line.upper().strip()=='*NODE';continue
        if inside and line.strip():
            ns=line.split(',');model.nodes[int(ns[0])]=list(map(float,ns[1:4]))
    ids,nodes,grad,weights,checks=spatial(model);rec=json.loads((r/'fields.json').read_text())['records'][-1];assert abs(rec['time']-1)<1e-12
    raw=np.memmap(r/'fields.bin',dtype='<f8',mode='r')
    def field(name,width):
        m=rec['fields'][name];data=raw[m['offset']//8:m['offset']//8+m['count']]
        return data.reshape((-1,width)) if name=='u' else data.reshape((rec['elements'],rec['ips'],width))[np.array(ids)-1,:8]
    u=field('u',4)[:,1:];deformation=np.eye(3)+np.einsum('eni,epnj->epij',u[nodes-1],grad);inverse=np.linalg.inv(deformation);J=np.linalg.det(deformation)
    sigma=tensors(field('stress',6));S=J[...,None,None]*(inverse@sigma@np.swapaxes(inverse,-1,-2));P=deformation@S
    elementforces=np.einsum('epij,epnj,ep->eni',P,grad,weights);internal=np.zeros_like(u);np.add.at(internal,nodes-1,elementforces)
    deck=(r/'job.inp').read_text();assert not any(k in deck for k in ('*EQUATION','*MPC','*CONTACT','*CLOAD'))
    loads=deck.split('*DLOAD\n')[1].split('*',1)[0];magnitudes={(int(e),int(f.strip()[1:])):float(p) for e,f,p in (line.split(',') for line in loads.strip().splitlines())}
    q,w=np.polynomial.legendre.leggauss(2);external=np.zeros_like(u);surface_totals={}
    for surface in ('INNER','END_MINUS','END_PLUS'):
        faces=[(face_nodes(model,e,f),magnitudes[e,f]) for e,f in model.surfaces[surface]];ns=np.array([n for n,p in faces]);pressure=np.array([-p for n,p in faces])
        points=np.array([[model.nodes[int(n)]+u[n-1] for n in face] for face in ns]);force=np.zeros_like(points)
        for a,wa in zip(q,w):
            for b,wb in zip(q,w):
                N=shape8(a,b);da=shape8(a+1e-30j,b).imag/1e-30;db=shape8(a,b+1e-30j).imag/1e-30
                normal=np.cross(np.einsum('n,fni->fi',da,points),np.einsum('n,fni->fi',db,points))
                force+=wa*wb*np.einsum('f,n,fi->fni',pressure,N,normal)
        np.add.at(external,ns-1,force);surface_totals[surface]=force.sum(axis=(0,1)).tolist()
    pins=json.loads((F/'results/output/plastic_tangential_work_p24_20261011T070440Z_terminal.json').read_text())['pins'];free=np.ones_like(u,dtype=bool)
    for n,d in pins:free[n-1,d]=False
    residual=internal-external;free_residual=np.where(free,abs(residual),0);index=np.unravel_index(np.argmax(free_residual),free_residual.shape)
    maximum=float(free_residual[index]);scale=max(float(abs(internal).max()),float(abs(external).max()))
    # Retain original force allowance as a diagnostic; no fitted tolerances.
    limits=1e-4+2e-5*np.maximum(abs(internal),abs(external));limit=float(limits[index]);excess=np.where(free,free_residual-limits,-np.inf);worst=float(excess.max())
    assert all(sha256(r/n)==h for n,h in report['input_output_hashes'].items())
    return dict(lane=lane,run=report['run'],pressure_mpa=24.,maximum_free_component_residual_n=maximum,maximum_residual_node=int(index[0]+1),maximum_residual_component=int(index[1]+1),maximum_nodal_force_n=scale,diagnostic_absolute_plus_relative_limit_n=limit,diagnostic_limit_passed=worst<=0,maximum_componentwise_allowance_excess_n=worst,internal_resultant_n=internal.sum(axis=0).tolist(),external_resultant_n=external.sum(axis=0).tolist(),external_resultant_by_surface_n=surface_totals,constrained_net_internal_minus_external_n=residual[~free].tolist(),work_evidence_sha256=sha256(evidence),input_output_hashes=report['input_output_hashes'],script_sha256=sha256(__file__),sources=['resultsmech.f:1130-1188 reference-volume total-Lagrangian internal force','shape20h.f','gauss.f:95-110,324','retained surface load coupons/audits'],verdict='Independent endpoint force equilibrium quantified on actual serialized geometry. Endpoint equilibrium cannot establish nonlinear path work convergence or validate cap/contact/MPC/Abaqus transfer.',limitations='This intact diagnostic has no contact/MPC/CLOAD; constrained net force requires native load equivalence before general support-transfer acceptance. Absolute/relative limit is diagnostic only, not native convergence criterion or burst.')
if __name__=='__main__':
    report=audit(sys.argv[1]);out=F/'results/output'/f'internal_endpoint_force_componentwise_{sys.argv[1]}_20261011T093005Z.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('input_output_hashes','constrained_net_internal_minus_external_n')},indent=2))
